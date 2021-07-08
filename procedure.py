from config import *
from model import *
from dataset import DataSet 
from logger import Log

import torch
import torch.nn as nn

from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

class BaseProcessor:

    @ex.capture
    def load_data(self, train_list, train_label, train_frame, test_list, test_label, test_frame, batch_size, train_clip, label_clip):
        self.dataset = dict()
        self.data_loader = dict()
        self.auto_data_loader = dict()

        self.dataset['train'] = DataSet(train_list, train_label, train_frame)

        full_len = len(self.dataset['train'])
        train_len = int(train_clip * full_len)
        val_len = full_len - train_len
        self.dataset['train'], self.dataset['val'] = torch.utils.data.random_split(self.dataset['train'], [train_len, val_len])

        self.data_loader['train'] = torch.utils.data.DataLoader(
            dataset=self.dataset['train'],
            batch_size=batch_size,
            shuffle=False)

        self.data_loader['val'] = torch.utils.data.DataLoader(
            dataset=self.dataset['val'],
            batch_size=batch_size,
            shuffle=False)

        if label_clip != 1.0:
            label_len = int(label_clip * train_len)
            unlabel_len = train_len - label_len
            self.dataset['label'], self.dataset['unlabel'] = torch.utils.data.random_split(self.dataset['train'], [label_len, unlabel_len])

            self.data_loader['label'] = torch.utils.data.DataLoader(
                dataset=self.dataset['label'],
                batch_size=batch_size,
                shuffle=False)

            self.data_loader['unlabel'] = torch.utils.data.DataLoader(
                dataset=self.dataset['unlabel'],
                batch_size=batch_size,
                shuffle=False)
        else:
            self.data_loader['label'] = torch.utils.data.DataLoader(
                dataset=self.dataset['train'],
                batch_size=batch_size,
                shuffle=False)

        self.dataset['test'] = DataSet(test_list, test_label, test_frame)

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=self.dataset['test'],
            batch_size=batch_size,
            shuffle=False)

    def load_weights(self, model=None, weight_path=None):
        if weight_path:
            pretrained_dict = torch.load(weight_path)
            model.load_state_dict(pretrained_dict)

    def initialize(self):
        self.load_data()
        self.load_model()
        self.load_optim()
        self.log = Log()

    @ex.capture
    def optimize(self, epoch_num):
        for epoch in range(epoch_num):
            self.epoch = epoch
            self.train_epoch()
            self.val_epoch()
            self.test_epoch()
            self.log.update_epoch()

    @ex.capture
    def save_model(self, train_mode):
        torch.save(self.encoder.state_dict(), f"output/model/{train_mode}.pt")

    def start(self):
        self.initialize()
        self.optimize()
        self.save_model()

# %%
class RecognitionProcessor(BaseProcessor):

    @ex.capture
    def load_model(self, train_mode, weight_path):
        self.encoder = Encoder()
        self.encoder = torch.nn.DataParallel(self.encoder).cuda()
        self.classifier = Linear()
        self.classifier = torch.nn.DataParallel(self.classifier).cuda()
        if 'loadweight' in train_mode:
            self.load_weights(self.encoder, weight_path)

    @ex.capture
    def load_optim(self):
        self.optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters()},
            {'params': self.classifier.parameters(), 'lr': 1e-3}],lr = 1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        self.CrossEntropyLoss = torch.nn.CrossEntropyLoss().cuda()

    @ex.capture
    def train_epoch(self, clip_gradient):
        self.encoder.train()
        self.classifier.train()
        loader = self.data_loader['label']
        for data, label, frame in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            frame = frame.type(torch.LongTensor).cuda()
            loss = self.train_batch(data, label, frame)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip_gradient)
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), clip_gradient)
            self.optimizer.step()
        self.scheduler.step()

    @ex.capture
    def train_batch(self, data, label, frame, train_mode):
        Z = self.encoder(data)
        if "linear" in train_mode:
            Z = Z.detach()
        Z = mask_mean(Z, frame)
        predict = self.classifier(Z)

        _, pred = torch.max(predict, 1)
        acc = pred.eq(label.view_as(pred)).float().mean()

        cls_loss = self.CrossEntropyLoss(predict, label)
        loss = cls_loss

        self.log.update_batch("log/train/cls_acc", acc.item())
        self.log.update_batch("log/train/cls_loss", loss.item())

        return loss

    def test_epoch(self):
        self.encoder.eval()
        self.classifier.eval()

        loader = self.data_loader['test']
        for data, label, frame in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            frame = frame.type(torch.LongTensor).cuda()
            # inference
            with torch.no_grad():
                Z = self.encoder(data)
                Z = mask_mean(Z, frame)
                predict = self.classifier(Z)
            _, pred = torch.max(predict, 1)
            acc = pred.eq(label.view_as(pred)).float().mean()
            cls_loss = self.CrossEntropyLoss(predict, label)
            loss = cls_loss
            self.log.update_batch("log/test/cls_acc", acc.item())
            self.log.update_batch("log/test/cls_loss", loss.item())

    def val_epoch(self):
        self.encoder.eval()
        self.classifier.eval()

        loader = self.data_loader['val']
        for data, label, frame in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            frame = frame.type(torch.LongTensor).cuda()
            # inference
            with torch.no_grad():
                Z = self.encoder(data)
                Z = mask_mean(Z, frame)
                predict = self.classifier(Z)
            _, pred = torch.max(predict, 1)
            acc = pred.eq(label.view_as(pred)).float().mean()
            cls_loss = self.CrossEntropyLoss(predict, label)
            loss = cls_loss
            self.log.update_batch("log/val/cls_acc", acc.item())
            self.log.update_batch("log/val/cls_loss", loss.item())
            
class MS2LProcessor(BaseProcessor):

    @ex.capture
    def contrastive_loss(self, X, Y, temp):
        shape = X.shape
        X_norm = nn.functional.normalize(X, dim=1)
        Y_norm = nn.functional.normalize(Y, dim=1)

        S12 = X_norm.mm(Y_norm.t())
        S21 = S12.t()
        S11 = X_norm.mm(X_norm.t())
        S22 = Y_norm.mm(Y_norm.t())

        S11[range(shape[0]), range(shape[0])] = -1.
        S22[range(shape[0]), range(shape[0])] = -1.

        S1 = torch.cat([S12, S11], dim = 1)
        S2 = torch.cat([S22, S21], dim = 1)

        S = torch.cat([S1, S2], dim = 0) / temp

        Mask = torch.arange(S.shape[0], dtype=torch.long).cuda()

        _, pred = torch.max(S, 1)
        ctr_acc = pred.eq(Mask.view_as(pred)).float().mean()
        ctr_loss = self.CrossEntropyLoss(S, Mask)

        return ctr_acc, ctr_loss

    def load_model(self):
        self.temp_mask = TemporalMask()
        self.temp_jigsaw = TemporalJigsaw()
        self.encoder = Encoder()
        self.encoder = torch.nn.DataParallel(self.encoder).cuda()
        self.contra_head = Projector()
        self.contra_head = torch.nn.DataParallel(self.contra_head).cuda()
        self.jigsaw_head = Projector(feature_num=self.temp_jigsaw.jig_num)
        self.jigsaw_head = torch.nn.DataParallel(self.jigsaw_head).cuda()
        self.motion_head = Decoder()
        self.motion_head = torch.nn.DataParallel(self.motion_head).cuda()

    def load_optim(self):
        self.optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters()},
            {'params': self.contra_head.parameters(), 'lr': 1e-3},
            {'params': self.jigsaw_head.parameters(), 'lr': 1e-3},
            {'params': self.motion_head.parameters(), 'lr': 1e-3}],lr = 1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        self.CrossEntropyLoss = torch.nn.CrossEntropyLoss().cuda()
        self.MSELoss = torch.nn.MSELoss().cuda()


    def motion_batch(self, data, feat_mask, frame):
        predict = self.motion_head(feat_mask)
        predict = mask_empty_frame(predict, frame)
        mse_loss = self.MSELoss(predict, data)
        self.log.update_batch("log/train/mse_loss", mse_loss.item())
        return  mse_loss

    def jigsaw_batch(self, feat_jigs, label_jigs, frame):
        predict = self.jigsaw_head(mask_mean(feat_jigs, frame))
        jig_loss = self.CrossEntropyLoss(predict, label_jigs)
        _, pred = torch.max(predict, 1)
        jig_acc = pred.eq(label_jigs.view_as(pred)).float().mean()
        self.log.update_batch("log/train/jig_acc", jig_acc.item())
        self.log.update_batch("log/train/jig_loss", jig_loss.item())
        return  jig_loss

    def contra_batch(self, feat, feat_mask, feat_jigs, frame):
        feat = self.contra_head(mask_mean(feat, frame))
        feat_mask = self.contra_head(mask_mean(feat_mask, frame))
        feat_jigs = self.contra_head(mask_mean(feat_jigs, frame))
        feat_mean = (feat + feat_mask + feat_jigs) / 3
        ctr_acc, ctr_loss = zip(*[self.contrastive_loss(feat, feat_mean), self.contrastive_loss(feat_mask, feat_mean), self.contrastive_loss(feat_jigs, feat_mean)])
        ctr_acc = sum(ctr_acc) / len(ctr_acc)
        ctr_loss = sum(ctr_loss) / len(ctr_loss)
        self.log.update_batch("log/train/ctr_acc", ctr_acc.item())
        self.log.update_batch("log/train/ctr_loss", ctr_loss.item())
        return  ctr_loss

    @ex.capture
    def train_epoch(self, clip_gradient, train_mode):
        self.encoder.train()
        loader = self.data_loader['train']

        for data, label, frame in tqdm(loader):
            data = data.type(torch.FloatTensor)
            label = label.type(torch.LongTensor)
            frame = frame.type(torch.LongTensor)
            data_mask = self.temp_mask(data, frame)
            data_jigs, label_jigs = self.temp_jigsaw(data, frame)

            data = data.cuda()
            label = label.cuda()
            frame = frame.cuda()
            data_mask = data_mask.cuda()
            data_jigs = data_jigs.cuda()
            label_jigs = label_jigs.cuda()

            feat = self.encoder(data)
            feat_mask = self.encoder(data_mask)
            feat_jigs = self.encoder(data_jigs)

            loss = self.motion_batch(data, feat_mask, frame) + self.jigsaw_batch(feat_jigs, label_jigs, frame) + self.contra_batch(feat, feat_mask, feat_jigs, frame)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip_gradient)
            torch.nn.utils.clip_grad_norm_(self.motion_head.parameters(), clip_gradient)
            torch.nn.utils.clip_grad_norm_(self.jigsaw_head.parameters(), clip_gradient)
            torch.nn.utils.clip_grad_norm_(self.contra_head.parameters(), clip_gradient)
            self.optimizer.step()

        self.scheduler.step()

    @ex.capture
    def optimize(self, epoch_num):
        for epoch in range(epoch_num):
            self.epoch = epoch
            self.train_epoch()
            self.log.update_epoch()

# %%
@ex.automain
def main(train_mode):

    if "pretrain" in train_mode:
        p = MS2LProcessor()
        p.start()

    if "loadweight" in train_mode:
        p = RecognitionProcessor()
        p.start()
