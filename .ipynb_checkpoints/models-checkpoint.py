import torch.nn as nn
import torch
from torchvision import models
from sys import path
import torch.nn.functional as F
import numpy as np
from net.st_gcn import Model as GCN

class ENC(nn.Module):
    def __init__(self, input_size, frame):
        super(ENC, self).__init__()
        self.hidden_size = input_size // 3
        self.input_size = input_size
        self.frame = frame
        self.encoder = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.fc_mean = nn.Linear(self.hidden_size, self.hidden_size)
        # self.fc_mean = nn.GRU(
        #     input_size=self.hidden_size,
        #     hidden_size=self.hidden_size,
        #     num_layers=2,
        #     batch_first=True
        # )
        self.fc_lvar = nn.Linear(self.hidden_size, self.hidden_size)
        # self.fc_lvar = nn.GRU(
        #     input_size=self.hidden_size,
        #     hidden_size=self.hidden_size,
        #     num_layers=2,
        #     batch_first=True
        # )

    def forward(self, input):
        shape = input.shape
        self.encoder.flatten_parameters()
        # self.fc_mean.flatten_parameters()
        # self.fc_lvar.flatten_parameters()
        X, _ = self.encoder(input)
        mean = self.fc_mean(X)
        lvar = self.fc_lvar(X)
        X = torch.sum(X, dim = 1)
        mean = torch.mean(mean, dim = 1)
        lvar = torch.mean(lvar, dim = 1)
        eps = torch.randn(X.shape).cuda()
        Z = mean + eps * torch.exp(lvar / 2)
        return Z, mean, lvar

class NSY(nn.Module):
    def __init__(self, input_size):
        super(NSY, self).__init__()
        self.hidden_size = input_size // 3
        self.noisy_mean = nn.Linear(self.hidden_size, self.hidden_size)
        self.noisy_lvar = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, mean, lvar):
        mean = self.noisy_mean(mean)
        lvar = self.noisy_lvar(lvar)
        eps = torch.randn(mean.shape).cuda()
        Z = mean + eps * torch.exp(lvar / 2)
        Z = torch.clamp(Z, min=-0.1, max=0.1)
        return Z, mean, lvar
    

class CLS(nn.Module):
    def __init__(self, input_size, num_label, scale=None): 
        super(CLS, self).__init__()
        self.hidden_size = 600
        self.scale = scale
        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=self.hidden_size//2,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.decoder = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            batch_first=True
        )
        self.classifier = nn.Linear(self.hidden_size, num_label)
        self.fc_mean = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_lvar = nn.Linear(self.hidden_size, self.hidden_size)


    def forward(self, input):
        self.encoder.flatten_parameters()
        self.decoder.flatten_parameters()
        X, _ = self.encoder(input)
        mean = self.fc_mean(X)
        lvar = self.fc_lvar(X)
        eps = torch.randn(X.shape).cuda()
        X = mean + eps * torch.exp(lvar / 2)
        if self.scale != 0.0:
            X, _ = self.decoder(X)
        X = torch.sum(X, dim = 1)
        X = self.classifier(X)
        return X, mean, lvar

class CTR(nn.Module):
    def __init__(self, cls):
        super(CTR, self).__init__()
        self.cls = cls
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.projection = nn.Linear(self.cls.module.hidden_size, self.cls.module.hidden_size // 2)
        self.attention = nn.Linear(self.cls.module.hidden_size // 2, self.cls.module.hidden_size // 2)

    def forward(self, input):
        self.cls.module.encoder.flatten_parameters()
        X, _ = self.cls.module.encoder(input)
        X = torch.sum(X, dim = 1)
        X = self.relu(X)
        X = self.projection(X)
        return X
    
    def contrastive(self, input):
        length = len(input)
        delta = [input[i] - input[0] for i in range(1, length)]
        MLP_delta = torch.stack([self.attention(d) for d in delta], 1)
        S = self.softmax(torch.einsum('ijk,ijk->ij', [MLP_delta, MLP_delta]))
        I = torch.stack(input[1::], 1)
        Y = torch.einsum('ij,ijk->ik', [S, I])
        return Y


class DEC(nn.Module):
    def __init__(self, input_size, frame, dataset):
        super(DEC, self).__init__()
        self.frame = frame
        self.input_size = input_size
        self.dataset = dataset
        if self.dataset == "NTU" or self.dataset == "PKUMMD2":
            self.m = 2
        elif self.dataset == "UCLA":
            self.m = 1
        self.decoder = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.input_size,
            num_layers=2,
            batch_first=True
        )
        self.softmax = nn.Softmax(dim=2)
        A = torch.ones(self.frame, self.frame).cuda()
        A = torch.tril(A)
        B = A
        for i in range(self.frame - 1):
            B[i + 1, 0] = 0
        self.register_buffer('A', A)
        self.register_buffer('B', B)
        # self.reconstruction = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
        #                                     nn.Linear(self.hidden_size, self.hidden_size * self.frame))
        if self.dataset == "NTU" or self.dataset == "PKUMMD2":
            self.reconstruction = GCN(1, 3 * self.frame, {'layout': 'ntu-rgb+d', 'strategy': 'spatial'}, False)
        elif self.dataset == "UCLA":
            self.reconstruction = GCN(1, 3 * self.frame, {'layout': 'ucla', 'strategy': 'spatial'}, False)
        # self.attention1 = nn.Linear(self.input_size, self.input_size)
        # self.attention2 = nn.Linear(self.input_size, self.input_size)

    def forward(self, input):
        shape = input.shape
        self.decoder.flatten_parameters()
        input = input.view(shape[0], 1, self.m, -1, 1)
        input = input.permute(0, 4, 1, 3, 2)
        X = self.reconstruction(input)
        X = X.reshape(shape[0], 3, self.frame, -1, self.m)
        X = X.permute(0, 2, 4, 3, 1)
        X = X.reshape(shape[0], self.frame, -1)
        # X = X.view(shape[0], self.frame, self.hidden_size)
        X, _ = self.decoder(X)
        # X = torch.clamp(X, -1, 1)
        # Q = self.attention1(X)
        # K = self.attention2(X)
        Y = torch.einsum('ik,jkl->jil', [self.B, X])
        Z = torch.einsum('ik,jkl->jil', [self.A, Y])
        # S = self.softmax(torch.einsum('ijk,itk->ijt', [Q, K]))
        # Y = torch.einsum('ijt,itk->ijk', [S, V])
        return Z

class DIS(nn.Module):
    def __init__(self, input_size, channel=1):
        super(DIS, self).__init__()
        self.channel = channel
        self.conv = nn.Conv2d(self.channel, 1, kernel_size=1)
        self.discriminator = nn.Linear(input_size, 1)

    def forward(self, input):
        shape = input.shape
        if self.channel == 1:
            input = input.view(shape[0], self.channel, shape[1], shape[2])
        X = self.conv(input)
        X = X.view(shape[0], -1)
        X = self.discriminator(X)
        return X