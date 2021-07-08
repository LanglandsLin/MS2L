# sys
import os
import sys
import numpy as np
import random
import pickle

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from itertools import permutations
from tqdm import tqdm
# visualization
import time


class DataSet(torch.utils.data.Dataset):
    """ Dataset for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 debug=False):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.eps = 1e-10
        self.init_joint_map()
        self.load_data()
        if "UCLA" in self.data_path:
            self.origin_transfer()
            self.normalize()
            self.default_rotate()

    def init_joint_map(self):
        self.joint_map = {'torso':1, 'left_hip': 13, 'right_hip': 17, 
         'shoulder_center':3, 'spine':2, 'left_shoulder':5, 'right_shoulder':9}
        self.origins = [1, 13, 17]
        # all minus one
        for joint in self.joint_map:
            self.joint_map[joint] = self.joint_map[joint] - 1
        for i in range(len(self.origins)):
            self.origins[i] = self.origins[i] - 1


    def load_data(self):
        # data: N C T V M

        # load label
        with open(self.label_path, 'rb') as f:
          self.sample_name, self.label = pickle.load(f)

        # load data
        self.data = np.load(self.data_path)
        if "NTU" in self.data_path:
            self.data = self.data[:, ::5]

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        # if self.model == "train":
        #     self.label = self.label[::10]
        #     self.data = self.data[::10]
        #     self.sample_name = self.sample_name[::10]

        #PKUMMD1
        # self.N, self.C, self.T, self.V, self.M = self.data.shape
        # self.data = np.transpose(self.data, (0, 2, 1, 3, 4))
        # self.data = self.data.reshape(self.N, self.T, 150)
        #PKUMMD1

        self.size, self.max_frame, self.feature_dim = self.data.shape

    def origin_transfer(self):
        data_numpy = np.reshape(self.data, (self.size, self.max_frame, -1, 3))
        origin = np.mean(data_numpy[:, :, self.origins,:], axis = 2, keepdims= True)
        data_numpy = data_numpy - origin
        self.data = np.reshape(data_numpy, (self.size, self.max_frame, self.feature_dim))

    def normalize(self):
        data_numpy = np.reshape(self.data, (self.size, self.max_frame, -1, 3))
        norm = np.linalg.norm(data_numpy, ord = 'fro', axis = (2,3), keepdims= True)
        data_numpy = data_numpy / (norm + self.eps)
        self.data = np.reshape(data_numpy, (self.size, self.max_frame, self.feature_dim))

    def my_rotate(self):
        data_numpy = np.reshape(self.data, (self.size, self.max_frame, -1, 3))
        dst = np.zeros_like(data_numpy)
        for i in tqdm(range(self.size)):
            for j in range(self.max_frame):
                shoulder_center = data_numpy[i,j,self.joint_map['shoulder_center'],:]
                spine = data_numpy[i,j,self.joint_map['spine'], :]
                right_shoulder = data_numpy[i,j, self.joint_map['right_shoulder'],:]
                left_shoulder = data_numpy[i,j, self.joint_map['left_shoulder'],:]

                new_z = shoulder_center - spine
                unit_z = new_z / np.linalg.norm(new_z + self.eps)
                new_x = right_shoulder - left_shoulder
                new_x = new_x - unit_z * np.inner(new_x, unit_z) 
                unit_x = new_x / np.linalg.norm(new_x + self.eps)
                unit_y = - np.cross(unit_x, unit_z)
                unit_y = unit_y / np.linalg.norm(unit_y + self.eps)
                x_axis = unit_x
                y_axis = unit_y
                z_axis = unit_z
                dst[i,j,:,0] = np.inner(data_numpy[i,j,:,:], x_axis)
                dst[i,j,:,1] = np.inner(data_numpy[i,j,:,:], y_axis)
                dst[i,j,:,2] = np.inner(data_numpy[i,j,:,:], z_axis)
        self.data = np.reshape(dst, (self.size, self.max_frame, self.feature_dim))
    def default_rotate(self):
        data_numpy = np.reshape(self.data, (self.size, self.max_frame, -1, 3))
        dst = np.zeros_like(data_numpy)
        for i in tqdm(range(self.size)):
            for j in range(self.max_frame):
                x_axis = data_numpy[i,j, self.joint_map['left_hip'],:]
                x_axis = x_axis / (np.linalg.norm(x_axis) + self.eps)
                y_axis = data_numpy[i,j, self.joint_map['torso'],:]
                y_axis = y_axis - x_axis * np.inner(y_axis,x_axis)
                y_axis = y_axis / (np.linalg.norm(y_axis) + self.eps)
                z_axis = np.cross(x_axis, y_axis)
                z_axis = z_axis / (np.linalg.norm(z_axis) + self.eps)
                dst[i,j,:,0] = np.inner(data_numpy[i,j,:,:], x_axis)
                dst[i,j,:,1] = np.inner(data_numpy[i,j,:,:], y_axis)
                dst[i,j,:,2] = np.inner(data_numpy[i,j,:,:], z_axis)
                    
        self.data = np.reshape(dst, (self.size, self.max_frame, self.feature_dim))

    # to implement
    def quick_rotate(self):
        data_numpy = np.reshape(self.data, (self.size, self.max_frame, -1, 3))
        x_axis = data_numpy[:,:,self.joint_map['left_hip'],:]
        x_axis = x_axis / (np.linalg.norm(x_axis, axis = 2) + self.eps)
        y_axis = data_numpy[:,:,self.joint_map['torso'],:]
        y_axis = y_axis - np.einsum('ijk,ij->ijk',x_axis, np.einsum('ijk,ijk->ij', y_axis, x_axis))
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / (np.linalg.norm(z_axis, axis = 2) + self.eps)  
        M = np.stack([x_axis,y_axis,z_axis], axis = 2) # (size, max_frame, 3(stack), 3)
        #(size, max_frame, joint, 3)
        #(size,max_frame, joint, 3)
        data_numpy = np.einsum('ijkt,ijlk->ijlk', M, data_numpy)
        for i in tqdm(range(self.size)):
            for j in range(self.max_frame):
                x_axis = data_numpy[i,j, self.joint_map['left_hip'],:]
                x_axis = x_axis / (np.linalg.norm(x_axis) + self.eps)
                y_axis = data_numpy[i,j, self.joint_map['torso'],:]
                y_axis = y_axis - x_axis * np.inner(y_axis,x_axis)
                y_axis = y_axis / (np.linalg.norm(y_axis) + self.eps)
                z_axis = np.cross(x_axis, y_axis)
                z_axis = z_axis / (np.linalg.norm(z_axis) + self.eps)
                dst[i,j,:,0] = np.inner(data_numpy[i,j,:,:], x_axis)
                dst[i,j,:,1] = np.inner(data_numpy[i,j,:,:], y_axis)
                dst[i,j,:,2] = np.inner(data_numpy[i,j,:,:], z_axis)
                    
        self.data = np.reshape(data_numpy, (self.size, self.max_frame, self.feature_dim))

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        data = np.array(self.data[index])
        label = self.label[index]
        if "NTU" in self.data_path:
            return data, label
        else:
            return data, label - 1
