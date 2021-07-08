import numpy as np
import pickle
import torch
from tqdm import tqdm

class AutoDataLoader():
    def __init__(self, dataloader):
        self.loader = dataloader
        self.loader_iterator = iter(self.loader)

    def __call__(self):
        try:
            data = next(self.loader_iterator)
        except:
            self.loader_iterator = iter(self.loader)
            data = next(self.loader_iterator)
        return data
        
class DataSet(torch.utils.data.Dataset):

    def __init__(self,
                 data_path: str,
                 label_path: str,
                 frame_path: str):
        self.data_path = data_path
        self.label_path = label_path
        self.frame_path = frame_path
        self.load_data()
        

    def load_data(self):
        if self.frame_path:
            self.frame = np.load(self.frame_path)

        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        self.data = np.load(self.data_path)

        self.size, self.max_frame, self.feature_dim = self.data.shape


    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple:
        data = np.array(self.data[index])
        label = self.label[index]
        if hasattr(self, 'frame'):
            frame = self.frame[index]
        else:
            frame = self.max_frame
        return data, label, frame

class NormDataSet(DataSet):
    def __init__(self,
                 data_path: str,
                 label_path: str,
                 frame_path: str):
        self.data_path = data_path
        self.label_path = label_path
        self.frame_path = frame_path
        self.eps = 1e-10
        self.init_joint_map()
        self.load_data()
        self.origin_transfer()
        self.normalize()
        self.default_rotate()

    def init_joint_map(self):
        self.joint_map = {'torso':1, 'left_hip': 13, 'right_hip': 17, 
         'shoulder_center':3, 'spine':2, 'left_shoulder':5, 'right_shoulder':9}
        self.origins = [1, 13, 17]
        for joint in self.joint_map:
            self.joint_map[joint] = self.joint_map[joint] - 1
        for i in range(len(self.origins)):
            self.origins[i] = self.origins[i] - 1
        
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
