import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio,os
import os
class Ske2pic():
    def __init__(self, mode):
        if mode == "fake":
            self.data_path = "/home/linlilang/ContrastiveGAN/output/fake_data.npy"
            self.label_path = "/home/linlilang/ContrastiveGAN/output/label.npy"
            self.output_dirs = "/home/linlilang/ContrastiveGAN/visual/fake/picture/{}_{}"
            self.output = "/home/linlilang/ContrastiveGAN/visual/fake/picture/{}_{}/{}.png"
            self.root = "/home/linlilang/ContrastiveGAN/visual/fake/gif"

        if mode == "true":
            self.data_path = "/home/linlilang/ContrastiveGAN/output/data.npy"
            self.label_path = "/home/linlilang/ContrastiveGAN/output/label.npy"
            self.output_dirs = "/home/linlilang/ContrastiveGAN/visual/true/picture/{}_{}"
            self.output = "/home/linlilang/ContrastiveGAN/visual/true/picture/{}_{}/{}.png"
            self.root = "/home/linlilang/ContrastiveGAN/visual/true/gif"

        self.load_data()
    
    def load_data(self):
        self.label = np.load(self.label_path)
        self.data = np.load(self.data_path)
        self.size, self.max_frame, self.feature_dim = self.data.shape
        self.data_numpy = np.reshape(self.data, (self.size, self.max_frame, 2, -1, 3))

    def draw_frame(self, index, time):
        datum = self.data_numpy[index, time, 0, :, :]
        # datum = datum.transpose(1, 0)
        label = self.label[index]
        # np.save("ske_ntu.npy", datum)

        # 创建画布
        x = datum[:,0]
        y = datum[:,1]
        z = datum[:,2]
        fig = plt.figure(time)
        ax = Axes3D(fig)

        ax.scatter(x, y, z)
        for i in range(len(x)):
            ax.text(x[i],y[i],z[i],i + 1)
            
        links = [(3, 2), (2, 20), (20, 1), (1, 0), (0, 12), (0, 16), (12, 13), (13, 14),
                (14, 15), (16, 17), (17, 18), (18, 19), (20, 4), (4, 5), (5, 6), (6, 7), (20, 8), (8, 9), (9, 10), (10, 11)]
        for link in links:
            i = link[0]
            j = link[1]
            ax.plot([x[i],x[j]], [y[i],y[j]], [z[i],z[j]], c='r')

        # ax.view_init(elev=90, azim=-144)

        ax.view_init(elev=-90, azim=40)
        plt.axis('off')
        dirs = self.output_dirs.format(label, index)
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        plt.savefig(self.output.format(label, index, "{0:03d}".format(time)))
        plt.close(time)

    def draw_frames(self, index):
        for i in range(60):
            self.draw_frame(index, i)
        label = self.label[index]
        dirs = self.output_dirs.format(label, index)
        images = []
        filenames=sorted((os.path.join(dirs, fn) for fn in os.listdir(dirs) if fn.endswith('.png')))
        for filename in filenames:
            images.append(imageio.imread(filename))
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        imageio.mimsave(os.path.join(self.root, "{}_{}.gif".format(label, index)), images,duration=0.1)

if __name__ == "__main__":
    # modes = ["original", "mask", "recover"]
    modes = ["fake", "true"]
    for mode in modes:
        ske2pic = Ske2pic(mode)
        for i in range(0, 20):
            ske2pic.draw_frames(i)

