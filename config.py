import os
from sacred import Experiment

ex = Experiment("MS2L")

@ex.config
def my_config():
    epoch_num = 60
    train_clip = 0.95
    label_clip = 1.0
    clip_gradient = 10
    hidden_size = 600
    temp = 0.9
    lamb = 0.9
    weight_path = None
    train_mode = 'pretrain'
    mask_ratio = 0.3
    seg_num = 3
    gpus = "0"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    dataset = "NTU"
    batch_size = 256
    input_size = 150
    channel_num = 3
    person_num = 2
    joint_num = 25
    label_num = 60
    max_frame = 300
    train_list = './data/NTU/xsub/train_data.npy'
    test_list = './data/NTU/xsub/val_data.npy'
    train_label = './data/NTU/xsub/train_label.pkl'
    test_label = './data/NTU/xsub/val_label.pkl'
    train_frame = './data/NTU/xsub/train_frame.npy'
    test_frame = './data/NTU/xsub/val_frame.npy'
    self = None
# %%
