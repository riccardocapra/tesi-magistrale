from regNet import RegNet
from utils import data_formatter, perturbation
import torch
import pykitti
from dataset import RegnetDataset

device = torch.device("cuda:2")
# Specify the dataset to load
basedir = '/media/RAIDONE/DATASETS/KITTI/ODOMETRY/'
sequence = "00"


dataset = RegnetDataset(basedir, sequence)
print(len(dataset))

# imageTensor, lidar_tensor = data_formatter(basedir)
model = RegNet()
model.train()
# imageTensor2 = imageTensor[:, :1, :, :]
# transl, rot = model(imageTensor[:, :3, :, :1216], lidar_tensor)
