from regNet import RegNet
from pyKItty import data_formatter
import torch

device = torch.device("cuda")
# Specify the dataset to load
basedir = '/media/RAIDONE/DATASETS/KITTI/ODOMETRY/'
imageTensor, lidar_tensor=data_formatter(basedir)
model = RegNet()
model.train()
model(imageTensor, lidar_tensor)
