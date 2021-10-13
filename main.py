from regNet import RegNet
from utils import data_formatter, perturbation
import torch

device = torch.device("cuda")
# Specify the dataset to load
basedir = '/media/RAIDONE/DATASETS/KITTI/ODOMETRY/'
imageTensor, lidar_tensor = data_formatter(basedir)
model = RegNet()
model.train()
# imageTensor2 = imageTensor[:, :1, :, :]
transl, rot = model(imageTensor[:, :3, :, :1216], lidar_tensor)
