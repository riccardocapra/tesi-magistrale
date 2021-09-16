from regNet import RegNet
from utils import data_formatter, get_calib
import torch

device = torch.device("cuda")
# Specify the dataset to load
basedir = '/media/RAIDONE/DATASETS/KITTI/ODOMETRY/'
get_calib(basedir)
imageTensor, lidar_tensor=data_formatter(basedir)
model = RegNet()
model.train()
imageTensor2 = imageTensor[:,:1,:,:]
#transl, rot = model(imageTensor, lidar_tensor)
