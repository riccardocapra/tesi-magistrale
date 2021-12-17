import argparse

import utils
from regNet import RegNet
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from dataset import RegnetDataset
import numpy as np
import math
import random
# import pykitti
from scipy.spatial.transform import Rotation as R
device = torch.device("cuda:1")

# Specify the dataset to load
basedir = '/media/RAIDONE/DATASETS/KITTI/ODOMETRY/'
sequence = ["00"]
# Set the rando seed used for the permutations
random.seed(1)

dataset = RegnetDataset(basedir, sequence)
dataset_size = len(dataset)
item = dataset.__getitem__(0)
quat = R.from_euler('zyx', item["rot_error"].tolist(), degrees=True)
dist = utils.quaternion_distance([0,0,0],[1,0,0])

