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


device = torch.device("cuda:1")

model = RegNet()
model = model.to(device)

model.load_state_dict(torch.load("./models/model.pt"))
model.eval()
