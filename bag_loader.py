import torch
from regNet import RegNet
import numpy as np
from utils import perturbation
import cupy
from scipy.spatial.transform import Rotation as R
from dataset import RegnetDataset
from math import radians
import random
print(torch.cuda.is_available())
model = RegNet()
device = torch.device("cuda:0")
model = model.to(device)


