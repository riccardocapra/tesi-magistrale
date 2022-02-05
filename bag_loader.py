import torch
from regNet import RegNet
import pykitti
import numpy as np
import utils
import cv2
from PIL import Image
import cupy
import bagpy
from bagpy import bagreader
# from dataset import RegnetDataset
# import cupy
# from scipy.spatial.transform import Rotation as R
# from dataset import RegnetDataset
# from math import radians
# import random

bag = bagreader('/media/RAIDONE/DATASETS/rosbag/iralab_A4-5_0.bag')
bag.topic_table