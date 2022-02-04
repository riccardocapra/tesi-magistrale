import torch
from regNet import RegNet
import pykitti
import numpy as np
import utils
import cv2
from PIL import Image
import cupy

# from dataset import RegnetDataset
# import cupy
# from scipy.spatial.transform import Rotation as R
# from dataset import RegnetDataset
# from math import radians
# import random
print(torch.cuda.is_available())
model = RegNet()
device = torch.device("cuda:0")

basedir = '/media/RAIDONE/DATASETS/KITTI/ODOMETRY/'

sequence_train = ["00", "02", "03", "04", "05", "06", "07"]
sequence_test = ["08", "09"]
dataset = pykitti.odometry(basedir, "00")
idx = 0
# print("---- VELO_IMAGE "+str(idx)+" FORMATTING BEGUN ---")
depth = utils.scan_loader(dataset.velo_files[idx]).T
# print(velo_files[idx])
# Find the sequence number in the file name
# print(sequence)
# print(path[-1])
# depth = dataset.get_velo(idx).T
h_init_ = np.copy(dataset.calib.T_cam2_velo)
depth_n = utils.pcl_rt(depth, h_init_, dataset.calib.K_cam2)
h, w = 352, 1216
depth_image = utils.depth_image_creation(depth_n, h, w)
cv2.imwrite('./images/Original.jpeg', depth_image)

# perturbation_vector = [0, 0, 45]

rot_error = [0,0,0]
tr_error = [0,0,0]

new_h_init = utils.perturbation(h_init_, rot_error, tr_error)

# depth_p = utils.pcl_rt(depth, new_h_init, dataset.calib.K_cam2)
depth_p = utils.pcl_rt(depth, new_h_init, dataset.calib.K_cam2)
depth_image_p = utils.depth_image_creation(depth_p, h, w)
cv2.imwrite('./images/Perturbated.jpeg', depth_image_p)

rgb_img = Image.open(dataset.cam2_files[idx])
rgb_img_cropped = rgb_img.crop((0, 0, 1216, 352))
rgb_img_cropped.save("./images/camera.jpg")
# print("---- VELO_IMAGE FORMATTING ENDED ---")
to_tensor = utils.transforms.ToTensor()
depth_image_tensor = to_tensor(depth_image_p)