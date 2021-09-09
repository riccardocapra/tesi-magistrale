import pykitti
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageMath
from skimage import io, transform


def calib():
    print("Cam left grey:")
    f = open(basedir + "sequences/00/calib.txt", "r")
    Lines = f.readlines()
    print(Lines[3])
    cam00Param = Lines[3].split()

    print(cam00Param[1] + " " + cam00Param[2] + " " + cam00Param[3])
    print(cam00Param[4] + " " + cam00Param[5] + " " + cam00Param[6])
    print(cam00Param[7] + " " + cam00Param[8] + " " + cam00Param[9])
    # fx = cam00Param[1]
    # fy = cam00Param[5]
    # cx = cam00Param[3]
    # cy = cam00Param[6]

    camera_matrixL = np.array([[cam00Param[1], cam00Param[2], cam00Param[3]],
                               [cam00Param[4], cam00Param[5], cam00Param[6]],
                               [cam00Param[1], cam00Param[8], cam00Param[9]]])
    f.close

    print("\n Cam right grey:")
    f = open(basedir + "sequences/00/calib.txt", "r")
    Lines = f.readlines()
    print(Lines[11])
    cam01Param = Lines[11].split()

    print(cam01Param[1] + " " + cam01Param[2] + " " + cam01Param[3])
    print(cam01Param[4] + " " + cam01Param[5] + " " + cam01Param[6])
    print(cam01Param[7] + " " + cam01Param[8] + " " + cam01Param[9])
    # fx = cam01Param[1]
    # fy = cam01Param[5]
    # cx = cam01Param[3]
    # cy = cam01Param[6]

    camera_matrixR = np.array([[cam01Param[1], cam01Param[2], cam01Param[3]],
                               [cam01Param[4], cam01Param[5], cam01Param[6]],
                               [cam01Param[1], cam01Param[8], cam01Param[9]]])
    f.close

    return


# We need to pass to regNet a rgb and a velo flow
def data_formatter(basedir):
    sequence = '00'
    dataset = pykitti.odometry(basedir, sequence)
    velo = dataset.velo_files
    rgb_files = dataset.cam2_files
    to_tensor = transforms.ToTensor()
    normalization = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])

    # c=0m
    rgb = []
    rgb_img = Image.open(rgb_files[0])
    print(rgb_img)
    rgb = to_tensor(rgb_img)
    print(rgb.shape)
    rgb = normalization(rgb)
    # for img in rgb_files:
    #    rgb_img = Image.open(img)
    #    rgb_img = to_tensor(rgb_img)
    #        rgb_img = normalization(rgb_img)
    #    rgb.append(rgb_img)
    # print(c)
    # c+=1
    return rgb, velo[0]


def dataset_construction(rgb, lidar):
    sample = {'rgb': rgb.float(), 'lidar': lidar.float()}
    return sample
