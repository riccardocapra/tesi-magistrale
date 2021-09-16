import pykitti
import torch
import numpy as np
from torchvision import transforms
from PIL import Image


def data_formatter(basedir):
    sequence = '00'
    dataset = pykitti.odometry(basedir, sequence)
    depth = dataset.get_velo(0)
    # depth = torch.from_numpy(depth / (2 ** 16)).float()
    # Le camere 2 e 3 sono quelle a colori, verificato. Mi prendo la 2.
    rgb_files = dataset.cam2_files
    # print(rgb_files)
    to_tensor = transforms.ToTensor()
    normalization = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # c=0m
    rgb = []
    rgb_img = Image.open(rgb_files[0])
    # print(rgb_img)
    rgb = to_tensor(rgb_img)
    # print("Dimensione tensore: "+str(rgb.shape))
    # rgb = normalization(rgb)
    rgb = rgb.unsqueeze(0)
    return rgb, depth


def get_calib(basedir):
    sequence = '00'
    dataset = pykitti.odometry(basedir, sequence)
    print(dataset.calib.T_cam2_velo)
    print("Cam left color:")
    f = open(basedir + "sequences/00/calib.txt", "r")
    lines = f.readlines()
    print(lines[3])
    cam02Param = lines[3].split()

    print(cam02Param[1] + " " + cam02Param[2] + " " + cam02Param[3])
    print(cam02Param[4] + " " + cam02Param[5] + " " + cam02Param[6])
    print(cam02Param[7] + " " + cam02Param[8] + " " + cam02Param[9])
    # fx = cam00Param[1]
    # fy = cam00Param[5]
    # cx = cam00Param[3]
    # cy = cam00Param[6]

    camera_matrixL = np.array([[cam02Param[1], cam02Param[2], cam02Param[3]],
                               [cam02Param[4], cam02Param[5], cam02Param[6]],
                               [cam02Param[1], cam02Param[8], cam02Param[9]]])
    f.close

    print("\n Cam right color:")
    f = open(basedir + "sequences/"+sequence+"/calib.txt", "r")
    lines = f.readlines()
    print(lines[4])
    cam03Param = lines[4].split()

    print(cam03Param[1] + " " + cam03Param[2] + " " + cam03Param[3])
    print(cam03Param[4] + " " + cam03Param[5] + " " + cam03Param[6])
    print(cam03Param[7] + " " + cam03Param[8] + " " + cam03Param[9])
    # fx = cam01Param[1]
    # fy = cam01Param[5]
    # cx = cam01Param[3]
    # cy = cam01Param[6]

    camera_matrixR = np.array([[cam03Param[1], cam03Param[2], cam03Param[3]],
                               [cam03Param[4], cam03Param[5], cam03Param[6]],
                               [cam03Param[1], cam03Param[8], cam03Param[9]]])
    f.close

    return camera_matrixL, camera_matrixR


# We need to pass to regNet a rgb and a velo flow


def dataset_construction(rgb, lidar):
    sample = {'rgb': rgb.float(), 'lidar': lidar.float()}
    return sample
