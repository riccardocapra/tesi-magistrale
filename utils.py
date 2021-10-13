import pykitti
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2
from datetime import datetime


def perturbation(H_init, p_factor):
    H_init[:3, 3] += p_factor
    return H_init


def depth_rototraslation(dataset):
    print("-- VELO_DATA FORMATTING BEGUN ---")
    depth_array = []
    cam2_velo = dataset.calib.T_cam2_velo
    i = 0
    while i < len(dataset.velo_files):
        depth = dataset.get_velo(i)[:, :3].T
        padding_vector = np.ones(depth.shape[1])
        depth = np.r_[depth, [padding_vector]]
        depth = np.dot(cam2_velo, depth)
        depth_array.append(depth)
        print(i)
        i += 1
    print("-- VELO_DATA FORMATTING ENDED ---")
    return depth_array


def pcl_rt(depth, H, K):
    depth = H.dot(depth)
    depth = depth.T
    depth = depth[depth[:, 2] > 0]
    depth = K @ depth[:, :3].T
    depth[0] = (depth[0] / depth[2]).astype(int)
    depth[1] = (depth[1] / depth[2]).astype(int)
    return depth


def depth_image_creation(depth, h, w):
    depth_image = np.zeros((h, w))
    z_max = np.amax(depth[2])
    mask1 = depth[0] >= 0
    mask2 = depth[0] < w
    mask3 = depth[1] >= 0
    mask4 = depth[1] < h
    final_mask = mask1 * mask2 * mask3 * mask4
    us = depth[0, final_mask]
    vs = depth[1, final_mask]
    zs = depth[2, final_mask]
    for i in range(0, len(us)):
        depth_image[int(vs[i]), int(us[i])] = (zs[i] / z_max) * 255
    return depth_image


def depth_rototranslation_single(dataset):
    print("---- VELO_IMAGE FORMATTING BEGUN ---")
    depth = dataset.get_velo(500).T
    h_init = perturbation(dataset.calib.T_cam2_velo, 1)
    depth_p = pcl_rt(depth, h_init, dataset.calib.K_cam2)
    depth = pcl_rt(depth, dataset.calib.T_cam2_velo, dataset.calib.K_cam2)
    h, w = 352, 1216
    depth_image = depth_image_creation(depth, h, w)
    depth_image_p = depth_image_creation(depth_p, h, w)
    cv2.imwrite('Original.jpeg', depth_image)
    cv2.imwrite('Perturbated.jpeg', depth_image_p)
    print("---- VELO_IMAGE FORMATTING ENDED ---")
    to_tensor = transforms.ToTensor()
    depth_image_tensor = to_tensor(depth_image)
    depth_image_tensor = depth_image_tensor.unsqueeze(0)
    return depth_image_tensor.float()


def data_formatter_pcl(dataset):
    print("---- VELO_IMAGES FORMATTING BEGUN ---")
    depths = dataset.velo
    depth_images = []
    h, w = 352, 1216
    start_time = datetime.now()
    for depth in depths:
        depth = depth.T
        depth = pcl_rt(depth, dataset.calib.T_cam2_velo, dataset.calib.K_cam2)
        depth_image = depth_image_creation(depth, h, w)
        depth_images.append(depth_image)
    end_time = datetime.now()
    end_time = end_time - start_time
    print("---- Secondi passati: "+str(end_time.total_seconds()))
    cv2.imwrite('filename.jpeg', depth_images[0])
    print("---- VELO_IMAGES FORMATTING ENDED ---")
    return depth_images


def data_formatter(basedir):
    print("-- DATA FORMATTING BEGUN ---")
    sequence = '00'
    dataset = pykitti.odometry(basedir, sequence)
    # depth_array = data_formatter_pcl(dataset)
    depth_array = depth_rototranslation_single(dataset)
    # depth = torch.from_numpy(depth / (2 ** 16)).float()
    # Le camere 2 e 3 sono quelle a colori, verificato. Mi prendo la 2.
    rgb_files = dataset.cam2_files
    # print(rgb_files)
    to_tensor = transforms.ToTensor()
    normalization = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # c=0m
    # rgb = []
    rgb_img = Image.open(rgb_files[0])
    # print(rgb_img)
    rgb = to_tensor(rgb_img)
    # print("Dimensione tensore: "+str(rgb.shape))
    # rgb = normalization(rgb)
    rgb = rgb.unsqueeze(0)
    print("-- DATA FORMATTING ENDED ---")
    return rgb.float(), depth_array


# The velodyne point clouds are stored in the folder 'velodyne_points'. To
# save space, all scans have been stored as Nx4 float matrix into a binary
# file.
# Here, data contains 4*num values, where the first 3 values correspond to
# x,y and z, and the last value is the reflectance information. All scans
# are stored row-aligned, meaning that the first 4 values correspond to the
# first measurement.


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
    f = open(basedir + "sequences/" + sequence + "/calib.txt", "r")
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
