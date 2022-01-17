import pykitti
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import cv2
from datetime import datetime
from scipy.spatial.transform import Rotation as R
import cupy
import multiprocessing
import math
from os import getpid
import os


def quaternion_distance(q, r):
    r = r.cpu()
    q = q.cpu()

    r_quat = R.from_euler('zyx', r, degrees=True)
    r_quat = r_quat.as_quat()
    r_quat[0] *= -1

    q_quat = R.from_euler('zyx', q, degrees=True)
    q_quat = q_quat.as_quat()

    # t = torch.zeros(4).to(device)
    #rinv = r.clone()
    #rinv[0] *= -1
    t = r_quat[0]*q_quat[0] - r_quat[1]*q_quat[1] - r_quat[2]*q_quat[2] - r_quat[3]*q_quat[3]
    # t[1] = r[0]*q[1] + r[1]*q[0] - r[2]*q[3] + r[3]*q[2]
    # t[2] = r[0]*q[2] + r[1]*q[3] + r[2]*q[0] - r[3]*q[1]
    # t[3] = r[0]*q[3] - r[1]*q[2] + r[2]*q[1] + r[3]*q[0]
    dist = 2*math.acos(np.clip(math.fabs(t.item()), 0., 1.))
    dist = 180. * dist / math.pi
    return dist


def pcl_rt(depth_pts, H, K):
    cupy.cuda.Device(2)
    # depth = H.dot(depth_pts)
    depth = cupy.dot(H, depth_pts)
    depth = depth.T
    depth = depth[depth[:, 2] > 0]
    depth = K @ depth[:, :3].T
    depth[0] = (depth[0] / depth[2]).astype(int)
    depth[1] = (depth[1] / depth[2]).astype(int)
    return depth


def depth_image_creation(depth, h, w):
    cupy.cuda.Device(2)
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


def perturbation(h_init, rot_error, tr_error):
    cupy.cuda.Device(2)
    new_h_init = np.copy(h_init)
    # print("Rotazione matrice originale:")
    # print(new_h_init[:3, :3])
    # H_init[:2, 3] += p_factor
    # extract roll, pitch and yaw
    # h_mat = R.from_matrix(new_h_init[:3, :3])
    # quat_rot = h_mat.as_quat()
    # print("Quaternioni originali:")
    # print(quat_rot)

    # quat_rot_matrix = R.from_quat(quat_rot).as_matrix()
    # print("Matrice che genererebbe quei quaternioni:")
    # print(quat_rot_matrix)

    rotation_array = R.from_euler('ZXY', rot_error)
    # h_mat = R.from_matrix(new_h_init[:3, :3].dot(rotation_array.as_matrix()))
    h_mat = R.from_matrix(cupy.dot(new_h_init[:3, :3], rotation_array.as_matrix()))
    # quat_rot = h_mat.as_quat()
    # print("Quaternioni della matrice che ruoterà H:")
    # print(quat_rot)

    # print("h_mat ruotata:")
    # print(h_mat.as_matrix())
    # r = R.from_rotvec(r.apply(rotation_array))
    new_h_init[:3, :3] = h_mat.as_matrix()
    # print("Rotazione della nuova matrice H che la fz ritorna:")
    # print(new_h_init[:3, :3])
    return new_h_init


def data_formatter_pcl_single(datasets, velo_files, idx, tr_error, rot_error):
    cupy.cuda.Device(2)
    # print("---- VELO_IMAGE "+str(idx)+" FORMATTING BEGUN ---")
    depth = scan_loader(velo_files[idx]).T
    # print(velo_files[idx])
    # FInd the sequence number in the file name
    sequence = velo_files[idx].split('/')[-3]
    # print(sequence)
    dataset = datasets[sequence]
    # print(path[-1])
    # depth = dataset.get_velo(idx).T
    h_init = np.copy(dataset.calib.T_cam2_velo)
    depth_n = pcl_rt(depth, h_init, dataset.calib.K_cam2)
    h, w = 352, 1216
    depth_image = depth_image_creation(depth_n, h, w)
    # cv2.imwrite('Original.jpeg', depth_image)

    # perturbation_vector = [0, 0, 45]

    new_h_init = perturbation(h_init, rot_error, tr_error)

    depth_p = pcl_rt(depth, new_h_init, dataset.calib.K_cam2)
    depth_image_p = depth_image_creation(depth_p, h, w)
    # cv2.imwrite('Perturbated.jpeg', depth_image_p)

    # print("---- VELO_IMAGE FORMATTING ENDED ---")
    to_tensor = transforms.ToTensor()
    depth_image_tensor = to_tensor(depth_image_p)
    return depth_image_tensor


def depth_tensor_creation(depth):
    # print("point cloud", c, " in esecuzione.")
    h, w = 352, 1216
    to_tensor = transforms.ToTensor()
    depth = depth.T
    perturbation_vector = [0, 0, 45]
    new_h_init = perturbation(global_dataset.calib.T_cam2_velo, perturbation_vector)
    depth = pcl_rt(depth, new_h_init, global_dataset.calib.K_cam2)
    depth_image = depth_image_creation(depth, h, w)
    depth_image_tensor = to_tensor(depth_image)
    # depth_images_tensor.append(depth_image_tensor)
    # c+=1
    return depth_image_tensor


def scan_loader(file):
    #print(file)
    scan = np.fromfile(file, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan


def data_formatter_pcl(dataset):
    print("---- VELO_IMAGES FORMATTING BEGUN ---")
    velo_files = dataset.velo_files[:50]
    start_time = datetime.now()
    depths = []
    for file in velo_files:
        scan = np.fromfile(file, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        depths.append(scan)
    # scan = np.fromfile(velo_files, dtype=np.float32)
    # scan.reshape((-1, 4))
    end_time = datetime.now()
    end_time = end_time - start_time
    print("---- Secondi passati per conversione lista: " + str(end_time.total_seconds()))

    start_time = datetime.now()
    # c = 1
    depth_images_tensor = []
    h, w = 352, 1216
    perturbation_vector = [0, 0, 45]
    to_tensor = transforms.ToTensor()
    for depth in depths:
        depth = depth.T
        new_h_init = perturbation(dataset.calib.T_cam2_velo, perturbation_vector)
        depth = pcl_rt(depth, new_h_init, dataset.calib.K_cam2)
        depth_image = depth_image_creation(depth, h, w)
        depth_image_tensor = depth_image_tensor.append(to_tensor(depth_image))

    end_time = datetime.now()
    end_time = end_time - start_time
    print("---- Secondi passati: " + str(end_time.total_seconds()))
    print("---- VELO_IMAGES FORMATTING ENDED ---")
    return depth_images_tensor


def data_formatter_pcl_multiprocessing(dataset):
    print("---- VELO_IMAGES FORMATTING BEGUN ---")
    velo_files = dataset.velo_files[:50]
    start_time = datetime.now()
    with multiprocessing.Pool(12) as p:
        depths = p.map(scan_loader, velo_files)
        p.terminate()
    # scan = np.fromfile(velo_files, dtype=np.float32)
    # scan.reshape((-1, 4))
    end_time = datetime.now()
    end_time = end_time - start_time
    print("---- Secondi passati per conversione lista: " + str(end_time.total_seconds()))

    depth_images_tensor = []
    start_time = datetime.now()
    # c = 1
    with multiprocessing.Pool(12) as p:
        depth_images_tensor = depth_images_tensor.append(p.map(depth_tensor_creation, depths))

    end_time = datetime.now()
    end_time = end_time - start_time
    print("---- Secondi passati: " + str(end_time.total_seconds()))
    print("---- VELO_IMAGES FORMATTING ENDED ---")
    return depth_images_tensor


global_dataset = []


def data_formatter(basedir):
    print("-- DATA FORMATTING BEGUN ---")
    sequence = '00'
    global global_dataset
    global_dataset = pykitti.odometry(basedir, sequence)
    # depth_array = data_formatter_pcl(dataset)
    depth_array = data_formatter_pcl_multiprocessing(global_dataset)
    # depth = torch.from_numpy(depth / (2 ** 16)).float()
    # Le camere 2 e 3 sono quelle a colori, verificato. Mi prendo la 2.
    rgb_files = global_dataset.cam2_files
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
    print("-- DATA FORMATTING ENDED ---")
    return rgb.float(), depth_array.float()

