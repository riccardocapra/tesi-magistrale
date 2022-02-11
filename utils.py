# import pykitti
import copy

import numpy as np
# import torch
from torchvision import transforms
from PIL import Image
import cv2
from datetime import datetime
from scipy.spatial.transform import Rotation as R
import cupy
# import multiprocessing
import math
# from os import getpid
# import os


def quaternion_distance(q, r):
    r = r.cpu()
    q = q.cpu()

    r_quat = R.from_euler('zyx', r, degrees=True)
    r_quat = r_quat.as_quat()
    r_quat[0] *= -1

    q_quat = R.from_euler('zyx', q, degrees=True)
    q_quat = q_quat.as_quat()

    # t = torch.zeros(4).to(device)
    # rinv = r.clone()
    # rinv[0] *= -1
    t = r_quat[0]*q_quat[0] - r_quat[1]*q_quat[1] - r_quat[2]*q_quat[2] - r_quat[3]*q_quat[3]
    # t[1] = r[0]*q[1] + r[1]*q[0] - r[2]*q[3] + r[3]*q[2]
    # t[2] = r[0]*q[2] + r[1]*q[3] + r[2]*q[0] - r[3]*q[1]
    # t[3] = r[0]*q[3] - r[1]*q[2] + r[2]*q[1] + r[3]*q[0]
    dist = 2*math.acos(np.clip(math.fabs(t.item()), 0., 1.))
    dist = 180. * dist / math.pi
    return dist


def pcl_rt(depth_pts, cam_to_velo, camera_matrix):
    depth_pts_cp = copy.deepcopy(depth_pts)
    depth_pts_cp[3] = np.ones(depth_pts_cp.shape[1])
    depth_pt = cupy.dot(cam_to_velo, depth_pts_cp)
    # rimuovere le x<0?
    # depth = depth[depth[:, 2] > 0]
    # creo una riga di uni da agganciare ai punti per moltiplicarli omogeneamente

    depth_pt = cupy.dot(camera_matrix, depth_pt[:3, :])
    depth_pt[0] = (depth_pt[0] / depth_pt[2]).astype(int)
    depth_pt[1] = (depth_pt[1] / depth_pt[2]).astype(int)
    return depth_pt


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
    rotation_array = R.from_euler('zyx', rot_error)
    # h_mat = R.from_matrix(cupy.dot(new_h_init[:3, :3], rotation_array.as_matrix()))
    # new_h_init[:3, :3] = h_mat.as_matrix()
    new_h_init[:3, :3] = cupy.dot(rotation_array.as_matrix(), new_h_init[:3, :3])
    new_h_init[:3, 3] += tr_error
    new_h_init[3,3] = 1
    # print("Rotazione della nuova matrice H che la fz ritorna:")
    # print(new_h_init[:3, :3])
    return new_h_init

def rototranslation_matrix_initializator(rot, trasl,fix_orientation=True):
    #La rotazione è espressa in radianti con gli assi in ordine zyx
    mat = np.eye(4)
    rot_mat = R.from_euler('zyx', rot)
    if fix_orientation:
        fix_mat = R.from_euler('zyx', [math.radians(90), 0, math.radians(90)])
        fix_mat = fix_mat.as_matrix()
        mat[:3,:3]=fix_mat
    rot_mat = rot_mat.as_matrix()
    mat[:3,:3] = np.dot(rot_mat, mat[:3,:3])
    mat[:3,3] = trasl
    return mat

def scan_loader(file):
    # print(file)
    scan = np.fromfile(file, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan


def data_formatter_pcl_single(datasets, velo_files, idx, tr_error, rot_error):
    cupy.cuda.Device(2)
    # print("---- VELO_IMAGE "+str(idx)+" FORMATTING BEGUN ---")
    depth = scan_loader(velo_files[idx]).T
    # print(velo_files[idx])
    # Find the sequence number in the file name
    sequence = velo_files[idx].split('/')[-3]
    # print(sequence)
    # dataset = datasets[sequence]
    # print(path[-1])
    # depth = dataset.get_velo(idx).T
    h_init = np.copy(datasets[sequence].calib.T_cam2_velo)
    # depth_n = pcl_rt(depth, h_init, dataset.calib.K_cam2)
    h, w = 352, 1216
    # depth_image = depth_image_creation(depth_n, h, w)
    # cv2.imwrite('Original.jpeg', depth_image)

    # perturbation_vector = [0, 0, 45]

    new_h_init = perturbation(h_init, rot_error, tr_error)

    depth_p = pcl_rt(depth, new_h_init, datasets[sequence].calib.K_cam2)
    depth_image_p = depth_image_creation(depth_p, h, w)
    # cv2.imwrite('Perturbated.jpeg', depth_image_p)

    # print("---- VELO_IMAGE FORMATTING ENDED ---")
    to_tensor = transforms.ToTensor()
    depth_image_tensor = to_tensor(depth_image_p)
    return depth_image_tensor

# Funzione che prende un sample del dataset, legge img_file, velo_file e proietta il velo sulla immagine
# con la decalib impostata per quella sample
def show_couple(sample, k_cam2, h, w, model_name, fix_orientation=True):
    depth = scan_loader(sample["velo_file"]).T

    print("La decalibrazione di rotazione su Z, Y, X è: "+str(math.degrees(sample["rot_error"][0]))+" "+
          str(math.degrees(sample["rot_error"][1]))+" "+str(math.degrees(sample["rot_error"][2])))

    print("La decalibrazione di traslazione su Z, Y, X è: "+str(sample["tr_error"][0])+" "+
          str(sample["tr_error"][1])+" "+str(sample["tr_error"][2]))

    new_h_init = rototranslation_matrix_initializator(sample["rot_error"], sample["tr_error"], fix_orientation)
    print(new_h_init)
    depth_p = pcl_rt(depth, new_h_init, k_cam2)
    depth_image_p = depth_image_creation(depth_p, h, w)
    cv2.imwrite('./images/'+model_name+'_decalibration.jpeg', depth_image_p)

    rgb_img = Image.open(sample["rgb_file"])
    rgb_img_cropped = rgb_img.crop((0, 0, 1216, 352))
    rgb_img_cropped.save("./images/"+model_name+"_camera.jpg")
    return
