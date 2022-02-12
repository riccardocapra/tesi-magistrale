import math

from dataset import RegnetDataset
import cv2
from PIL import Image
import torch
import pykitti
import utils
import numpy as np
from regNet import RegNet
from math import radians
from scipy.spatial.transform import Rotation as R

def main():
    # device = torch.device("cuda:0")
    basedir = '/media/RAIDONE/DATASETS/KITTI/ODOMETRY/'

    # sequence_train = ["00", "02", "03", "04", "05", "06", "07"]
    sequence_test = ["08"]
    # dataset = pykitti.odometry(basedir, "00")
    dataset_model = RegnetDataset(basedir,sequence_test)
    idx = 0
    model_name = "test"
    h, w = 352, 1216
    zero = np.zeros((len(dataset_model), 3))
    zero = np.zeros((len(dataset_model), 3))

    rot = dataset_model.rot_errors
    tr = dataset_model.tr_errors

    rot[0] = [math.radians(20),math.radians(0),math.radians(0)]

    dataset_model.set_decalibrations(rot, tr)
    #mostra la decalibrazione originale
    utils.show_couple(dataset_model[0], dataset_model.datasets["08"].calib.K_cam2, h, w, model_name, True)
    # set test
    rot[0] = [math.radians(10), math.radians(0), math.radians(0)]
    dataset_model.set_decalibrations(rot, tr)
    # mostra la decalibrazione impostata
    utils.show_couple(dataset_model[0], dataset_model.datasets["08"].calib.K_cam2, h, w, "decal", True)
    #correct test
    rot = np.zeros((len(dataset_model), 3))
    tr = np.zeros((len(dataset_model), 3))
    rot[0] = [math.radians(10), math.radians(0), math.radians(0)]
    dataset_model.correct_decalibrations(rot, tr)
    # mostra la decalibrazione impostata corretta
    utils.show_couple(dataset_model[0], dataset_model.datasets["08"].calib.K_cam2, h, w, "correct", True)


    # print("---- VELO_IMAGE "+str(idx)+" FORMATTING BEGUN ---")
    # depth = utils.scan_loader(dataset.velo_files[idx]).T
    # print(velo_files[idx])
    # Find the sequence number in the file name
    # print(sequence)
    # print(path[-1])
    # depth = dataset.get_velo(idx).T
    # h_init_ = np.copy(dataset.calib.T_cam2_velo)

    # depth_n = utils.pcl_rt(depth, dataset.calib.T_cam2_velo, dataset.calib.K_cam2)
    # h, w = 352, 1216
    # depth_image = utils.depth_image_creation(depth_n, h, w)
    # cv2.imwrite('./images/real_calibration.jpeg', depth_image)
    #
    # # perturbation_vector = [0, 0, 45]
    # h_init_ = dataset.calib.T_cam2_velo.copy()
    # # h_init_ = np.eye(4)
    # rot_error = [radians(0),radians(0),radians(0)]
    # tr_error = [0,0,0]
    #
    # new_h_init = utils.perturbation(h_init_, rot_error, tr_error)
    #
    # # depth_p = utils.pcl_rt(depth, new_h_init, dataset.calib.K_cam2)
    # depth_p = utils.pcl_rt(depth, new_h_init, dataset.calib.K_cam2)
    # depth_image_p = utils.depth_image_creation(depth_p, h, w)
    # cv2.imwrite('./images/just_rotated.jpeg', depth_image_p)
    #
    # rgb_img = Image.open(dataset.cam2_files[idx])
    # rgb_img_cropped = rgb_img.crop((0, 0, 1216, 352))
    # rgb_img_cropped.save("./images/camera.jpg")
    # # print("---- VELO_IMAGE FORMATTING ENDED ---")
    # to_tensor = utils.transforms.ToTensor()
    # depth_image_tensor = to_tensor(depth_image_p)
    print("end")

if __name__ == "__main__":
    main()