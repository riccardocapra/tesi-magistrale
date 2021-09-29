import pykitti
import torch
import numpy as np
from torchvision import transforms
from PIL import Image


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


def depth_rototraslation_single(dataset):
    print("---- VELO_IMAGE FORMATTING BEGUN ---")
    depth_array = []
    # cam2_velo = dataset.calib.T_cam2_velo
    # cam2_P = dataset.calib.P_rect_20
    # print("Cam 02 rectified matrix:")
    # print(dataset.calib.P_rect_20)
    depth = dataset.get_velo(0)[:, :3].T
    # ???
    padding_vector = np.ones(depth.shape[1])
    depth = np.r_[depth, [padding_vector]]
    # Pre-moltiplico i punti della pcl per la matrice H_init e per la P_rect dela camera 2
    # calib_matrix = np.dot(dataset.calib.P_rect_20, np.linalg.inv(dataset.calib.T_cam2_velo))
    # Una volta fatto ciò ottengo una matrice (3,n) dove n sono i punti della pcl.
    # Ogni riga della matrice rappresenta 3 valori (u,v,w), per farla combaciare all'imamgine divido tutto per w.
    # Ottengo w*(x,y,1) dove x ed y sono le coordinate nel frame immagine delle profondità w.
    # Creo una matrice di zeri (vedi paper) delle dimensioni dell'immagine
    # Alle varie coorinate (x,y) assegno la w corrispondente
    print("Z min: " + str(np.amin(depth[2])) + "Z max: " + str(np.amax(depth[2])))
    #T_velo_cam2 = np.linalg.inv(dataset.calib.T_cam2_velo)

    P_rect_20 = np.vstack([dataset.calib.P_rect_20, [0, 0, 0, 1]])
    Tr = dataset.calib.T_cam0_velo
    T_cam2_velo = P_rect_20.dot(Tr)
    T_velo_cam2 = np.linalg.inv(T_cam2_velo)

    depth = T_velo_cam2.dot(depth)
    print(dataset.calib.T_cam2_velo)
    depth[0] = depth[0] / depth[2]
    depth[1] = depth[1] / depth[2]
    c = 0
    for i in depth[0]:
        if i > 352 or i < 0:
            c += 1
    print("X oob: " + str(c) + "/" + str(depth[0].shape[0]) + " il " + str(int(c / depth[0].shape[0] * 100)) + "%")
    c = 0
    for i in depth[1]:
        if i > 1216 or i < 0:
            c += 1
    print("Y oob: " + str(c) + "/" + str(depth[1].shape[0]) + " il " + str(int(c / depth[1].shape[0] * 100)) + "%")
    c = 0
    for i in depth.T:
        if i[0] > 352 or i[0] < 0 or i[1] > 1216 or i[1] < 0:
            c += 1
    print(
        "X and Y oob: " + str(c) + "/" + str(depth[0].shape[0]) + " il " + str(int(c / depth[0].shape[0] * 100)) + "%")
    print("X oob max: " + str(np.amax(depth[0])) + " Y oob max: " + str(np.amax(depth[1])))
    # print(str(depth.T[0][0]/depth.T[0][2])+" "+str(depth.T[0][1]/depth.T[0][2])+" "+str(depth.T[0][2]/depth.T[0][2]))
    print("Z max: " + str(np.amax(depth[2])) + " Z min: " + str(np.amin(depth[2])))
    zMin = np.amin(depth[2])
    zMax = np.amax(depth[2])
    depth_image = np.zeros((352, 1216))
    w, h = 352, 1216
    for i in depth.T: #pipopipopipo
        if 352 > i[0] > 0 and 1216 > i[1] > 0:
            i[2] = ((i[2] - zMin) * 255) / (zMax - zMin)
            depth_image[int(i[0]), int(i[1])] = int(i[2])
    # print("depth_image max: " + str(np.amax(depth_image)) + " depth_image min: " + str(np.amin(depth_image)))
    data = np.zeros((h, w, 3), dtype=np.uint8)
    img = Image.fromarray(depth_image, 'L')
    img.save('./my.png')
    # img.show()
    print("---- VELO_IMAGE FORMATTING ENDED ---")
    return depth


def data_formatter(basedir):
    print("-- DATA FORMATTING BEGUN ---")
    sequence = '00'
    dataset = pykitti.odometry(basedir, sequence)
    depth_array = depth_rototraslation_single(dataset)
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
    print("-- DATA FORMATTING ENDED ---")
    return rgb, depth_array


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
