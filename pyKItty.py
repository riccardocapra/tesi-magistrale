import pykitti
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

def show_PC(disp_rgb, velo):
    i = 0
    for PC in disp_rgb:
        f, ax = plt.subplots(2, figsize=(15, 5))
        ax[0].imshow(dataset.get_rgb(i)[0])
        ax[0].set_title(str(i)+' Left RGB Image (cam2)')

        ax[1].imshow(PC, cmap='viridis')
        ax[1].set_title('RGB Stereo Disparity')
        
        plt.show()

        # f2 = plt.figure()
        # ax2 = f2.add_subplot(111, projection='3d')
        # velo_range = range(0, velo[i].shape[0], 100)
        # ax2.scatter(velo[i][velo_range, 0],
        #             velo[i][velo_range, 1],
        #             velo[i][velo_range, 2],
        #             c=velo[i][velo_range, 3],
        #     cmap='gray')
        # i+=1
    return

basedir = './kitti/dataset/'
date = '2011_09_26'
drive = '0001'
dataset = pykitti.raw(basedir, date, drive, frames=range(0, 50, 5))

#print(data.calib)
# Grab some data
#first_gray = dataset.get_gray(0)
#first_rgb = dataset.get_rgb(0)
velo = dataset.get_velo(0)

pcd = o3d.io.read_point_cloud("/home/capra/Scrivania/tesi/kitti/dataset/2011_09_26/2011_09_26_drive_0001_extract/velodyne_points/data/0000000110.txt", format='xyz')
#o3d.visualization.draw_geometries([pcd])
color_raw = o3d.io.read_image("/home/capra/Scrivania/tesi/kitti/dataset/2011_09_26/2011_09_26_drive_0001_extract/image_00/data/0000000000.png")
imageL_Path="/home/capra/Scrivania/tesi/kitti/dataset/2011_09_26/2011_09_26_drive_0001_extract/image_02/data/0000000000.png"
imageR_Path="/home/capra/Scrivania/tesi/kitti/dataset/2011_09_26/2011_09_26_drive_0001_extract/image_03/data/0000000000.png"
imgL = cv2.imread(imageL_Path,0)
imgR = cv2.imread(imageR_Path,0)
h,w = imgR.shape[:2]
stereo = cv2.StereoBM_create()
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()


print("Cam left grey:")
f = open("/home/capra/Scrivania/tesi/kitti/dataset/2011_09_26/calib_cam_to_cam.txt", "r")
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
    [cam00Param[1], cam00Param[8],cam00Param[9]]])
f.close

print("\n Cam right grey:")
f = open("/home/capra/Scrivania/tesi/kitti/dataset/2011_09_26/calib_cam_to_cam.txt", "r")
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
    [cam01Param[1], cam01Param[8],cam01Param[9]]])
f.close


# Do some stereo processing
stereo = cv2.StereoBM_create()
#disp_gray = stereo.compute(np.array(first_gray[0]), np.array(first_gray[1]))
disp_rgb = []

f2 = plt.figure()
ax2 = f2.add_subplot(111, projection='3d')
velo_range = range(0, velo.shape[0], 10)
ax2.scatter(velo[velo_range, 0],
            velo[velo_range, 1],
            velo[velo_range, 2],
            c=velo[velo_range, 3],
    cmap='gray')

# for img in dataset.rgb:        
#     disp_rgb.append(stereo.compute(
#         cv2.cvtColor(np.array(img[0]) , cv2.COLOR_RGB2GRAY),
#         cv2.cvtColor(np.array(img[1]), cv2.COLOR_RGB2GRAY)))
# show_PC(disp_rgb, velo)
