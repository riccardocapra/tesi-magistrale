import pykitti
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d

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

for img in dataset.rgb:        
    disp_rgb.append(stereo.compute(
        cv2.cvtColor(np.array(img[0]) , cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(np.array(img[1]), cv2.COLOR_RGB2GRAY)))
show_PC(disp_rgb, velo)
