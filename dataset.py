from torch.utils.data import Dataset
import torch
# import pandas as pd
# import os, os.path
from PIL import Image
# from skimage import io, transform
import numpy as np
import random
from torchvision import transforms
from utils import data_formatter_pcl_single
import pykitti
from math import radians


class RegnetDataset(Dataset):
    """RegNet dataset."""

    def __init__(self, dataset_dir, sequences, pose_type=None, seed=None, transform=None):

        """
        :param dataset_dir: directory where dataset is located
        :param pose_type: type of pose representation: Euler angles, Quaternions, Dual quaternions
        :param transform: transform operations applied on the input images
        """

        self.root_dir = dataset_dir
        self.datasets = dict()
        self.datasets = dict.fromkeys(sequences, [])
        for sequence in sequences:
            # print(sequence)
            dataset = pykitti.odometry(dataset_dir, sequence)
            self.datasets[sequence] = dataset

        self.rgb_files = []
        self.velo_files = []
        self.rot_errors = []
        self.tr_errors = []
        self.sizes = []
        self.length = 0
        for key in self.datasets:
            # print(key)
            self.rgb_files = [*self.rgb_files, *self.datasets[key].cam2_files]
            self.velo_files = [*self.velo_files, *self.datasets[key].velo_files]
            self.length = self.length + len(self.datasets[key].velo_files)
            self.sizes.append(len(self.datasets[key]))
        self.initialize_decalibrations()

            # print(len(self.datasets[key]))
        # self.csv_file = pd.read_csv(os.path.join(dataset_dir, "dataset.csv"),
        #                             sep=',',
        #                             header=None,
        #                             skiprows=1)

        # self.pose_type = pose_type
        # self.transform = transform
    def initialize_decalibrations(self):
        for idx in range(self.length):
            rot_error = [0, 0, 0]
            tr_error = [0, 0, 0]
            rot_error[0] = radians(random.randrange(-20, 20))
            rot_error[1] = radians(random.randrange(-20, 20))
            rot_error[2] = radians(random.randrange(-20, 20))

            tr_error[0] = random.randrange(-150, 150) / 100
            tr_error[1] = random.randrange(-150, 150) / 100
            tr_error[2] = random.randrange(-150, 150) / 100
            self.rot_errors.append(rot_error)
            self.tr_errors.append(tr_error)

    def set_decalibrations(self,rot_error_input ,tr_error_input):
        self.tr_errors = tr_error_input
        self.rot_errors = rot_error_input


    def correct_decalibrations(self,rot_error_input ,tr_error_input):
        c=0
        for i in self.tr_errors:
            i[0]-=tr_error_input[c][0]
            i[1]-=tr_error_input[c][1]
            i[2]-=tr_error_input[c][2]
            self.rot_errors[c][0]-=rot_error_input[c][0]
            self.rot_errors[c][2]-=rot_error_input[c][1]
            self.rot_errors[c][1]-=rot_error_input[c][2]
            c+=1

    def custom_transform(rgb_input):
        # Tenere img size originale
        # resize = transforms.Resize((352, 1216))
        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        # rgb = resize(rgb)
        # if random.random() > 0.5:
        #     brightness = transforms.ColorJitter(brightness=0.4)
        #     rgb = brightness(rgb)
        rgb = to_tensor(rgb_input)
        rgb = normalization(rgb)
        return rgb

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # rot_error = [0, 0, 0]
        # tr_error = [0, 0, 0]
        # rot_error[0] = radians(random.randrange(-20, 20))
        # rot_error[1] = radians(random.randrange(-20, 20))
        # rot_error[2] = radians(random.randrange(-20, 20))
        #
        # tr_error[0] = random.randrange(-150, 150)/100
        # tr_error[1] = random.randrange(-150, 150)/100
        # tr_error[2] = random.randrange(-150, 150)/100

        depth = data_formatter_pcl_single(self.datasets, self.velo_files, idx, self.tr_errors[idx], self.rot_errors[idx])

        # Image have to be resized to

        rgb_img = Image.open(self.rgb_files[idx])
        # rgb_img_cropped = rgb_img.crop((left, top, right, bottom))
        rgb_img_cropped = rgb_img.crop((0, 0, 1216, 352))
        # print(str(width)+" "+str(height))

        to_tensor = transforms.ToTensor()
        rgb = to_tensor(rgb_img_cropped)

        # error on the z, y, x-axis

        # print("rot error: "+str(rot_error))

        # If tensor has odd number of values, it's not possible to split it using an elegant way
        # Now rotation and translation are considered separately
        tr = torch.from_numpy(np.array(self.tr_errors[idx])).float()
        rot = torch.from_numpy(np.array(self.rot_errors[idx])).float()

        sample = {'idx': idx, 'rgb': rgb.float(), 'lidar': depth.float(), 'tr_error': tr, 'rot_error': rot,
                  'velo_file':self.velo_files[idx], "rgb_file":self.rgb_files[idx]}
        return sample

