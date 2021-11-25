from torch.utils.data import Dataset
import torch
import pandas as pd
import os, os.path
from PIL import Image, ImageMath
from skimage import io, transform
import numpy as np
import random
from torchvision import transforms
from utils import data_formatter_pcl_single
import pykitti


class RegnetDataset(Dataset):
    """RegNet dataset."""

    def __init__(self, dataset_dir, sequence, pose_type=None, seed=None, transform=None):

        """

        :param dataset_dir: directory where dataset is located
        :param pose_type: type of pose representation: Euler angles, Quaternions, Dual quaternions
        :param transform: transform operations applied on the input images
        """

        self.root_dir = dataset_dir
        self.dataset = pykitti.odometry(dataset_dir, sequence)
        # self.csv_file = pd.read_csv(os.path.join(dataset_dir, "dataset.csv"),
        #                             sep=',',
        #                             header=None,
        #                             skiprows=1)

        # self.pose_type = pose_type
        # self.transform = transform

    def custom_transform(rgb):

        # Tenere img size originale
        # resize = transforms.Resize((352, 1216))
        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

        # rgb = resize(rgb)

        # if random.random() > 0.5:
        #     brightness = transforms.ColorJitter(brightness=0.4)
        #     rgb = brightness(rgb)

        rgb = to_tensor(rgb)
        rgb = normalization(rgb)
        return rgb


    def __len__(self):
        return len(self.dataset.velo_files)

    def __getitem__(self, idx):

        rgb_files = self.dataset.cam2_files


        # rgb_name = os.path.join(self.root_dir, self.csv_file.iloc[idx, 0])
        # refl_name = os.path.join(self.root_dir, self.csv_file.iloc[idx, 1])
        # depth_name = os.path.join(self.root_dir, self.csv_file.iloc[idx, 2])
        #
        # camera_roll = str(self.csv_file.iloc[idx, 3])

        rgb_img = Image.open(rgb_files[idx])
        to_tensor = transforms.ToTensor()
        rgb = to_tensor(rgb_img)

        depth = data_formatter_pcl_single(self.dataset, idx)

        # if self.pose_type == "quaternions":
        #     translation = self.csv_file.iloc[idx, 4]
        #     rotation = self.csv_file.iloc[idx, 6]
        # elif self.pose_type == "dual_quaternions":
        #     translation = self.csv_file.iloc[idx, 8]
        #     rotation = self.csv_file.iloc[idx, 7]
        # else:
        #     translation = self.csv_file.iloc[idx, 4]
        #     rotation = self.csv_file.iloc[idx, 5]

        # translation = translation.split(';')
        # rotation = rotation.split(';')

        tr = [0, 0, 0]
        rot = [0, 0, 0]

        # for el in translation:
        #     tr.append(float(el))
        #
        # for el in rotation:
        #     rot.append(float(el))

        # If tensor has odd number of values, it's not possible to split it using an elegant way
        # Now rotation and translation are considered separately
        tr = torch.from_numpy(np.array(tr)).float()
        rot = torch.from_numpy(np.array(rot)).float()

        sample = {'idx': idx, 'rgb': rgb.float(), 'lidar': depth.float(), 'tr_error': tr, 'rot_error': rot}

        return sample
