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

    def __init__(self, dataset_dir, sequences, pose_type=None, seed=None, transform=None):

        """
        :param dataset_dir: directory where dataset is located
        :param pose_type: type of pose representation: Euler angles, Quaternions, Dual quaternions
        :param transform: transform operations applied on the input images
        """

        self.root_dir = dataset_dir
        self.datasets = []
        self.calibs = []
        for sequence in sequences:
            print(sequence)
            dataset = pykitti.odometry(dataset_dir, sequence)
            self.datasets.append(dataset)
            self.calibs.append([sequence, dataset.calib])

        self.rgb_files = []
        self.velo_files = []

        self.sizes = []
        self.lenght = 0
        for dataset in self.datasets:

            self.rgb_files = [*self.rgb_files, *dataset.cam2_files]
            self.velo_files = [*self.velo_files, *dataset.velo_files]
            self.lenght = self.lenght + len(dataset.velo_files)
            self.sizes.append(len(dataset.velo_files))
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
        return self.lenght

    def __getitem__(self, idx):
        rgb_filess = self.rgb_files
        depth = []
        rot_error = [0, 0, 0]
        tr_error = [0, 0, 0]
        z_error = random.randrange(-5, 5)
        rot_error[0] = z_error
        depth.append(data_formatter_pcl_single(self.velo_files, idx, tr_error, rot_error))

        rgb_img = Image.open(self.rgb_files[idx])
        to_tensor = transforms.ToTensor()
        rgb = to_tensor(rgb_img)

        # error on the z,y,x axis

        # print("rot error: "+str(rot_error))

        # If tensor has odd number of values, it's not possible to split it using an elegant way
        # Now rotation and translation are considered separately
        tr = torch.from_numpy(np.array(tr_error)).float()
        rot = torch.from_numpy(np.array(rot_error)).float()

        sample = {'idx': idx, 'rgb': rgb.float(), 'lidar': depth.float(), 'tr_error': tr, 'rot_error': rot}

        return sample
