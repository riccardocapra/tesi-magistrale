# from d2l import torch as d2l
import torch.cuda
from torch import nn
import torch.nn.functional as F


def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(inplace=True))


class RegNet(torch.nn.Module):

    def __init__(self):
        super(RegNet, self).__init__()
        # self.sx = nn.Parameter(torch.Tensor([0.0]))
        # self.sq = nn.Parameter(torch.Tensor([-3.0]))

        self.rgb_features_n1 = nn.Sequential(
            nin_block(3, 96, kernel_size=11, strides=4, padding=5))
        self.rgb_features_n2 = nn.Sequential(
            nin_block(96, 256, kernel_size=5, strides=1, padding=2))
        self.rgb_features_n3 = nn.Sequential(
            nin_block(256, 384, kernel_size=3, strides=1, padding=1))

        self.refl_features_n1 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.MaxPool2d(3, stride=1, padding=1),
            nin_block(1, 48, kernel_size=11, strides=4, padding=5))
        self.refl_features_n2 = nn.Sequential(
            nin_block(48, 128, kernel_size=5, strides=1, padding=2))
        self.refl_features_n3 = nn.Sequential(
            nin_block(128, 192, kernel_size=3, strides=1, padding=1))

        # Feature matching
        self.fuse1 = nn.Sequential(nn.Conv2d(576, 512, 5, 1, 2),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, 1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, 1),
                                   nn.ReLU(inplace=True))

        self.fuse2 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, 1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(512, 512, 1),
                                   nn.ReLU(inplace=True))

        # self.feature_extraction = nn.Sequential(
        #    nin_block(512, 512, kernel_size=3, strides=1, padding=1),
        #    nn.MaxPool2d(3, stride=2),
        #    nin_block(512, 256, kernel_size=5, strides=1, padding=2),
        #    nn.MaxPool2d(3, stride=2),
        #    nn.Dropout(0.5))

        self.fc1 = nn.Linear(512 * 2 * 9, 512)
        self.fc1_transl = nn.Linear(512, 256)
        self.fc1_rot = nn.Linear(512, 256)
        self.fc2_transl = nn.Linear(256, 3)
        self.fc2_rot = nn.Linear(256, 3)

    def forward(self, img_data, lidar_data):
        # IMG branch
        img_data = F.max_pool2d(self.rgb_features_n1(img_data), 3, 2)
        img_data = F.max_pool2d(self.rgb_features_n2(img_data), 3, 2)
        img_data = F.max_pool2d(self.rgb_features_n3(img_data), 3, 2, 1)

        # Lidar branch
        lidar_data = F.max_pool2d(self.refl_features_n1(lidar_data), 3, 2, 1)
        lidar_data = F.max_pool2d(self.refl_features_n2(lidar_data), 3, 2, 1)
        lidar_data = F.max_pool2d(self.refl_features_n3(lidar_data), 3, 2, 1)

        # Data fusion
        fused_data = torch.cat((lidar_data, img_data), 1)
        fused_data = F.max_pool2d(self.fuse1(fused_data), 3, 2, 1)
        fused_data = F.max_pool2d(self.fuse2(fused_data), 3, 2)

        # fused_data = F.max_pool2d(self.feature_extraction(fused_data), 3, 2)
        fused_data = fused_data.view(-1, 512 * 2 * 9)
        fused_data = F.relu(self.fc1(fused_data))
        transl = F.relu(self.fc1_transl(fused_data))
        rot = F.relu(self.fc1_rot(fused_data))
        transl = self.fc2_transl(transl)
        rot = self.fc2_rot(rot)
        return transl, rot
