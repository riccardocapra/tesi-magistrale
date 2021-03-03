from d2l import torch as d2l
import torch
from torch import nn
import torch.nn.functional as F

def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())

class RegNet(torch.nn.Module):
    
    def __init__(self):

        super(RegNet, self).__init__()
        self.sx = nn.Parameter(torch.Tensor([0.0]))
        self.sq = nn.Parameter(torch.Tensor([-3.0]))

        self.rgb_features = nn.Sequential(
            nin_block(1, 96, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2d(3, stride=2),
            nin_block(96, 256, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            nin_block(256, 384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.5))

        self.depth_features = nn.Sequential(
            nin_block(1, 48, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2d(3, stride=2),
            nin_block(48, 128, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            nin_block(128, 192, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.5))

        self.feature_matching = nn.Sequential(
            nin_block(576, 512, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2d(3, stride=2),
            nin_block(512, 512, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.5))
        #
        #self.feature_extraction = nn.Sequential(
        #    nin_block(512, 512, kernel_size=11, strides=4, padding=0),
        #    nn.MaxPool2d(3, stride=2),
        #    nin_block(512, 256, kernel_size=5, strides=1, padding=2),
        #    nn.MaxPool2d(3, stride=2),
        #    nn.Dropout(0.5))
        #
        self.fc1 = nn.Linear(512 * 2 * 9, 512)
        self.fc1_trasl = nn.Linear(512, 256)
        self.fc1_rot = nn.Linear(512, 256)
        self.fc2_trasl = nn.Linear(256, 3)  # euler and quaternions   
        self.fc2_rot = nn.Linear(256, 4)  # quaternions and dual quaternions 


    def forward(self, img_data, lidar_data):
        # Lidar branch
        lidar_data = F.max_pool2d(self.depth_feature(lidar_data), 3, 2, 1)
        # IMG branch
        img_data = F.max_pool2d(self.rgb_features(img_data), 3, 2, 1)
        # Data fusion
        fused_data = torch.cat((lidar_data, img_data), 1)
        fused_data = F.max_pool2d(self.feature_matching(fused_data), 3, 2, 1)
        fused_data = F.max_pool2d(self.feature_extraction(fused_data), 3, 2)
        fused_data = fused_data.view(-1, 512 * 2 * 9)
        transl = F.relu(self.fc1_trasl(fused_data))
        rot = F.relu(self.fc1_rot(fused_data))
        transl = self.fc2_trasl(transl)
        rot = self.fc2_rot(rot)
        return transl, rot
