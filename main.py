import argparse
from regNet import RegNet
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from dataset import RegnetDataset
import numpy as np
import pykitti
from utils import data_formatter, perturbation

parser = argparse.ArgumentParser(description='RegNet')
parser.add_argument('--loss', default='simple',
                    help='Type of loss used')

args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()


device = torch.device("cuda:2")
# Specify the dataset to load
basedir = '/media/RAIDONE/DATASETS/KITTI/ODOMETRY/'
sequence = "00"


dataset = RegnetDataset(basedir, sequence)
dataset_size = len(dataset)
# print(dataset.__getitem__(0))
imageTensor = dataset.__getitem__(0)["rgb"]

validation_split = .2
# training_split = .8
shuffle_dataset = False
random_seed = 42
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
train_indices, val_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

TrainImgLoader = torch.utils.data.DataLoader(dataset=dataset,
                                             sampler=train_sampler,
                                             batch_size=10,
                                             num_workers=4,
                                             drop_last=False,
                                             pin_memory=True)

TestImgLoader = torch.utils.data.DataLoader(dataset=dataset,
                                            sampler=valid_sampler,
                                            batch_size=10,
                                            num_workers=4,
                                            drop_last=False,
                                            pin_memory=True)

# print(len(TrainImgLoader))
# print(len(TestImgLoader))
loss = nn.SmoothL1Loss(reduction='none')
rescale_param = 751.0

model = RegNet()
model.train()
# imageTensor2 = imageTensor[:, :1, :, :]


def train(model, optimizer, rgb_img, refl_img, target_transl, target_rot):
    model.train()


    # rgb = rgb_img.to(device)
    # lidar = refl_img.to(device)
    # target_transl = target_transl.to(device)
    # target_rot = target_rot.to(device)

    optimizer.zero_grad()
    print("qui1")
    # Run model
    transl_err, rot_err = model(rgb_img, refl_img)

    # Translation and rotation euclidean loss
    # Check euclidean distance between error predicted and the real one
    # loss = nn.MSELoss(reduction='none')

    loss_transl = loss(transl_err, target_transl).sum().mean()
    loss_rot = loss(rot_err, target_rot).sum()

    # Somma pesata???
    if args.loss == 'learnable':
        total_loss = loss_transl * torch.exp(-model.sx) + model.sx + loss_rot * torch.exp(-model.sq) + model.sq
    else:
        total_loss = torch.add(loss_transl, rescale_param * loss_rot)

    total_loss.backward()
    optimizer.step()

    return total_loss.item()
sample = dataset.__getitem__(0)
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(parameters, lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

total_train_loss = 0
local_loss = 0.
total_iter = 0
for batch_idx, sample in enumerate(TrainImgLoader):
    loss_train = train(model, optimizer, sample['rgb'], sample['lidar'], sample['tr_error'], sample['rot_error'])
print("end")
