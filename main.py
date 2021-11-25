import argparse
from regNet import RegNet
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from dataset import RegnetDataset
import numpy as np
import math
# import pykitti
from scipy.spatial.transform import Rotation as R

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

validation_split = .8#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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


def quaternion_distance(q, r):
    r = r.cpu()
    q = q.cpu()

    r_quat = R.from_euler('zyx', r, degrees=True)
    r_quat = r_quat.as_quat()
    r_quat[0] *= -1

    q_quat = R.from_euler('zyx', q, degrees=True)
    q_quat = q_quat.as_quat()

    # t = torch.zeros(4).to(device)
    #rinv = r.clone()
    #rinv[0] *= -1
    t = r_quat[0]*q_quat[0] - r_quat[1]*q_quat[1] - r_quat[2]*q_quat[2] - r_quat[3]*q_quat[3]
    # t[1] = r[0]*q[1] + r[1]*q[0] - r[2]*q[3] + r[3]*q[2]
    # t[2] = r[0]*q[2] + r[1]*q[3] + r[2]*q[0] - r[3]*q[1]
    # t[3] = r[0]*q[3] - r[1]*q[2] + r[2]*q[1] + r[3]*q[0]
    dist = 2*math.acos(np.clip(math.fabs(t.item()), 0., 1.))
    dist = 180. * dist / math.pi
    return dist


def train(model, optimizer, rgb_img, refl_img, target_transl, target_rot, c):
    model.train()

    rgb = rgb_img.to(device)
    lidar = refl_img.to(device)
    target_transl = target_transl.to(device)
    target_rot = target_rot.to(device)

    optimizer.zero_grad()
    print(c)
    # Run model
    transl_err, rot_err = model(rgb, lidar)

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


def test(model, rgb_img, refl_img, target_transl, target_rot, c):
    model.eval()
    print(c)
    rgb = rgb_img.to(device)
    lidar = refl_img.to(device)
    target_transl = target_transl.to(device)
    target_rot = target_rot.to(device)

    # target_transl = tr_error
    # target_rot = rot_error

    # if args.cuda:
    #    rgb, lidar, target_transl, target_rot = rgb.cuda(), lidar.cuda(), target_transl.cuda(), target_rot.cuda()

    # Run model
    with torch.no_grad():
        transl_err, rot_err = model(rgb, lidar)

    # Translation and rotation euclidean loss
    # Sum errors computed with the input pose and check the distance with the target one
    loss = nn.MSELoss(reduction='none')

    loss_transl = loss(transl_err, target_transl).sum(1).mean()
    loss_rot = loss(rot_err, target_rot).sum(1).mean()

    if args.loss == 'learnable':
        total_loss = loss_transl * torch.exp(-model.sx) + model.sx + loss_rot * torch.exp(-model.sq) + model.sq
    else:
        total_loss = torch.add(loss_transl, rescale_param * loss_rot)

    total_trasl_error = 0.0
    total_rot_error = 0.0
    for j in range(rgb.shape[0]):
        total_trasl_error += torch.norm(target_transl[j] - transl_err[j]) * 100.
        total_rot_error += quaternion_distance(target_rot[j], rot_err[j])

    return total_loss.item(), total_trasl_error.item(), total_rot_error


parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(parameters, lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
model = model.to(device)
total_train_loss = 0
local_loss = 0.
total_iter = 0
c = 0
for batch_idx, sample in enumerate(TrainImgLoader):
    loss_train = train(model, optimizer, sample['rgb'], sample['lidar'], sample['tr_error'], sample['rot_error'], c)
    c = c+1
print("end training")

## Test ##
total_test_loss = 0
total_test_t = 0.
total_test_r = 0.
c = 0
local_loss = 0.0
for batch_idx, sample in enumerate(TestImgLoader):
            loss_test, trasl_e, rot_e = test(model, sample['rgb'], sample['lidar'], sample['tr_error'], sample['rot_error'], c)
            total_test_t += trasl_e
            total_test_r += rot_e
            local_loss += loss_test
            c = c+1
print("end test")
