import argparse
import utils
from regNet import RegNet
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from dataset import RegnetDataset
import numpy as np
import math
import random
# import pykitti
from scipy.spatial.transform import Rotation as R


def test(model, rgb_img, refl_img, target_transl, target_rot):
    model.eval()
    rgb = rgb_img.to(device)
    lidar = refl_img.to(device)
    target_transl = target_transl.to(device)
    target_rot = target_rot.to(device)

    # target_transl = tr_error
    # target_rot = rot_error

    # if args.cuda:
    #    rgb, lidar, target_transl, target_rot = rgb.cuda(), lidar.cuda(), target_transl.cuda(), target_rot.cuda()

    # Run model disabling learning
    with torch.no_grad():
        transl_err, rot_err = model(rgb, lidar)

    # Translation and rotation euclidean loss
    # Sum errors computed with the input pose and check the distance with the target one
    loss = nn.MSELoss(reduction='none')

    loss_transl = loss(transl_err, target_transl).sum(1).mean()
    loss_rot = loss(rot_err, target_rot).sum(1).mean()

    total_loss = torch.add(loss_transl, rescale_param * loss_rot)

    total_trasl_error = 0.0
    total_rot_error = 0.0
    for j in range(rgb.shape[0]):
        total_trasl_error += torch.norm(target_transl[j] - transl_err[j]) * 100.
        total_rot_error += utils.quaternion_distance(target_rot[j], rot_err[j])

    return total_loss.item(), total_trasl_error.item(), total_rot_error, rot_err


# Specify the dataset to load
basedir = '/media/RAIDONE/DATASETS/KITTI/ODOMETRY/'
sequence = ["07", "08", "09"]
rescale_param = 751.0
# Set the random seed used for the permutations
random.seed(1)
# export CUDA_VISIBLE_DEVICES=2
# echo $CUDA_VISIBLE_DEVICES
# 2

print('torch device', torch.cuda.current_device(),torch.cuda.device(0),torch.cuda.device_count())
device = torch.device('cuda:0')

print("begin test")

# test model load
model = RegNet()
model = model.to(device)
model.load_state_dict(torch.load("./models/model_20-epochs.pt", map_location='cuda:0'))
model.eval()
dataset = RegnetDataset(basedir, sequence)
dataset_size = len(dataset)


validation_split = 1
# training_split = .8
shuffle_dataset = False
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
train_indices, val_indices = indices[split:], indices[:split]
valid_sampler = SubsetRandomSampler(val_indices)


TestImgLoader = torch.utils.data.DataLoader(dataset=dataset,
                                            sampler=valid_sampler,
                                            batch_size=32,
                                            num_workers=4,
                                            drop_last=False,
                                            pin_memory=True)

total_test_loss = 0
total_test_t = 0.
total_test_r = 0.
c = 0
local_loss = 0.0
len_TestImgLoader = len(TestImgLoader)
for batch_idx, sample in enumerate(TestImgLoader):
            loss_test, trasl_e, total_rot_error, rot_error = test(model, sample['rgb'], sample['lidar'], sample['tr_error'], sample['rot_error'])
            total_test_t += trasl_e
            total_test_r += total_rot_error
            local_loss += loss_test
            c = c+1
            print(str(c) + "/" + str(len_TestImgLoader))

print("end test")
print(rot_error)
print("loss: "+str(loss_test/len_TestImgLoader))
print("total_test_t: "+str(total_test_t))
print("total_test_r: "+str(total_test_r))


print("end test 2")
