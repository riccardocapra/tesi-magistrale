# import argparse
# import utils
from regNet import RegNet
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
# import torch.optim as optim
from dataset import RegnetDataset
import numpy as np
# import math
import random
# import pykitti
from scipy.spatial.transform import Rotation as R
import wandb
import torch.optim as optim

def test(test_model, rgb_img, refl_img, target_transl, target_rot):
    test_model.eval()
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
        transl_err, rot_err = test_model(rgb, lidar)

    # Translation and rotation euclidean loss
    # Sum errors computed with the input pose and check the distance with the target one
    loss_test = nn.MSELoss(reduction='none')

    # The following code is used to show the expected rotation vs the computed ones

    rot_err_np = rot_err.cpu()
    rot_err_np = rot_err_np.numpy()
    rot_err_euler = R.from_euler('zyx', rot_err_np)
    rot_err_euler = rot_err_euler.as_euler('zxy', degrees=True)
    # print("rot err: ", rot_err_euler)

    target_rot_np = target_rot.cpu()
    target_rot_np = target_rot_np.numpy()
    target_rot_euler = R.from_euler('zyx', target_rot_np)
    target_rot_euler = target_rot_euler.as_euler('zxy', degrees=True)
    # print("rot target: ", target_rot_euler)

    loss_transl_test = loss_test(transl_err, target_transl).sum(1).mean()

    loss_rot_test = loss_test(rot_err, target_rot).sum(1).mean()

    total_loss_test = torch.add(loss_transl_test, rescale_param * loss_rot_test)

    # total_trasl_error = 0.0
    # total_rot_error = 0.0
    # for j in range(rgb.shape[0]):
    #     total_trasl_error += torch.norm(target_transl[j] - transl_err[j]) * 100.
    #     total_rot_error += utils.quaternion_distance(target_rot[j], rot_err[j])

    # return total_loss.item(), total_trasl_error.item(), total_rot_error, rot_err
    test_comparator = [target_rot_euler, rot_err_euler]

    return total_loss_test.item(), loss_rot_test, test_comparator


# Specify the dataset to load
basedir = '/media/RAIDONE/DATASETS/KITTI/ODOMETRY/'
sequence = ["08", "09"]
dataset = RegnetDataset(basedir, sequence)
# print('torch device', torch.cuda.current_device(), torch.cuda.device(0), torch.cuda.device_count())
device = torch.device('cuda:1')
# Set the random seed used for the permutations
random.seed(1)
epoch_number = 1
learning_ratio = 0.00001
batch_size = 32
rescale_param = 751.0
dataset_size = len(dataset)
# export CUDA_VISIBLE_DEVICES=2
# echo $CUDA_VISIBLE_DEVICES
# 2



print("begin test 2")

# test model load
model = RegNet()
model = model.to(device)
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(parameters, lr=learning_ratio, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# model.load_state_dict(torch.load("./models/model_20-epochs.pt", map_location='cuda:0'))
checkpoint = torch.load("./models/partial_model_epoch-40.pt", map_location='cuda:0')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
model.eval()

wandb.init(project="thesis-project_test", entity="capra")
wandb.run.name = "test from epoch:"+str(epoch)
wandb.config = {
    "learning_rate": learning_ratio,
    "epochs": epoch_number,
    "batch_size": batch_size,
    "sample_quantity": dataset_size
}

TestImgLoader = torch.utils.data.DataLoader(dataset=dataset,
                                            shuffle=True,
                                            batch_size=batch_size,
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
    total_loss, loss_rot, comparator = test(model, sample['rgb'], sample['lidar'], sample['tr_error'],
                                                      sample['rot_error'])

    for i in range(comparator[0].shape[0]):
        a = abs(comparator[0][i][0] - comparator[1][i][0])
        wandb.log({"z-axis rotation error": a})
        a = abs(comparator[0][i][1] - comparator[1][i][1])
        wandb.log({"y-axis rotation error": a})
        a = abs(comparator[0][i][2] - comparator[1][i][2])
        wandb.log({"x-axis rotation error": a})

    c = c+1
    if c % 10 == 0:
        wandb.log({"total loss test": total_loss})
        wandb.log({"loss rot test": loss_rot})
    print(str(c) + "/" + str(len_TestImgLoader))

# print("end test")
# print(rot_error)
# print("loss: "+str(loss_test/len_TestImgLoader))
# print("total_test_t: "+str(total_test_t))
# print("total_test_r: "+str(total_test_r))


print("end test 2")
