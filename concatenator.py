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
# import pykitti
from scipy.spatial.transform import Rotation as R
import wandb
import torch.optim as optim
# from main import test

def test(test_model, rgb_img, refl_img, target_transl, target_rot, velo_file, rgb_file):
    test_model.eval()
    rgb = rgb_img.to(device)
    lidar = refl_img.to(device)
    target_transl_device = target_transl.to(device)
    target_rot_device = target_rot.to(device)

    # target_transl = tr_error
    # target_rot = rot_error

    # if args.cuda:
    #    rgb, lidar, target_transl, target_rot = rgb.cuda(), lidar.cuda(), target_transl.cuda(), target_rot.cuda()

    # Run model disabling learning
    with torch.no_grad():
        transl_err, rot_err = test_model(rgb, lidar)

    # Translation and rotation euclidean loss
    # Sum errors computed with the input pose and check the distance with the target one
    loss = nn.MSELoss(reduction='none')

    # The following code is used to show the expected rotation vs the computed ones

    rot_err_np = rot_err.cpu()
    rot_err_np = rot_err_np.numpy()
    rot_err_euler = R.from_euler('zyx', rot_err_np)
    rot_err_euler = rot_err_euler.as_euler('zxy', degrees=True)
    # print("rot err: ", rot_err_euler)
    target_rot_euler = R.from_euler('zyx', target_rot)
    target_rot_euler = target_rot_euler.as_euler('zxy', degrees=True)

    transl_err_np = transl_err.cpu().numpy()


    # print("rot target: ", target_rot_euler)

    loss_transl_test = loss(transl_err, target_transl_device).sum(1).mean()

    loss_rot_test = loss(rot_err, target_rot_device).sum(1).mean()

    total_loss_test = torch.add(loss_transl_test, rescale_param * loss_rot_test)

    # total_trasl_error = 0.0
    # total_rot_error = 0.0
    # for j in range(rgb.shape[0]):
    #     total_trasl_error += torch.norm(target_transl[j] - transl_err[j]) * 100.
    #     total_rot_error += utils.quaternion_distance(target_rot[j], rot_err[j])

    # return total_loss.item(), total_trasl_error.item(), total_rot_error, rot_err
    rot_test_comparator = [target_rot_euler, rot_err_euler]
    tr_test_comparator = [transl_err_np, target_transl]

    return total_loss_test.item(), loss_rot_test, loss_transl_test, rot_test_comparator, tr_test_comparator, \
           velo_file, rgb_file, rot_err_euler, transl_err


# Specify the dataset to load
basedir = '/media/RAIDONE/DATASETS/KITTI/ODOMETRY/'
sequence = ["08"]
dataset_20 = RegnetDataset(basedir, sequence)
# print('torch device', torch.cuda.current_device(), torch.cuda.device(0), torch.cuda.device_count())
device = torch.device('cuda:0')
# Set the random seed used for the permutations
batch_size = 32
dataset_size = len(dataset_20)
# uni = np.ones((dataset_size,3)).tolist()
# zeros = np.zeros((dataset_size,3)).tolist()
rescale_param = 1.

# dataset.set_decalibtarions(uni,uni)
# dataset.correct_decalibrartions(zeros,zeros)

print("begin test")

# test model_20 load
model_20 = RegNet()
parameters = filter(lambda p: p.requires_grad, model_20.parameters())
learning_ratio = 0.00001
optimizer = optim.Adam(parameters, lr=learning_ratio, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
model_20 = model_20.to(device)
parameters = filter(lambda p: p.requires_grad, model_20.parameters())
# model.load_state_dict(torch.load("./models/model_20-epochs.pt", map_location='cuda:0'))
checkpoint = torch.load("./models/model_20.pt", map_location='cuda:0')
model_20.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
model_20.eval()

wandb.init(project="thesis-project_test", entity="capra")
wandb.run.name = "test from epoch:"+str(epoch)
wandb.config = {
    "batch_size": batch_size,
    "sample_quantity": dataset_size
}

TestImgLoader = torch.utils.data.DataLoader(dataset=dataset_20,
                                            batch_size=batch_size,
                                            num_workers=4,
                                            drop_last=False,
                                            pin_memory=True)

total_loss = 0.
total_loss_rot = 0.
total_loss_transl = 0
# total_test_t = 0.
# total_test_r = 0.

local_loss = 0.0

len_TestImgLoader = len(TestImgLoader)
c = 0

predicted_rot_decals=[]
predicted_tr_decals=[]
for batch_idx, sample in enumerate(TestImgLoader):
    test_loss, loss_rot, loss_transl, rot_comparator, tr_comparator, velo_file, rgb_image, predicted_rot_decal, predicted_tr_decal = \
        test(model_20, sample['rgb'], sample['lidar'], sample['tr_error'], sample['rot_error'], sample["velo_file"],
             sample["rgb_image"])
    total_loss+=test_loss
    total_loss_rot +=loss_rot
    total_loss_transl +=loss_transl
    for i in range(rot_comparator[0].shape[0]):
        wandb.log({"z-axis rotation error": abs(rot_comparator[0][i][0] - rot_comparator[1][i][0])})
        wandb.log({"y-axis rotation error": abs(rot_comparator[0][i][1] - rot_comparator[1][i][1])})
        wandb.log({"x-axis rotation error": abs(rot_comparator[0][i][2] - rot_comparator[1][i][2])})
        wandb.log({"z-axis translation error": abs(tr_comparator[0][i][0] - tr_comparator[1][i][0])})
        wandb.log({"y-axis translation error": abs(tr_comparator[0][i][1] - tr_comparator[1][i][1])})
        wandb.log({"x-axis translation error":  abs(tr_comparator[0][i][2] - tr_comparator[1][i][2])})
        predicted_rot_decals.append(predicted_rot_decal[i])
        predicted_tr_decals.append(predicted_tr_decal[i])


    c = c + 1
    # if c % 10 == 0:
    #     wandb.log({"loss trasl test": loss_transl})
    #     wandb.log({"loss rot test": loss_rot})
    # print("testing" + str(c) + "/" + str(len_TestImgLoader))
wandb.log({"total loss test": total_loss / len_TestImgLoader})
wandb.log({"loss rot test": total_loss_rot / len_TestImgLoader})
wandb.log({"loss trasl test": total_loss_transl / len_TestImgLoader})

dataset_10  = dataset_20.copy()
dataset_10.correct_decalibrations(predicted_rot_decals,predicted_tr_decals)
model_10 = RegNet()
optimizer = optim.Adam(parameters, lr=learning_ratio, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
model_10 = model_10.to(device)
parameters = filter(lambda p: p.requires_grad, model_10.parameters())
# model.load_state_dict(torch.load("./models/model_20-epochs.pt", map_location='cuda:0'))
checkpoint = torch.load("./models/model_10.pt", map_location='cuda:0')
model_10.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
model_10.eval()


#QUI ORA DEVI PRENDERE LE DECAL. PREDETTE, CORREGGERE IL DATASET E RIMANDARLO AL MODEL_10#

#
# # print("end test")
# # print(rot_error)
# # print("loss: "+str(loss_test/len_TestImgLoader))
# # print("total_test_t: "+str(total_test_t))
# # print("total_test_r: "+str(total_test_r))
#
#
print("end CONCAT")
