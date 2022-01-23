import argparse
from regNet import RegNet
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from dataset import RegnetDataset
import numpy as np
import utils
import random
# import pykitti
from scipy.spatial.transform import Rotation as R


def train(model, optimizer, rgb_img, refl_img, target_transl_error, target_rot_error, c):
    model.train()

    rgb = rgb_img.to(device)
    lidar = refl_img.to(device)
    target_transl_error = target_transl_error.to(device)
    target_rot_error = target_rot_error.to(device)

    optimizer.zero_grad()
    # print(c)
    # Run model
    transl_err, rot_err = model(rgb, lidar)

    # Translation and rotation euclidean loss
    # Check euclidean distance between error predicted and the real one
    # loss = nn.MSELoss(reduction='none')
    # t = target_rot_error.cpu()
    # t = t.detach().numpy()
    # t_euler = R.from_euler('ZYX', t)
    # t_euler = t_euler.as_euler('ZYX', degrees=True)
    #
    # r = rot_err.cpu()
    # r = r.detach().numpy()
    # r_euler = R.from_euler('ZYX', r)
    # r_euler = r_euler.as_euler('ZYX', degrees=True)
    #
    # # print("trasl err: "+str(transl_err) + "target rot:  " + str(target_transl))
    # print("target rot:  " + str(t_euler))
    # print("rot err: " + str(r_euler))


    loss_transl = loss(transl_err, target_transl_error).sum(1).mean()

    loss_rot = loss(rot_err, target_rot_error).sum(1).mean()

    total_loss = torch.add(loss_transl, rescale_param * loss_rot)

    total_loss.backward()
    optimizer.step()
    # print(total_loss)

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
        total_rot_error += utils.quaternion_distance(target_rot[j], rot_err[j])

    return total_loss.item(), total_trasl_error.item(), total_rot_error


parser = argparse.ArgumentParser(description='RegNet')
parser.add_argument('--loss', default='simple',
                    help='Type of loss used')
args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()
# gpu +1
device = torch.device("cuda:0")

# Specify the dataset to load
basedir = '/media/RAIDONE/DATASETS/KITTI/ODOMETRY/'
sequence = ["00", "02", "03", "04", "05", "06", "07"]
# Set the rando seed used for the permutations
random.seed(1)

dataset = RegnetDataset(basedir, sequence)
dataset_size = len(dataset)
# print(dataset.__getitem__(0))
# imageTensor = dataset.__getitem__(0)["rgb"]

validation_split = 0
# training_split = .8
shuffle_dataset = False
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
train_indices, val_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

TrainImgLoader = torch.utils.data.DataLoader(dataset=dataset,
                                             sampler=train_sampler,
                                             shuffle=True,
                                             batch_size=32,
                                             num_workers=4,
                                             drop_last=False,
                                             pin_memory=True)

TestImgLoader = torch.utils.data.DataLoader(dataset=dataset,
                                            sampler=valid_sampler,
                                            shuffle=True,
                                            batch_size=32,
                                            num_workers=4,
                                            drop_last=False,
                                            pin_memory=True)

# print(len(TrainImgLoader))
# print(len(TestImgLoader))
loss = nn.SmoothL1Loss(reduction='none')
rescale_param = 751.0

model = RegNet()
model = model.to(device)
# imageTensor2 = imageTensor[:, :1, :, :]
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(parameters, lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
epoch_number = 20
len_TrainImgLoader = len(TrainImgLoader)
for epoch in range(0, epoch_number):
    print('This is %d-th epoch' % epoch)
    total_train_loss = 0
    local_loss = 0.
    total_iter = 0
    c = 0
    for batch_idx, sample in enumerate(TrainImgLoader):
        loss_train = train(model, optimizer, sample['rgb'], sample['lidar'], sample['tr_error'], sample['rot_error'], c)
        local_loss = local_loss+loss_train
        c = c+1
        print(str(c)+"/"+str(len_TrainImgLoader))

    print("epoch "+str(epoch)+" loss_train: "+str(local_loss/len(train_sampler)))

    ## Test ##
    total_test_loss = 0
    total_test_t = 0.
    total_test_r = 0.
    c = 0
    local_loss = 0.0
    # for batch_idx, sample in enumerate(TestImgLoader):
    #             loss_test, trasl_e, rot_e = test(model, sample['rgb'], sample['lidar'], sample['tr_error'], sample['rot_error'], c)
    #             total_test_t += trasl_e
    #             total_test_r += rot_e
    #             local_loss += loss_test
    #             c = c+1
    # print("end test")
    # print("total_test_t: "+str(total_test_t))
    # print("total_test_r: "+str(total_test_r))
# save the model
print("saving the model...")
torch.save(model.state_dict(), "./models/model_"+str(epoch_number)+"-epochs_V2.pt")
print("model saved")
# test model load
# model = RegNet()
# model.load_state_dict(torch.load("./models/model.pt"))
# model.eval()


