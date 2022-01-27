import argparse
from regNet import RegNet
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from dataset import RegnetDataset
import numpy as np
import random
import wandb
# import utils
# import pykitti
# from scipy.spatial.transform import Rotation as R


def train(train_model, train_optimizer, rgb_img, refl_img, target_transl_error, target_rot_error):
    train_model.train()

    rgb = rgb_img.to(device)
    lidar = refl_img.to(device)
    target_transl_error = target_transl_error.to(device)
    target_rot_error = target_rot_error.to(device)

    train_optimizer.zero_grad()
    # print(c)
    # Run model
    transl_err, rot_err = train_model(rgb, lidar)

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

    loss = nn.MSELoss(reduction='none')

    loss_transl = loss(transl_err, target_transl_error).sum(1).mean()

    loss_rot = loss(rot_err, target_rot_error).sum(1).mean()

    total_loss = torch.add(loss_transl, rescale_param * loss_rot)

    total_loss.backward()
    train_optimizer.step()
    # print(total_loss)

    return total_loss.item()


wandb.init(project="thesis-project_train", entity="capra")

parser = argparse.ArgumentParser(description='RegNet')
parser.add_argument('--loss', default='simple',
                    help='Type of loss used')
args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()
# gpu +1
device = torch.device("cuda:0")

# Specify the dataset to load
basedir = '/media/RAIDONE/DATASETS/KITTI/ODOMETRY/'

# sequence = ["00", "02", "03", "04", "05", "06", "07"]
sequence = ["00"]
# Set the rando seed used for the permutations
random.seed(1)
epoch_number = 1
learning_ratio = 0.00001
batch_size = 32
rescale_param = 751.0

dataset = RegnetDataset(basedir, sequence)
dataset_size = len(dataset)
print("Saranno considerate ", dataset_size, " coppie pcl-immgine.")
# print(dataset.__getitem__(0))
# imageTensor = dataset.__getitem__(0)["rgb"]

validation_split = 0.9
# training_split = .8
shuffle_dataset = False
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
train_indices, val_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

TrainImgLoader = torch.utils.data.DataLoader(dataset=dataset,
                                             shuffle=True,
                                             batch_size=batch_size,
                                             num_workers=4,
                                             drop_last=False,
                                             pin_memory=True)

# print(len(TrainImgLoader))
# print(len(TestImgLoader))

len_TrainImgLoader = len(TrainImgLoader)


model = RegNet()
model = model.to(device)
# imageTensor2 = imageTensor[:, :1, :, :]
parameters = filter(lambda p: p.requires_grad, model.parameters())


optimizer = optim.Adam(parameters, lr=learning_ratio, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

wandb.config = {
    "learning_rate": learning_ratio,
    "epochs": epoch_number,
    "batch_size": batch_size,
    "sample_quantity": dataset_size
}

for epoch in range(0, epoch_number):
    print('This is %d-th epoch' % epoch)
    total_train_loss = 0
    local_loss = 0.
    total_iter = 0
    best_loss = 0
    c = 0
    for batch_idx, sample in enumerate(TrainImgLoader):
        loss_train = train(model, optimizer, sample['rgb'], sample['lidar'], sample['tr_error'], sample['rot_error'])
        local_loss = local_loss+loss_train
        c = c+1
        wandb.log({"loss training": loss_train})
        print(str(c)+"/"+str(len_TrainImgLoader)+" epoch:"+str(epoch))

    print("epoch "+str(epoch)+" loss_train: "+str(local_loss/len(train_sampler)))
    if epoch == 0:
        best_loss = local_loss
    if local_loss <= best_loss:
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
        }, "./models/partial_model_epoch-"+str(epoch_number)+".pt")
        best_loss=local_loss

# save the model
print("saving the model...")
torch.save(model.state_dict(), "./models/model_"+str(epoch_number)+"-epochs_V2.pt")
print("model saved")
# test model load
# model = RegNet()
# model.load_state_dict(torch.load("./models/model.pt"))
# model.eval()
