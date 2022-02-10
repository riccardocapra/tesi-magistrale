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
from tqdm import tqdm
import copy
import utils
import cv2
from PIL import Image

def test(test_model, device, rgb_img, refl_img, target_transl, target_rot, velo_file, rgb_file, rescale_param=1.):
    test_model.eval()
    rgb = rgb_img.to(device)
    lidar = refl_img.to(device)
    target_transl_device = target_transl.to(device)
    target_rot_device = target_rot.to(device)

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
           velo_file, rgb_file, rot_err_euler, transl_err_np

def test_model(dataset, device, checkpoint_model, model_name_param="unnamed", rescale_param=1.):
    print("begin test model_" + model_name_param)
    model = RegNet()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    learning_ratio = 0.00001
    optimizer = optim.Adam(parameters, lr=learning_ratio, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    model = model.to(device)
    # parameters = filter(lambda p: p.requires_grad, model.parameters())
    model.load_state_dict(checkpoint_model)
    # optimizer.load_state_dict(checkpoint_model['optimizer_state_dict'])
    model.eval()
    testImgLoader = torch.utils.data.DataLoader(dataset=dataset,
                                                   batch_size=32,
                                                   num_workers=4,
                                                   drop_last=False,
                                                   pin_memory=True)
    total_loss = 0.
    total_loss_rot = 0.
    total_loss_transl = 0
    c = 0
    model_predicted_rot_decals = []
    model_predicted_tr_decals = []
    pbar_train = tqdm(total=len(testImgLoader))
    for batch_idx, sample in enumerate(testImgLoader):
        test_loss, loss_rot, loss_transl, rot_comparator, tr_comparator, velo_file, rgb_image, predicted_rot_decal, predicted_tr_decal = \
            test(model, device, sample['rgb'], sample['lidar'], sample['tr_error'], sample['rot_error'], sample["velo_file"],
                 sample["rgb_file"], rescale_param)
        total_loss += test_loss
        total_loss_rot += loss_rot
        total_loss_transl += loss_transl
        for i in range(rot_comparator[0].shape[0]):
            wandb.log({"Z rotation error model_"+model_name_param: abs(rot_comparator[0][i][0] - rot_comparator[1][i][0])})
            wandb.log({"Y rotation error model_"+model_name_param: abs(rot_comparator[0][i][1] - rot_comparator[1][i][1])})
            wandb.log({"X rotation error model_"+model_name_param: abs(rot_comparator[0][i][2] - rot_comparator[1][i][2])})
            wandb.log({"Z translation error model_"+model_name_param: abs(tr_comparator[0][i][0] - tr_comparator[1][i][0])})
            wandb.log({"Y translation error model_"+model_name_param: abs(tr_comparator[0][i][1] - tr_comparator[1][i][1])})
            wandb.log({"x translation error model_"+model_name_param: abs(tr_comparator[0][i][2] - tr_comparator[1][i][2])})
            model_predicted_rot_decals.append(predicted_rot_decal[i])
            model_predicted_tr_decals.append(predicted_tr_decal[i])
        c = c + 1
        pbar_train.update(1)
    pbar_train.close()
    wandb.log({"total loss test model": total_loss / len(testImgLoader)})
    wandb.log({"loss rot test model": total_loss_rot / len(testImgLoader)})
    wandb.log({"loss trasl test model": total_loss_transl / len(testImgLoader)})
    print("ending test model_" + model_name_param)
    return model_predicted_rot_decals, model_predicted_tr_decals


def main():
    print("main")

    device = torch.device('cuda:0')
    basedir = '/media/RAIDONE/DATASETS/KITTI/ODOMETRY/'
    h, w = 352, 1216
    rescale_param = 1.
    sequence = ["08"]
    model_name="20"
    dataset_20 = RegnetDataset(basedir, sequence)
    dataset_size = len(dataset_20)

    # utils.data_formatter_pcl_single(dataset_20, dataset_20.velo_files, dataset_20)
    # cv2.imwrite('./images/'+model_name+'_decalibration.jpeg', depth_image_p)


    # zero = np.zeros((dataset_size, 3))
    # zeroe = np.zeros((dataset_size, 3))
    # zero[0]=[90,0,90]
    # dataset_20.set_decalibrations(zero,zeroe)
    # utils.show_couple(dataset_20[0], dataset_20.datasets["08"].calib.K_cam2, h, w, model_name)

    checkpoint = torch.load("./models/model_200-epochs_V4.pt", map_location='cuda:0')
    # epoch = checkpoint['epoch']
    epoch = 200
    wandb.init(project="thesis-project_test", entity="capra")
    wandb.run.name = "test from epoch:"+str(epoch)
    wandb.config = {
        # "batch_size": checkpoint["batch_size"],
        "batch_size": 32,
        "sample_quantity": dataset_size
    }
    predicted_rot_decals,predicted_tr_decals = test_model(dataset_20, device, checkpoint, model_name, rescale_param)

    #INIZIO MODELLO 10#

    model_name = "10"
    dataset_10  = copy.deepcopy(dataset_20)
    dataset_10.correct_decalibrations(predicted_rot_decals,predicted_tr_decals)
    # checkpoint = torch.load("./models/model_200-epochs_V4.pt", map_location='cuda:0')
    predicted_rot_decals,predicted_tr_decals = test_model(dataset_10, device, checkpoint, model_name)

    #INIZIO MODELLO 05#

    model_name = "05"
    dataset_05  = copy.deepcopy(dataset_10)
    # dataset_05.correct_decalibrations(predicted_rot_decals,predicted_tr_decals)


    # creare un loop che cicla le dacal predette e tira fuori delle matrici

    print("end CONCAT")

if __name__ == "__main__":
    main()