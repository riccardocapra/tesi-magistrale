# import argparse
# import utils
import math

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
from main import test
from tqdm import tqdm
import copy
import utils
import cv2
from PIL import Image



def test_model(dataset, device, checkpoint_model, model_name_param="unnamed", complete_checkpoint=False,rescale_param=1.):
    print("begin test model_" + model_name_param)
    model = RegNet()
    # parameters = filter(lambda p: p.requires_grad, model.parameters())
    # learning_ratio = 0.00001
    # optimizer = optim.Adam(parameters, lr=learning_ratio, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    model = model.to(device)
    # parameters = filter(lambda p: p.requires_grad, model.parameters())
    if complete_checkpoint==True:
        model.load_state_dict(checkpoint_model["model_state_dict"])
    else:
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
            # print(abs(rot_comparator[0][i][0] - rot_comparator[1][i][0]))
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

def confront_decalib(dataset_decalib, predicted_decalib, accettable_range):
    c = 0
    #contains indexes of oor decals
    out_of_range = []
    confronter_total = []
    for i in dataset_decalib:
        z = abs(i[0]-predicted_decalib[c][0])
        # print("z: "+str(i[0])+" "+str(predicted_decalib[c][0]))
        y = abs(i[1]-predicted_decalib[c][1])
        # print("y: "+str(i[1])+" "+str(predicted_decalib[c][1]))
        x = abs(i[2]-predicted_decalib[c][2])
        # print("x: "+str(i[2])+" "+str(predicted_decalib[c][2]))
        confronter = (z, y, x)
        if z > accettable_range or y > accettable_range or x > accettable_range:
            out_of_range.append(c)
        c += 1
        confronter_total.append(confronter)

    return confronter_total, out_of_range

def main():
    print("main")

    device = torch.device('cuda:0')
    basedir = '/media/RAIDONE/DATASETS/KITTI/ODOMETRY/'
    # h, w = 352, 1216
    rescale_param = 1.
    sequence = ["08"]
    model_name="20"
    dataset_20 = RegnetDataset(basedir, sequence)
    dataset_size = len(dataset_20)

    # utils.data_formatter_pcl_single(dataset_20, dataset_20.velo_files, dataset_20)
    # cv2.imwrite('./images/'+model_name+'_decalibration.jpeg', depth_image_p)

    checkpoint = torch.load("./models/model_200-epochs_V4.pt", map_location='cuda:0')
    # epoch = checkpoint['epoch']
    epoch = 200
    wandb.init(project="thesis-project_test", entity="capra")
    wandb.run.name = "Test of model_20 NET - full std decal"
    wandb.config = {
        # "batch_size": checkpoint["batch_size"],
        "batch_size": 32,
        "sample_quantity": dataset_size
    }
    predicted_rot_decals,predicted_tr_decals = test_model(dataset_20, device, checkpoint, model_name)
    confront, out_of_range = confront_decalib(dataset_20.rot_errors_euler, predicted_rot_decals, 10)
    print("Su "+str(dataset_size)+" elementi ci sono: "+str(len(out_of_range))+" O.O.R. per le rotazioni")
    print(out_of_range[:10])
    #INIZIO MODELLO 10#

    model_name = "10"
    dataset_10  = copy.deepcopy(dataset_20)
    dataset_10.correct_decalibrations(predicted_rot_decals,predicted_tr_decals)
    checkpoint = torch.load("./models/model_10.pt", map_location='cuda:0')
    predicted_rot_decals,predicted_tr_decals = test_model(dataset_10, device, checkpoint, model_name, True)
    confront, out_of_range = confront_decalib(dataset_10.rot_errors_euler, predicted_rot_decals, 5)
    print("Su "+str(dataset_size)+" elementi ci sono: "+str(len(out_of_range))+" O.O.R. per le rotazioni")

    #INIZIO MODELLO 05#

    model_name = "05"
    dataset_05  = copy.deepcopy(dataset_10)
    dataset_05.correct_decalibrations(predicted_rot_decals,predicted_tr_decals)
    predicted_rot_decals,predicted_tr_decals = test_model(dataset_05, device, checkpoint, model_name, True)
    confront, out_of_range = confront_decalib(dataset_05.rot_errors_euler, predicted_rot_decals, 2)
    print("Su "+str(dataset_size)+" elementi ci sono: "+str(len(out_of_range))+" O.O.R. per le rotazioni")

    #INIZIO MODELLO 02#

    model_name = "02"
    dataset_02  = copy.deepcopy(dataset_05)
    dataset_02.correct_decalibrations(predicted_rot_decals,predicted_tr_decals)
    predicted_rot_decals,predicted_tr_decals = test_model(dataset_02, device, checkpoint, model_name, True)
    confront, out_of_range = confront_decalib(dataset_02.rot_errors_euler, predicted_rot_decals, 1)
    print("Su "+str(dataset_size)+" elementi ci sono: "+str(len(out_of_range))+" O.O.R. per le rotazioni")


    #INIZIO MODELLO 01#

    model_name = "01"
    dataset_01  = copy.deepcopy(dataset_02)
    dataset_01.correct_decalibrations(predicted_rot_decals,predicted_tr_decals)
    predicted_rot_decals,predicted_tr_decals = test_model(dataset_01, device, checkpoint, model_name, True)
    confront, out_of_range = confront_decalib(dataset_01.rot_errors_euler, predicted_rot_decals, 0.5)
    print("Su "+str(dataset_size)+" elementi ci sono: "+str(len(out_of_range))+" O.O.R. per le rotazioni")

    # creare un loop che cicla le dacal predette e tira fuori delle matrici

    print("end CONCAT")

if __name__ == "__main__":
    main()