import argparse
from regNet import RegNet
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
# import dataset
# import numpy as np
import random
import wandb
# import utils
# import pykitti
from scipy.spatial.transform import Rotation as R
from dataset import RegnetDataset
from tqdm import tqdm

def train(train_model, device, train_optimizer, rgb_img, refl_img, target_transl_error, target_rot_error, rescale_param=1):
    train_model.train()

    rgb = rgb_img.to(device)
    lidar = refl_img.to(device)
    target_transl_error = target_transl_error.to(device)
    target_rot_error = target_rot_error.to(device)

    train_optimizer.zero_grad()
    # print(c)
    # Run model
    transl_err, rot_err = train_model(rgb, lidar)

    loss = nn.MSELoss(reduction='none')

    loss_transl_train = loss(transl_err, target_transl_error).sum(1).mean()

    loss_rot_train = loss(rot_err, target_rot_error).sum(1).mean()

    total_loss_train = torch.add(loss_transl_train, rescale_param * loss_rot_train)

    total_loss_train.backward()
    train_optimizer.step()
    # print(total_loss)

    return total_loss_train.item()


def test(test_model, device, rgb_img, refl_img, target_transl, target_rot, rescale_param=1):
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

    return total_loss_test.item(), loss_rot_test, loss_transl_test, rot_test_comparator, tr_test_comparator

def main():
    parser = argparse.ArgumentParser(description='RegNet')
    parser.add_argument('--loss', default='simple',
                    help='Type of loss used')
    args = parser.parse_args()
    # args.cuda = not args.no_cuda and torch.cuda.is_available()


    # Specify the dataset to load
    basedir = '/media/RAIDONE/DATASETS/KITTI/ODOMETRY/'

    sequence_train = ["00", "02", "03", "04", "05", "06", "07"]
    sequence_test = ["08", "09"]
    dataset_train = RegnetDataset(basedir, sequence_train)
    dataset_test = RegnetDataset(basedir, sequence_test)

    # sequence = ["00"]
    # Set the rando seed used for the permutations
    random.seed(1)
    epoch_number = 200
    learning_ratio = 0.00001
    batch_size = 32
    # rescale_param = 751.0
    rescale_param = 1.


    wandb.init(project="thesis-project_train", entity="capra")
    wandb.run.name = "Train run "+str(epoch_number)+" epochs "+str(batch_size)+" batch size"

    dataset_train_size = len(dataset_train)
    print("Saranno considerate per il training ", dataset_train_size, " coppie pcl-immgine. Le epoche sono: ",epoch_number)
    # print(dataset.__getitem__(0))
    # imageTensor = dataset.__getitem__(0)["rgb"]

    validation_split = 0
    # training_split = .8

    TrainImgLoader = torch.utils.data.DataLoader(dataset=dataset_train,
                                                 shuffle=True,
                                                 batch_size=batch_size,
                                                 num_workers=4,
                                                 drop_last=False,
                                                 pin_memory=True)
    TestImgLoader = torch.utils.data.DataLoader(dataset=dataset_test,
                                                batch_size=batch_size,
                                                num_workers=4,
                                                drop_last=False,
                                                pin_memory=True)

    # print(len(TrainImgLoader))
    # print(len(TestImgLoader))

    len_TrainImgLoader = len(TrainImgLoader)
    len_TestImgLoader = len(TestImgLoader)
    # gpu +1
    device = torch.device("cuda:1")
    model = RegNet()
    model = model.to(device)
    # imageTensor2 = imageTensor[:, :1, :, :]
    parameters = filter(lambda p: p.requires_grad, model.parameters())


    optimizer = optim.Adam(parameters, lr=learning_ratio, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    wandb.config = {
        "learning_rate": learning_ratio,
        "epochs": epoch_number,
        "batch_size": batch_size,
        "sample_quantity": dataset_train_size
    }

    best_loss = 0
    total_loss = 0
    for epoch in range(0, epoch_number):
        print('This is %d-th epoch' % epoch)

        pbar_train = tqdm(total=len_TrainImgLoader)

        loss_train = 0
        total_loss = 0.
        # total_iter = 0

        c = 0
        for batch_idx, sample in enumerate(TrainImgLoader):
            loss_train = train(model, device, optimizer, sample['rgb'], sample['lidar'], sample['tr_error'], sample['rot_error'])
            total_loss+= loss_train
            # local_loss_train = total_iter+loss_train
            c = c+1
            pbar_train.update(1)
            # print("training "+str(c)+"/"+str(len_TrainImgLoader)+" epoch:"+str(epoch))
        pbar_train.close()
        wandb.log({"loss training": total_loss/len_TrainImgLoader})
        print("epoch "+str(epoch)+" loss_train: "+str(total_loss/len_TrainImgLoader))

        ## Test ##

        pbar_test = tqdm(total=len_TestImgLoader)
        total_loss = 0.
        total_loss_rot = 0.
        total_loss_transl = 0
        # total_test_t = 0.
        # total_test_r = 0.

        local_loss = 0.0

        c = 0
        for batch_idx, sample in enumerate(TestImgLoader):
            test_loss, loss_rot, loss_transl, rot_comparator, tr_comparator = test(model, device, sample['rgb'], sample['lidar'], sample['tr_error'],
                                                    sample['rot_error'])
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
            c = c + 1
            pbar_test.update(1)
            # if c % 10 == 0:
            #     wandb.log({"loss trasl test": loss_transl})
            #     wandb.log({"loss rot test": loss_rot})
            # print("testing" + str(c) + "/" + str(len_TestImgLoader))
        pbar_test.close()
        wandb.log({"total loss test": total_loss / len_TestImgLoader})
        wandb.log({"loss rot test": total_loss_rot / len_TestImgLoader})
        wandb.log({"loss trasl test": total_loss_transl / len_TestImgLoader})

        if epoch == 0:
            best_loss = total_loss / len_TestImgLoader
        if total_loss / len_TestImgLoader <= best_loss/ len_TestImgLoader:
            print("Salvato modello nuovo migliore del precedente alla apoca "+str(epoch))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss':total_loss / len_TestImgLoader,
            }, "./models/partial_model_epoch.pt")
            best_loss=total_loss / len_TestImgLoader

    # save the model
    print("saving the last model...")
    torch.save({
                'epoch': epoch_number,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss':total_loss / len_TestImgLoader,
            }, "./models/partial_model_epoch.pt")
    print("model saved")
    # test model load
    # model = RegNet()
    # model.load_state_dict(torch.load("./models/model.pt"))
    # model.eval()

if __name__ == "__main__":
    main()