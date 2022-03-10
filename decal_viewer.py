from dataset import RegnetDataset
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("decal_viewer")
    basedir = '/media/RAIDONE/DATASETS/KITTI/ODOMETRY/'
    sequence_train = ["00", "02", "03", "04", "05", "06", "07"]
    dataset_train = RegnetDataset(basedir, sequence_train)
    rot_decals = np.array(dataset_train.rot_errors_euler).T
    plt.hist(rot_decals[0], density=False, bins=40)  # density=False would make counts
    plt.xlabel('Z rot error');
    plt.savefig('./images/Z_rot_error.png')
    plt.close()
    plt.hist(rot_decals[1], density=False, bins=40)  # density=False would make counts
    plt.xlabel('Y rot error');
    plt.savefig('./images/Y_rot_error.png')
    plt.close()
    plt.hist(rot_decals[2], density=False, bins=40)  # density=False would make counts
    plt.xlabel('X rot error');
    plt.savefig('./images/X_rot_error.png')
    plt.close()

    tr_decals = np.array(dataset_train.tr_errors).T
    plt.hist(tr_decals[0], density=False)  # density=False would make counts
    plt.xlabel('Z tr error');
    plt.savefig('./images/Z_tr_error.png')
    plt.close()
    plt.hist(tr_decals[1], density=False)  # density=False would make counts
    plt.xlabel('Y tr error');
    plt.savefig('./images/Y_tr_error.png')
    plt.close()
    plt.hist(tr_decals[2], density=False)  # density=False would make counts
    plt.xlabel('X tr error');
    plt.savefig('./images/X_tr_error.png')

    print("end")

if __name__ == "__main__":
    main()