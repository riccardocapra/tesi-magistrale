from utils import *


def depth_tensor_creation(depth):
    # print("point cloud", c, " in esecuzione.")
    h, w = 352, 1216
    to_tensor = transforms.ToTensor()
    depth = depth.T
    perturbation_vector = [0, 0, 45]
    new_h_init = perturbation(global_dataset.calib.T_cam2_velo, perturbation_vector, [0, 0, 0])
    depth = pcl_rt(depth, new_h_init, global_dataset.calib.K_cam2)
    depth_image = depth_image_creation(depth, h, w)
    depth_image_tensor = to_tensor(depth_image)
    # depth_images_tensor.append(depth_image_tensor)
    # c+=1
    return depth_image_tensor


def data_formatter_pcl(dataset):
    print("---- VELO_IMAGES FORMATTING BEGUN ---")
    velo_files = dataset.velo_files[:50]
    start_time = datetime.now()
    depths = []
    for file in velo_files:
        scan = np.fromfile(file, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        depths.append(scan)
    # scan = np.fromfile(velo_files, dtype=np.float32)
    # scan.reshape((-1, 4))
    end_time = datetime.now()
    end_time = end_time - start_time
    print("---- Secondi passati per conversione lista: " + str(end_time.total_seconds()))

    start_time = datetime.now()
    # c = 1
    depth_images_tensor = []
    h, w = 352, 1216
    perturbation_vector = [0, 0, 45]
    to_tensor = transforms.ToTensor()
    for depth in depths:
        depth = depth.T
        new_h_init = perturbation(dataset.calib.T_cam2_velo, perturbation_vector, [0, 0, 0])
        depth = pcl_rt(depth, new_h_init, dataset.calib.K_cam2)
        depth_image = depth_image_creation(depth, h, w)
        depth_image_tensor = depth_image_tensor.append(to_tensor(depth_image))

    end_time = datetime.now()
    end_time = end_time - start_time
    print("---- Secondi passati: " + str(end_time.total_seconds()))
    print("---- VELO_IMAGES FORMATTING ENDED ---")
    return depth_images_tensor


def data_formatter_pcl_multiprocessing(dataset):
    print("---- VELO_IMAGES FORMATTING BEGUN ---")
    velo_files = dataset.velo_files[:50]
    start_time = datetime.now()
    with multiprocessing.Pool(12) as p:
        depths = p.map(scan_loader, velo_files)
        p.terminate()
    # scan = np.fromfile(velo_files, dtype=np.float32)
    # scan.reshape((-1, 4))
    end_time = datetime.now()
    end_time = end_time - start_time
    print("---- Secondi passati per conversione lista: " + str(end_time.total_seconds()))

    depth_images_tensor = []
    start_time = datetime.now()
    # c = 1
    with multiprocessing.Pool(12) as p:
        depth_images_tensor = depth_images_tensor.append(p.map(depth_tensor_creation, depths))

    end_time = datetime.now()
    end_time = end_time - start_time
    print("---- Secondi passati: " + str(end_time.total_seconds()))
    print("---- VELO_IMAGES FORMATTING ENDED ---")

    return depth_images_tensor


global_dataset = []


def data_formatter(basedir):
    print("-- DATA FORMATTING BEGUN ---")
    sequence = '00'
    global global_dataset
    global_dataset = pykitti.odometry(basedir, sequence)
    # depth_array = data_formatter_pcl(dataset)
    depth_array = data_formatter_pcl_multiprocessing(global_dataset)
    # depth = torch.from_numpy(depth / (2 ** 16)).float()
    # Le camere 2 e 3 sono quelle a colori, verificato. Mi prendo la 2.
    rgb_files = global_dataset.cam2_files
    # print(rgb_files)
    to_tensor = transforms.ToTensor()
    normalization = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # c=0m
    # rgb = []
    rgb_img = Image.open(rgb_files[0])
    # print(rgb_img)
    rgb = to_tensor(rgb_img)
    # print("Dimensione tensore: "+str(rgb.shape))
    # rgb = normalization(rgb)
    print("-- DATA FORMATTING ENDED ---")
    return rgb.float(), depth_array.float()
