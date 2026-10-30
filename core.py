import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim

from models import pointnetpp

from tools import train_test
from data.load.kitti360 import Kitti

import os

import dotenv

from pathlib import Path

from data.collect.kitti360 import project_cylindrical_data, project_ground_truth_data, project_pinhole_data

dotenv.load_dotenv()
KITTI_360_DATASET_DIR = os.getenv("KITTI_360_DATASET_DIR")

WEIGHTS= [
1.4129499197006226,
0.0,
0.0,
0.0,
0.0,
0.0,
0.01741851307451725,
7.1900129318237305,
0.2356705367565155,
0.019624188542366028,
0.20727452635765076,
0.13224773108959198,
0.030789798125624657, 
0.012895463965833187,
0.19053854048252106,
1.008994460105896,
0.6853728294372559,
0.009955069981515408,
0.0, 
0.08083534240722656,
0.004743135999888182,
0.8439980149269104,
0.04647771269083023,
0.0,
0.016481032595038414, 
1.0697057247161865,
0.05648650974035263,
0.16119875013828278,
5.597163200378418,
0.1295274794101715,
0.022218089550733566,
2.653687000274658,
0.025828640908002853,
0.015011333860456944,
0.017142565920948982,
0.008132226765155792,
0.2913314700126648,
0.008886005729436874,
0.0036444910801947117,
0.007300559431314468,
0.04280197620391846,
0.006305180490016937,
0.031746309250593185,
0.24166610836982727,
0.012485727667808533,
0.0
]

def collect_kitti_data(clean_existing_files : bool = False):
    dotenv.load_dotenv()

    KITTI_360_DATASET_DIR= Path(os.getenv("KITTI_360_DATASET_DIR"))

    (KITTI_360_DATASET_DIR / "all_ground_truth_projections").mkdir(parents= True, exist_ok= True)
    (KITTI_360_DATASET_DIR / "all_pinhole_projections").mkdir(parents= True, exist_ok= True)
    (KITTI_360_DATASET_DIR / "all_cylindrical_projections").mkdir(parents= True, exist_ok= True)

    if clean_existing_files:
        print("-- Cleaning --")
        ground_truth_items= next(os.walk(KITTI_360_DATASET_DIR / "all_ground_truth_projections"))[2]
        pinhole_items= next(os.walk(KITTI_360_DATASET_DIR / "all_pinhole_projections"))[2]
        cylindrical_items= next(os.walk(KITTI_360_DATASET_DIR / "all_cylindrical_projections"))[2]

        for item in ground_truth_items:
            (KITTI_360_DATASET_DIR/"all_ground_truth_projections"/item).unlink(missing_ok= True)

        for item in pinhole_items:
                    (KITTI_360_DATASET_DIR/"all_pinhole_projections"/item).unlink(missing_ok= True)

        for item in cylindrical_items:
                    (KITTI_360_DATASET_DIR/"all_cylindrical_projections"/item).unlink(missing_ok= True)


    for i in range(0,11):
        if i == 1:
            continue
        number = ""
        if i < 10:
            number = f"0{i}"
        else:
            number = f"{i}"

        print(f"-- Collecting From Drive {i+1}/{11} --")

        project_ground_truth_data(
            absolute_mesh_directory= Path(KITTI_360_DATASET_DIR / f'data_3d_semantics/train/2013_05_28_drive_00{number}_sync/'),  
            absolute_scan_directory= Path(KITTI_360_DATASET_DIR / f'data_3d_raw/2013_05_28_drive_00{number}_sync/velodyne_points/data'),
            absolute_destination_directory= Path(KITTI_360_DATASET_DIR / f'all_ground_truth_projections'),
            prefix= f"{number}_",
            absolute_calibration_directory= Path(KITTI_360_DATASET_DIR / f'calibration'),
            absolute_poses_directory= Path(KITTI_360_DATASET_DIR / f'data_poses/2013_05_28_drive_00{number}_sync'),
            ignore_duplicates = True
            )
        
        project_pinhole_data(
            absolute_image_directory =   Path(KITTI_360_DATASET_DIR / f'data_2d_semantics/train/2013_05_28_drive_00{number}_sync/image_00/instance'),
            absolute_scan_directory = Path(KITTI_360_DATASET_DIR / f'data_3d_raw/2013_05_28_drive_00{number}_sync/velodyne_points/data'),
            absolute_destination_directory = Path(KITTI_360_DATASET_DIR / f'all_pinhole_projections'),
            prefix= f"{number}_",
            absolute_calibration_directory= Path(KITTI_360_DATASET_DIR / f'calibration'),
            data_format = 'KITTI',
            sample_only = False,
            ignore_duplicates=True
        )

        project_cylindrical_data(
            absolute_image_directory =   Path(KITTI_360_DATASET_DIR / f'data_2d_semantics/train/2013_05_28_drive_00{number}_sync/image_00/instance'),
            absolute_scan_directory = Path(KITTI_360_DATASET_DIR / f'data_3d_raw/2013_05_28_drive_00{number}_sync/velodyne_points/data'),
            absolute_destination_directory = Path(KITTI_360_DATASET_DIR / f'all_cylindrical_projections'),
            prefix= f"{number}_",
            LIDAR_relative_xyz = (0.79, 0.32, -0.18),
            image_margins = (0.10, 0.00, 0.10, 0.0),
            LIDAR_margins = (0.0, 0.3, 115.0, 235.0), # remember, angle goes counterclockwise
            data_format = 'KITTI',
            sample_only = False,
            ignore_duplicates=True
        )

    print('-- Removing Mismatched Files --')
    collected_cylindrical_projections= next(os.walk(KITTI_360_DATASET_DIR / f'all_cylindrical_projections'))[2]
    marked_for_deletion_cylindrical= []

    collected_ground_truth_projections= next(os.walk(KITTI_360_DATASET_DIR / f'all_ground_truth_projections'))[2]
    marked_for_deletion_ground_truth= []

    for item in collected_cylindrical_projections:
        if item not in collected_ground_truth_projections:
            marked_for_deletion_cylindrical.append(item)

    for item in collected_ground_truth_projections:
        if item not in collected_cylindrical_projections:
            marked_for_deletion_ground_truth.append(item)

    for item in marked_for_deletion_cylindrical:
        (KITTI_360_DATASET_DIR/f'all_cylindrical_projections/{item}').unlink(missing_ok= True)

    for item in marked_for_deletion_ground_truth:
        (KITTI_360_DATASET_DIR/f'all_ground_truth_projections/{item}').unlink(missing_ok= True)

    cylindrical_and_ground_truth_projections= next(os.walk(KITTI_360_DATASET_DIR / f'all_cylindrical_projections'))[2]
    marked_for_deletion_cylindrical_and_ground_truth= []

    collected_pinhole_projections= next(os.walk(KITTI_360_DATASET_DIR / f'all_pinhole_projections'))[2]
    marked_for_deletion_pinhole= []

    for item in cylindrical_and_ground_truth_projections:
        if item not in collected_pinhole_projections:
            marked_for_deletion_cylindrical_and_ground_truth.append(item)

    for item in collected_pinhole_projections:
        if item not in cylindrical_and_ground_truth_projections:
            marked_for_deletion_pinhole.append(item)

    for item in marked_for_deletion_cylindrical_and_ground_truth:
        (KITTI_360_DATASET_DIR/f'all_cylindrical_projections/{item}').unlink(missing_ok= True)
        (KITTI_360_DATASET_DIR/f'all_ground_truth_projections/{item}').unlink(missing_ok= True)

    for item in marked_for_deletion_pinhole:
        (KITTI_360_DATASET_DIR/f'all_pinhole_projections/{item}').unlink(missing_ok= True)


async def train_pointnetpp(input_model= None, batch_size= 20, learning_rate= 1e-3, optimizer_choice= "Adam", num_epochs= 1, start_epoch= 0, print_batch_data= True):
    dataset= 'kitti360'

    results = {'dataset': 'KITTI360'}
    epochs = num_epochs

    dataloader = DataLoader(
        dataset=Kitti(KITTI_360_DATASET_DIR, projection_type='ground_truth', set='train', std_num_points=90000, num_samples=61280), #124792 | 61280
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    testing_dataloader = DataLoader(
        dataset=Kitti(KITTI_360_DATASET_DIR, projection_type='ground_truth', set='test', std_num_points=90000, num_samples=61280), #124792 | 61280
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    num_features = 1
    num_classes = 46

    model= input_model
    if model== None:
        model = pointnetpp.Model(
            num_classes=num_classes,
            num_features=num_features,
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to train this model")

    device = torch.device("cuda")
    model.to(device)

    loss_function = nn.CrossEntropyLoss(ignore_index=-1, weight= torch.tensor(WEIGHTS).to(device))

    optimizer= optim.Adam(model.parameters(), lr= learning_rate) if optimizer_choice == "Adam" else optim.SGD(model.parameters(), lr= learning_rate)

    for i in range(epochs):
        print(f"-Epoch {i+1+start_epoch}-")
        result = await train_test.train(
                model=model,
                dataloader=dataloader,
                device=device,
                loss_fn=loss_function,
                optimizer=optimizer,
                print_batch_data= print_batch_data
            )
        print(f"Training | loss= {result['loss']} | overall accuracy= {result['oAcc']} | mean class accuracy= {result['mAcc']} | mean intersection over union= {result['mIoU']} | time= {result['time']} hours")

        result = await train_test.test(
            model= model,
            dataloader=testing_dataloader,
            device=device,
            loss_fn=loss_function,
        )
        print(f"Testing | loss= {result['loss']}| overall accuracy= {result['oAcc']} | mean class accuracy= {result['mAcc']} | mean intersection over union= {result['mIoU']}")

    results['status'] = True
    results['loss']= result['loss']
    results['model']= model
    
    return results