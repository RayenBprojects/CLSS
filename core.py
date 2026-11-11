import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim

from models import pointnetpp

from tools import train_test
from tools.model_viewer import load_saved_model
from tools.accuracy import IGNORE_INDEX, OAcc, MAcc, MIoU
from data.load.kitti360 import Kitti

import os

import dotenv

from pathlib import Path

from data.collect.kitti360 import project_cylindrical_data, project_ground_truth_data, project_pinhole_data

dotenv.load_dotenv()
KITTI_360_DATASET_DIR = os.getenv("KITTI_360_DATASET_DIR")


WEIGHTS= [
    0.019375182688236237, 
    0.03226417675614357, 
    0.0325198695063591, 
    0.04837179183959961, 
    0.0878184363245964, 
    0.21050558984279633, 
    2.262850761413574, 
    0.3857010304927826, 
    0.019117509946227074, 
    0.04239395260810852, 
    0.0, 
    0.6294781565666199, 
    10.990584373474121, 
    0.04600956663489342, 
    0.18214745819568634, 
    0.7701876163482666, 
    0.7574699521064758, 
    0.590023398399353, 
    0.8931823372840881
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


def train_pointnetpp(input_model= None, projection_type= "ground_truth", batch_size= 32, learning_rate= 1e-3, optimizer_choice= "Adam", num_epochs= 1, start_epoch= 0, print_batch_data= True, input_optimizer= None, input_scheduler= None):
    results = {}
    epochs = num_epochs

    dataloader = DataLoader(
        dataset=Kitti(KITTI_360_DATASET_DIR, projection_type=projection_type, set='train', std_num_points=124792, num_samples=61280), #124792 | 61280
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    testing_dataloader = DataLoader(
        dataset=Kitti(KITTI_360_DATASET_DIR, projection_type=projection_type, set='test', std_num_points=124792, num_samples=61280), #124792 | 61280
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    num_features = 1
    num_classes = len(WEIGHTS)

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

    loss_function = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, weight= torch.tensor(WEIGHTS).to(device))

    optimizer = input_optimizer
    if optimizer is None:
        optimizer = (
            optim.Adam(model.parameters(), lr=learning_rate)
            if optimizer_choice == "Adam"
            else optim.SGD(model.parameters(), lr=learning_rate)
        )

    scheduler= input_scheduler
    if scheduler is None:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.7)

    for i in range(epochs):
        print(f"-Epoch {i+1+start_epoch}-")
        result = train_test.train(
                model=model,
                dataloader=dataloader,
                device=device,
                loss_fn=loss_function,
                optimizer=optimizer,
                print_batch_data= print_batch_data
            )
        print(f"Training | loss= {result['loss']} | overall accuracy= {result['oAcc']} | mean class accuracy= {result['mAcc']} | mean intersection over union= {result['mIoU']} | time= {result['time']} hours")

        scheduler.step()

        result = train_test.test(
            model= model,
            dataloader=testing_dataloader,
            device=device,
            loss_fn=loss_function,
            print_batch_data=print_batch_data
        )
        print(f"Testing | loss= {result['loss']}| overall accuracy= {result['oAcc']} | mean class accuracy= {result['mAcc']} | mean intersection over union= {result['mIoU']}\n")

    results['status'] = True
    results['loss']= result['loss']
    results['model']= model
    results['optimizer'] = optimizer
    results['scheduler'] = scheduler
    
    return results


def evaluate_model(model_name, projection_type= "ground_truth", own_data= True, batch_size= 16, print_batch_data= True):
    truth_dataloader= DataLoader(
            dataset= Kitti(KITTI_360_DATASET_DIR, projection_type="ground_truth", set='eval', std_num_points=124792, num_samples=61280),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        ) if not own_data else None
    
    dataloader= DataLoader(
        dataset= Kitti(KITTI_360_DATASET_DIR, projection_type=projection_type, set='eval', std_num_points=124792, num_samples=61280),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to train this model")
    
    device = torch.device("cuda")

    models_directory= Path("./models/saved/")
    model= load_saved_model(model_path= models_directory/model_name, device= device)[0]

    loss_function = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, weight= torch.tensor(WEIGHTS).to(device))

    result = train_test.test(
        model= model,
        dataloader=dataloader,
        device=device,
        loss_fn=loss_function,
        truth_dataloader= truth_dataloader,
        print_batch_data=print_batch_data,
        model_eval= False
    )

    print(f"Evaluation | overall accuracy= {result['oAcc']} | mean class accuracy= {result['mAcc']} | mean intersection over union= {result['mIoU']}")


def compare_models(truth_model_name, model_name, projection_type, batch_size= 16, print_batch_data= True):
    truth_dataloader= DataLoader(
        dataset= Kitti(KITTI_360_DATASET_DIR, projection_type="ground_truth", set='eval', std_num_points=124792, num_samples=61280),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    dataloader= DataLoader(
        dataset= Kitti(KITTI_360_DATASET_DIR, projection_type= projection_type, set='eval', std_num_points=124792, num_samples=61280),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to train this model")
    
    device = torch.device("cuda")

    models_directory= Path("./models/saved/")
    truth_model= load_saved_model(model_path= models_directory/truth_model_name, device= device)[0]
    model= load_saved_model(model_path= models_directory/model_name, device= device)[0]

    loss_function= nn.CrossEntropyLoss(ignore_index= IGNORE_INDEX, weight= torch.tensor(WEIGHTS).to(device))

    result= train_test.compare(truth_model=truth_model, model=model, truth_dataloader= truth_dataloader, dataloader=dataloader, device= device, loss_fn=loss_function, print_batch_data=print_batch_data)

    print(f"Comparison | overall accuracy= {result['oAcc']} | mean class accuracy= {result['mAcc']} | mean intersection over union= {result['mIoU']}")


def evaluate_data(projection_type, batch_size= 16, print_batch_data= True):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = torch.device("cuda")

    ground_truth_dataset= Kitti(
        directory= KITTI_360_DATASET_DIR,
        projection_type= 'ground_truth',
        set= 'train',
        split= False
    )

    ground_truth_dataloader= DataLoader(dataset= ground_truth_dataset, batch_size= batch_size)

    dataset= Kitti(
        directory= KITTI_360_DATASET_DIR,
        projection_type= projection_type,
        set= 'train',
        split= False
    )

    dataloader= DataLoader(dataset= dataset, batch_size= batch_size)

    oAcc= OAcc()
    mAcc= MAcc()
    mIoU= MIoU()

    num_batches= len(ground_truth_dataloader)

    for index, (ground_truth_data, data) in enumerate(zip(ground_truth_dataloader, dataloader)):
        for i in range(2):
            ground_truth_data[i], data[i]= ground_truth_data[i].to(device), data[i].to(device)

        valid_points= ((ground_truth_data[0] != 0).any(dim= -1)) & ((data[0] != 0).any(dim= -1))

        if not valid_points.any():
                    raise ValueError("A batch contains no valid points")

        ground_truth_data, data= ground_truth_data[1].to(device), data[1].to(device) # saves memory

        labels= torch.arange(len(WEIGHTS)).to(device)

        oAcc.calculate_oAcc(ground_truth_data, data, valid_points)
        mAcc.calculate_mAcc(ground_truth_data, data, labels, valid_points)
        mIoU.calculate_mIoU(ground_truth_data, data, labels, valid_points)

        if print_batch_data:
            print(f"\r{projection_type[0].upper()+projection_type[1:]} batch {index+1}/{num_batches}: overall accuracy= {oAcc.get_oAcc()} | mean class accuracy= {mAcc.get_mAcc()} | mean intersection over union= {mIoU.get_mIoU()}", end= "")
        

    if print_batch_data:
        print("\r")
    else:
        print("")

    print(f"{projection_type[0].upper()+projection_type[1:]}: overall accuracy= {oAcc.get_oAcc()} | mean class accuracy= {mAcc.get_mAcc()} | mean intersection over union= {mIoU.get_mIoU()}")