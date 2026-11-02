from data.load.kitti360 import Kitti

from tools.accuracy import OAcc, MAcc, MIoU

from pathlib import Path

import dotenv

import os

import torch

from torch.utils.data import DataLoader

from torch import accelerator

import argparse

def main():
    parser= argparse.ArgumentParser()
    parser.add_argument("--batch_size", default= 16)
    args= parser.parse_args()

    dotenv.load_dotenv()

    KITTI_360_DATASET_DIR= Path(os.getenv("KITTI_360_DATASET_DIR"))

    device= accelerator.current_accelerator() if accelerator.is_available() else "cpu"

    ground_truth_dataset= Kitti(
        directory= KITTI_360_DATASET_DIR,
        projection_type= 'ground_truth',
        set= 'train',
        split= False
    )

    ground_truth_dataloader= DataLoader(dataset= ground_truth_dataset, batch_size= int(args.batch_size))

    pinhole_dataset= Kitti(
        directory= KITTI_360_DATASET_DIR,
        projection_type= 'pinhole',
        set= 'train',
        split= False
    )

    pinhole_dataloader= DataLoader(dataset= pinhole_dataset, batch_size= int(args.batch_size))

    cylindrical_dataset= Kitti(
        directory= KITTI_360_DATASET_DIR,
        projection_type= 'cylindrical',
        set= 'train',
        split= False
    )

    cylindrical_dataloader= DataLoader(dataset= cylindrical_dataset, batch_size= int(args.batch_size))

    metrics= {
        'pinhole': {'oAcc': OAcc(), 'mAcc': MAcc(), 'mIoU' : MIoU()},
        'cylindrical' : {'oAcc': OAcc(), 'mAcc': MAcc(), 'mIoU' : MIoU()}
    }

    print("-- Running Evaluation --")
    for ground_truth_data, pinhole_data, cylindrical_data in zip(ground_truth_dataloader, pinhole_dataloader, cylindrical_dataloader):
        for i in range(2):
            ground_truth_data[i], pinhole_data[i], cylindrical_data[i]= ground_truth_data[i].to(device), pinhole_data[i].to(device), cylindrical_data[i].to(device)

        valid_points= ((ground_truth_data[0] != 0).any(dim= -1)) & ((pinhole_data[0] != 0).any(dim= -1) )& ((cylindrical_data[0] != 0).any(dim= -1))
        if not valid_points.any():
                    raise ValueError("A batch contains no valid points")

        ground_truth_data, pinhole_data, cylindrical_data= ground_truth_data[1].to(device), pinhole_data[1].to(device), cylindrical_data[1].to(device) # saves memory

        labels= torch.arange(ground_truth_data.shape[-1]).to(device)

        metrics['pinhole']['oAcc'].calculate_oAcc(ground_truth_data, pinhole_data, valid_points)
        metrics['pinhole']['mAcc'].calculate_mAcc(ground_truth_data, pinhole_data, labels, valid_points)
        metrics['pinhole']['mIoU'].calculate_mIoU(ground_truth_data, pinhole_data, labels, valid_points)

        metrics['cylindrical']['oAcc'].calculate_oAcc(ground_truth_data, cylindrical_data, valid_points)
        metrics['cylindrical']['mAcc'].calculate_mAcc(ground_truth_data, cylindrical_data, labels, valid_points)
        metrics['cylindrical']['mIoU'].calculate_mIoU(ground_truth_data, cylindrical_data, labels, valid_points)

    print(f"Pinhole projection: overall accuracy= {metrics['pinhole']['oAcc'].get_oAcc()} | mean class accuracy= {metrics['pinhole']['mAcc'].get_mAcc()} | mean intersection over union= {metrics['pinhole']['mIoU'].get_mIoU()}")
    print(f"Cylindrical projection: overall accuracy= {metrics['cylindrical']['oAcc'].get_oAcc()} | mean class accuracy= {metrics['cylindrical']['mAcc'].get_mAcc()} | mean intersection over union= {metrics['cylindrical']['mIoU'].get_mIoU()}")

if __name__ == "__main__":
    main()