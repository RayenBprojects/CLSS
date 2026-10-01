import torch
from torch import nn
from torch.utils.data import TensorDataset, dataloader
from torch import optim
from torchvision import datasets

from models import pointnetpp

from tools import train_test
from tools.s3dis import create_s3dis_dataloader

import numpy as np

import asyncio
import os
import dotenv


dotenv.load_dotenv()
KITTI_360_DATASET_DIR= os.getenv("KITTI_360_DATASET_DIR")
S3DIS_DATASET_DIR = os.getenv("S3DIS_DATASET_DIR")


async def train_pointnetpp(dataset='s3dis'):
    results= {}
    dataloader= 0
    testing_dataloader= 0
    num_features= 0
    epochs= 20

    if dataset== 'kitti360': # WIP
        results['status']= False
        return results
        num_features= 19
    elif dataset == 's3dis':
        results['dataset'] = 's3dis'
        dataloader = create_s3dis_dataloader(
            root=S3DIS_DATASET_DIR or './data/s3dis',
            batch_size=4,
            train=True,
            test_area=6,
        )
        testing_dataloader = create_s3dis_dataloader(
            root=S3DIS_DATASET_DIR or './data/s3dis',
            batch_size=4,
            train=False,
            test_area=6,
            shuffle=False,
        )
        num_features = dataloader.dataset.num_features


    model = pointnetpp.Model(
        num_classes=13,
        num_features=num_features,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to train this model")

    device = torch.device("cuda")
    model.to(device)

    loss_function = nn.CrossEntropyLoss(ignore_index=-1)

    optimizer= optim.SGD(model.parameters(), lr= 1e-1)

    print('---Training---')
    for i in range(epochs):
        print(f"-Epoch {i+1}-")
        result = await train_test.train(
                model=model,
                dataloader=dataloader,
                device=device,
                loss_fn=loss_function,
                optimizer=optimizer,
            )
        print(f"Training | loss= {result['loss']} | time= {result['time']} hours")

        result = await train_test.test(
            model= model,
            dataloader=testing_dataloader,
            device=device,
            loss_fn=loss_function,
        )
        print(f"Testing | loss= {result['loss']}")

    results['status'] = True
    results['loss']= result['loss']
    results['model']= model
    return results