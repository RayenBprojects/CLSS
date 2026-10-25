import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim

from models import pointnetpp

from tools import train_test
from data.load.kitti360 import Kitti

import os
import dotenv


dotenv.load_dotenv()
KITTI_360_DATASET_DIR = os.getenv("KITTI_360_DATASET_DIR")


async def train_pointnetpp(dataset='kitti360'):
    if dataset != 'kitti360':
        raise ValueError("Only the KITTI-360 dataset is supported")

    results = {'dataset': 'KITTI360'}
    epochs = 3

    dataloader = DataLoader(
        dataset=Kitti(KITTI_360_DATASET_DIR, projection_type='ground_truth', set='train', std_num_points=90000, num_samples=61280),
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    testing_dataloader = DataLoader(
        dataset=Kitti(KITTI_360_DATASET_DIR, projection_type='ground_truth', set='test', std_num_points=90000, num_samples=61280),
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    num_features = 1
    num_classes = 46


    model = pointnetpp.Model(
        num_classes=num_classes,
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