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
        print(f"Testing | loss= {result['loss']}")

    results['status'] = True
    results['loss']= result['loss']
    results['model']= model
    
    return results
