import torch

import time

from tools.accuracy import OAcc, MAcc, MIoU

async def train(model, dataloader, device, loss_fn, optimizer, print_batch_data= True):
    start_time= time.perf_counter()
    model.train()

    num_batches = len(dataloader)
    if num_batches == 0:
        raise ValueError("The training dataloader is empty")

    returned_loss = 0.0

    oAcc = OAcc()
    mAcc= MAcc()
    mIoU= MIoU()
    
    for index, (X, Y) in enumerate(dataloader):
        X, Y= X.to(device), Y.to(device)
        valid_points = (X != 0.0).any(dim=-1)
        if not valid_points.any():
            raise ValueError("A training batch contains no valid points")

        Y_pred = model(X)

        labels= torch.arange(Y_pred.shape[-1]).to(device)
        Y_pred_choice= torch.argmax(Y_pred, dim= -1)
        oAcc.calculcate_oAcc(Y_pred_choice, Y, valid_points)
        mAcc.calculate_mAcc(Y_pred_choice, Y, labels, valid_points)
        mIoU.calculate_mIoU(Y_pred_choice, Y, labels, valid_points)

        loss= loss_fn(Y_pred[valid_points], Y[valid_points])
        returned_loss += loss.item()
        loss.backward()
        if print_batch_data:
            print(f"\rBatch {index+1} | loss= {loss.item()} | overall accuracy= {oAcc.get_oAcc()} | mean class accuracy= {mAcc.get_mAcc()} | mean intersection over union= {mIoU.get_mIoU()}", end= "")
        optimizer.step()
        optimizer.zero_grad()

    if print_batch_data:
        print("\r")
    else:
        print("")
    end_time = time.perf_counter()
    execution_time = float(end_time - start_time)/3600

    return {'loss': returned_loss / num_batches, 'oAcc' : oAcc.get_oAcc(), 'mAcc' : mAcc.get_mAcc(), 'mIoU' : mIoU.get_mIoU(), 'time': execution_time}


async def test(model, dataloader, device, loss_fn) -> dict[str: float]:
    model.eval()

    num_batches = len(dataloader)
    if num_batches == 0:
        raise ValueError("The testing dataloader is empty")

    loss = 0.0

    oAcc = OAcc()
    mAcc= MAcc()
    mIoU= MIoU()

    with torch.no_grad():
        for X, Y in dataloader:
            X, Y= X.to(device), Y.to(device)
            valid_points = (X != 0.0).any(dim=-1)
            if not valid_points.any():
                raise ValueError("A testing batch contains no valid points")

            Y_pred = model(X)

            labels= torch.arange(Y_pred.shape[-1]).to(device)
            Y_pred_choice= torch.argmax(Y_pred, dim= -1)
            oAcc.calculcate_oAcc(Y_pred_choice, Y, valid_points)
            mAcc.calculate_mAcc(Y_pred_choice, Y, labels, valid_points)
            mIoU.calculate_mIoU(Y_pred_choice, Y, labels, valid_points)

            loss += loss_fn(Y_pred[valid_points], Y[valid_points]).item()

    return {'loss' : float(loss)/num_batches, 'oAcc' : oAcc.get_oAcc(), 'mAcc' : mAcc.get_mAcc(), 'mIoU' : mIoU.get_mIoU()}