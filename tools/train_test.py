import asyncio

import torch

import time


async def train(model, dataloader, device, loss_fn, optimizer, settings= {}):
    start_time= time.perf_counter()
    model.train()

    num_batches = len(dataloader)
    if num_batches == 0:
        raise ValueError("The training dataloader is empty")

    returned_loss = 0.0
    
    for X, Y in dataloader:
        X, Y= X.to(device), Y.to(device)
        valid_points = (X != 0.0).any(dim=-1)
        if not valid_points.any():
            raise ValueError("A training batch contains no valid points")

        Y_pred = model(X)

        loss= loss_fn(Y_pred[valid_points], Y[valid_points])
        returned_loss += loss.item()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    end_time = time.perf_counter()
    execution_time = float(end_time - start_time)/3600

    return {'loss': returned_loss / num_batches, 'accuracy': 0.0, 'time': execution_time}


async def test(model, dataloader, device, loss_fn, settings= {}) -> dict[str: float]:
    model.eval()

    num_batches = len(dataloader)
    if num_batches == 0:
        raise ValueError("The testing dataloader is empty")

    loss = 0.0

    with torch.no_grad():
        for X, Y in dataloader:
            X, Y= X.to(device), Y.to(device)
            valid_points = (X != 0.0).any(dim=-1)
            if not valid_points.any():
                raise ValueError("A testing batch contains no valid points")

            Y_pred = model(X)

            loss += loss_fn(Y_pred[valid_points], Y[valid_points]).item()

    return {'loss' : float(loss)/num_batches, 'accuracy' : 0.0} # Accuracy not implemented yet