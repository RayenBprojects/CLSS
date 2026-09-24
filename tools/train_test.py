import asyncio

import torch


async def train(model, dataloader, device, loss_fn, optimizer, settings= {}):
    model.train()

    num_batches = len(dataloader)
    if num_batches == 0:
        raise ValueError("The training dataloader is empty")

    returned_loss = 0.0

    for X, Y in dataloader:
        X, Y= X.to(device), Y.to(device)

        Y_pred = model(X).transpose(1, 2)

        loss= loss_fn(Y_pred, Y)
        returned_loss += loss.item()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    return {'loss': returned_loss / num_batches, 'accuracy': 0.0}


async def test(model, dataloader, device, loss_fn, settings= {}) -> dict[str: float]:
    model.eval()

    num_batches = len(dataloader)
    if num_batches == 0:
        raise ValueError("The testing dataloader is empty")

    loss = 0.0

    with torch.no_grad():
        for X, Y in dataloader:
            X, Y= X.to(device), Y.to(device)

            Y_pred = model(X).transpose(1, 2)

            loss += loss_fn(Y_pred, Y).item()

    return {'loss' : float(loss)/num_batches, 'accuracy' : 0.0} # Accuracy not implemented yet