import asyncio

import torch
from torch import nn


def train(model, dataloader, device, loss_fn, optimizer, settings= {}):
    model.train()

    for X, Y in dataloader:
        X, Y= X.to(device), Y.to(device)

        Y_pred= model(X)

        loss= loss_fn(Y_pred, Y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()


def test(model, dataloader, device, loss_fn, settings= {}) -> dict[str: float]:
    model.eval()

    num_batches= len(dataloader)

    loss = 0.0

    with torch.no_grad():
        for X, Y in dataloader:
            X, Y= X.to(device), Y.to(device)

            Y_pred= model(X)

            loss += loss_fn(Y_pred, Y).item()

    return {'loss' : float(loss)/num_batches, 'accuracy' : 0.0} # Accuracy not implemented yet