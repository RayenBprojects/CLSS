import torch

import time

from tools.accuracy import OAcc, MAcc, MIoU


def _validate_batch_sizes(batch_index, **tensors):
    batch_sizes = {name: tensor.shape[0] for name, tensor in tensors.items()}
    if len(set(batch_sizes.values())) > 1:
        sizes = ", ".join(f"{name}={size}" for name, size in batch_sizes.items())
        raise ValueError(f"Mismatched batch sizes at batch {batch_index}: {sizes}")


def _valid_points(X, truth_X=None, batch_index=None):
    valid_points = (X != 0.0).any(dim=-1)
    if truth_X is None:
        return valid_points

    truth_valid_points = (truth_X != 0.0).any(dim=-1)
    if valid_points.shape != truth_valid_points.shape:
        raise ValueError(
            f"Paired dataloaders yielded mismatched point batches at batch "
            f"{batch_index}: input mask has shape {tuple(valid_points.shape)}, "
            f"truth mask has shape {tuple(truth_valid_points.shape)}"
        )
    return valid_points & truth_valid_points


def train(model, dataloader, device, loss_fn, optimizer, print_batch_data= True) -> dict[str: float]:
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
        _validate_batch_sizes(index, X=X, Y=Y)
        valid_points = (X != 0.0).any(dim=-1)
        if not valid_points.any():
            raise ValueError("A training batch contains no valid points")

        Y_pred = model(X)

        labels= torch.arange(Y_pred.shape[-1]).to(device)
        Y_pred_choice= torch.argmax(Y_pred, dim= -1)
        oAcc.calculate_oAcc(Y_pred_choice, Y, valid_points)
        mAcc.calculate_mAcc(Y_pred_choice, Y, labels, valid_points)
        mIoU.calculate_mIoU(Y_pred_choice, Y, labels, valid_points)

        loss= loss_fn(Y_pred[valid_points], Y[valid_points])
        returned_loss += loss.item()
        loss.backward()
        if print_batch_data:
            print(f"\rBatch {index+1}/{num_batches}: loss= {loss.item()} | overall accuracy= {oAcc.get_oAcc()} | mean class accuracy= {mAcc.get_mAcc()} | mean intersection over union= {mIoU.get_mIoU()}", end= "")
        optimizer.step()
        optimizer.zero_grad()

    if print_batch_data:
        print("\r")
    else:
        print("")
    end_time = time.perf_counter()
    execution_time = float(end_time - start_time)/3600

    return {'loss': returned_loss / num_batches, 'oAcc' : oAcc.get_oAcc(), 'mAcc' : mAcc.get_mAcc(), 'mIoU' : mIoU.get_mIoU(), 'time': execution_time}


def test(model, dataloader, device, loss_fn, truth_dataloader= None, print_batch_data= True, model_eval= True) -> dict[str: float]:
    if model_eval:
        model.eval()

    num_batches = len(dataloader)
    if num_batches == 0:
        raise ValueError("The dataloader is empty")
    if truth_dataloader is not None and len(truth_dataloader) != num_batches:
        raise ValueError("The truth and input dataloaders have different numbers of batches")

    loss = 0.0

    oAcc = OAcc()
    mAcc= MAcc()
    mIoU= MIoU()

    with torch.no_grad():
        if truth_dataloader is None:
            batches = ((None, batch) for batch in dataloader)
        else:
            batches = zip(truth_dataloader, dataloader)

        for index, (truth_batch, batch) in enumerate(batches):
            X, Y = batch[0].to(device), batch[1].to(device)
            if truth_batch is None:
                truth_X = None
                _validate_batch_sizes(index, X=X, Y=Y)
            else:
                truth_X, truth_Y = truth_batch[0].to(device), truth_batch[1].to(device)
                _validate_batch_sizes(
                    index,
                    input_X=X,
                    input_Y=Y,
                    truth_X=truth_X,
                    truth_Y=truth_Y,
                )
                Y = truth_Y

            valid_points = _valid_points(X, truth_X, index)
            if not valid_points.any():
                raise ValueError("A batch contains no valid points")

            Y_pred = model(X)

            labels= torch.arange(Y_pred.shape[-1]).to(device)
            Y_pred_choice= torch.argmax(Y_pred, dim= -1)
            oAcc.calculate_oAcc(Y_pred_choice, Y, valid_points)
            mAcc.calculate_mAcc(Y_pred_choice, Y, labels, valid_points)
            mIoU.calculate_mIoU(Y_pred_choice, Y, labels, valid_points)

            loss += loss_fn(Y_pred[valid_points], Y[valid_points]).item()

            if print_batch_data:
                print(f"\rBatch {index+1}/{num_batches}: overall accuracy= {oAcc.get_oAcc()} | mean class accuracy= {mAcc.get_mAcc()} | mean intersection over union= {mIoU.get_mIoU()}", end= "")
    if print_batch_data:
        print("\r")
    else:
        print("")

    return {'loss' : float(loss)/num_batches, 'oAcc' : oAcc.get_oAcc(), 'mAcc' : mAcc.get_mAcc(), 'mIoU' : mIoU.get_mIoU()}


def compare(truth_model, model, truth_dataloader, dataloader, device, loss_fn, print_batch_data= True) -> dict[str: float]:
    #truth_model.eval()
    #model.eval()

    num_batches = len(dataloader)
    if num_batches == 0:
        raise ValueError("The dataloader is empty")

    oAcc = OAcc()
    mAcc= MAcc()
    mIoU= MIoU()

    loss= 0.0

    with torch.no_grad():
        for index, ((truth_X, truth_Y), (X, Y)) in enumerate(zip(truth_dataloader, dataloader)):
            truth_X, truth_Y, X, Y= truth_X.to(device), truth_Y.to(device), X.to(device), Y.to(device)
            _validate_batch_sizes(index, truth_X=truth_X, truth_Y=truth_Y, X=X, Y=Y)
            valid_points = _valid_points(X, truth_X, index)
            if not valid_points.any():
                raise ValueError("A batch contains no valid points")
            
            truth_Y, Y= truth_model(X), model(X)

            labels= torch.arange(truth_Y.shape[-1]).to(device)
            truth_Y_choice= torch.argmax(truth_Y, dim= -1)
            Y_choice= torch.argmax(Y, dim= -1)
            oAcc.calculate_oAcc(Y_choice, truth_Y_choice, valid_points)
            mAcc.calculate_mAcc(Y_choice, truth_Y_choice, labels, valid_points)
            mIoU.calculate_mIoU(Y_choice, truth_Y_choice, labels, valid_points)

            loss += loss_fn(Y[valid_points], truth_Y_choice[valid_points]).item()

            if print_batch_data:
                print(f"\rBatch {index+1}/{num_batches}: overall accuracy= {oAcc.get_oAcc()} | mean class accuracy= {mAcc.get_mAcc()} | mean intersection over union= {mIoU.get_mIoU()}", end= "")
    if print_batch_data:
        print("\r")
    else:
        print("")

    return {'loss' : float(loss)/num_batches, 'oAcc' : oAcc.get_oAcc(), 'mAcc' : mAcc.get_mAcc(), 'mIoU' : mIoU.get_mIoU()}