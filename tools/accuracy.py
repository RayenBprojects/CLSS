import torch


def get_mIoU(predicted_Y : torch.Tensor, ground_truth_Y : torch.Tensor) -> float:
    '''Expects input in batches with label integer'''

    labels= range(torch.min(predicted_Y), torch.max(predicted_Y))
    scores= torch.zeros()
    for i in labels:
        pass
