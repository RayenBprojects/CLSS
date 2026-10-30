import torch

from tools.accuracy import MAcc, MIoU, OAcc


def test_metrics_accumulate_counts_and_ignore_invalid_points():
    labels = torch.tensor([0, 1, 2])
    batches = [
        (
            torch.tensor([[0, 0, 1, 0]]),
            torch.tensor([[0, 1, 1, 99]]),
            torch.tensor([[True, True, True, False]]),
        ),
        (
            torch.tensor([[1, 1, 2, 2]]),
            torch.tensor([[1, 2, 2, 99]]),
            torch.tensor([[True, True, True, False]]),
        ),
    ]

    overall_accuracy = OAcc()
    mean_accuracy = MAcc()
    mean_iou = MIoU()
    for predicted, ground_truth, valid_mask in batches:
        overall_accuracy.calculcate_oAcc(predicted, ground_truth, valid_mask)
        mean_accuracy.calculate_mAcc(predicted, ground_truth, labels, valid_mask)
        mean_iou.calculate_mIoU(predicted, ground_truth, labels, valid_mask)

    assert overall_accuracy.get_oAcc() == 2 / 3
    assert abs(mean_accuracy.get_mAcc() - 13 / 18) < 1e-6
    assert mean_iou.get_mIoU() == 0.5
