import torch
from torch import nn

from tools.train_test import compare, test, train


class FixedLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([[2.0, -1.0]]))

    def forward(self, points):
        return self.logits.expand(*points.shape[:2], -1)


def test_train_and_test_ignore_dummy_point_labels():
    points = torch.tensor([[[1.0], [0.0]]])
    labels = torch.tensor([[0, 99]])
    dataloader = [(points, labels)]
    loss_fn = nn.CrossEntropyLoss()

    model = FixedLogits()
    expected_loss = loss_fn(model.logits, torch.tensor([0])).item()

    training_result = train(
        model=model,
        dataloader=dataloader,
        device=torch.device("cpu"),
        loss_fn=loss_fn,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
    )
    testing_result = test(
        model=FixedLogits(),
        dataloader=dataloader,
        device=torch.device("cpu"),
        loss_fn=loss_fn,
    )

    assert training_result["loss"] == expected_loss
    assert testing_result["loss"] == expected_loss
    assert training_result["oAcc"] == 1.0
    assert training_result["mAcc"] == 1.0
    assert training_result["mIoU"] == 1.0
    assert testing_result["oAcc"] == 1.0
    assert testing_result["mAcc"] == 1.0
    assert testing_result["mIoU"] == 1.0


def test_train_and_test_ignore_unlabeled_valid_points():
    points = torch.tensor([[[1.0], [2.0]]])
    labels = torch.tensor([[0, -1]])
    dataloader = [(points, labels)]
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    model = FixedLogits()

    training_result = train(
        model=model,
        dataloader=dataloader,
        device=torch.device("cpu"),
        loss_fn=loss_fn,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
    )
    testing_result = test(
        model=FixedLogits(),
        dataloader=dataloader,
        device=torch.device("cpu"),
        loss_fn=loss_fn,
    )

    assert training_result["oAcc"] == 1.0
    assert training_result["mAcc"] == 1.0
    assert training_result["mIoU"] == 1.0
    assert testing_result["oAcc"] == 1.0
    assert testing_result["mAcc"] == 1.0
    assert testing_result["mIoU"] == 1.0


def test_test_rejects_mismatched_truth_batch_sizes():
    points = torch.ones((16, 1, 1))
    labels = torch.zeros((16, 1), dtype=torch.long)
    truth_points = torch.ones((32, 1, 1))
    truth_labels = torch.zeros((32, 1), dtype=torch.long)

    try:
        test(
            model=FixedLogits(),
            dataloader=[(points, labels)],
            truth_dataloader=[(truth_points, truth_labels)],
            device=torch.device("cpu"),
            loss_fn=nn.CrossEntropyLoss(),
            print_batch_data=False,
        )
    except ValueError as error:
        assert "Mismatched batch sizes" in str(error)
        assert "input_X=16" in str(error)
        assert "truth_X=32" in str(error)
    else:
        raise AssertionError("Mismatched batch sizes should raise ValueError")


def test_compare_reports_metrics():
    points = torch.tensor([[[1.0]]])
    labels = torch.tensor([[0]])
    result = compare(
        truth_model=FixedLogits(),
        model=FixedLogits(),
        dataloader=[(points, labels)],
        device=torch.device("cpu"),
        loss_fn=nn.CrossEntropyLoss(),
        print_batch_data=False,
    )

    assert result["oAcc"] == 1.0
    assert result["mAcc"] == 1.0
    assert result["mIoU"] == 1.0
