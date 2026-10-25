import torch

from models.pointnetpp import Model, _farthest_point_sample, _knn_indices


def test_pointnetpp_logits_are_finite_with_padded_points():
    torch.manual_seed(0)
    model = Model(num_classes=13, num_features=9)
    points = torch.cat(
        (torch.randn(1, 64, 12), torch.zeros(1, 64, 12)),
        dim=1,
    )

    logits = model(points)

    assert logits.shape == (1, 128, 13)
    assert torch.isfinite(logits).all()

    valid_points = (points != 0.0).any(dim=-1)
    targets = torch.zeros((1, 128), dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(logits[valid_points], targets[valid_points])
    loss.backward()

    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_pointnetpp_sampling_and_grouping_exclude_zero_padding():
    points = torch.tensor(
        [[[100.0, 0.0, 0.0, 1.0], [101.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]
    )
    valid_points = (points != 0.0).any(dim=-1)

    center_indices = _farthest_point_sample(points, num_samples=2, valid_points=valid_points)
    centers = points[torch.arange(points.shape[0])[:, None], center_indices]
    neighbor_indices, neighbor_valid = _knn_indices(
        points[:, :, :3],
        centers[:, :, :3],
        valid_points,
        num_neighbors=8,
    )
    groups = points[
        torch.arange(points.shape[0])[:, None, None],
        neighbor_indices,
    ]

    assert ((centers != 0.0).any(dim=-1)).all()
    assert groups.shape == (1, 2, 8, 4)
    assert int(neighbor_valid.sum()) == 4
    assert ((groups != 0.0).any(dim=-1) | ~neighbor_valid).all()
