import torch

from models.pointnetpp import Model


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


def test_pointnetpp_fps_excludes_zero_padding():
    points = torch.tensor(
        [[[100.0, 0.0, 0.0, 1.0], [101.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]
    )

    centers, groups, _ = Model._fps(points, num_samples=2, num_features=1)

    assert ((centers != 0.0).any(dim=-1)).all()
    assert int((groups != 0.0).any(dim=-1).sum()) == 2
