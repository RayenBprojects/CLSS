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
