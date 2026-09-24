from collections.abc import Sequence
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.datasets import S3DIS


def collate_s3dis(
    samples: Sequence[Data],
    num_points: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create dense point-cloud batches from PyG S3DIS samples.

    Returns point features with shape ``(batch_size, num_points, 3 + num_features)``
    and labels with shape ``(batch_size, num_points)``.
    """
    if not samples:
        raise ValueError("Cannot collate an empty S3DIS batch")
    if num_points <= 0:
        raise ValueError("num_points must be greater than zero")

    feature_count = samples[0].x.shape[-1]
    input_batch = torch.zeros(
        len(samples),
        num_points,
        3 + feature_count,
        dtype=samples[0].pos.dtype,
    )
    label_batch = torch.full(
        (len(samples), num_points),
        fill_value=-1,
        dtype=samples[0].y.dtype,
    )

    for batch_index, sample in enumerate(samples):
        if sample.x.shape[-1] != feature_count:
            raise ValueError("All S3DIS samples must have the same feature count")

        point_count = min(sample.pos.shape[0], num_points)
        input_batch[batch_index, :point_count] = torch.cat(
            (sample.pos[:point_count], sample.x[:point_count]),
            dim=-1,
        )
        label_batch[batch_index, :point_count] = sample.y[:point_count]

    return input_batch, label_batch


def create_s3dis_dataloader(
    root: str,
    batch_size: int,
    *,
    train: bool = True,
    test_area: int = 6,
    num_points: int = 4096,
    shuffle: bool | None = None,
    num_workers: int = 0,
    drop_last: bool = True,
) -> DataLoader:
    """Load S3DIS through PyG and yield dense batches for PointNet++."""
    dataset = S3DIS(root=root, test_area=test_area, train=train)
    if shuffle is None:
        shuffle = train

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=partial(collate_s3dis, num_points=num_points),
        pin_memory=torch.cuda.is_available(),
    )