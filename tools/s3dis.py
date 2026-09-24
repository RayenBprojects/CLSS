import os
import shutil
import ssl
from collections.abc import Sequence
from functools import partial
from urllib.error import URLError
from urllib.request import urlopen

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import extract_zip
from torch_geometric.datasets import S3DIS


class DownloadableS3DIS(S3DIS):
    """S3DIS loader with configurable and bounded archive downloads."""

    def download(self) -> None:
        configured_url = os.getenv("S3DIS_URL")
        urls = [configured_url] if configured_url else [
            "https://huggingface.co/datasets/cminst/S3DIS/resolve/main/"
            "indoor3d_sem_seg_hdf5_data.zip",
            self.url,
        ]
        timeout = float(os.getenv("S3DIS_DOWNLOAD_TIMEOUT", "60"))
        retries = int(os.getenv("S3DIS_DOWNLOAD_RETRIES", "3"))
        archive_path = os.path.join(self.root, "indoor3d_sem_seg_hdf5_data.zip")
        os.makedirs(self.root, exist_ok=True)

        if not os.path.exists(archive_path):
            last_error = None
            for url in urls:
                for attempt in range(retries):
                    partial_path = f"{archive_path}.part"
                    try:
                        with urlopen(
                            url,
                            context=ssl._create_unverified_context(),
                            timeout=timeout,
                        ) as response, open(partial_path, "wb") as archive:
                            shutil.copyfileobj(
                                response,
                                archive,
                                length=10 * 1024 * 1024,
                            )
                        os.replace(partial_path, archive_path)
                        break
                    except (OSError, URLError) as error:
                        last_error = error
                        if os.path.exists(partial_path):
                            os.unlink(partial_path)
                else:
                    continue
                break
            else:
                raise RuntimeError(
                    "Unable to download S3DIS from the configured sources. "
                    "Set S3DIS_URL to another reachable copy of "
                    "indoor3d_sem_seg_hdf5_data.zip, or place that archive "
                    "in the dataset root."
                ) from last_error

        extract_zip(archive_path, self.root)
        os.unlink(archive_path)
        shutil.rmtree(self.raw_dir, ignore_errors=True)
        os.rename(
            os.path.join(self.root, "indoor3d_sem_seg_hdf5_data"),
            self.raw_dir,
        )


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
    dataset = DownloadableS3DIS(root=root, test_area=test_area, train=train)
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