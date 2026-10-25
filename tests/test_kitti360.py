import numpy as np
import pytest
import torch

from data.collect.kitti360 import _get_projection_output_path
from data.load.kitti360 import Kitti


PROJECTION_VALUES = {
    "ground_truth": 11,
    "pinhole": 22,
    "cylindrical": 33,
}


@pytest.fixture
def projection_dataset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data/saved/kitti360_selection").mkdir(parents=True)

    dataset_dir = tmp_path / "dataset"
    for projection in PROJECTION_VALUES:
        (dataset_dir / f"all_{projection}_projections").mkdir(parents=True)

    for scene_index in range(10):
        filename = f"scene_{scene_index:02}.npy"
        for projection, value in PROJECTION_VALUES.items():
            point_cloud = np.full((8, 6), value, dtype=np.float32)
            point_cloud[:, 4] = 2
            np.save(dataset_dir / f"all_{projection}_projections" / filename, point_cloud)

    return dataset_dir


@pytest.mark.parametrize("projection_type", PROJECTION_VALUES)
def test_kitti_loads_selected_projection(projection_dataset, projection_type):
    dataset = Kitti(
        directory=projection_dataset,
        projection_type=projection_type,
        set="train",
        force_organize=True,
        std_num_points=8,
        num_samples=10,
    )

    assert len(dataset) == 8
    selected_names = [path.name for path in dataset.file_paths[projection_type]]
    assert all(
        [path.name for path in dataset.file_paths[projection]] == selected_names
        for projection in PROJECTION_VALUES
    )

    points, labels = dataset[0]
    assert points.shape == (8, 4)
    assert labels.shape == (8,)
    assert labels.dtype == torch.long
    assert torch.all(points[:, 0] == PROJECTION_VALUES[projection_type])


def test_kitti_reports_mismatched_projection_files(projection_dataset):
    (projection_dataset / "all_cylindrical_projections" / "scene_09.npy").unlink()

    with pytest.raises(Exception, match="Different projection datasets do not match"):
        Kitti(
            directory=projection_dataset,
            projection_type="ground_truth",
            set="train",
            force_organize=True,
            std_num_points=8,
            num_samples=10,
        )


def test_projection_output_path_overwrites_or_appends_duplicate_suffix(tmp_path):
    existing_output = tmp_path / "drive_0001.npy"
    existing_output.write_bytes(b"existing")

    assert _get_projection_output_path(
        tmp_path, "drive_", "0001", ignore_duplicates=True
    ) == existing_output
    assert _get_projection_output_path(
        tmp_path, "drive_", "0001", ignore_duplicates=False
    ) == tmp_path / "drive_0001d.npy"

    (tmp_path / "drive_0001d.npy").write_bytes(b"existing duplicate")
    assert _get_projection_output_path(
        tmp_path, "drive_", "0001", ignore_duplicates=False
    ) == tmp_path / "drive_0001dd.npy"
