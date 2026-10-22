import numpy as np
import torch

from data.load.kitti360 import Kitti


def create_projection_directories(root):
    for projection in ("2d", "3d"):
        for split in ("", "_test", "_eval"):
            (root / f"all_{projection}_projections{split}").mkdir(parents=True)


def test_kitti_only_loads_npy_files(tmp_path):
    create_projection_directories(tmp_path)
    projection_dir = tmp_path / "all_2d_projections"
    np.save(projection_dir / "a.npy", np.random.rand(8, 6))
    np.save(projection_dir / "b.npy", np.random.rand(12, 6))
    (projection_dir / "notes.txt").write_text("not a point cloud")

    dataset = Kitti(
        directory=tmp_path,
        projection="2d",
        set="train",
        organize=False,
        std_num_points=12,
    )

    assert [p.name for p in dataset.file_paths] == ["a.npy", "b.npy"]
    assert len(dataset) == 2
    assert dataset[0][0].shape == (12, 4)
    assert dataset[0][1].shape == (12,)
    assert dataset[0][1].dtype == torch.long
    assert dataset[1][0].shape == (12, 4)
    assert dataset[1][1].shape == (12,)


def test_kitti_reports_mismatched_projection_files(tmp_path):
    create_projection_directories(tmp_path)
    np.save(tmp_path / "all_2d_projections" / "a.npy", np.zeros((8, 6)))
    np.save(tmp_path / "all_3d_projections" / "b.npy", np.zeros((8, 6)))

    try:
        Kitti(
            directory=tmp_path,
            projection="2d",
            set="train",
            organize=True,
            std_num_points=8,
        )
    except ValueError as error:
        assert str(error) == "2D and 3D projection directories must contain matching .npy files"
    else:
        raise AssertionError("Mismatched projection files should be rejected")