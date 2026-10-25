import argparse
import os
import tempfile
from pathlib import Path

import dotenv
import numpy as np

from data.collect.kitti360 import project_ground_truth_data


DRIVE_NAME = "2013_05_28_drive_0000_sync"


def _mesh_range(mesh_path):
    start, end = mesh_path.stem.split("_", maxsplit=1)
    return int(start), int(end)


def main():
    parser = argparse.ArgumentParser(description="Project one KITTI scan against ground truth and inspect its labels.")
    parser.add_argument("--index", type=int, default=0, help="Index of an eligible scan to display")
    args = parser.parse_args()

    dotenv.load_dotenv(Path(__file__).resolve().parent / ".env")
    dataset_directory_value = os.getenv("KITTI_360_DATASET_DIR")
    if not dataset_directory_value:
        raise RuntimeError("KITTI_360_DATASET_DIR is missing from the project .env file")
    dataset_directory = Path(dataset_directory_value)
    drive_directory = dataset_directory / "data_3d_semantics/train" / DRIVE_NAME
    static_mesh_directory = drive_directory / "static"
    scan_directory = dataset_directory / "data_3d_raw" / DRIVE_NAME / "velodyne_points/data"
    calibration_directory = dataset_directory / "calibration"
    poses_directory = dataset_directory / "data_poses" / DRIVE_NAME

    scans = sorted(scan_directory.glob("*.bin"))
    static_meshes = sorted(static_mesh_directory.glob("*.ply"))
    mesh_ranges = [(mesh, *_mesh_range(mesh)) for mesh in static_meshes]
    pose_indices = set(np.loadtxt(poses_directory / "poses.txt", usecols=(0,), dtype=int).reshape(-1).tolist())
    eligible_scans = [
        scan
        for scan in scans
        if scan.stem.isdigit()
        and int(scan.stem) in pose_indices
        and any(start <= int(scan.stem) <= end for _, start, end in mesh_ranges)
    ]
    if not eligible_scans:
        raise FileNotFoundError(
            f"No scans have both a pose and static-mesh coverage in {scan_directory} and {static_mesh_directory}"
        )
    if args.index < 0 or args.index >= len(eligible_scans):
        parser.error(f"--index must be between 0 and {len(eligible_scans) - 1}")

    selected_scan = eligible_scans[args.index]
    frame_number = int(selected_scan.stem)
    selected_mesh = next(mesh for mesh, start, end in mesh_ranges if start <= frame_number <= end)

    with tempfile.TemporaryDirectory(prefix="ground-truth-visualization-") as temporary_directory:
        temporary_path = Path(temporary_directory)
        staged_mesh_directory = temporary_path / "mesh"
        staged_static_directory = staged_mesh_directory / "static"
        staged_scans = temporary_path / "scans"
        output_directory = temporary_path / "output"
        staged_static_directory.mkdir(parents=True)
        staged_scans.mkdir()
        (staged_static_directory / selected_mesh.name).symlink_to(selected_mesh.resolve())
        (staged_scans / selected_scan.name).symlink_to(selected_scan.resolve())

        result = project_ground_truth_data(
            absolute_mesh_directory=str(staged_mesh_directory),
            absolute_scan_directory=str(staged_scans),
            absolute_destination_directory=str(output_directory),
            prefix="",
            absolute_calibration_directory=str(calibration_directory),
            absolute_poses_directory=str(poses_directory),
            ignore_duplicates=True,
        )
        if result != 0:
            raise RuntimeError(f"Ground-truth projection failed for scan {selected_scan.stem}")

        output_path = output_directory / f"{selected_scan.stem}.npy"
        if not output_path.is_file():
            raise RuntimeError(
                f"Ground-truth projection did not create {selected_scan.stem}; "
                f"verify pose and mesh coverage for {selected_mesh.name}"
            )
        labeled_points = np.load(output_path)
        if labeled_points.ndim != 2 or labeled_points.shape[1] < 6:
            raise ValueError(f"Expected XYZ, reflectivity, semantic, and instance columns; got {labeled_points.shape}")

        labeled_indices = np.flatnonzero(labeled_points[:, -2] >= 0)
        if labeled_indices.size == 0:
            raise ValueError(f"No points received ground-truth labels for scan {selected_scan.stem}")

        import open3d as o3d

        labeled_cloud_points = labeled_points[labeled_indices]
        semantics = labeled_cloud_points[:, -2].astype(int)
        colors = np.empty((len(labeled_cloud_points), 3), dtype=float)
        colors[:, 0] = ((semantics * 67 + 53) % 255) / 255.0
        colors[:, 1] = ((semantics * 131 + 97) % 255) / 255.0
        colors[:, 2] = ((semantics * 197 + 193) % 255) / 255.0

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(labeled_cloud_points[:, :3])
        cloud.colors = o3d.utility.Vector3dVector(colors)
        app = o3d.visualization.gui.Application.instance
        app.initialize()
        visualizer = o3d.visualization.O3DVisualizer(f"Ground truth: {selected_scan.stem}", 1280, 800)
        visualizer.add_geometry("Ground-truth labeled points", cloud)
        visualizer.point_size = 5
        visualizer.show_settings = True
        visualizer.mouse_mode = o3d.visualization.gui.SceneWidget.Controls.PICK_POINTS

        def print_selected_labels(window):
            selection_sets = window.get_selection_sets()
            picked_count = 0
            for selection_set in selection_sets:
                for selected_points in selection_set.values():
                    for selected_point in selected_points:
                        point_index = labeled_indices[selected_point.index]
                        print(
                            f"Picked point {point_index}: xyz={labeled_points[point_index, :3]}, "
                            f"semantic={int(labeled_points[point_index, -2])}, "
                            f"instance={int(labeled_points[point_index, -1])}"
                        )
                        picked_count += 1
            if picked_count == 0:
                print("No points selected. Choose point-picking mode and click points in the viewer.")

        visualizer.add_action("Print selected point labels", print_selected_labels)
        print(f"Using mesh {selected_mesh.name}. Settings panel is open; select points, then print their labels from Actions.")
        app.add_window(visualizer)
        app.run()


if __name__ == "__main__":
    main()
