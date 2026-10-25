import argparse
import os
import tempfile
from pathlib import Path

import dotenv
import numpy as np

from data.collect.kitti360 import project_pinhole_data


DRIVE_NAME = "2013_05_28_drive_0000_sync"


def main():
    parser = argparse.ArgumentParser(description="Project one KITTI scan and inspect it with Open3D point picking.")
    parser.add_argument("--index", type=int, default=0, help="Index of the matched image/scan pair to display")
    args = parser.parse_args()

    dotenv.load_dotenv(Path(__file__).resolve().parent / ".env")
    dataset_directory_value = os.getenv("KITTI_360_DATASET_DIR")
    if not dataset_directory_value:
        raise RuntimeError("KITTI_360_DATASET_DIR is missing from the project .env file")
    dataset_directory = Path(dataset_directory_value)
    image_directory = dataset_directory / "data_2d_semantics/train" / DRIVE_NAME / "image_00/instance"
    scan_directory = dataset_directory / "data_3d_raw" / DRIVE_NAME / "velodyne_points/data"
    calibration_directory = dataset_directory / "calibration"

    images = {path.stem: path for path in image_directory.glob("*.png")}
    scans = {path.stem: path for path in scan_directory.glob("*.bin")}
    matching_stems = sorted(images.keys() & scans.keys())
    if not matching_stems:
        raise FileNotFoundError(f"No matching image/scan pairs in {image_directory} and {scan_directory}")
    if args.index < 0 or args.index >= len(matching_stems):
        parser.error(f"--index must be between 0 and {len(matching_stems) - 1}")

    selected_stem = matching_stems[args.index]
    with tempfile.TemporaryDirectory(prefix="pinhole-visualization-") as temporary_directory:
        temporary_path = Path(temporary_directory)
        staged_images = temporary_path / "images"
        staged_scans = temporary_path / "scans"
        output_directory = temporary_path / "output"
        staged_images.mkdir()
        staged_scans.mkdir()
        (staged_images / images[selected_stem].name).symlink_to(images[selected_stem].resolve())
        (staged_scans / scans[selected_stem].name).symlink_to(scans[selected_stem].resolve())

        result = project_pinhole_data(
            absolute_image_directory=str(staged_images),
            absolute_scan_directory=str(staged_scans),
            absolute_destination_directory=str(output_directory),
            prefix="",
            absolute_calibration_directory=str(calibration_directory),
            data_format="KITTI",
            sample_only=False,
            ignore_duplicates=True,
        )
        if result != 0:
            raise RuntimeError(f"Projection failed for matched pair {selected_stem}")

        labeled_points = np.load(output_directory / f"{selected_stem}.npy")
        if labeled_points.ndim != 2 or labeled_points.shape[1] < 6:
            raise ValueError(f"Expected XYZ, reflectivity, semantic, and instance columns; got {labeled_points.shape}")

        labeled_indices = np.flatnonzero(labeled_points[:, -2] >= 0)
        if labeled_indices.size == 0:
            raise ValueError(f"No points received labels in {selected_stem}; check image/scan pairing and calibration")

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
        visualizer = o3d.visualization.O3DVisualizer(f"Pinhole projection: {selected_stem}", 1280, 800)
        visualizer.add_geometry("Labeled points", cloud)
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
        print("Settings panel is open. Click points, then choose Actions > Print selected point labels.")
        app.add_window(visualizer)
        app.run()


if __name__ == "__main__":
    main()
