from pathlib import Path

from data.collect.kitti360 import project_cylindrical_data, project_ground_truth_data, project_pinhole_data

import os
import dotenv

dotenv.load_dotenv()

KITTI_360_DATASET_DIR= Path(os.getenv("KITTI_360_DATASET_DIR"))

def main():
    project_cylindrical_data(
        absolute_image_directory =   Path(KITTI_360_DATASET_DIR / f'data_2d_semantics/train/2013_05_28_drive_0000_sync/image_00/instance'),
        absolute_scan_directory = Path(KITTI_360_DATASET_DIR / f'data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data'),
        absolute_destination_directory = Path(KITTI_360_DATASET_DIR / f'all_cylindrical_projections'),
        prefix= f"00_",
        LIDAR_relative_xyz = (0.79, 0.32, -0.18),
        image_margins = (0.10, 0.00, 0.10, 0.0),
        LIDAR_margins = (0.0, 0.3, 115.0, 235.0), # remember, angle goes counterclockwise
        data_format = 'KITTI',
        sample_only = True,
        ignore_duplicates=True
    )

if __name__ == "__main__":
    main()