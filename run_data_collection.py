from pathlib import Path

from data.collect.kitti360 import project_2d_semantic_instance_data, project_KITTI_3d_semantic_instance_data

import os
import dotenv


REMOVE_MISMATCH= True


dotenv.load_dotenv()
KITTI_360_DATASET_DIR= Path(os.getenv("KITTI_360_DATASET_DIR"))


for i in range(0,11):
    if i == 1:
        continue
    number = ""
    if i < 10:
        number = f"0{i}"
    else:
        number = f"{i}"

    print(f"collecting drive {i+1}/{11}")

    project_2d_semantic_instance_data(
        absolute_image_directory =   Path(KITTI_360_DATASET_DIR / f'data_2d_semantics/train/2013_05_28_drive_00{number}_sync/image_00/instance'),           #f'D:\\Datasets\\KITTI-360\\data_2d_semantics\\train\\2013_05_28_drive_00{number}_sync\\image_00\\instance'),
        absolute_scan_directory = Path(KITTI_360_DATASET_DIR / f'data_3d_raw/2013_05_28_drive_00{number}_sync/velodyne_points/data'), #Path(f'D:\\Datasets\\KITTI-360\\data_3d_raw\\2013_05_28_drive_00{number}_sync\\velodyne_points\\data'),
        absolute_destination_directory = Path(KITTI_360_DATASET_DIR / f'all_2d_projections'), #Path(f'D:\\lidar-data-helper\\projects\\2013_05_28_drive_00{number}_sync\\projected_2d_labels'),
        LIDAR_relative_xyz = (0.79, 0.32, -0.18),
        image_margins = (0.10, 0.00, 0.10, 0.0),
        LIDAR_margins = (0.0, 0.3, 115.0, 235.0), # remember, angle goes counterclockwise
        data_format = 'KITTI',
        sample_only = False,
        ignore_duplicates=True
    )

    project_KITTI_3d_semantic_instance_data(
        absolute_mesh_directory= Path(KITTI_360_DATASET_DIR / f'data_2d_semantics/train/2013_05_28_drive_00{number}_sync/'),  
        absolute_scan_directory= Path(KITTI_360_DATASET_DIR / f'data_3d_raw/2013_05_28_drive_00{number}_sync/velodyne_points/data'),
        absolute_destination_directory= Path(KITTI_360_DATASET_DIR / f'all_3d_projections'),
        absolute_calibration_directory= Path(KITTI_360_DATASET_DIR / f'calibration'),
        absolute_poses_directory= Path(KITTI_360_DATASET_DIR / f'data_poses/2013_05_28_drive_00{number}_sync'), #f'D:\\Datasets\\KITTI-360\\data_poses\\2013_05_28_drive_00{number}_sync',
        ignore_duplicates = True
        )