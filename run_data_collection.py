from pathlib import Path

from data.collect.kitti360 import project_2d_semantic_instance_data, project_KITTI_3d_semantic_instance_data

import os
import dotenv


REMOVE_MISMATCH= True


def main():
    dotenv.load_dotenv()
    KITTI_360_DATASET_DIR= Path(os.getenv("KITTI_360_DATASET_DIR"))

    (KITTI_360_DATASET_DIR / "all_2d_projections").mkdir(parents= True, exist_ok= True)


    for i in range(0,11):
        if i == 1:
            continue
        number = ""
        if i < 10:
            number = f"0{i}"
        else:
            number = f"{i}"

        print(f"collecting from drive {i+1}/{11}")

        project_2d_semantic_instance_data(
            absolute_image_directory =   Path(KITTI_360_DATASET_DIR / f'data_2d_semantics/train/2013_05_28_drive_00{number}_sync/image_00/instance'),           #f'D:\\Datasets\\KITTI-360\\data_2d_semantics\\train\\2013_05_28_drive_00{number}_sync\\image_00\\instance'),
            absolute_scan_directory = Path(KITTI_360_DATASET_DIR / f'data_3d_raw/2013_05_28_drive_00{number}_sync/velodyne_points/data'), #Path(f'D:\\Datasets\\KITTI-360\\data_3d_raw\\2013_05_28_drive_00{number}_sync\\velodyne_points\\data'),
            absolute_destination_directory = Path(KITTI_360_DATASET_DIR / f'all_2d_projections'), #Path(f'D:\\lidar-data-helper\\projects\\2013_05_28_drive_00{number}_sync\\projected_2d_labels'),
            prefix= f"{number}_",
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
            prefix= f"{number}_",
            absolute_calibration_directory= Path(KITTI_360_DATASET_DIR / f'calibration'),
            absolute_poses_directory= Path(KITTI_360_DATASET_DIR / f'data_poses/2013_05_28_drive_00{number}_sync'), #f'D:\\Datasets\\KITTI-360\\data_poses\\2013_05_28_drive_00{number}_sync',
            ignore_duplicates = True
            )

    if REMOVE_MISMATCH:
        print('-- Removing Mismatched Files --')
        collected_2d_projections= next(os.walk(KITTI_360_DATASET_DIR / f'all_2d_projections'))[2]
        marked_for_deletion_2d= []

        collected_3d_projections= next(os.walk(KITTI_360_DATASET_DIR / f'all_3d_projections'))[2]
        marked_for_deletion_3d= []

        for item in collected_2d_projections:
            if item not in collected_3d_projections:
                marked_for_deletion_2d.append(item)

        for item in collected_3d_projections:
            if item not in collected_2d_projections:
                marked_for_deletion_3d.append(item)

        for item in marked_for_deletion_2d:
            (KITTI_360_DATASET_DIR/f'all_2d_projections/{item}').unlink(missing_ok= True)

        for item in marked_for_deletion_3d:
            (KITTI_360_DATASET_DIR/f'all_3d_projections/{item}').unlink(missing_ok= True)


if __name__ == "__main__":
    main()