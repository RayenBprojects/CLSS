from pathlib import Path

from data.collect.kitti360 import project_cylindrical_data, project_ground_truth_data, project_pinhole_data

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

        project_pinhole_data(
            absolute_image_directory =   Path(KITTI_360_DATASET_DIR / f'data_2d_semantics/train/2013_05_28_drive_00{number}_sync/image_00/instance'),           #f'D:\\Datasets\\KITTI-360\\data_2d_semantics\\train\\2013_05_28_drive_00{number}_sync\\image_00\\instance'),
            absolute_scan_directory = Path(KITTI_360_DATASET_DIR / f'data_3d_raw/2013_05_28_drive_00{number}_sync/velodyne_points/data'), #Path(f'D:\\Datasets\\KITTI-360\\data_3d_raw\\2013_05_28_drive_00{number}_sync\\velodyne_points\\data'),
            absolute_destination_directory = Path(KITTI_360_DATASET_DIR / f'all_cylindrical_projections'), #Path(f'D:\\lidar-data-helper\\projects\\2013_05_28_drive_00{number}_sync\\projected_2d_labels'),
            prefix= f"{number}_",
            absolute_calibration_directory= Path(KITTI_360_DATASET_DIR / f'calibration'),
            data_format = 'KITTI',
            sample_only = False,
            ignore_duplicates=True
        )

        project_cylindrical_data(
            absolute_image_directory =   Path(KITTI_360_DATASET_DIR / f'data_2d_semantics/train/2013_05_28_drive_00{number}_sync/image_00/instance'),           #f'D:\\Datasets\\KITTI-360\\data_2d_semantics\\train\\2013_05_28_drive_00{number}_sync\\image_00\\instance'),
            absolute_scan_directory = Path(KITTI_360_DATASET_DIR / f'data_3d_raw/2013_05_28_drive_00{number}_sync/velodyne_points/data'), #Path(f'D:\\Datasets\\KITTI-360\\data_3d_raw\\2013_05_28_drive_00{number}_sync\\velodyne_points\\data'),
            absolute_destination_directory = Path(KITTI_360_DATASET_DIR / f'all_cylindrical_projections'), #Path(f'D:\\lidar-data-helper\\projects\\2013_05_28_drive_00{number}_sync\\projected_2d_labels'),
            prefix= f"{number}_",
            LIDAR_relative_xyz = (0.79, 0.32, -0.18),
            image_margins = (0.10, 0.00, 0.10, 0.0),
            LIDAR_margins = (0.0, 0.3, 115.0, 235.0), # remember, angle goes counterclockwise
            data_format = 'KITTI',
            sample_only = False,
            ignore_duplicates=True
        )

        project_ground_truth_data(
            absolute_mesh_directory= Path(KITTI_360_DATASET_DIR / f'data_3d_semantics/train/2013_05_28_drive_00{number}_sync/'),  
            absolute_scan_directory= Path(KITTI_360_DATASET_DIR / f'data_3d_raw/2013_05_28_drive_00{number}_sync/velodyne_points/data'),
            absolute_destination_directory= Path(KITTI_360_DATASET_DIR / f'all_ground_truth_projections'),
            prefix= f"{number}_",
            absolute_calibration_directory= Path(KITTI_360_DATASET_DIR / f'calibration'),
            absolute_poses_directory= Path(KITTI_360_DATASET_DIR / f'data_poses/2013_05_28_drive_00{number}_sync'), #f'D:\\Datasets\\KITTI-360\\data_poses\\2013_05_28_drive_00{number}_sync',
            ignore_duplicates = True
            )

    if REMOVE_MISMATCH:
        print('-- Removing Mismatched Files --')
        collected_cylindrical_projections= next(os.walk(KITTI_360_DATASET_DIR / f'all_cylindrical_projections'))[2]
        marked_for_deletion_cylindrical= []

        collected_ground_truth_projections= next(os.walk(KITTI_360_DATASET_DIR / f'all_ground_truth_projections'))[2]
        marked_for_deletion_ground_truth= []

        for item in collected_cylindrical_projections:
            if item not in collected_ground_truth_projections:
                marked_for_deletion_cylindrical.append(item)

        for item in collected_ground_truth_projections:
            if item not in collected_cylindrical_projections:
                marked_for_deletion_ground_truth.append(item)

        for item in marked_for_deletion_cylindrical:
            (KITTI_360_DATASET_DIR/f'all_cylindrical_projections/{item}').unlink(missing_ok= True)

        for item in marked_for_deletion_ground_truth:
            (KITTI_360_DATASET_DIR/f'all_ground_truth_projections/{item}').unlink(missing_ok= True)

        cylindrical_and_ground_truth_projections= next(os.walk(KITTI_360_DATASET_DIR / f'all_cylindrical_projections'))[2]
        marked_for_deletion_cylindrical_and_ground_truth= []

        collected_pinhole_projections= next(os.walk(KITTI_360_DATASET_DIR / f'all_pinhole_projections'))[2]
        marked_for_deletion_pinhole= []

        for item in cylindrical_and_ground_truth_projections:
            if item not in collected_pinhole_projections:
                marked_for_deletion_cylindrical_and_ground_truth.append(item)

        for item in collected_pinhole_projections:
            if item not in cylindrical_and_ground_truth_projections:
                marked_for_deletion_pinhole.append(item)

        for item in marked_for_deletion_cylindrical_and_ground_truth:
            (KITTI_360_DATASET_DIR/f'all_cylindrical_projections/{item}').unlink(missing_ok= True)
            (KITTI_360_DATASET_DIR/f'all_ground_truth_projections/{item}').unlink(missing_ok= True)

        for item in marked_for_deletion_pinhole:
            (KITTI_360_DATASET_DIR/f'all_pinhole_projections/{item}').unlink(missing_ok= True)



if __name__ == "__main__":
    main()