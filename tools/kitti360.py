import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import shutil
import sys


class Kitti(Dataset):
    def __init__(self, directory, projection, set, organize= True, std_num_points=124792):

        if projection not in ['2d', '3d'] or set not in ['train', 'test', 'eval']:
            raise Exception("Invalid arguments.")

        self.directory= Path(directory)

        directory_2d= (self.directory/"all_2d_projections")
        directory_2d_items= next(os.walk(directory_2d))

        directory_2d_test= (self.directory/"all_2d_projections_test")
        directory_2d_test.mkdir(parents= True, exist_ok= True)
        directory_2d_test_items= next(os.walk(directory_2d_test))

        directory_2d_eval= (self.directory/"all_2d_projections_eval")
        directory_2d_eval.mkdir(parents= True, exist_ok= True)
        directory_2d_eval_items= next(os.walk(directory_2d_eval))

        directory_3d= (self.directory/"all_3d_projections")
        directory_3d_items= next(os.walk(directory_3d))

        directory_3d_test= (self.directory/"all_3d_projections_test")
        directory_3d_test.mkdir(parents= True, exist_ok= True)
        directory_3d_test_items= next(os.walk(directory_3d_test))

        directory_3d_eval= (self.directory/"all_3d_projections_eval")
        directory_3d_eval.mkdir(parents= True, exist_ok= True)
        directory_3d_eval_items= next(os.walk(directory_3d_eval))

        if organize:
            self._clean()

            files_2d = {path.name for path in directory_2d.glob("*.npy")}
            files_3d = {path.name for path in directory_3d.glob("*.npy")}
            if files_2d != files_3d:
                raise ValueError("2D and 3D projection directories must contain matching .npy files")

            file_names = sorted(files_2d)
            np.random.default_rng().shuffle(file_names)
            test_count = len(file_names) // 10
            eval_count = len(file_names) // 10

            for file_name in file_names[:test_count]:
                shutil.move(directory_2d / file_name, directory_2d_test / file_name)
                shutil.move(directory_3d / file_name, directory_3d_test / file_name)

            for file_name in file_names[test_count:test_count + eval_count]:
                shutil.move(directory_2d / file_name, directory_2d_eval / file_name)
                shutil.move(directory_3d / file_name, directory_3d_eval / file_name)

        directories= {
            ('2d', 'train'): directory_2d,
            ('2d', 'test'): directory_2d_test,
            ('2d', 'eval'): directory_2d_eval,
            ('3d', 'train'): directory_3d,
            ('3d', 'test'): directory_3d_test,
            ('3d', 'eval'): directory_3d_eval
        }

        self.file_paths = sorted(
            path for path in directories[(projection, set)].iterdir()
            if path.is_file() and path.suffix.lower() == ".npy"
        )

        if not self.file_paths:
            raise FileNotFoundError(f"No .npy files found in {self.directory}")

        self.std_num_points = std_num_points

    def __del__(self):
        self._clean()

    def _clean(self):
        directory_2d= (self.directory/"all_2d_projections")
        directory_2d_items= next(os.walk(directory_2d))

        directory_2d_test= (self.directory/"all_2d_projections_test")
        directory_2d_test.mkdir(parents= True, exist_ok= True)
        directory_2d_test_items= next(os.walk(directory_2d_test))

        directory_2d_eval= (self.directory/"all_2d_projections_eval")
        directory_2d_eval.mkdir(parents= True, exist_ok= True)
        directory_2d_eval_items= next(os.walk(directory_2d_eval))

        directory_3d= (self.directory/"all_3d_projections")
        directory_3d_items= next(os.walk(directory_3d))

        directory_3d_test= (self.directory/"all_3d_projections_test")
        directory_3d_test.mkdir(parents= True, exist_ok= True)
        directory_3d_test_items= next(os.walk(directory_3d_test))

        directory_3d_eval= (self.directory/"all_3d_projections_eval")
        directory_3d_eval.mkdir(parents= True, exist_ok= True)
        directory_3d_eval_items= next(os.walk(directory_3d_eval))

        for item_2d_test  in directory_2d_test_items[2]:
            shutil.move(directory_2d_test/item_2d_test, directory_2d/item_2d_test)

        for item_2d_eval in directory_2d_eval_items[2]:
            shutil.move(directory_2d_eval/item_2d_eval, directory_2d/item_2d_eval)

        for item_3d_test in directory_3d_test_items[2]:
            shutil.move(directory_3d_test/item_3d_test, directory_3d/item_3d_test)

        for item_3d_eval in directory_3d_eval_items[2]:
            shutil.move(directory_3d_eval/item_3d_eval, directory_3d/item_3d_eval)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        array = np.load(file_path, mmap_mode="r", allow_pickle=False)

        if array.ndim != 2:
            raise ValueError(
                f"Expected a 2D point-cloud array in {file_path}, got shape {array.shape}"
            )

        array = torch.from_numpy(array.copy())
        num_points = array.shape[0]

        if num_points > self.std_num_points:
            selected_indices = torch.randperm(num_points)[: self.std_num_points]
            array = array[selected_indices]
        elif num_points < self.std_num_points:
            padding = torch.zeros(
                (self.std_num_points - num_points, array.shape[1]),
                dtype=array.dtype,
            )
            array = torch.cat((array, padding), dim=0).to(torch.float32)

        return (array[:, :4], array[:, 4])