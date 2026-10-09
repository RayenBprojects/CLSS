import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import shutil
import sys

class Kitti(Dataset):
    def __init__(self, directory, label_type, set, force_organize= False, std_num_points=124792, num_samples= 61280): #124792 | 61280
        
        if label_type not in ['2d', '3d'] or set not in ['train', 'test', 'eval']:
            raise Exception("Invalid label type or set selection.")

        self.directory= Path(directory)

        directories= {
            '2d': self.directory/"all_2d_projections",
            '3d': self.directory/"all_3d_projections"
        }

        items= {
            '2d': next(os.walk(directories['2d']))[2],
            '3d': next(os.walk(directories['3d']))[2]
            }

        if sorted(items['2d']) != sorted(items['3d']):
            raise Exception(f"2D and 3D datasets do not match. Please organize the data before initializing the dataset.")
        
        max_num_samples= len(items['2d'])

        if num_samples > max_num_samples:
            raise Exception(f"Invalid number of samples. Only {max_num_samples} samples found.")

        if not force_organize:
            try:
                self.selection= np.load("./tools/kitti_selection.npy")

                if self.selection.shape[0] != num_samples:
                    raise Exception()
            except:
                if set != 'train':
                    print(f"KITTI360 WARNING: No valid existing {set} selection found; {set} samples will be selected randomly.")

                self.selection= np.random.choice(max_num_samples, size= num_samples, replace=False)
        else:
            if set != 'train':
                print(f"KITTI360 WARNING: force_organize is set to true for non train set; {set} samples will be selected randomly.")

            self.selection= np.random.choice(max_num_samples, size= num_samples, replace=False)

        np.save("./tools/kitti_selection.npy", self.selection)

        indices= {'train': (0, int(0.8*num_samples)),
                  'test': (int(0.8*num_samples), int(0.9*num_samples)),
                  'eval': (int(0.9*num_samples), num_samples)}

        self.file_paths= []

        for index in self.selection[indices[set][0] : indices[set][1]]:
            self.file_paths.append(directories[label_type]/items[label_type][index])

        self.std_num_points = std_num_points

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        array = np.load(file_path, mmap_mode="r", allow_pickle=False)

        if array.ndim != 2:
            raise ValueError(
                f"Expected a 2D point-cloud array in {file_path}, got shape {array.shape}"
            )

        array = torch.from_numpy(array.copy()).to(torch.float32)
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

        return (array[:, :4], array[:, 4].to(torch.long))