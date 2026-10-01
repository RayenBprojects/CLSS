import glob
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset



class Kitti(Dataset):
    def __init__(self, directory):
        directory_items = next(os.walk(directory))[2]
        directory_items.sort()

        self.file_paths= []

        for item in directory_items:
            self.file_paths.append(directory / item)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        file_path= self.file_paths[index]
        array= np.load(file_path, mmap_mode= "r")

        return torch.from_numpy(array)

        # need to fix unequal point numbers!!!!