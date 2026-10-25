import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

_semantic_ID_dict = {
    -1 : "unlabeled",
    0 : "unlabeled",
    1 : "ego vehicle",
    2 : "rectification border",
    3 : "out of roi",
    4 : "static",
    5 : "dynamic",
    6 : "ground",
    7 : "road",
    8 : "sidewalk",
    9 : "parking",
    10 : "rail track",
    11 : "building",
    12 : "wall",
    13 : "fence",
    14 : "guard rail",
    15 : "bridge",
    16 : "tunnel",
    17 : "pole",
    18 : "polegroup",
    19 : "traffic light",
    20 : "traffic sign",
    21 : "vegetation",
    22 : "terrain",
    23 : "sky",
    24 : "person",
    25 : "rider",
    26 : "car",
    27 : "truck",
    28 : "bus",
    29 : "caravan",
    30 : "trailer",
    31 : "train",
    32 : "motorcycle",
    33 : "bicycle",
    34 : "garage",
    35 : "gate",
    36 : "stop",
    37 : "smallpole",
    38 : "lamp",
    39 : "trash bin",
    40 : "vending machine",
    41 : "box",
    42 : "unknown construction",
    43 : "unknown vehicle",
    44 : "unknown object",
    45 : "license plate"
}

class Kitti(Dataset):
    def __init__(self, directory, projection_type, set, force_organize= False, std_num_points=124792, num_samples= 61280, force_num_samples= False): #124792 | 61280

        if num_samples < 10:
                    raise Exception(f"Invalid number of samples. Minimum 10 samples needed.")
        
        if projection_type not in ['ground_truth', 'pinhole', 'cylindrical'] or set not in ['train', 'test', 'eval']:
            raise Exception("Invalid label type or set selection.")

        self.projection_type= projection_type

        self.std_num_points = std_num_points

        self.directory= Path(directory)

        directories= {
            'ground_truth': self.directory/"all_ground_truth_projections",
            'pinhole': self.directory/"all_pinhole_projections",
            'cylindrical': self.directory/"all_cylindrical_projections",
        }

        items= {
            'ground_truth': sorted(next(os.walk(directories['ground_truth']))[2]),
            'pinhole': sorted(next(os.walk(directories['pinhole']))[2]),
            'cylindrical' : sorted(next(os.walk(directories['cylindrical']))[2])
            }

        if (sorted(items['ground_truth']) != sorted(items['pinhole'])) or (sorted(items['ground_truth']) != sorted(items['cylindrical'])):
            raise Exception(f"Different projection datasets do not match. Please organize the data before initializing the dataset.")

        items= items['ground_truth']
        
        max_num_samples= len(items)

        if num_samples > max_num_samples:
            if force_num_samples:
                raise Exception(f"Invalid number of samples. Only {max_num_samples} samples found.")
            else:
                print(f"Warning: Invalid number of samples. Only {max_num_samples} samples found.")
                num_samples= max_num_samples

        if not force_organize:
            try:
                self.selection= np.load(Path("./data/saved/kitti360_selection/kitti_selection.npy"))

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

        np.save(Path("./data/saved/kitti360_selection/kitti_selection.npy"), self.selection)

        indices= {'train': (0, int(0.8*num_samples)),
                  'test': (int(0.8*num_samples), int(0.9*num_samples)),
                  'eval': (int(0.9*num_samples), num_samples)}

        self.file_paths= {
            'ground_truth': [],
            'pinhole': [],
            'cylindrical': [],
        }

        for index in self.selection[indices[set][0] : indices[set][1]]:
            self.file_paths['ground_truth'].append(directories['ground_truth']/items[index])
            self.file_paths['pinhole'].append(directories['pinhole']/items[index])
            self.file_paths['cylindrical'].append(directories['cylindrical']/items[index])

    def __len__(self):
        return len(self.file_paths['ground_truth'])

    def __getitem__(self, index):
        array= {
            'ground_truth': np.load(self.file_paths['ground_truth'][index], allow_pickle=False).copy(),
            'pinhole': np.load(self.file_paths['pinhole'][index], allow_pickle=False).copy(),
            'cylindrical': np.load(self.file_paths['cylindrical'][index], allow_pickle=False).copy()
        }

        if (len(array['ground_truth']) != len(array['pinhole'])) or (len(array['ground_truth']) != len(array['cylindrical'])):
            print(f"WARNING: data projections corresponding to scene {index+1} are mismatched in point number.")

        mask = (
            (array['cylindrical'][:, 4] != -1)
            & (array['cylindrical'][:, 4] != 0)
            & (array['cylindrical'][:, 4] != 1)
            & (array['pinhole'][:, 4] != -1)
            & (array['pinhole'][:, 4] != 0)
            & (array['pinhole'][:, 4] != 1)
            & (array['ground_truth'][:, 4] != 0)
            & (array['ground_truth'][:, 4] != 1)
        )

        array= array[self.projection_type][mask]

        if array.ndim != 2:
            raise ValueError(
                f"Expected a 2D point-cloud array in {self.file_paths[self.projection_type][index]}, got shape {array.shape}"
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