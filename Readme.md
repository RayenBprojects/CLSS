# Cylindrical Label Scene Segmentation

This repository contains code for exploring cylindrical 2D label projections for point-cloud segmentation. It includes KITTI-360 data collection and loading, PointNet models, training utilities, tests, and visualization scripts.

## Repository structure

| Path | Contents |
| --- | --- |
| `core.py` | PointNet++ training setup |
| `run_data_collection.py` | Collect ground-truth, pinhole, and cylindrical KITTI-360 projections |
| `run_model_training.py` | Train and export a model |
| `data/collect/` | KITTI-360 projection and data-collection code |
| `data/load/` | Dataset loading and preprocessing |
| `data/saved/` | Saved dataset-selection metadata |
| `models/` | PointNet model definitions |
| `models/saved/` | Exported models |
| `tools/` | Training/testing helpers and model utilities |
| `visualization/` | Scripts for viewing projections, labels, and model output |
| `tests/` | Automated tests |
| `requirements.txt` | Python package dependencies |

## Setup

Use a Python environment with a PyTorch build compatible with your hardware. Install the listed dependencies:

```bash
python -m pip install -r requirements.txt
```

Set `KITTI_360_DATASET_DIR` to the location of your KITTI-360 dataset. You can start from the provided environment template:

```bash
cp .env.template .env
```

Then edit `.env` and set the variable to your dataset directory. The collection and training scripts load this setting from `.env`.

## Common commands

Collect KITTI-360 projections:

```bash
python run_data_collection.py
```

**Warning:** `run_data_collection.py` currently removes existing files from its generated projection directories before collecting data.

Train and export the PointNet++ model:

```bash
python run_model_training.py
```

Training currently requires CUDA and uses the KITTI-360 dataset.

Run the tests:

```bash
python -m pytest
```

Visualization scripts are in `visualization/`; run an individual script with Python to inspect its corresponding data or model output.
