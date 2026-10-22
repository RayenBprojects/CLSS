# Scene Labeling Experiments

This repository contains point cloud scene labeling experiments based on PointNet++ for semantic segmentation on datasets such as KITTI-360 and S3DIS.

## Overview

The project trains a PointNet++ model for 3D scene understanding and saves a checkpoint plus a TorchScript export after training. The main training entry point is in `run.py`, which calls the training routine defined in `core.py`.

## Project structure

- `core.py` – training pipeline and model export logic
- `run.py` – entry point used to start training
- `models/` – model implementations
- `tools/` – dataset loaders and training/test utilities
- `data/` – local dataset storage
- `requirements.txt` – project dependencies

## Requirements

- Python 3.9+
- CUDA-capable GPU for training
- PyTorch and related dependencies from `requirements.txt`

## Setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure dataset paths. The code reads environment variables for dataset directories:

   ```bash
   export KITTI_360_DATASET_DIR="/path/to/kitti360"
   export S3DIS_DATASET_DIR="/path/to/s3dis"
   ```

   You can also create a `.env` file in the project root with these values.

## Training

Start training with:

```bash
python run.py
```

This runs the default training configuration for `KITTI-360` and saves the model to:

- `models/saved/PointnetPPKITTI360.pth`
- `models/saved/PointnetPPKITTI360.pt`

You can also call the training routine directly from Python:

```python
from core import train_pointnetpp

results = train_pointnetpp(dataset='kitti360')
# or
results = train_pointnetpp(dataset='s3dis')
```

## Notes

- Training is designed for CUDA and will raise an error if CUDA is unavailable.
- The project currently focuses on point cloud segmentation experiments and model evaluation.
- Some dataset logic and environment variables may need to be adapted to your local folder layout.

## License

This project does not include a license file. Please confirm with the repository owner before using it in a production or public environment.
