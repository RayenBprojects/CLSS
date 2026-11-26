# Cylindrical Label Scene Segmentation

CLSS is a KITTI-360 point-cloud semantic-segmentation project. It generates
ground-truth, pinhole, and cylindrical projections of LiDAR data, trains
PointNet/PointNet++ models, and evaluates projections and saved models.

## Repository layout

| Path | Purpose |
| --- | --- |
| `core.py` | Data collection, training, and evaluation orchestration |
| `run_data_collection.py` | Generate projection arrays from KITTI-360 data |
| `run_model_training.py` | Train PointNet++ and export `.pt2` models |
| `run_data_evaluation.py` | Compare pinhole and cylindrical projection labels with ground truth |
| `run_model_evaluation.py` | Evaluate and compare saved models |
| `data/collect/` | KITTI-360 projection generation |
| `data/load/` | Dataset loading, filtering, and train/test/eval splits |
| `data/saved/` | Saved dataset-selection metadata |
| `models/` | PointNet and PointNet++ model definitions |
| `models/saved/` | Exported `.pt2` model files |
| `tools/` | Accuracy metrics, model loading/viewing, and train/test loops |
| `visualization/` | Projection and model-output viewers |
| `scripts/` | Environment setup, KITTI-360 downloads, and experiment scripts |

## Requirements and setup

The setup script creates a `.venv` with Python 3.14 and installs the packages
listed in `requirements.txt`:

```bash
bash scripts/set_up_environment.sh
source .venv/bin/activate
```

To create the environment without installing packages, use:

```bash
bash scripts/set_up_environment_no_pip_install.sh
source .venv/bin/activate
```

Alternatively, create an environment with a Python version supported by your
installed PyTorch, torchvision, and Open3D packages, then run:

```bash
python -m pip install -r requirements.txt
```

**Important:** `torch` and `torchvision` are not installed from
`requirements.txt`. Install both separately using the command selected for
your operating system, package manager, Python version, and hardware
acceleration on the
[PyTorch Get Started locally page](https://pytorch.org/get-started/locally/).
Use compatible PyTorch and torchvision builds from the same CPU, CUDA, or other
acceleration configuration.

Training, model evaluation, and data evaluation require PyTorch with CUDA
support and an available CUDA device. Projection collection can run without
CUDA. The model-output viewer uses CUDA when available and falls back to CPU.

Run project commands from the repository root. Python code reads
`KITTI_360_DATASET_DIR` from `.env` using `python-dotenv`; the download script
sources the same file as shell configuration.

## Dataset setup

Create `.env` from the template and set the dataset directory to an existing
directory:

```bash
cp .env.template .env
mkdir -p /absolute/path/to/kitti360
```

Set this line in `.env`:

```dotenv
KITTI_360_DATASET_DIR=/absolute/path/to/kitti360
```

Install `wget` and `unzip`, then download and extract the KITTI-360 components:

```bash
bash scripts/download_dataset.sh
```

The downloader expects the configured directory to exist and creates these
folders under it:

```text
$KITTI_360_DATASET_DIR/
  calibration/
  data_2d_semantics/
  data_3d_raw/
  data_3d_semantics/
  data_poses/
```

The loader also uses three generated projection directories:

```text
$KITTI_360_DATASET_DIR/
  all_ground_truth_projections/
  all_pinhole_projections/
  all_cylindrical_projections/
```

Data collection creates these directories and removes mismatched filenames.
The three directories must contain matching sample filenames before training or
evaluation.

## Collect projection data

Generate the ground-truth, pinhole, and cylindrical projections:

```bash
python run_data_collection.py
```

To remove existing projection files before collecting again:

```bash
python run_data_collection.py --clean_existing_files
```

This option deletes existing generated `.npy` projection files, but not the
downloaded KITTI-360 source folders.

## Train a model

Training, evaluation, and the model-output viewer use the shared seed `42` from
`tools/reproducibility.py` for Python, NumPy, and PyTorch random operations.
Change `SEED` there to use a different seed. Some GPU operations may still
produce different results across hardware or software versions.

Train PointNet++ with the default ground-truth projection, batch size 16,
learning rate 0.001, and five epochs:

```bash
python run_model_training.py
```

Available options are:

| Option | Values / default | Meaning |
| --- | --- | --- |
| `--projection_type` | `ground_truth`, `pinhole`, `cylindrical` / `ground_truth` | Projection used for training |
| `--batch_size` | `32` | Batch size |
| `--learning_rate` | `0.001` | Optimizer learning rate |
| `--num_epochs` | `120` | Number of epochs |
| `--optim` | `Adam`, `SGD` / `Adam` | Optimizer |
| `--print_batch_data` | off by default | Print per-batch metrics |

Training requires CUDA. The script exports a numbered `.pt2` model into
`models/saved/` after each successfully completed epoch. Exported graphs retain
the CUDA device used during export, so re-export models on the target CUDA
setup when moving them between systems.

The repository contains checked-in `.pt2` model files under `models/saved/`.
Generated model files are normally added as `NNNN_kitti360.pt2`.

## Evaluate

Compare pinhole and cylindrical projection labels with ground truth:

```bash
python run_data_evaluation.py
```

Evaluate saved models and compare projection models against the ground-truth
model:

```bash
python run_model_evaluation.py \
  --ground_truth_model GROUND_TRUTH_MODEL.pt2 \
  --pinhole_model PINHOLE_MODEL.pt2 \
  --cylindrical_model CYLINDRICAL_MODEL.pt2
```

Replace the example filenames with `.pt2` files in `models/saved/`. The CLI
defaults are `ground_truth_model.pt2`, `pinhole_model.pt2`, and
`cylindrical_model.pt2`; a run skips evaluations for model files that it cannot
find.

Both evaluation commands require CUDA. `run_data_evaluation.py` accepts
`--batch_size` and `--print_batch_data`. `run_model_evaluation.py` accepts
`--print_batch_data`; model evaluations use a fixed batch size of 1.

## Visualization

Scripts in `visualization/` view ground-truth, pinhole, and cylindrical
projections, and projection alignment. Run a specific viewer from
the repository root, for example:

```bash
python -m visualization.view_cylindrical_visualization
```
