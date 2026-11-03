# Cylindrical Label Scene Segmentation

CLSS is a KITTI-360 point-cloud semantic-segmentation project. It generates ground-truth, pinhole, and cylindrical projections of LiDAR data, trains PointNet++ models, and evaluates the projections and saved models.

## Repository layout

| Path | Purpose |
| --- | --- |
| `core.py` | Data collection, training, and evaluation orchestration |
| `run_data_collection.py` | Generate projection arrays from KITTI-360 data |
| `run_model_training.py` | Train PointNet++ and export models |
| `run_data_evaluation.py` | Compare pinhole and cylindrical projection labels with ground truth |
| `run_model_evaluation.py` | Evaluate and compare saved models |
| `data/collect/` | KITTI-360 projection generation |
| `data/load/` | Dataset loading, filtering, and train/test/eval splits |
| `data/saved/` | Saved dataset-selection metadata |
| `models/` | PointNet and PointNet++ model definitions |
| `models/saved/` | Exported models |
| `tools/` | Accuracy metrics, model loading/viewing, and train/test loops |
| `visualization/` | Projection and model-output viewers |
| `scripts/` | Environment setup, KITTI-360 downloads, and experiment scripts |
| `tests/` | Automated tests |

## Requirements and setup

The setup script uses Python 3.14 and installs the packages in `requirements.txt`:

```bash
bash scripts/set_up_environment.sh
source .venv/bin/activate
```

Alternatively, create an environment with a Python version supported by the installed PyTorch and Open3D packages, then run `python -m pip install -r requirements.txt`. Training and both evaluation commands require a CUDA-capable PyTorch installation and an available CUDA device.

Run project commands from the repository root. Python code reads `KITTI_360_DATASET_DIR` from `.env` using `python-dotenv`; the download script sources the same file as shell configuration.

## Dataset setup

Create `.env` from the template and set the dataset directory to an existing directory. Replace the example path with an absolute path on your machine:

```bash
cp .env.template .env
mkdir -p /path/to/kitti360
```

Set this line in `.env`:

```dotenv
KITTI_360_DATASET_DIR=/path/to/kitti360
```

Download and extract the dataset components used by this project:

```bash
bash scripts/download_dataset.sh
```

The downloader requires `wget` and `unzip`. It expects the configured directory to exist and places the downloaded data under it. The resulting input folders include:

```text
$KITTI_360_DATASET_DIR/
  calibration/
  data_2d_semantics/
  data_3d_raw/
  data_3d_semantics/
  data_poses/
```

The dataset loader uses three generated projection directories. Data collection creates them:

```text
all_ground_truth_projections/
all_pinhole_projections/
all_cylindrical_projections/
```

The three projection directories must contain matching sample filenames for training and evaluation.

## Collect projection data

Generate the ground-truth, pinhole, and cylindrical projections:

```bash
python run_data_collection.py
```

To remove existing files from the generated projection directories before collecting again, use:

```bash
python run_data_collection.py --clean_existing_files
```

This option deletes existing generated projection files; it does not delete the downloaded KITTI-360 source folders.

## Train a model

Train PointNet++ with the default ground-truth projection, batch size 16, learning rate 0.001, and five epochs:

```bash
python run_model_training.py
```

Available options are:

| Option | Values / default | Meaning |
| --- | --- | --- |
| `--projection_type` | `ground_truth`, `pinhole`, `cylindrical` / `ground_truth` | Projection used for training |
| `--batch_size` | `16` | Batch size |
| `--learning_rate` | `0.001` | Optimizer learning rate |
| `--num_epochs` | `5` | Number of epochs |
| `--optim` | `Adam`, `SGD` / `Adam` | Optimizer |
| `--print_batch_data` | off by default | Print per-batch metrics |

Training requires CUDA. The script exports a numbered `.pt2` model into `models/saved/` after each successfully completed epoch. Exported graphs retain the CUDA device used during export, so models exported by earlier versions must be retrained and re-exported before CUDA evaluation. Model files are ignored by Git except for the checked-in sample model.

## Evaluate

Compare pinhole and cylindrical projection labels with ground truth:

```bash
python run_data_evaluation.py
```

Evaluate saved models and compare each projection model against the ground-truth model:

```bash
python run_model_evaluation.py \
  --ground_truth_model GROUND_TRUTH_MODEL.pt2 \\
  --pinhole_model PINHOLE_MODEL.pt2 \\
  --cylindrical_model CYLINDRICAL_MODEL.pt2
```

Replace the example filenames with model files located in `models/saved/`. The command's defaults are `sample_model_1.pt2`, `sample_model_2.pt2`, and `sample_model_3.pt2`; those three files are not included in this repository. A run skips evaluations for model files that it cannot find. Both evaluation commands require CUDA.

Both evaluation scripts also accept `--batch_size` and `--print_batch_data`.

## Visualization and tests

Scripts in `visualization/` view ground-truth, pinhole, and cylindrical projections, projection alignment, and model output. Run a specific viewer from the repository root, for example:

```bash
python visualization/view_cylindrical_visualization.py
```

Run the automated tests with:

```bash
python -m pytest
```