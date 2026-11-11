#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# All training runs use the same CUDA device, so run them sequentially.
python -u run_model_training.py -pt ground_truth -bs 16 -ne 200 > temp_experiment_ground_truth.log & #2>&1
python -u run_model_training.py -pt pinhole -bs 16 -ne 200 > temp_experiment_pinhole.log & #2>&1
python -u run_model_training.py -pt cylindrical -bs 16 -ne 200 > temp_experiment_cylindrical.log & #2>&1