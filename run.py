import asyncio
import argparse
from pathlib import Path

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

from core import train_pointnetpp


def next_model_path(save_dir: Path, dataset: str) -> Path:
    existing = save_dir.glob("[0-9][0-9][0-9][0-9]_*.pt")
    next_number = max(
        (int(path.name.split("_", maxsplit=1)[0]) for path in existing),
        default=0,
    ) + 1
    model_path = save_dir / f"{next_number:04d}_{dataset}.pt"
    while model_path.exists():
        next_number += 1
        model_path = save_dir / f"{next_number:04d}_{dataset}.pt"
    return model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("s3dis", "kitti360"), default="kitti360")
    args = parser.parse_args()

    results = asyncio.run(train_pointnetpp(dataset=args.dataset))

    if results["status"]:
        dataset = results['dataset']
        model = results["model"]
        model.eval()

        save_dir = Path("./models/saved")
        save_dir.mkdir(parents=True, exist_ok=True)
        model_path = next_model_path(save_dir, dataset)
        standalone_model = torch.jit.script(model.cpu().eval())
        torch.jit.save(standalone_model, str(model_path))
        print(f"Saved standalone model to {model_path}")

if __name__ == "__main__":
    main()