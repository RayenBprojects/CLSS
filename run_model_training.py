import asyncio
import argparse
from pathlib import Path

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

from core import train_pointnetpp


def next_model_path(save_dir: Path, dataset: str, prefix: str | None= None) -> Path:
    existing = (
        *save_dir.glob("[0-9][0-9][0-9][0-9]_*.pt"),
        *save_dir.glob("[0-9][0-9][0-9][0-9]_*.pt2"),
    )
    next_number = max(
        (int(path.name.split("_", maxsplit=1)[0]) for path in existing),
        default=0,
    ) + 1
    model_path = save_dir / f"{next_number:04d}_{dataset}.pt2"
    while model_path.exists():
        next_number += 1
        model_path = save_dir / f"{next_number:04d}_{dataset}.pt2"
    return model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default= 16)
    parser.add_argument("--learning_rate", default= 1e-3)
    parser.add_argument("--num_epochs", default= 5)
    parser.add_argument("--optim", choices= ("Adam", "SGD"), default= "Adam")
    parser.add_argument("-p", "--print_batch_data", action= 'store_true')
    args = parser.parse_args()

    model= None
    print(f"Optim: {args.optim} | Learning Rate: {args.learning_rate}")
    print("-- Training --")
    for i in range(int(args.num_epochs)):
        
        results = asyncio.run(train_pointnetpp(input_model= model, batch_size= int(args.batch_size), learning_rate= float(args.learning_rate), optimizer_choice= args.optim, num_epochs= 1, start_epoch= i, print_batch_data= args.print_batch_data))

        if results["status"]:
            dataset = results['dataset']
            model = results["model"]
            model.eval()

            save_dir = Path("./models/saved")
            save_dir.mkdir(parents=True, exist_ok=True)
            model_path = next_model_path(save_dir, dataset)
            model = model.cpu().eval()
            input_features = model.input_features
            example_input = torch.rand(1, 64, input_features)
            exported_model = torch.export.export(
                model,
                (example_input,),
                dynamic_shapes={
                    "X": {1: torch.export.Dim("num_points", min=64)},
                },
            )
            torch.export.save(exported_model, str(model_path))
            print(f"Saved standalone model to {model_path}")
    print("-- DONE --")
    
if __name__ == "__main__":
    main()
