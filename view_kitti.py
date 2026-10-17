import argparse
import os

import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from tools.kitti360 import Kitti
from tools.model_viewer import (
    get_sample,
    load_saved_model,
    predict_labels,
    show_point_cloud,
)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="View a KITTI-360 sample prediction")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--model", help="TorchScript model file; defaults to latest KITTI360 model")
    parser.add_argument("--data-root", default=os.getenv("KITTI_360_DATASET_DIR"))
    parser.add_argument("--split", choices=("train", "test", "eval"), default="train")
    parser.add_argument("--num-samples", type=int, default=10)
    args = parser.parse_args()
    if args.data_root is None:
        parser.error("Set KITTI_360_DATASET_DIR or provide --data-root")

    dataset = Kitti(
        args.data_root,
        label_type="2d",
        set=args.split,
        std_num_points=90000,
        num_samples=args.num_samples,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_path = load_saved_model("KITTI360", args.model, device)
    points, labels = get_sample(dataloader, args.sample_index)
    prediction = predict_labels(model, points, device)
    point_array = points.numpy()
    label_array = labels.numpy()

    print(f"Model: {model_path}")
    show_point_cloud(
        point_array,
        label_array,
        prediction,
        label_array > 1,
        f"KITTI-360 {args.split} sample {args.sample_index}",
        class_count=46,
    )


if __name__ == "__main__":
    main()