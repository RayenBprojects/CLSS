import argparse
import os
from pathlib import Path

import torch
from dotenv import load_dotenv

from tools.model_viewer import (
    get_sample,
    load_saved_model,
    predict_labels,
    show_point_cloud,
)
from data.manage.s3dis import create_s3dis_dataloader


ROOT = Path(__file__).resolve().parent


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="View an S3DIS sample prediction")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--model", help="TorchScript model file; defaults to latest S3DIS model")
    parser.add_argument(
        "--dataset-root",
        default=os.getenv("S3DIS_DATASET_DIR", str(ROOT / "data" / "s3dis")),
    )
    parser.add_argument("--test-area", type=int, default=6)
    args = parser.parse_args()

    dataloader = create_s3dis_dataloader(
        root=args.dataset_root,
        batch_size=1,
        train=False,
        test_area=args.test_area,
        shuffle=False,
        drop_last=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_path = load_saved_model("S3DIS", args.model, device)
    points, labels = get_sample(dataloader, args.sample_index)
    prediction = predict_labels(model, points, device)
    point_array = points.numpy()
    label_array = labels.numpy()

    print(f"Model: {model_path}")
    show_point_cloud(
        point_array,
        label_array,
        prediction,
        label_array >= 0,
        f"S3DIS sample {args.sample_index}",
        class_count=13,
    )


if __name__ == "__main__":
    main()