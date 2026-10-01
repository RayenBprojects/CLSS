import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from torch_geometric.datasets import S3DIS


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "s3dis"
DEFAULT_MODEL = ROOT / "models" / "saved" / "PointnetPPS3DIS.pt"
CLASS_COLORS = np.array(
	[
		[0.90, 0.12, 0.12],
		[0.12, 0.65, 0.18],
		[0.10, 0.30, 0.95],
		[0.96, 0.76, 0.12],
		[0.65, 0.16, 0.78],
		[0.10, 0.80, 0.78],
		[1.00, 0.40, 0.08],
		[0.40, 0.70, 0.10],
		[0.95, 0.30, 0.58],
		[0.40, 0.40, 0.40],
		[0.55, 0.33, 0.17],
		[0.05, 0.55, 0.45],
		[0.20, 0.55, 0.88],
	],
	dtype=np.float64,
)


def make_cloud(points: np.ndarray, labels: np.ndarray) -> o3d.geometry.PointCloud:
	cloud = o3d.geometry.PointCloud()
	cloud.points = o3d.utility.Vector3dVector(points)
	colors = np.full((len(labels), 3), 0.15, dtype=np.float64)
	valid = (labels >= 0) & (labels < len(CLASS_COLORS))
	colors[valid] = CLASS_COLORS[labels[valid]]
	cloud.colors = o3d.utility.Vector3dVector(colors)
	return cloud


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Compare S3DIS ground-truth labels with TorchScript inference."
	)
	parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
	parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
	parser.add_argument("--test-area", type=int, default=6)
	parser.add_argument("--sample", type=int)
	args = parser.parse_args()

	dataset = S3DIS(root=str(args.data), test_area=args.test_area, train=False)
	sample_index = args.sample
	if sample_index is None:
		try:
			sample_index = int(input(f"Enter sample index (0-{len(dataset) - 1}): "))
		except ValueError:
			parser.error("Sample index must be an integer")
	if not 0 <= sample_index < len(dataset):
		parser.error(f"Sample index must be between 0 and {len(dataset) - 1}")
	if not args.model.is_file():
		parser.error(
			f"TorchScript model not found: {args.model}. "
			"Run run.py to train and export it first."
		)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = torch.jit.load(str(args.model), map_location=device).eval()

	sample = dataset[sample_index]
	points = sample.pos.cpu().numpy()
	model_input = torch.cat((sample.pos, sample.x), dim=-1).unsqueeze(0).to(device)
	with torch.inference_mode():
		predictions = model(model_input).argmax(dim=-1)[0].cpu().numpy()

	ground_truth_cloud = make_cloud(points, sample.y.cpu().numpy())
	prediction_cloud = make_cloud(points, predictions)
	x_offset = float(np.ptp(points[:, 0]) + 0.25)
	prediction_cloud.translate((x_offset, 0.0, 0.0))

	print(f"Showing test sample {sample_index}: ground truth (left), inference (right).")
	app = o3d.visualization.gui.Application.instance
	app.initialize()
	viewer = o3d.visualization.O3DVisualizer(
		"S3DIS: Ground Truth | TorchScript Inference", 1024, 768
	)
	viewer.add_geometry("Ground Truth", ground_truth_cloud)
	viewer.add_geometry("TorchScript Inference", prediction_cloud)
	viewer.show_settings = True
	app.add_window(viewer)
	app.run()


if __name__ == "__main__":
	main()
