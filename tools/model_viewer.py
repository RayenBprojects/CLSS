from pathlib import Path
import colorsys

import numpy as np
import open3d as o3d
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_saved_model(dataset: str, model_path: str | None, device: torch.device):
    if model_path is None:
        saved_models = sorted(
            (ROOT / "models" / "saved").glob(f"[0-9][0-9][0-9][0-9]_{dataset}.pt"),
            key=lambda path: int(path.name.split("_", maxsplit=1)[0]),
        )
        if not saved_models:
            raise FileNotFoundError(
                f"No saved {dataset} model found in {ROOT / 'models' / 'saved'}; "
                "train one with run.py or pass --model."
            )
        model_path = str(saved_models[-1])

    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    return torch.jit.load(str(path), map_location=device).eval(), path


def get_sample(dataloader, sample_index: int):
    if sample_index < 0:
        raise ValueError("sample index must be zero or greater")
    for index, sample in enumerate(dataloader):
        if index == sample_index:
            return sample[0][0], sample[1][0]
    raise IndexError(f"sample index {sample_index} is outside the dataset")


def predict_labels(model, points: torch.Tensor, device: torch.device) -> np.ndarray:
    with torch.inference_mode():
        logits = model(points.unsqueeze(0).to(device))
    return logits.argmax(dim=-1)[0].cpu().numpy()


def _class_colors(labels: np.ndarray, class_count: int) -> np.ndarray:
    palette = np.asarray(
        [
            colorsys.hsv_to_rgb((index * 0.61803398875) % 1.0, 0.78, 0.95)
            for index in range(class_count)
        ],
        dtype=np.float64,
    )
    colors = np.full((len(labels), 3), (0.45, 0.45, 0.45), dtype=np.float64)
    valid = (labels >= 0) & (labels < class_count)
    colors[valid] = palette[labels[valid].astype(np.int64)]
    return colors


def show_point_cloud(
    points: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    valid_mask: np.ndarray,
    title: str,
    class_count: int,
) -> None:
    points = np.asarray(points)[:, :3]
    ground_truth = np.asarray(ground_truth).reshape(-1)
    prediction = np.asarray(prediction).reshape(-1)
    valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
    valid_mask &= np.isfinite(points).all(axis=1)
    points = points[valid_mask]
    ground_truth = ground_truth[valid_mask]
    prediction = prediction[valid_mask]
    if not len(points):
        raise ValueError("The selected sample contains no valid points")

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64, copy=False))
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.point_size = 2.0

    app = o3d.visualization.gui.Application.instance
    app.initialize()
    window = app.create_window(title, 1280, 800)
    scene = o3d.visualization.gui.SceneWidget()
    scene.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
    scene.scene.set_background((0.055, 0.065, 0.075, 1.0))

    controls = o3d.visualization.gui.Horiz(
        8,
        o3d.visualization.gui.Margins(12, 6, 12, 6),
    )
    ground_truth_button = o3d.visualization.gui.Button("Ground truth")
    prediction_button = o3d.visualization.gui.Button("Prediction")
    selection_label = o3d.visualization.gui.Label("")
    controls.add_child(ground_truth_button)
    controls.add_child(prediction_button)
    controls.add_stretch()
    controls.add_child(selection_label)
    window.add_child(controls)
    window.add_child(scene)

    displayed = [False]

    def select_view(name: str, labels: np.ndarray) -> None:
        cloud.colors = o3d.utility.Vector3dVector(_class_colors(labels, class_count))
        if displayed[0]:
            scene.scene.remove_geometry("sample")
        scene.scene.add_geometry("sample", cloud, material)
        displayed[0] = True
        selection_label.text = f"{name} | {len(points):,} points"
        scene.force_redraw()

    ground_truth_button.set_on_clicked(
        lambda: select_view("Ground truth", ground_truth)
    )
    prediction_button.set_on_clicked(lambda: select_view("Prediction", prediction))

    bounds = cloud.get_axis_aligned_bounding_box()
    scene.setup_camera(60.0, bounds, bounds.get_center())
    select_view("Ground truth", ground_truth)

    def layout(_context) -> None:
        rect = window.content_rect
        control_height = 48
        controls.frame = o3d.visualization.gui.Rect(
            rect.x, rect.y, rect.width, control_height
        )
        scene.frame = o3d.visualization.gui.Rect(
            rect.x,
            rect.y + control_height,
            rect.width,
            max(0, rect.height - control_height),
        )

    window.set_on_layout(layout)
    app.run()