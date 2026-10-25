import torch
from torch import nn


def _gather_points(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch_indices = torch.arange(points.shape[0], device=points.device)
    batch_indices = batch_indices.view(points.shape[0], *([1] * (indices.ndim - 1)))
    return points[batch_indices, indices]


def _farthest_point_sample(
    points: torch.Tensor,
    num_samples: int,
    valid_points: torch.Tensor,
) -> torch.Tensor:
    """Return FPS indices while retaining only one nearest-distance value per point."""
    batch_size, point_count = points.shape[:2]
    if num_samples <= 0:
        raise ValueError("num_samples must be greater than zero")

    with torch.no_grad():
        first_indices = torch.multinomial(valid_points.float(), num_samples=1).squeeze(1)
        sample_indices = torch.empty(
            (batch_size, num_samples),
            dtype=torch.long,
            device=points.device,
        )
        sample_indices[:, 0] = first_indices
        nearest_distances = torch.full(
            (batch_size, point_count),
            float("inf"),
            dtype=points.dtype,
            device=points.device,
        )
        batch_indices = torch.arange(batch_size, device=points.device)

        for sample_index in range(num_samples):
            center = points[batch_indices, sample_indices[:, sample_index], :3]
            distances = ((points[:, :, :3] - center[:, None, :]) ** 2).sum(dim=-1)
            nearest_distances = torch.minimum(nearest_distances, distances)
            nearest_distances = nearest_distances.masked_fill(~valid_points, -1.0)

            if sample_index + 1 < num_samples:
                sample_indices[:, sample_index + 1] = nearest_distances.argmax(dim=1)

    return sample_indices


def _knn_indices(
    source_xyz: torch.Tensor,
    query_xyz: torch.Tensor,
    source_valid: torch.Tensor,
    num_neighbors: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find bounded neighborhoods without materializing all query/source pairs."""
    if num_neighbors <= 0:
        raise ValueError("num_neighbors must be greater than zero")
    if source_xyz.shape[1] == 0:
        raise ValueError("Cannot find neighbors in an empty point set")

    neighbor_count = min(num_neighbors, source_xyz.shape[1])
    with torch.no_grad():
        distances = torch.cdist(query_xyz, source_xyz)
        distances = distances.masked_fill(~source_valid[:, None, :], float("inf"))
        nearest_distances, nearest_indices = distances.topk(
            neighbor_count,
            dim=-1,
            largest=False,
            sorted=True,
        )
        neighbor_valid = torch.isfinite(nearest_distances)

        if neighbor_count < num_neighbors:
            padding_count = num_neighbors - neighbor_count
            nearest_indices = torch.nn.functional.pad(
                nearest_indices,
                (0, padding_count),
            )
            neighbor_valid = torch.nn.functional.pad(
                neighbor_valid,
                (0, padding_count),
                value=False,
            )

    return nearest_indices, neighbor_valid


def _masked_max(features: torch.Tensor, valid_neighbors: torch.Tensor) -> torch.Tensor:
    features = features.masked_fill(~valid_neighbors.unsqueeze(-1), float("-inf"))
    pooled = features.max(dim=2).values
    return pooled.masked_fill(~valid_neighbors.any(dim=2, keepdim=True), 0.0)


class _SharedMLP(nn.Module):
    def __init__(self, input_features: int, output_features: tuple[int, ...]):
        super().__init__()
        layers = []
        for feature_count in output_features:
            layers.extend((nn.Linear(input_features, feature_count), nn.ReLU()))
            input_features = feature_count
        self.layers = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class Model(nn.Module):
    """PointNet++ segmentation with bounded neighborhoods and MRG-style fusion."""

    level1_centers = 256
    level1_neighbors = 32
    level2_centers = 64
    level2_neighbors = 16
    direct_context_neighbors = 32

    def __init__(self, num_classes: int, num_features: int):
        super().__init__()
        self.input_features = 3 + num_features

        self.level1_mlp = _SharedMLP(self.input_features, (64, 64, 128))
        self.level2_mlp = _SharedMLP(3 + 128, (128, 128))
        self.direct_context_mlp = _SharedMLP(
            self.input_features,
            (64, 128),
        )
        self.merge_context = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.feature_propagation = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.segmentation_head = nn.Sequential(
            nn.Linear(3 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim != 3 or X.shape[-1] != self.input_features:
            raise ValueError(
                f"Expected input shape (batch, points, {self.input_features}), "
                f"got {tuple(X.shape)}"
            )

        valid_points = (X != 0.0).any(dim=-1)

        level1_indices = _farthest_point_sample(
            X,
            self.level1_centers,
            valid_points,
        )
        level1_xyz = _gather_points(X[:, :, :3], level1_indices)
        level1_group_indices, level1_group_valid = _knn_indices(
            X[:, :, :3],
            level1_xyz,
            valid_points,
            self.level1_neighbors,
        )
        level1_group_xyz = (
            _gather_points(X[:, :, :3], level1_group_indices)
            - level1_xyz.unsqueeze(2)
        )
        level1_group_features = _gather_points(X[:, :, 3:], level1_group_indices)
        level1_groups = torch.cat((level1_group_xyz, level1_group_features), dim=-1)
        level1_group_features = self.level1_mlp(level1_groups)
        level1_features = _masked_max(level1_group_features, level1_group_valid)

        level1_valid = torch.ones(
            level1_xyz.shape[:2],
            dtype=torch.bool,
            device=X.device,
        )
        level2_indices = _farthest_point_sample(
            level1_xyz,
            self.level2_centers,
            level1_valid,
        )
        level2_xyz = _gather_points(level1_xyz, level2_indices)
        level2_valid = torch.ones(
            level2_xyz.shape[:2],
            dtype=torch.bool,
            device=X.device,
        )

        level2_group_indices, level2_group_valid = _knn_indices(
            level1_xyz,
            level2_xyz,
            level1_valid,
            self.level2_neighbors,
        )
        level2_group_xyz = (
            _gather_points(level1_xyz, level2_group_indices)
            - level2_xyz.unsqueeze(2)
        )
        level2_group_features = _gather_points(level1_features, level2_group_indices)
        level2_groups = torch.cat((level2_group_xyz, level2_group_features), dim=-1)
        subregion_features = _masked_max(
            self.level2_mlp(level2_groups),
            level2_group_valid,
        )

        direct_indices, direct_valid = _knn_indices(
            X[:, :, :3],
            level2_xyz,
            valid_points,
            self.direct_context_neighbors,
        )
        direct_xyz = (
            _gather_points(X[:, :, :3], direct_indices)
            - level2_xyz.unsqueeze(2)
        )
        direct_input_features = _gather_points(X[:, :, 3:], direct_indices)
        direct_groups = torch.cat((direct_xyz, direct_input_features), dim=-1)
        direct_features = _masked_max(
            self.direct_context_mlp(direct_groups),
            direct_valid,
        )

        level2_features = self.merge_context(
            torch.cat((subregion_features, direct_features), dim=-1)
        )

        level1_context_indices, _ = _knn_indices(
            level2_xyz,
            level1_xyz,
            level2_valid,
            num_neighbors=1,
        )
        level1_context = _gather_points(level2_features, level1_context_indices).squeeze(2)
        level1_features = self.feature_propagation(
            torch.cat((level1_features, level1_context), dim=-1)
        )

        point_context_indices, _ = _knn_indices(
            level1_xyz,
            X[:, :, :3],
            level1_valid,
            num_neighbors=1,
        )
        point_context = _gather_points(level1_features, point_context_indices).squeeze(2)
        point_features = torch.cat((X[:, :, :3], point_context), dim=-1)
        return self.segmentation_head(point_features)
