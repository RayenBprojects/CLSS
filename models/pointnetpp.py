import torch
from torch import nn


class PointNetPP(nn.Module):
    def __init__(self):
        super().__init__()


    def _fps(X, num_samples, num_features):
        batch_size= X.shape[0]
        num_points= X.shape[1]

        sample_points= torch.zeros(batch_size, num_samples, 3)
        initial_indices = torch.multinomial((X != 0.0).any(dim= -1).to(float), num_samples= 1).squeeze(1)

        sample_points[:,0,:]= X[initial_indices.reshape(initial_indices, 1)]

        distances= torch.zeros(batch_size, num_points, num_samples)

        





























    @staticmethod
    def _fps(X: torch.Tensor, num_samples: int, num_features: int) -> torch.Tensor:
        """Select well-spread points with batched farthest point sampling.

        ``X`` is expected to have shape ``[batch_size, num_points, features]``.
        The first point in each batch is selected as the initial centroid so
        that sampling is deterministic.
        """
        if X.ndim != 3:
            raise ValueError("X must have shape [batch_size, num_points, features]")
        if X.shape[-1] != num_features:
            raise ValueError(
                f"num_features must match X.shape[-1] ({X.shape[-1]})"
            )
        if not 1 <= num_samples <= X.shape[1]:
            raise ValueError(
                "num_samples must be between 1 and the number of input points"
            )

        batch_size, num_points, _ = X.shape
        sampled_indices = torch.empty(
            (batch_size, num_samples), dtype=torch.long, device=X.device
        )
        min_distances = torch.full(
            (batch_size, num_points), float("inf"), dtype=X.dtype, device=X.device
        )
        farthest_indices = torch.zeros(
            batch_size, dtype=torch.long, device=X.device
        )
        batch_indices = torch.arange(batch_size, device=X.device)

        for sample_index in range(num_samples):
            sampled_indices[:, sample_index] = farthest_indices
            centroid = X[batch_indices, farthest_indices].unsqueeze(1)
            distances = torch.sum((X - centroid) ** 2, dim=-1)
            min_distances = torch.minimum(min_distances, distances)
            farthest_indices = torch.argmax(min_distances, dim=1)

        gather_indices = sampled_indices.unsqueeze(-1).expand(-1, -1, num_features)
        return torch.gather(X, dim=1, index=gather_indices)