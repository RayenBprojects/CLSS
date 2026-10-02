import torch
from torch import nn

from .pointnet import Model as PointNet

class Model(nn.Module):
    def __init__(self, num_classes, num_features):
        super().__init__()

        self.pointnet_1= PointNet(1024, input_features=3 + num_features)

        self.pointnet_2 = PointNet(
            1024,
            input_features=3 + 1024,
            use_input_transform=False,
        )

        self.unit_pointnet_1= nn.Sequential(
            nn.Linear(3+1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 64)
        )

        self.unit_pointnet_2= nn.Sequential(
            nn.Linear(3+64, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )


    def forward(self, X):
        fps_1= self._fps(X, 64, X.shape[-1] - 3)
        
        batch_size, sample_count, point_count, feature_count = fps_1[1].shape
        local_features = self.pointnet_1(
            fps_1[1].reshape(batch_size * sample_count, point_count, feature_count)
        ).reshape(batch_size, sample_count, -1)
        abstraction_1= torch.concat((fps_1[0][:,:,:3], local_features), dim= 2)

        fps_2= self._fps(abstraction_1, 16, 1024)

        batch_size, sample_count, point_count, feature_count = fps_2[1].shape
        local_features = self.pointnet_2(
            fps_2[1].reshape(batch_size * sample_count, point_count, feature_count)
        ).reshape(batch_size, sample_count, -1)
        abstraction_2= torch.concat((fps_2[0][:,:,:3], local_features), dim= 2)

        nearest_indices = fps_2[2].argmin(dim=2)

        batch_indices = torch.arange(
            abstraction_2.shape[0],
            device=X.device
        )[:, None]

        selected_features = abstraction_2[batch_indices, nearest_indices, 3:]

        interpolation_1 = torch.concat((abstraction_1[:, :, :3], selected_features), dim=2)

        interpolation_1= torch.concat((interpolation_1[:,:,:3], self.unit_pointnet_1(interpolation_1)), dim= 2)

        nearest_indices = fps_1[2].argmin(dim=2)

        batch_indices = torch.arange(
            abstraction_1.shape[0],
            device=X.device
        )[:, None]

        selected_features = interpolation_1[batch_indices, nearest_indices, 3:]

        interpolation_2= torch.concat((X[:,:,:3], selected_features), dim= 2)

        interpolation_2= torch.concat((interpolation_2[:,:,:3], self.unit_pointnet_2(interpolation_2)), dim= 2)

        return interpolation_2[:, :, 3:]
        



    # Massive PITA; I don't think I can even read this.
    @staticmethod
    def _fps(
        X: torch.Tensor,
        num_samples: int,
        num_features: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size= X.shape[0]
        num_points= X.shape[1]
        feature_count= 3+num_features

        sample_points= torch.zeros(batch_size, num_samples, feature_count, device= X.device, dtype= X.dtype)

        initial_indices = torch.multinomial((X != 0.0).any(dim= -1).float(), num_samples= 1).squeeze(1)

        sample_points[:,0,:]= X[torch.arange(0, batch_size, device= X.device), initial_indices]

        distances= torch.zeros(batch_size, num_points, num_samples, device= X.device, dtype= X.dtype)

        for i in range(num_samples-1):
            distances[:,:,i]= ((X[:,:,:3]-sample_points[:,i,:3].unsqueeze(1))**2).sum(dim= -1, keepdim= False)

            farthest_indices= distances[:,:,:i+1].min(dim= -1, keepdim= False).values.argmax(dim= -1, keepdim= False)
            sample_points[:,i+1,:]= X[torch.arange(0, batch_size, device= X.device), farthest_indices]

        distances[:, :, -1] = (
            (X[:, :, :3] - sample_points[:, -1, :3].unsqueeze(1)) ** 2
        ).sum(dim=-1)

        # Assign every point to its nearest sampled point, then compact each
        # group into the fixed point dimension. Unused rows remain dummies.
        # AI generated; must learn:
        nearest_samples= ((X[:, :, None, :3] - sample_points[:, None, :, :3])**2).sum(dim= -1).argmin(dim= -1)
        groupings= torch.zeros(batch_size, num_samples, num_points, feature_count, device= X.device, dtype= X.dtype)

        for sample_index in range(num_samples):
            belongs_to_sample= nearest_samples == sample_index
            compact_indices= belongs_to_sample.cumsum(dim= -1) - 1
            valid_points= belongs_to_sample & (compact_indices < num_points)

            valid_indices = valid_points.nonzero()
            batch_indices = valid_indices[:, 0]
            source_indices = valid_indices[:, 1]
            target_indices= compact_indices[batch_indices, source_indices]
            groupings[batch_indices, sample_index, target_indices]= X[batch_indices, source_indices]
            groupings[
                batch_indices, sample_index, target_indices, :3
            ] -= sample_points[batch_indices, sample_index, :3]

        return sample_points, groupings, distances