import torch
from torch import nn


class PointNetPP(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _fps(X, num_samples, num_features):
        batch_size= X.shape[0]
        num_points= X.shape[1]
        feature_count= 3+num_features

        sample_points= torch.zeros(batch_size, num_samples, feature_count, device= X.device, dtype= X.dtype)

        initial_indices = torch.multinomial((X != 0.0).any(dim= -1).to(float), num_samples= 1).squeeze(1)

        sample_points[:,0,:]= X[torch.arange(0, batch_size, device= X.device), initial_indices]

        distances= torch.zeros(batch_size, num_points, num_samples, device= X.device, dtype= X.dtype)

        for i in range(num_samples-1):
            distances[:,:,i]= ((X[:,:,:3]-sample_points[:,i,:3])**2).sum(dim= -1, keepdim= False)

            sample_points[:,i+1,:]= X[torch.arange(0, batch_size, device= X.device), distances[:,:,:i+1].min(dim= -1, keepdim= False).argmax(dim= -1, keepdim= False)]

        # Assign every point to its nearest sampled point, then compact each
        # group into the fixed point dimension. Unused rows remain dummies.
        nearest_samples= ((X[:, :, None, :3] - sample_points[:, None, :, :3])**2).sum(dim= -1).argmin(dim= -1)
        groupings= torch.zeros(batch_size, num_samples, num_points, feature_count, device= X.device, dtype= X.dtype)

        for sample_index in range(num_samples):
            belongs_to_sample= nearest_samples == sample_index
            compact_indices= belongs_to_sample.cumsum(dim= -1) - 1
            valid_points= belongs_to_sample & (compact_indices < num_points)

            batch_indices, source_indices= valid_points.nonzero(as_tuple= True)
            target_indices= compact_indices[batch_indices, source_indices]
            groupings[batch_indices, sample_index, target_indices]= X[batch_indices, source_indices]

        return sample_points, groupings
        










