import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepRBFNetwork(nn.Module):
    def __init__(self, feature_extractor, num_classes, feature_dim):
        """
        Args:
            feature_extractor (nn.Module): Feature extractor network.
            num_classes (int): Number of classes.
            feature_dim (int): Dimension of the feature vector output by the feature extractor.
        """
        super(DeepRBFNetwork, self).__init__()
        self.feature_extractor = feature_extractor
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # Define trainable parameters for each class
        self.A = nn.Parameter(torch.randn(num_classes, feature_dim, feature_dim))  # A_k matrices
        self.b = nn.Parameter(torch.randn(num_classes, feature_dim))              # b_k vectors

    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)  # Shape: (batch_size, feature_dim)

        # Compute distances for each class
        distances = []
        for k in range(self.num_classes):
            A_k = self.A[k]  # Shape: (feature_dim, feature_dim)
            b_k = self.b[k]  # Shape: (feature_dim,)
            d_k = torch.norm(torch.matmul(A_k.T, features.T).T + b_k, p=2, dim=1)  # Shape: (batch_size,)
            distances.append(d_k)
        distances = torch.stack(distances, dim=1)  # Shape: (batch_size, num_classes)

        return distances