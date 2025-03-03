import torch
import torch.nn as nn

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
        features = self.feature_extractor(x)
        features = torch.squeeze(features)  # Shape: (batch_size, feature_dim)
        # print("features:", features.shape)

        # Compute distances for each class
        distances = []
        for k in range(self.num_classes):
            A_k = self.A[k]  # Shape: (feature_dim, feature_dim)
            # print("A_k:",A_k.shape)
            b_k = self.b[k]  # Shape: (feature_dim,)
            # print("b_k:",b_k.shape)

            # Compute (A_k^T * features^T)^T + b_k
            transformed_features = torch.matmul(features, A_k.T) + b_k  # Shape: (batch_size, feature_dim)

            # Compute L2 norm (distance)
            d_k = torch.norm(transformed_features, p=2, dim=1)  # Shape: (batch_size,)
            distances.append(d_k)

        distances = torch.stack(distances, dim=1)  # Shape: (batch_size, num_classes)

        return distances