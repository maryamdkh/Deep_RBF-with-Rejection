import torch
import torch.nn as nn

class DeepRBFNetwork(nn.Module):
    def __init__(self, feature_extractor,args):
        """
        Args:
            feature_extractor (nn.Module): Feature extractor network.
            num_classes (int): Number of classes.
            feature_dim (int): Dimension of the feature vector output by the feature extractor..
        Improve:
            distance -> l1
            init - > b: lambda_max /2 ; A: torch.randn() * 0.001 (or smaller) ; lambda 500 -> 200
        """
        super(DeepRBFNetwork, self).__init__()
        self.feature_extractor = feature_extractor
        self.num_classes = args.num_classes
        self.feature_dim = args.feature_dim

        # Define trainable parameters for each class
        self.A = nn.Parameter(torch.randn(self.num_classes, self.feature_dim, self.feature_dim) * 0.001)  # A_k matrices
        self.b = nn.Parameter(torch.full((self.num_classes, self.feature_dim), args.lambda_margin / 2))      # b_k vectors

    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)  # Shape: (batch_size, feature_dim)
        features = torch.squeeze(features)  # Shape: (batch_size, feature_dim)

        # Debugging: Print the shape of features
        # print(f"Features shape: {features.shape}")
# 
        # Ensure the batch dimension is preserved
        if features.dim() == 1:
            features = features.unsqueeze(0)  # Add batch dimension if missing

        # Debugging: Print the shape of features after unsqueeze
        # print(f"Features shape after unsqueeze: {features.shape}")

        # Ensure features has the correct shape (batch_size, feature_dim)
        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected features to have shape (batch_size, {self.feature_dim}), "
                f"but got {features.shape}"
            )

        # Compute distances for each class
        distances = []
        for k in range(self.num_classes):
            A_k = self.A[k]  # Shape: (feature_dim, feature_dim)
            b_k = self.b[k]  # Shape: (feature_dim,)

            # Debugging: Print the shapes of A_k and b_k
            # print(f"A_k shape: {A_k.shape}")
            # print(f"b_k shape: {b_k.shape}")

            # Compute (A_k^T * features^T)^T + b_k
            transformed_features = torch.matmul(features, A_k.T) + b_k  # Shape: (batch_size, feature_dim)

            # Debugging: Print the shape of transformed_features
            # print(f"Transformed features shape: {transformed_features.shape}")

            # Compute L1 norm (distance)
            d_k = torch.norm(transformed_features, p=1, dim=1)  # Shape: (batch_size,)
            distances.append(d_k)

        distances = torch.stack(distances, dim=1)  # Shape: (batch_size, num_classes)

        return distances

    def inference(self, x, threshold=1.0):
        """
        Perform inference for a batch of samples.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ...).
            threshold (float): Threshold for rejection. If the minimum distance is greater than this, reject the sample.

        Returns:
            predicted_labels (torch.Tensor): Predicted labels for each sample. Shape: (batch_size,).
            is_rejected (torch.Tensor): Boolean tensor indicating whether each sample is rejected. Shape: (batch_size,).
        """
        # Compute distances for all classes
        distances = self.forward(x)  # Shape: (batch_size, num_classes)

        # Find the minimum distance and corresponding class for each sample
        min_distances, predicted_labels = torch.min(distances, dim=1)  # Shapes: (batch_size,), (batch_size,)

        # Determine if the sample should be rejected
        is_rejected = min_distances > threshold  # Shape: (batch_size,)

        # Set predicted labels to -1 for rejected samples
        predicted_labels[is_rejected] = -1

        return predicted_labels, is_rejected