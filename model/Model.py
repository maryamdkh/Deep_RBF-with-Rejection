import torch
import torch.nn as nn

class DeepRBFNetwork(nn.Module):
    def __init__(self, feature_extractor, args):
        """
        Args:
            feature_extractor (nn.Module): Feature extractor network.
            args: Arguments containing num_classes, feature_dim, and lambda_margin.
        """
        super(DeepRBFNetwork, self).__init__()
        self.feature_extractor = feature_extractor
        self.num_classes = args.num_classes
        self.feature_dim = args.feature_dim

        # Define trainable parameters for each class
        self.A = nn.Parameter(torch.randn(self.num_classes, self.feature_dim, self.feature_dim) * 0.0001)  # A_k matrices
        self.b = nn.Parameter(torch.full((self.num_classes, self.feature_dim), args.lambda_margin / 2))      # b_k vectors

    def forward(self, x):
        """
        Compute distances for all classes.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ...).

        Returns:
            distances (torch.Tensor): Distances for each class. Shape: (batch_size, num_classes).
        """
        # Extract features
        features = self.feature_extractor(x)  # Shape: (batch_size, feature_dim)
        features = torch.squeeze(features)  # Shape: (batch_size, feature_dim)

        # Ensure the batch dimension is preserved
        if features.dim() == 1:
            features = features.unsqueeze(0)  # Add batch dimension if missing

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

            # Compute (A_k^T * features^T)^T + b_k
            transformed_features = torch.matmul(features, A_k.T) + b_k  # Shape: (batch_size, feature_dim)

            # Compute L1 norm (distance)
            d_k = torch.norm(transformed_features, p=2, dim=1)  # Shape: (batch_size,)
            distances.append(d_k)

        distances = torch.stack(distances, dim=1)  # Shape: (batch_size, num_classes)

        return distances

    def inference(self, x, threshold=1.0):
        """
        Perform inference for a batch of samples using the minimum distance approach.

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

        return distances,predicted_labels, is_rejected

    def inference_softml(self, x, lambda_eval=500):
        """
        Perform inference for a batch of samples using the SoftML probability approach.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ...).
            lambda_eval (float): Lambda parameter for evaluation (used to compute the rejection threshold).

        Returns:
            predicted_labels (torch.Tensor): Predicted labels for each sample. Shape: (batch_size,).
            is_rejected (torch.Tensor): Boolean tensor indicating whether each sample is rejected. Shape: (batch_size,).
        """
        # Compute distances for all classes
        distances = self.forward(x)  # Shape: (batch_size, num_classes)

        # Compute O_i = exp(-d_i(x))
        O = torch.exp(-distances)  # Shape: (batch_size, num_classes)

        # Convert lambda_eval to a tensor and move it to the same device as O
        lambda_eval_tensor = torch.tensor(lambda_eval, dtype=torch.float32, device=O.device)

        # Compute the rejection threshold T
        T = -torch.log((-1 + torch.sqrt(1 + 4 * torch.exp(lambda_eval_tensor))) / (2 * torch.exp(lambda_eval_tensor)))

        # Compute probabilities for each class
        numerator = O * (1 + torch.exp(lambda_eval_tensor) * O)  # Shape: (batch_size, num_classes)
        denominator = torch.prod(1 + torch.exp(lambda_eval_tensor) * O, dim=1, keepdim=True)  # Shape: (batch_size, 1)
        p_classes = numerator / denominator  # Shape: (batch_size, num_classes)

        # Compute probability for the rejection class
        p_rejection = 1 / denominator.squeeze(1)  # Shape: (batch_size,)

        # Combine probabilities for all classes and the rejection class
        p_all = torch.cat([p_classes, p_rejection.unsqueeze(1)], dim=1)  # Shape: (batch_size, num_classes + 1)

        # Find the class with the highest probability
        _, predicted_labels = torch.max(p_all, dim=1)  # Shape: (batch_size,)

        # Determine if the sample should be rejected
        is_rejected = predicted_labels == self.num_classes  # Rejection class is the last one

        # Set predicted labels to -1 for rejected samples
        predicted_labels[is_rejected] = -1

        return predicted_labels, is_rejected