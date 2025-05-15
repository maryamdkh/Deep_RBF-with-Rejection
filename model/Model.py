import torch
import torch.nn as nn

class DeepRBFNetwork(nn.Module):
    def __init__(self, args,feature_extractor=None):
        """
        Args:
            feature_extractor (nn.Module): Backbone CNN with convolutional layers.
            args: Arguments containing num_classes, feature_dim, lambda_margin, distance_metric.
        """
        super(DeepRBFNetwork, self).__init__()

        if feature_extractor:
            # Freeze feature extractor
            for param in feature_extractor.parameters():
                param.requires_grad = False
            self.feature_extractor = feature_extractor

        # Post-feature extraction processing
        self.post_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global pooling
            nn.Flatten(),  # [B, 2048]
            nn.Linear(args.input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, args.feature_dim),  # Final feature_dim (e.g., 512)
        )

        # Initialize the new layers with a fixed seed
        self._initialize_weights(seed=42)

        self.num_classes = args.num_classes
        self.feature_dim = args.feature_dim
        self.meteric_norm = 2 if args.distance_metric == 'l2' else 1

        # Learnable RBF parameters (randomly initialized)
        self.A = nn.Parameter(torch.randn(self.num_classes, self.feature_dim, self.feature_dim) * 0.0001)
        self.b = nn.Parameter(torch.full((self.num_classes, self.feature_dim), args.lambda_margin / 2))

    def _initialize_weights(self, seed=42):
        torch.manual_seed(seed)
        for m in self.post_extractor:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x):
        """
        Compute distances for all classes.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ...).

        Returns:
            distances (torch.Tensor): Distances for each class. Shape: (batch_size, num_classes).
        """
        if self.feature_extractor:
            # Extract frozen features
            with torch.no_grad():
                x = self.feature_extractor(x)

            # Process features to desired dimension
            features = self.post_extractor(x)  # Shape: [batch_size, feature_dim]
        else:
            features = x

        # Ensure the batch dimension is preserved
        if features.dim() == 1:
            features = features.unsqueeze(0)  # Add batch dimension if missing

        # Compute distances for each class
        distances = []
        for k in range(self.num_classes):
            A_k = self.A[k]
            b_k = self.b[k]
            transformed = torch.matmul(features, A_k.T) + b_k
            d_k = torch.norm(transformed, p=self.meteric_norm, dim=1)
            distances.append(d_k)

        distances = torch.stack(distances, dim=1)  # [batch_size, num_classes]
        return distances
    


    def inference(self, x, rejection_threshold=1.0, confidence_threshold=0.5):
        """
        Perform inference for a batch of samples using the minimum distance approach with rejection rules.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ...).
            rejection_threshold (float): Threshold for rejection based on distance. 
                If the minimum distance is greater than this, reject the sample.
            confidence_threshold (float): Threshold for rejection based on confidence.
                If the difference between distances to the two closest classes is less than this,
                reject the sample even if the minimum distance is below rejection_threshold.

        Returns:
            distances (torch.Tensor): Distances to all class centers. Shape: (batch_size, num_classes).
            predicted_labels (torch.Tensor): Predicted labels for each sample (-1 for unknown). Shape: (batch_size,).
            is_rejected (torch.Tensor): Boolean tensor indicating whether each sample is rejected. Shape: (batch_size,).
        """
        # Compute distances for all classes
        distances = self.forward(x)  # Shape: (batch_size, num_classes)

        # Find the minimum distance and corresponding class for each sample
        min_distances, predicted_labels = torch.min(distances, dim=1)  # Shapes: (batch_size,), (batch_size,)

        # For binary case, compute the difference between distances to the two classes
        if distances.shape[1] == 2:
            # Absolute difference between distances to the two classes
            confidence = torch.abs(distances[:, 0] - distances[:, 1])
        else:
            # For multi-class, get difference between top two closest classes
            sorted_distances, _ = torch.sort(distances, dim=1)
            confidence = sorted_distances[:, 1] - sorted_distances[:, 0]

        # Determine if the sample should be rejected based on two conditions:
        # 1. Minimum distance is too large (far from all known classes)
        # 2. Confidence is too low (difference between top two classes is small)
        
        is_rejected = (min_distances > rejection_threshold) | (confidence < confidence_threshold)
        # is_rejected = (min_distances > rejection_threshold)

        # Set predicted labels to -1 for rejected samples
        predicted_labels[is_rejected] = -1

        return distances, predicted_labels, is_rejected

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