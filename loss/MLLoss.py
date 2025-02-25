import torch
import torch.nn as nn

class MLLoss(nn.Module):
    def __init__(self, lambda_margin=1.0):
        """
        Args:
            lambda_margin (float): Margin for hinge loss.
        """
        super(MLLoss, self).__init__()
        self.lambda_margin = lambda_margin

    def forward(self, distances, doctor_labels, real_labels):
        """
        Args:
            distances (torch.Tensor): Shape (batch_size, num_classes).
            doctor_labels (torch.Tensor): Shape (batch_size,).
            real_labels (torch.Tensor): Shape (batch_size,).
        """
        batch_size = distances.size(0)
        loss = 0.0

        for i in range(batch_size):
            d = distances[i]  # Shape: (num_classes,)
            doctor_label = doctor_labels[i]
            real_label = real_labels[i]

            if doctor_label == "control":  # Group 1
                loss += d[0]  # Minimize distance to control
                for j in range(1, distances.size(1)):  # Maximize distance to other classes
                    loss += torch.max(torch.tensor(0.0), self.lambda_margin - d[j])
            elif doctor_label == "parkinson":  # Group 2
                loss += d[1]  # Minimize distance to parkinson
                for j in range(distances.size(1)):  # Maximize distance to other classes
                    if j != 1:
                        loss += torch.max(torch.tensor(0.0), self.lambda_margin - d[j])
            elif doctor_label == "unknown":  # Rejection groups
                if real_label == "control":  # Group 3
                    loss += torch.max(torch.tensor(0.0), self.lambda_margin - d[1])  # Maximize distance to parkinson
                elif real_label == "parkinson":  # Group 4
                    loss += torch.max(torch.tensor(0.0), self.lambda_margin - d[0])  # Maximize distance to control

        return loss / batch_size  # Average loss over the batch