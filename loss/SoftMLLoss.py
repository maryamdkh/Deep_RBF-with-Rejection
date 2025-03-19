import torch
import torch.nn as nn

class SoftMLLoss(nn.Module):
    def __init__(self, lambda_margin=500):
        """
        Args:
            lambda_margin (float): Margin for the log-sum-exp approximation.
        """
        super(SoftMLLoss, self).__init__()
        self.lambda_margin = lambda_margin

    def forward(self, distances, doctor_labels, real_labels):
        """
        Args:
            distances (torch.Tensor): Shape (batch_size, num_classes).
            doctor_labels (torch.Tensor): Shape (batch_size,).
            real_labels (torch.Tensor): Shape (batch_size,).
        """
        batch_size = distances.size(0)
        loss = torch.tensor(0.0, device=distances.device, requires_grad=True)

        for i in range(batch_size):
            d = distances[i]  # Shape: (num_classes,)
            doctor_label = doctor_labels[i]
            real_label = real_labels[i]

            # Compute O_i = exp(-d_i(x))
            O = torch.exp(-d)

            if doctor_label.item() == 0:  # Group 1
                # Minimize distance to control (first term: d_yi(x_i))
                loss = loss + d[0]

                # Maximize distance to other classes (second term: log(1 + exp(lambda - d_j(x_i))))
                for j in range(1, distances.size(1)):
                    loss = loss + torch.log1p(torch.exp(self.lambda_margin - d[j]))

            elif doctor_label.item() == 1:  # Group 2
                # Minimize distance to parkinson (first term: d_yi(x_i))
                loss = loss + d[1]

                # Maximize distance to other classes (second term: log(1 + exp(lambda - d_j(x_i))))
                for j in range(distances.size(1)):
                    if j != 1:
                        loss = loss +  torch.log1p(torch.exp(self.lambda_margin - d[j]))

            elif doctor_label.item() == 2:  # Rejection groups
                if real_label.item() == 0:  # Group 3
                    # Maximize distance to parkinson (second term: log(1 + exp(lambda - d_j(x_i))))
                    loss = loss + torch.log1p(torch.exp(self.lambda_margin - d[1]))
                elif real_label.item() == 1:  # Group 4
                    # Maximize distance to control (second term: log(1 + exp(lambda - d_j(x_i))))
                    loss = loss +torch.log1p(torch.exp(self.lambda_margin - d[0]))

        return loss / batch_size  # Average loss over the batch