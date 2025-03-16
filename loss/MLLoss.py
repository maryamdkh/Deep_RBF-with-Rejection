import torch
import torch.nn as nn

class MLLoss(nn.Module):
    def __init__(self, lambda_margin=500):
        """
        Args:
            lambda_margin (float): Margin for hinge loss.

        Improve:
            instead of max -> softplus 
            EDA -> group samples (balance group 1 2 more impo) (1,2 and 3,4 seperate balanced) 
            lambda_margin ->400-500  
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
        # print("batch_size",batch_size)
        loss = torch.tensor(0.0, device=distances.device, requires_grad=True)  # Initialize loss as a tensor

        for i in range(batch_size):
            d = distances[i]  # Shape: (num_classes,)
            # print(d.shape)
            # print(d)
            doctor_label = doctor_labels[i]
            real_label = real_labels[i]
            # print("doctor_label:",doctor_label)
            # print("real_label:",real_label)


            if doctor_label.item() == 0:  # Group 1
                loss = loss + d[0]  # Minimize distance to control

                for j in range(1, distances.size(1)):  # Maximize distance to other classes
                    loss = loss + torch.max(torch.tensor(0.0, device=distances.device), self.lambda_margin - d[j])
                    
            elif doctor_label.item()  == 1:  # Group 2
                loss = loss + d[1]  # Minimize distance to parkinson
                for j in range(distances.size(1)):  # Maximize distance to other classes
                    if j != 1:
                        loss = loss + torch.max(torch.tensor(0.0, device=distances.device), self.lambda_margin - d[j])

            elif doctor_label.item()  == 2:  # Rejection groups
                if real_label.item()  == 0:  # Group 3
                    loss = loss + torch.max(torch.tensor(0.0, device=distances.device), self.lambda_margin - d[1])  # Maximize distance to parkinson
                elif real_label.item()  == 1:  # Group 4
                    loss = loss + torch.max(torch.tensor(0.0, device=distances.device), self.lambda_margin - d[0])  # Maximize distance to control
        # print(loss)
        return loss / batch_size  # Average loss over the batch