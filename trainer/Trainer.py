import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, save_dir):
        """
        Args:
            model (nn.Module): The Deep-RBF network.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            criterion (nn.Module): Custom loss function.
            optimizer (optim.Optimizer): Optimizer for training.
            device (torch.device): Device to run the model on (e.g., 'cuda' or 'cpu').
            save_dir (str): Directory to save the best model checkpoints.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir
        self.best_val_loss = float('inf')

        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

        # Lists to store loss values for plotting
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self):
        """
        Train the model for one epoch.
        """
        self.model.train()
        epoch_loss = 0.0

        # Use tqdm for progress bar
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for images, doctor_labels, real_labels in pbar:
            # Move data to the correct device
            images = images.to(self.device)
            doctor_labels = doctor_labels.to(self.device)
            real_labels = real_labels.to(self.device)

            # Forward pass
            distances = self.model(images)
            loss = self.criterion(distances, doctor_labels, real_labels)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

            # Update progress bar
            pbar.set_postfix({"Train Loss": loss.item()})

        # Return average loss for the epoch
        return epoch_loss / len(self.train_loader)

    def validate_epoch(self):
        """
        Validate the model for one epoch.
        """
        self.model.eval()
        epoch_loss = 0.0

        with torch.no_grad():
            # Use tqdm for progress bar
            pbar = tqdm(self.val_loader, desc="Validation", leave=False)
            for images, doctor_labels, real_labels in pbar:
                # Move data to the correct device
                images = images.to(self.device)
                doctor_labels = doctor_labels.to(self.device)
                real_labels = real_labels.to(self.device)

                # Forward pass
                distances = self.model(images)
                loss = self.criterion(distances, doctor_labels, real_labels)

                # Accumulate loss
                epoch_loss += loss.item()

                # Update progress bar
                pbar.set_postfix({"Val Loss": loss.item()})

        # Return average loss for the epoch
        return epoch_loss / len(self.val_loader)

    def save_checkpoint(self, epoch, val_loss):
        """
        Save the model checkpoint if the validation loss improves.
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            checkpoint_path = os.path.join(self.save_dir, f"best_model_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Saved best model checkpoint at epoch {epoch} with val loss {val_loss:.4f}")

    def plot_losses(self):
        """
        Plot the training and validation losses over epochs.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Losses")
        plt.legend()
        plt.grid(True)
        plt.show()

    def train(self, num_epochs):
        """
        Train the model for a given number of epochs.
        """
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")

            # Train and validate
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()

            # Save losses for plotting
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Save the best model checkpoint
            self.save_checkpoint(epoch + 1, val_loss)

            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Plot losses after training
        self.plot_losses()