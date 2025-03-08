import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, save_dir,save_results):
        """
        Args:
            model (nn.Module): The Deep-RBF network.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            criterion (nn.Module): Custom loss function.
            optimizer (optim.Optimizer): Optimizer for training.
            device (torch.device): Device to run the model on (e.g., 'cuda' or 'cpu').
            save_dir (str): Directory to save the best model checkpoints.
            save_results (str): Directory to save the results plot.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir
        self.save_results = save_results
        self.best_val_loss = float('inf')
        self.best_model_weights = None  # Store the best model weights

        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.save_results, exist_ok=True)

        # Lists to store loss values for plotting
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self):
        """
        Train the model for one epoch.
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0  # Track the number of non-empty batches

        # Use tqdm for progress bar
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for images, doctor_labels, real_labels, _ in pbar:
            # Skip empty batches
            if len(images) == 0:
                continue

            # Move data to the correct device
            images = images.to(self.device)
            doctor_labels = doctor_labels.to(self.device)
            real_labels = real_labels.to(self.device)

            # Forward pass
            distances = self.model(images)
            loss = self.criterion(distances, doctor_labels, real_labels)
            # print(loss)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({"Train Loss": loss.item()})

        # Return average loss for the epoch (avoid division by zero)
        return epoch_loss / num_batches if num_batches > 0 else 0.0

    def validate_epoch(self):
        """
        Validate the model for one epoch.
        """
        self.model.eval()
        epoch_loss = 0.0
        num_batches = 0  # Track the number of non-empty batches

        with torch.no_grad():
            # Use tqdm for progress bar
            pbar = tqdm(self.val_loader, desc="Validation", leave=False)
            for images, doctor_labels, real_labels, _ in pbar:
                # Skip empty batches
                if len(images) == 0:
                    continue

                # Move data to the correct device
                images = images.to(self.device)
                doctor_labels = doctor_labels.to(self.device)
                real_labels = real_labels.to(self.device)

                # Forward pass
                distances = self.model(images)
                loss = self.criterion(distances, doctor_labels, real_labels)

                # Accumulate loss
                epoch_loss += loss.item()
                num_batches += 1

                # Update progress bar
                pbar.set_postfix({"Val Loss": loss.item()})

        # Return average loss for the epoch (avoid division by zero)
        return epoch_loss / num_batches if num_batches > 0 else 0.0

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
        plt.savefig(os.path.join(self.save_results,"loss.png"))
        plt.show()

    def load_best_model(self):
        """
        Load the best model weights.
        """
        if self.best_model_weights is not None:
            self.model.load_state_dict(self.best_model_weights)
            print("Loaded the best model weights.")
        else:
            print("No best model weights found. Training might not have started yet.")

    def plot_confusion_matrix(self, true_labels, predicted_labels, class_names):
        """
        Plot and save the confusion matrix.

        Args:
            true_labels (list or np.array): Ground truth labels.
            predicted_labels (list or np.array): Predicted labels.
            class_names (list): Names of the classes.
        """
        cm = confusion_matrix(true_labels, predicted_labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(self.save_results, "confusion_matrix.png"))
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

    def predict(self, dataloader, threshold=1.0):
        """
        Perform inference on a given DataLoader and generate a classification report.

        Args:
            dataloader (DataLoader): DataLoader for inference.
            threshold (float): Threshold for rejection. If the minimum distance is greater than this, reject the sample.

        Returns:
            report (str): Classification report.
        """
        self.model.eval()  # Set model to evaluation mode
        all_predicted_labels = []
        all_doctor_labels = []

        with torch.no_grad():
            for images, doctor_labels, _, _ in dataloader:
                # Skip empty batches
                if len(images) == 0:
                    continue

                # Move data to the correct device
                images = images.to(self.device)
                doctor_labels = doctor_labels.to(self.device)

                # Predict labels
                predicted_labels, is_rejected = self.model.inference(images, threshold=threshold)

                # Handle rejected samples (assign label 2)
                predicted_labels[is_rejected] = 2

                # Collect results
                all_predicted_labels.extend(predicted_labels.cpu().numpy())
                all_doctor_labels.extend(doctor_labels.cpu().numpy())

        # Generate classification report
        target_names = ["control", "parkinson", "rejected"]
        report = classification_report(all_doctor_labels, all_predicted_labels, target_names=target_names)

        # Print and save classification report
        print("Classification Report:")
        print(report)

        # Save classification report to a file
        report_path = os.path.join(self.save_results, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write(report)

        print(f"Classification report saved to {report_path}")

        # Plot confusion matrix
        self.plot_confusion_matrix(all_doctor_labels, all_predicted_labels, target_names)

        return report