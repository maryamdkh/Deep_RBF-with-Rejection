import matplotlib.pyplot as plt
import torch
import numpy as np

def show_images_from_dataloader(dataloader, num_images=10):
    """
    Display a grid of images from a DataLoader.

    Args:
        dataloader (DataLoader): PyTorch DataLoader containing images and doctor_labels, real_labels, group_labels.
        num_images (int): Number of images to display (default: 10).
    """
    # Ensure the number of images to display is valid
    if num_images <= 0:
        raise ValueError("Number of images must be greater than 0.")

    # Get a batch of images and doctor_labels, real_labels, group_labels from the DataLoader
    images, doctor_labels, real_labels, group_labels = next(iter(dataloader))

    # Ensure we don't try to display more images than are available in the batch
    num_images = min(num_images, len(images))

    # Create a grid of images
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    fig.suptitle("Sample Images from DataLoader", fontsize=16)

    # If only one image is to be displayed, axes will not be a list
    if num_images == 1:
        axes = [axes]

    # Display the images
    for i in range(num_images):
        # Convert tensor to numpy array and transpose to (H, W, C)
        img = images[i].numpy().transpose((1, 2, 0))

        # Undo normalization (if applicable)
        img = (img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)  # Clip values to [0, 1] range

        # Display the image
        axes[i].imshow(img)
        axes[i].set_title(f"Label: {doctor_labels[i].item()}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()