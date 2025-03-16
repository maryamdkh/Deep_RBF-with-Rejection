import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn as sns
import pandas as pd

def plot_group_distribution(dataset,file_path, title="Group Distribution"):
    """
    Plot the distribution of each group before and after oversampling.

    Args:
        title (str): Title of the plot.
    """
    dataframe = dataset.dataframe
    oversampled_indices = dataset.oversampled_indices

    # Create a DataFrame for the original data
    original_df = dataframe.copy()
    original_df["group_label"] = -1  # Initialize group labels

    # Assign group labels to the original DataFrame
    for idx in range(len(original_df)):
        _, _, group_label = dataset._get_labels_and_group(idx)
        original_df.at[idx, "group_label"] = group_label

    # Create a DataFrame for the oversampled data
    oversampled_df = pd.DataFrame({
        "group_label": [dataset._get_labels_and_group(idx)[2] for idx in oversampled_indices]
    })

    # Plot settings
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=16)

    # Plot original distribution
    sns.countplot(x="group_label", data=original_df, ax=axes[0], palette="Set2")
    axes[0].set_title("Before Oversampling")
    axes[0].set_xlabel("Group Label")
    axes[0].set_ylabel("Count")

    # Plot oversampled distribution
    sns.countplot(x="group_label", data=oversampled_df, ax=axes[1], palette="Set2")
    axes[1].set_title("After Oversampling")
    axes[1].set_xlabel("Group Label")
    axes[1].set_ylabel("Count")

    # Add annotations for clarity
    for ax in axes:
        for p in ax.patches:
            ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha="center", va="center", fontsize=10, color="black", xytext=(0, 5),
                        textcoords="offset points")

    plt.tight_layout()
    plt.savefig(file_path)
    plt.show()



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