import torch
import numpy as np

def balanced_collate_fn(batch):
    """
    Collate function to balance samples from each group in a batch.
    Args:
        batch (list): List of tuples (image, doctor_label, real_label, group_label).
    Returns:
        Balanced batch of images, doctor_labels, real_labels, and group_labels.
    """
    # Separate samples by group
    groups = {0: [], 1: [], 2: [], 3: []}  # Group 0: control, Group 1: parkinson, Group 2: unknown_control, Group 3: unknown_parkinson
    for sample in batch:
        image, doctor_label, real_label, group_label = sample
        groups[group_label].append((image, doctor_label, real_label, group_label))

    # Find the minimum number of samples across all groups
    min_samples = min(len(groups[0]), len(groups[1]), len(groups[2]), len(groups[3]))

    # Balance the groups by randomly sampling with replacement
    balanced_batch = []
    for group in groups.values():
        if len(group) > min_samples:
            # Sample indices with replacement
            indices = np.random.choice(len(group), size=min_samples, replace=True)
            # Use indices to select samples
            sampled_group = [group[i] for i in indices]
            balanced_batch.extend(sampled_group)
        else:
            balanced_batch.extend(group)

    # Shuffle the balanced batch
    np.random.shuffle(balanced_batch)

    # Separate images, doctor_labels, real_labels, and group_labels
    images = torch.stack([item[0] for item in balanced_batch])
    doctor_labels = torch.tensor([item[1] for item in balanced_batch])
    real_labels = torch.tensor([item[2] for item in balanced_batch])
    group_labels = torch.tensor([item[3] for item in balanced_batch])

    return images, doctor_labels, real_labels, group_labels