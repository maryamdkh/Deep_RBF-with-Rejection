import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import random

class PaHaWDataset(Dataset):
    def __init__(self, df, image_root_dir=None, transform=None, has_labels=True):
        """
        Args:
            df (pd.DataFrame): DataFrame containing at least:
                - 'image_path': relative or absolute paths to images
                - (optional) 'label': corresponding labels if has_labels=True
            image_root_dir (string, optional): Root directory to prepend to image paths.
                                              If None, paths are treated as absolute.
            transform (callable, optional): Optional transform to be applied on a sample.
            has_labels (bool): Whether the dataset contains labels (for training/validation)
                               or not (for inference).
        """
        self.df = df.copy()
        self.image_root_dir = image_root_dir
        self.has_labels = has_labels
        
        # Default transform if none provided
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((128, 128)),  # First resize the image
            transforms.ToTensor(),  # Then convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize tensor
                               std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get image path from DataFrame
        img_path = self.df.iloc[idx]['image_path']
        
        # Prepend root directory if specified
        if self.image_root_dir is not None:
            img_path = os.path.join(self.image_root_dir, img_path)
        
        # Load image
        image = Image.open(img_path).convert('RGB')  # Ensure RGB format
        
        # Apply transform (which now properly converts PIL to Tensor first)
        if self.transform:
            image = self.transform(image)
        
        # Return different items depending on whether we have labels
        if self.has_labels:
            label = self.df.iloc[idx]['label']
            return image, label, img_path  # Return image, label, and path
        else:
            return image, img_path  # Return image and path for inference

    def get_labels(self):
        """Return all labels if available, otherwise None"""
        if self.has_labels:
            return self.df['label'].values
        return None
    
class ParkinsonDataset(Dataset):
    def __init__(self, dataframe, data_dir, transform=None, is_train=True, oversample_option=1, k=None, class_weights=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing image file paths, doctor_label, and real_label.
            data_dir (str): Root directory where images are stored.
            transform (callable, optional): Transform to be applied to the images.
            is_train (bool): Whether the dataset is for training (default: True).
            oversample_option (int): 1 for equal and balanced sampling, 2 for weighted sampling.
            k (int, optional): Constant number to oversample to. If None, oversample to the max length within each pair.
            class_weights (dict, optional): Manual weights for Control and Parkinson classes. 
                                           Example: {"control": 0.3, "parkinson": 0.7}.
        """
        self.dataframe = dataframe
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        self.oversample_option = oversample_option
        self.k = k
        self.class_weights = class_weights if class_weights is not None else {"control": 0.5, "parkinson": 0.5}  # Default weights

        # Define label mappings
        self.label_mapping = {"control": 0, "parkinson": 1, "unknown": 2}

        # Define normalization for image data (typical for pre-trained models)
        self.normalize = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])

        # Define data augmentation for training data
        if self.is_train:
            self.augmentation = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation(degrees=360),
                transforms.ToTensor(),  # Convert PIL image to tensor
                self.normalize         # Apply normalization
            ])
        else:
            # For validation/test data, only apply normalization
            self.augmentation = transforms.Compose([
                transforms.ToTensor(),  # Convert PIL image to tensor
                self.normalize          # Apply normalization
            ])

        # Group indices by group_label
        self.group_indices = {0: [], 1: [], 2: [], 3: []}
        for idx in range(len(self.dataframe)):
            doctor_label, real_label, group_label = self._get_labels_and_group(idx)
            self.group_indices[group_label].append(idx)

         # Only oversample if this is training data
        if self.is_train:
            self.oversampled_indices = self._oversample_indices()
        else:
            # For validation/test, use original indices in order
            self.oversampled_indices = list(range(len(self.dataframe)))
            
    def _get_labels_and_group(self, idx):
        """
        Helper function to get labels and group label for a given index.
        """
        doctor_label_str = self.dataframe.iloc[idx, 1]  # Assuming the second column is doctor_label
        real_label_str = self.dataframe.iloc[idx, 2]    # Assuming the third column is real_label

        # Convert string labels to integers using the mapping
        doctor_label = self.label_mapping.get(doctor_label_str, -1)  # Default to -1 if label is invalid
        real_label = self.label_mapping.get(real_label_str, -1)      # Default to -1 if label is invalid

        # Validate labels
        if doctor_label == -1 or real_label == -1:
            raise ValueError(f"Invalid label found: doctor_label={doctor_label_str}, real_label={real_label_str}")

        # Determine the group label
        if doctor_label == 0:  # control
            group_label = 0  # Group 1: control
        elif doctor_label == 1:  # parkinson
            group_label = 1  # Group 2: parkinson
        elif doctor_label == 2 and real_label == 0:  # unknown + control
            group_label = 2  # Group 3: unknown_control
        elif doctor_label == 2 and real_label == 1:  # unknown + parkinson
            group_label = 3  # Group 4: unknown_parkinson
        else:
            raise ValueError("Invalid label combination")

        return doctor_label, real_label, group_label

    def _oversample_indices(self):
        """
        Oversample indices for each group to balance the dataset.
        """
        # Determine target lengths for each pair
        pair1_lengths = [len(self.group_indices[0]), len(self.group_indices[1])]
        pair2_lengths = [len(self.group_indices[2]), len(self.group_indices[3])]

        target_pair1 = self.k if self.k is not None else max(pair1_lengths)
        target_pair2 = self.k if self.k is not None else max(pair2_lengths)

        # Oversample indices for each group
        oversampled_indices = {}
        for group_label in self.group_indices:
            if group_label in [0, 1]:  # Group 1 (Control) and Group 2 (Parkinson)
                target_length = target_pair1
            else:  # Group 3 (Unknown_Control) and Group 4 (Unknown_Parkinson)
                target_length = target_pair2

            indices = self.group_indices[group_label]
            oversampled_indices[group_label] = self._oversample_group(indices, target_length, group_label)

        # Combine all oversampled indices into a single list
        combined_indices = []
        for group_label in oversampled_indices:
            combined_indices.extend(oversampled_indices[group_label])

        # Shuffle the combined indices
        random.shuffle(combined_indices)

        return combined_indices

    def _oversample_group(self, indices, target_length, group_label):
        """
        Oversample a group's indices to the target length.
        """
        if self.oversample_option == 1:
            # Equal and balanced sampling with weights for Group 1 and Group 2
            return self._oversample_equal_with_weights(indices, target_length, group_label)
        elif self.oversample_option == 2:
            # Weighted sampling
            return self._oversample_weighted(indices, target_length)
        else:
            raise ValueError(f"Invalid oversample_option: {self.oversample_option}")

    def _oversample_equal_with_weights(self, indices, target_length, group_label):
        """
        Oversample indices to the target length by repeating samples, considering weights for Group 1 and Group 2.
        """
        if len(indices) == 0:
            return []

        # Apply weights only to Group 1 (Control) and Group 2 (Parkinson)
        if group_label in [0, 1]:  # Group 1 or Group 2
            if group_label == 0:  # Group 1 (Control)
                weight = self.class_weights.get("control", 1.0)
            else:  # Group 2 (Parkinson)
                weight = self.class_weights.get("parkinson", 1.0)
            adjusted_target_length =target_length #int(target_length * weight)
        else:  # Group 3 or Group 4 (Unknown classes)
            adjusted_target_length = target_length  # No weighting

        return list(np.random.choice(indices, size=adjusted_target_length, replace=True))

    def _oversample_weighted(self, indices, target_length):
        """
        Oversample indices to the target length with weighted sampling.
        """
        if len(indices) == 0:
            return []
        weights = np.ones(len(indices)) / len(indices)  # Higher weight for shorter arrays
        return list(np.random.choice(indices, size=target_length, replace=True, p=weights))

    def __len__(self):
        return len(self.oversampled_indices)

    def __getitem__(self, idx):
        # Get the oversampled index
        oversampled_idx = self.oversampled_indices[idx]

        # Get image path and labels from the DataFrame
        img_path = os.path.join(self.data_dir, self.dataframe.iloc[oversampled_idx, 0])  # Assuming the first column is the image path
        doctor_label, real_label, group_label = self._get_labels_and_group(oversampled_idx)

        # Load the image
        image = Image.open(img_path).convert('RGB')  # Ensure image is in RGB format

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        else:
            image = self.augmentation(image)

        # Return image, doctor_label, real_label, and group_label
        return image, doctor_label, real_label, group_label