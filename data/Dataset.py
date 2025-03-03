import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ParkinsonDataset(Dataset):
    def __init__(self, dataframe, data_dir, transform=None, is_train=True):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing image file paths, doctor_label, and real_label.
            data_dir (str): Root directory where images are stored.
            transform (callable, optional): Transform to be applied to the images.
            is_train (bool): Whether the dataset is for training (default: True).
        """
        self.dataframe = dataframe
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train

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

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get image path and labels from the DataFrame
        img_path = os.path.join(self.data_dir, self.dataframe.iloc[idx, 0])  # Assuming the first column is the image path
        doctor_label_str = self.dataframe.iloc[idx, 1]  # Assuming the second column is doctor_label
        real_label_str = self.dataframe.iloc[idx, 2]    # Assuming the third column is real_label

        # Convert string labels to integers using the mapping
        doctor_label = self.label_mapping.get(doctor_label_str, -1)  # Default to -1 if label is invalid
        real_label = self.label_mapping.get(real_label_str, -1)      # Default to -1 if label is invalid

        # Validate labels
        if doctor_label == -1 or real_label == -1:
            raise ValueError(f"Invalid label found: doctor_label={doctor_label_str}, real_label={real_label_str}")

        # Load the image
        image = Image.open(img_path).convert('RGB')  # Ensure image is in RGB format

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        else:
            image = self.augmentation(image)

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

        # Return image, doctor_label, real_label, and group_label
        return image, doctor_label, real_label, group_label