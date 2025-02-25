import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CustomDataset(Dataset):
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

        # Define normalization for image data (typical for pre-trained models)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # Mean for ImageNet
            std=[0.229, 0.224, 0.225]    # Std for ImageNet
        )

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
        doctor_label = self.dataframe.iloc[idx, 1]  # Assuming the second column is doctor_label
        real_label = self.dataframe.iloc[idx, 2]    # Assuming the third column is real_label

        # Load the image
        image = Image.open(img_path).convert('RGB')  # Ensure image is in RGB format

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        else:
            image = self.augmentation(image)

        # Determine the group label
        if doctor_label == "control":
            group_label = 0  # Group 1: control
        elif doctor_label == "parkinson":
            group_label = 1  # Group 2: parkinson
        elif doctor_label == "unknown" and real_label == "control":
            group_label = 2  # Group 3: unknown_control
        elif doctor_label == "unknown" and real_label == "parkinson":
            group_label = 3  # Group 4: unknown_parkinson
        else:
            raise ValueError("Invalid label combination")

        # Return image, doctor_label, real_label, and group_label
        return image, doctor_label, real_label, group_label