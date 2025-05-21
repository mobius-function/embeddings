# src/dataset.py
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def get_transforms(img_size=128, is_training=True):
    """
    Get image transformations.

    Args:
        img_size (int): Size of the image
        is_training (bool): Whether the transformations are for training

    Returns:
        transforms.Compose: Composed transformations
    """
    if is_training:
        # Training data transformations with augmentation
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])
    else:
        # Validation/test data transformations without augmentation
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])

    return transform


class FaceDataset(Dataset):
    """
    Dataset for loading face images.
    """

    def __init__(self, image_dir, transform=None):
        """
        Initialize the dataset.

        Args:
            image_dir (str): Directory containing face images
            transform (callable, optional): Optional transform to be applied to images
        """
        self.image_dir = image_dir
        self.transform = transform

        # Get all image files
        self.image_paths = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, file))

        print(f"Found {len(self.image_paths)} images in {image_dir}")

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Args:
            idx (int): Index of the item

        Returns:
            torch.Tensor: Image tensor
        """
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image