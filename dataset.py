"""
Dataset Module - GPU Optimized Data Loading
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from PIL import Image
import random


class SSLDataset(Dataset):
    """Dataset for self-supervised learning - returns two augmented views."""
    
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, _ = self.base_dataset[idx]
        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1, view2


def get_transforms(config, mode='train'):
    """Get data transformations."""
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    elif mode == 'test':
        return transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:  # SSL
        return transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomResizedCrop(config.image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def get_balanced_subset(dataset, percentage, seed=42):
    """Create balanced subset with specified percentage per class."""
    random.seed(seed)
    np.random.seed(seed)
    
    targets = np.array(dataset.targets)
    indices = []
    
    for class_idx in range(len(dataset.classes)):
        class_indices = np.where(targets == class_idx)[0]
        n_samples = max(1, int(len(class_indices) * percentage))
        selected = np.random.choice(class_indices, size=n_samples, replace=False)
        indices.extend(selected.tolist())
    
    random.shuffle(indices)
    return Subset(dataset, indices)


def get_dataloaders(config, label_pct=1.0):
    """Get all data loaders."""
    # Training loader
    train_dataset = datasets.ImageFolder(config.train_dir, get_transforms(config, 'train'))
    if label_pct < 1.0:
        train_dataset = get_balanced_subset(train_dataset, label_pct, config.seed)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.supervised_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True
    )
    
    # Test loader
    test_dataset = datasets.ImageFolder(config.test_dir, get_transforms(config, 'test'))
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.supervised_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    # SSL loader
    ssl_base = datasets.ImageFolder(config.train_dir, transform=None)
    ssl_dataset = SSLDataset(ssl_base, get_transforms(config, 'ssl'))
    ssl_loader = DataLoader(
        ssl_dataset,
        batch_size=config.ssl_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True
    )
    
    return train_loader, test_loader, ssl_loader
