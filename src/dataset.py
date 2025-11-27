"""
Dataset utilities for CIFAR-10 with configurable transforms.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from .transforms import get_train_transform, get_eval_transform


def get_cifar10_loaders(
    mode='rgb',
    batch_size=128,
    num_workers=4,
    data_dir='./data'
):
    """
    Get CIFAR-10 train and test dataloaders.
    
    Args:
        mode: 'rgb' for V-JEPA, 'edge' for A-JEPA
        batch_size: Batch size for dataloaders
        num_workers: Number of dataloader workers
        data_dir: Directory to store/load CIFAR-10 data
    
    Returns:
        train_loader, test_loader
    """
    train_transform = get_train_transform(mode)
    test_transform = get_eval_transform(mode, perturb=False)
    
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def get_perturbed_test_loader(
    mode='rgb',
    batch_size=128,
    num_workers=4,
    data_dir='./data'
):
    """
    Get perturbed CIFAR-10 test dataloader for robustness evaluation.
    
    Args:
        mode: 'rgb' for V-JEPA, 'edge' for A-JEPA
        batch_size: Batch size for dataloader
        num_workers: Number of dataloader workers
        data_dir: Directory to store/load CIFAR-10 data
    
    Returns:
        perturbed_test_loader
    """
    perturbed_transform = get_eval_transform(mode, perturb=True)
    
    perturbed_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=perturbed_transform
    )
    
    perturbed_loader = DataLoader(
        perturbed_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return perturbed_loader

