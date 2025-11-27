"""
Custom transforms for V-JEPA and A-JEPA experiments.
"""

import cv2
import numpy as np
import torch
from torchvision import transforms


class CannyEdge:
    """Convert RGB image to Canny edge map."""
    
    def __init__(self, low_threshold=50, high_threshold=150):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
    
    def __call__(self, img):
        # img is a PIL Image or tensor
        if isinstance(img, torch.Tensor):
            # Convert tensor to numpy (C, H, W) -> (H, W, C)
            img_np = img.permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype(np.uint8)
        else:
            # PIL Image
            img_np = np.array(img)
        
        # Convert to grayscale if RGB
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)
        
        # Normalize to [0, 1] and add channel dimension
        edges = edges.astype(np.float32) / 255.0
        return torch.from_numpy(edges).unsqueeze(0)  # (1, H, W)


class TexturePerturbation:
    """Strong texture perturbation for robustness evaluation."""
    
    def __init__(self, noise_std=0.1):
        self.color_jitter = transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.2
        )
        self.noise_std = noise_std
    
    def __call__(self, img):
        # Apply color jitter
        img = self.color_jitter(img)
        
        # Convert to tensor if not already
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        
        # Add Gaussian noise
        noise = torch.randn_like(img) * self.noise_std
        img = torch.clamp(img + noise, 0, 1)
        
        return img


class RandomMask:
    """Apply random masking to input tensor for JEPA context encoder."""
    
    def __init__(self, mask_ratio=0.5, patch_size=4):
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
    
    def __call__(self, x):
        """
        Args:
            x: Tensor of shape (C, H, W)
        Returns:
            Masked tensor with same shape
        """
        C, H, W = x.shape
        
        # Calculate number of patches
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        num_patches = num_patches_h * num_patches_w
        
        # Number of patches to mask
        num_masked = int(num_patches * self.mask_ratio)
        
        # Generate mask indices
        indices = torch.randperm(num_patches)[:num_masked]
        
        # Create masked version
        masked = x.clone()
        for idx in indices:
            ph = (idx // num_patches_w) * self.patch_size
            pw = (idx % num_patches_w) * self.patch_size
            masked[:, ph:ph+self.patch_size, pw:pw+self.patch_size] = 0
        
        return masked


def get_train_transform(mode='rgb'):
    """Get training transforms for specified mode."""
    base_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    
    if mode == 'rgb':
        return transforms.Compose(base_transforms + [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    elif mode == 'edge':
        return transforms.Compose(base_transforms + [
            CannyEdge(),
        ])
    else:
        raise ValueError(f"Unknown mode: {mode}")


def get_eval_transform(mode='rgb', perturb=False):
    """Get evaluation transforms for specified mode."""
    if mode == 'rgb':
        if perturb:
            return transforms.Compose([
                TexturePerturbation(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])
    elif mode == 'edge':
        if perturb:
            # For edge mode with perturbation, first perturb RGB then convert to edges
            return transforms.Compose([
                TexturePerturbation(),
                CannyEdgeFromTensor(),
            ])
        else:
            return transforms.Compose([
                CannyEdge(),
            ])
    else:
        raise ValueError(f"Unknown mode: {mode}")


class CannyEdgeFromTensor:
    """Convert tensor to Canny edge map (for perturbed inputs)."""
    
    def __init__(self, low_threshold=50, high_threshold=150):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
    
    def __call__(self, img):
        # img is a tensor (C, H, W) in [0, 1]
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)
        
        # Normalize to [0, 1] and add channel dimension
        edges = edges.astype(np.float32) / 255.0
        return torch.from_numpy(edges).unsqueeze(0)


# =============================================================================
# CIFAR-10-C Style Corruptions for Robustness Testing
# =============================================================================

class GaussianBlur:
    """Apply Gaussian blur corruption."""
    
    def __init__(self, kernel_size=5, sigma=1.0):
        self.kernel_size = kernel_size
        self.sigma = sigma
    
    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        
        blurred = cv2.GaussianBlur(img_np, (self.kernel_size, self.kernel_size), self.sigma)
        
        blurred = blurred.astype(np.float32) / 255.0
        return torch.from_numpy(blurred).permute(2, 0, 1)


class GaussianNoise:
    """Add Gaussian noise corruption."""
    
    def __init__(self, std=0.1):
        self.std = std
    
    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        
        noise = torch.randn_like(img) * self.std
        return torch.clamp(img + noise, 0, 1)


class ContrastChange:
    """Change image contrast."""
    
    def __init__(self, factor=0.5):
        self.factor = factor
    
    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        
        mean = img.mean()
        return torch.clamp((img - mean) * self.factor + mean, 0, 1)


class BrightnessChange:
    """Change image brightness."""
    
    def __init__(self, delta=0.3):
        self.delta = delta
    
    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)
        
        return torch.clamp(img + self.delta, 0, 1)


CORRUPTIONS = {
    'blur': GaussianBlur(kernel_size=5, sigma=2.0),
    'noise': GaussianNoise(std=0.15),
    'contrast': ContrastChange(factor=0.4),
    'brightness': BrightnessChange(delta=0.3),
}


def get_corruption_transform(corruption_name, mode='rgb'):
    """
    Get transform with a specific corruption applied.
    
    Args:
        corruption_name: One of 'blur', 'noise', 'contrast', 'brightness'
        mode: 'rgb' or 'edge'
    
    Returns:
        Composed transform
    """
    if corruption_name not in CORRUPTIONS:
        raise ValueError(f"Unknown corruption: {corruption_name}. Choose from {list(CORRUPTIONS.keys())}")
    
    corruption = CORRUPTIONS[corruption_name]
    
    if mode == 'rgb':
        return transforms.Compose([
            corruption,
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    elif mode == 'edge':
        return transforms.Compose([
            corruption,
            CannyEdgeFromTensor(),
        ])
    else:
        raise ValueError(f"Unknown mode: {mode}")

