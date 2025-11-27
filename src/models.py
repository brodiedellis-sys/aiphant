"""
Model architectures for V-JEPA and A-JEPA.
"""

import torch
import torch.nn as nn


class RelationalBlock(nn.Module):
    """
    Lightweight relational reasoning block for abstract processing.
    
    Aggregates global context (mean of features across batch) and mixes it
    back into individual features via a small residual MLP.
    
    Args:
        dim: Feature dimension
        hidden_dim: Hidden layer size (default: 16 to keep overhead <5%)
    """
    
    def __init__(self, dim: int, hidden_dim: int = 16):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D) where B is batch size, D is feature dim
        # Mix individual features with global context
        global_context = x.mean(dim=0, keepdim=True)  # (1, D)
        mixed = x + global_context  # (B, D) - each sample gets global info
        
        # Apply small MLP and add residual
        out = x + self.mlp(mixed)
        return out


class SmallEncoder(nn.Module):
    """
    Lightweight CNN encoder for CIFAR-10 (32x32 images) with width scaling.
    
    Args:
        in_channels: 3 for RGB (V-JEPA), 1 for Edge/Depth/Segmentation (A-JEPA)
        emb_dim: Output embedding dimension (512 for V-JEPA, 128 for A-JEPA)
        width_mult: Channel width multiplier (1.0 for V-JEPA, 0.5 for A-JEPA)
        use_relational: Whether to apply RelationalBlock (for A-JEPA abstract reasoning)
    """
    
    def __init__(
        self, 
        in_channels: int = 3, 
        emb_dim: int = 512, 
        width_mult: float = 1.0,
        use_relational: bool = False
    ):
        super().__init__()
        c1 = int(64 * width_mult)
        c2 = int(128 * width_mult)
        c3 = int(256 * width_mult)
        c4 = int(256 * width_mult)
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, stride=2, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c4, 3, stride=2, padding=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )
        
        # Feature dim after conv: c4 * 2 * 2 (assuming 32x32 input -> 2x2 after 4 stride-2 convs)
        self.flat_dim = c4 * 2 * 2
        
        # Optional relational reasoning block (for A-JEPA)
        self.use_relational = use_relational
        if use_relational:
            self.relational = RelationalBlock(self.flat_dim, hidden_dim=16)
        
        self.fc = nn.Linear(self.flat_dim, emb_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        
        # Apply relational reasoning if enabled (A-JEPA path)
        if self.use_relational:
            h = self.relational(h)
        
        z = self.fc(h)
        # L2 normalize embeddings
        z = z / (z.norm(dim=1, keepdim=True) + 1e-6)
        return z


class Predictor(nn.Module):
    """
    MLP predictor for JEPA.
    Maps context embedding to target embedding space.
    
    Args:
        emb_dim: Input and output embedding dimension
        hidden_dim: Hidden layer dimension (default: 4x emb_dim)
    """
    
    def __init__(self, emb_dim: int, hidden_dim: int | None = None):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = emb_dim * 4  # e.g. 2048 for 512, 512 for 128
        
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, emb_dim),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def get_jepa_models(variant: str):
    """
    Create encoder and predictor for specified JEPA variant.
    
    Args:
        variant: 'v_jepa' (RGB, 512-dim, full width) or 'a_jepa' (abstract input, 128-dim, relational reasoning)
    
    Returns:
        encoder, predictor
    
    Notes:
        - V-JEPA: Standard visual processing with RGB input
        - A-JEPA: Abstract processing with single-channel input (edges/depth/segmentation)
                  plus relational reasoning block for global context aggregation
    """
    if variant == "v_jepa":
        # Visual JEPA: RGB input, larger model, no relational reasoning
        in_channels = 3
        emb_dim = 512
        width_mult = 1.0
        use_relational = False
    elif variant == "a_jepa":
        # Aphantasic JEPA: Single-channel abstract input, smaller model, relational reasoning
        in_channels = 1
        emb_dim = 128
        width_mult = 0.5
        use_relational = True
    else:
        raise ValueError(f"Unknown variant: {variant}")
    
    encoder = SmallEncoder(
        in_channels=in_channels,
        emb_dim=emb_dim,
        width_mult=width_mult,
        use_relational=use_relational,
    )
    predictor = Predictor(emb_dim=emb_dim)
    
    return encoder, predictor


class LinearProbe(nn.Module):
    """Linear classifier for evaluation."""
    
    def __init__(self, emb_dim, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(emb_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)
