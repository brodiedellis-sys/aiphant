"""
Temporal Prediction Task: Test abstract state modeling with bouncing balls.

This script trains V-JEPA and A-JEPA to predict future frame embeddings
from context frames, measuring prediction accuracy at different time horizons.
"""

import argparse
import os
import sys
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.datasets.bouncing_balls import get_bouncing_balls_loaders
from src.models import SmallEncoder, Predictor, RelationalBlock


class TemporalEncoder(nn.Module):
    """
    Encoder that processes multiple context frames and aggregates them.
    
    Takes N context frames, encodes each, and pools into a single context embedding.
    """
    
    def __init__(self, base_encoder: SmallEncoder, pool_type: str = 'mean'):
        super().__init__()
        self.encoder = base_encoder
        self.pool_type = pool_type
    
    def encode_single(self, frame: torch.Tensor) -> torch.Tensor:
        """Encode a single frame. frame: (B, 1, H, W)"""
        return self.encoder(frame)
    
    def encode_context(self, context: torch.Tensor) -> torch.Tensor:
        """
        Encode context frames and aggregate.
        
        Args:
            context: (B, T, 1, H, W) - batch of T context frames
        
        Returns:
            context_embedding: (B, D) - aggregated context
        """
        B, T, C, H, W = context.shape
        
        # Flatten batch and time: (B*T, C, H, W)
        frames_flat = context.view(B * T, C, H, W)
        
        # Encode all frames
        embeddings = self.encoder(frames_flat)  # (B*T, D)
        
        # Reshape: (B, T, D)
        D = embeddings.shape[-1]
        embeddings = embeddings.view(B, T, D)
        
        # Pool across time
        if self.pool_type == 'mean':
            context_emb = embeddings.mean(dim=1)  # (B, D)
        elif self.pool_type == 'last':
            context_emb = embeddings[:, -1, :]    # (B, D)
        else:
            raise ValueError(f"Unknown pool type: {self.pool_type}")
        
        return context_emb


def get_temporal_models(variant: str):
    """
    Create encoder and predictor for temporal prediction task.
    
    Args:
        variant: 'v_jepa' or 'a_jepa'
    
    Returns:
        temporal_encoder, predictor, emb_dim
    """
    # Note: For bouncing balls, both variants use 1-channel input (grayscale)
    # The difference is:
    # - V-JEPA: raw grayscale, larger model
    # - A-JEPA: edge-detected, smaller model with relational reasoning
    
    if variant == "v_jepa":
        base_encoder = SmallEncoder(
            in_channels=1,  # Grayscale
            emb_dim=256,    # Smaller than CIFAR version for efficiency
            width_mult=1.0,
            use_relational=False,
        )
        emb_dim = 256
    elif variant == "a_jepa":
        base_encoder = SmallEncoder(
            in_channels=1,  # Edge-detected
            emb_dim=128,
            width_mult=0.5,
            use_relational=True,
        )
        emb_dim = 128
    else:
        raise ValueError(f"Unknown variant: {variant}")
    
    temporal_encoder = TemporalEncoder(base_encoder, pool_type='mean')
    predictor = Predictor(emb_dim=emb_dim, hidden_dim=emb_dim * 2)
    
    return temporal_encoder, predictor, emb_dim


def cosine_similarity_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative cosine similarity loss."""
    pred = nn.functional.normalize(pred, dim=-1)
    target = nn.functional.normalize(target, dim=-1)
    return -torch.mean(torch.sum(pred * target, dim=-1))


def cosine_similarity(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cosine similarity (higher = better, range [-1, 1])."""
    pred = nn.functional.normalize(pred, dim=-1)
    target = nn.functional.normalize(target, dim=-1)
    return torch.sum(pred * target, dim=-1)


def train_epoch(
    temporal_encoder: TemporalEncoder,
    predictor: Predictor,
    dataloader,
    optimizer,
    device,
):
    """Train for one epoch on temporal prediction."""
    temporal_encoder.train()
    predictor.train()
    
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc='Training', leave=False):
        context = batch['context'].to(device)  # (B, T, 1, H, W)
        target = batch['target'].to(device)    # (B, 1, H, W)
        
        # Encode context frames
        context_emb = temporal_encoder.encode_context(context)  # (B, D)
        
        # Encode target frame (stop gradient - this is the target)
        with torch.no_grad():
            target_emb = temporal_encoder.encode_single(target)  # (B, D)
        
        # Predict target embedding from context
        pred_emb = predictor(context_emb)  # (B, D)
        
        # Compute loss
        loss = cosine_similarity_loss(pred_emb, target_emb)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate_by_horizon(
    temporal_encoder: TemporalEncoder,
    predictor: Predictor,
    dataloader,
    device,
    horizons: list = [1, 3, 5, 10],
):
    """
    Evaluate prediction error at different time horizons.
    
    Returns:
        dict mapping horizon -> mean cosine similarity
    """
    temporal_encoder.eval()
    predictor.eval()
    
    # Collect similarities by horizon
    horizon_similarities = defaultdict(list)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', leave=False):
            context = batch['context'].to(device)
            target = batch['target'].to(device)
            batch_horizons = batch['horizon'].numpy()
            
            # Encode and predict
            context_emb = temporal_encoder.encode_context(context)
            target_emb = temporal_encoder.encode_single(target)
            pred_emb = predictor(context_emb)
            
            # Compute similarities
            sims = cosine_similarity(pred_emb, target_emb).cpu().numpy()
            
            # Group by horizon
            for sim, h in zip(sims, batch_horizons):
                if h in horizons:
                    horizon_similarities[h].append(sim)
    
    # Compute mean similarity per horizon
    results = {}
    for h in horizons:
        if h in horizon_similarities:
            results[h] = np.mean(horizon_similarities[h])
        else:
            results[h] = None
    
    return results


def compute_drift(similarities_by_horizon: dict) -> float:
    """
    Compute drift as the degradation rate of similarity over horizon.
    
    Drift = (sim at horizon 1) - (sim at max horizon)
    Lower drift = more stable representations.
    """
    horizons = sorted(similarities_by_horizon.keys())
    if len(horizons) < 2:
        return 0.0
    
    first_sim = similarities_by_horizon[horizons[0]]
    last_sim = similarities_by_horizon[horizons[-1]]
    
    if first_sim is None or last_sim is None:
        return 0.0
    
    return first_sim - last_sim


def run_experiment(variant: str, device, args):
    """Run full training and evaluation for one variant."""
    print(f"\n{'='*60}")
    print(f"Running {variant.upper()}")
    print(f"{'='*60}")
    
    # Determine mode
    mode = 'raw' if variant == 'v_jepa' else 'edge'
    
    # Get data
    train_loader, test_loader = get_bouncing_balls_loaders(
        mode=mode,
        num_train=args.num_train,
        num_test=args.num_test,
        context_length=args.context_length,
        max_horizon=args.max_horizon,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    # Create models
    temporal_encoder, predictor, emb_dim = get_temporal_models(variant)
    temporal_encoder = temporal_encoder.to(device)
    predictor = predictor.to(device)
    
    # Count parameters
    enc_params = sum(p.numel() for p in temporal_encoder.parameters())
    pred_params = sum(p.numel() for p in predictor.parameters())
    print(f"Encoder params: {enc_params:,}")
    print(f"Predictor params: {pred_params:,}")
    print(f"Total params: {enc_params + pred_params:,}")
    
    # Optimizer
    params = list(temporal_encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    
    # Training
    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        loss = train_epoch(temporal_encoder, predictor, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {loss:.4f}")
    
    # Evaluation
    print("\nEvaluating by horizon...")
    horizons = [1, 3, 5, 10]
    results = evaluate_by_horizon(
        temporal_encoder, predictor, test_loader, device, horizons
    )
    
    # Compute drift
    drift = compute_drift(results)
    
    return {
        'variant': variant,
        'params': enc_params + pred_params,
        'similarities': results,
        'drift': drift,
    }


def print_comparison(v_results: dict, a_results: dict):
    """Print side-by-side comparison of results."""
    print("\n")
    print("=" * 70)
    print("        Temporal Prediction: V-JEPA vs A-JEPA")
    print("=" * 70)
    
    print(f"\n{'Metric':<30} {'V-JEPA':>18} {'A-JEPA':>18}")
    print("-" * 70)
    
    # Parameters
    print(f"{'Parameters':<30} {v_results['params']:>18,} {a_results['params']:>18,}")
    
    print("-" * 70)
    print("Cosine Similarity by Horizon (higher = better):")
    
    # Similarities at each horizon
    horizons = [1, 3, 5, 10]
    for h in horizons:
        v_sim = v_results['similarities'].get(h)
        a_sim = a_results['similarities'].get(h)
        v_str = f"{v_sim:.4f}" if v_sim is not None else "N/A"
        a_str = f"{a_sim:.4f}" if a_sim is not None else "N/A"
        print(f"  Horizon {h:<3} steps{'':<15} {v_str:>18} {a_str:>18}")
    
    print("-" * 70)
    
    # Drift
    print(f"{'Drift (lower = more stable)':<30} {v_results['drift']:>18.4f} {a_results['drift']:>18.4f}")
    
    print("=" * 70)
    
    # Analysis
    print("\nKey Findings:")
    
    # Parameter efficiency
    param_ratio = v_results['params'] / a_results['params']
    print(f"  - V-JEPA has {param_ratio:.1f}x more parameters than A-JEPA")
    
    # Drift comparison
    if a_results['drift'] < v_results['drift']:
        improvement = ((v_results['drift'] - a_results['drift']) / v_results['drift']) * 100
        print(f"  - A-JEPA has {improvement:.1f}% lower drift (more temporally stable)")
    else:
        print(f"  - V-JEPA has lower drift ({v_results['drift']:.4f} vs {a_results['drift']:.4f})")
    
    # Long-horizon performance
    v_long = v_results['similarities'].get(10)
    a_long = a_results['similarities'].get(10)
    if v_long and a_long:
        if a_long > v_long:
            print(f"  - A-JEPA maintains better coherence at horizon 10 ({a_long:.4f} vs {v_long:.4f})")
        else:
            print(f"  - V-JEPA maintains better coherence at horizon 10 ({v_long:.4f} vs {a_long:.4f})")


def parse_args():
    parser = argparse.ArgumentParser(description='Temporal Prediction Task')
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_train', type=int, default=500, help='Training videos')
    parser.add_argument('--num_test', type=int, default=100, help='Test videos')
    parser.add_argument('--context_length', type=int, default=5, help='Context frames')
    parser.add_argument('--max_horizon', type=int, default=10, help='Max prediction horizon')
    parser.add_argument('--num_workers', type=int, default=2, help='Dataloader workers')
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Settings: {args.epochs} epochs, {args.num_train} train videos, horizon up to {args.max_horizon}")
    
    # Run both variants
    v_results = run_experiment('v_jepa', device, args)
    a_results = run_experiment('a_jepa', device, args)
    
    # Print comparison
    print_comparison(v_results, a_results)


if __name__ == '__main__':
    main()

