"""
Compositional Generalization Test: CLEVR-style Scenes

Tests whether A-JEPA generalizes better to novel shape-color combinations.

Setup:
- Train on certain shape-color combinations (e.g., red circles, green squares)
- Test on novel combinations (e.g., green circles, red squares)

Hypothesis: A-JEPA, which ignores color/texture and focuses on shape, should
generalize better to novel combinations since it's not memorizing specific
visual appearances.
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.datasets.clevr_simple import (
    CLEVRSimpleDataset,
    TRAIN_COMBINATIONS,
    TEST_NOVEL_COMBINATIONS,
)
from src.models import SmallEncoder, Predictor, RelationalBlock
from torch.utils.data import DataLoader


class TemporalEncoderCLEVR(nn.Module):
    """Temporal encoder for CLEVR-style videos."""
    
    def __init__(self, in_channels: int, emb_dim: int, width_mult: float, use_relational: bool):
        super().__init__()
        self.encoder = SmallEncoder(
            in_channels=in_channels,
            emb_dim=emb_dim,
            width_mult=width_mult,
            use_relational=use_relational,
        )
    
    def encode_single(self, frame):
        """Encode a single frame. frame: (B, C, H, W)"""
        return self.encoder(frame)
    
    def encode_context(self, context):
        """Encode context frames and aggregate. context: (B, T, C, H, W)"""
        B, T, C, H, W = context.shape
        frames_flat = context.view(B * T, C, H, W)
        embeddings = self.encoder(frames_flat)
        D = embeddings.shape[-1]
        embeddings = embeddings.view(B, T, D)
        return embeddings.mean(dim=1)


def get_clevr_models(variant: str):
    """Create encoder and predictor for CLEVR task."""
    if variant == "v_jepa":
        # V-JEPA: grayscale input (average RGB), larger model
        encoder = TemporalEncoderCLEVR(
            in_channels=1,  # Grayscale
            emb_dim=256,
            width_mult=1.0,
            use_relational=False,
        )
        emb_dim = 256
    elif variant == "a_jepa":
        # A-JEPA: edge input, smaller model with relational reasoning
        encoder = TemporalEncoderCLEVR(
            in_channels=1,  # Edge
            emb_dim=128,
            width_mult=0.5,
            use_relational=True,
        )
        emb_dim = 128
    else:
        raise ValueError(f"Unknown variant: {variant}")
    
    predictor = Predictor(emb_dim=emb_dim, hidden_dim=emb_dim * 2)
    return encoder, predictor, emb_dim


def cosine_similarity(pred, target):
    """Compute cosine similarity."""
    pred = nn.functional.normalize(pred, dim=-1)
    target = nn.functional.normalize(target, dim=-1)
    return torch.sum(pred * target, dim=-1)


def cosine_similarity_loss(pred, target):
    """Negative cosine similarity loss."""
    return -torch.mean(cosine_similarity(pred, target))


def create_loader(mode, num_samples, combinations, batch_size, seed):
    """Create a dataloader with specific combinations."""
    dataset = CLEVRSimpleDataset(
        num_samples=num_samples,
        num_objects=2,
        mode=mode,
        combinations=combinations,
        include_motion=True,
        num_frames=20,
        context_length=5,
        max_horizon=5,
        seed=seed,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)


def train_epoch(encoder, predictor, dataloader, optimizer, device):
    """Train for one epoch."""
    encoder.train()
    predictor.train()
    
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc='Training', leave=False):
        context = batch['context'].to(device)
        target = batch['target'].to(device)
        
        context_emb = encoder.encode_context(context)
        
        with torch.no_grad():
            target_emb = encoder.encode_single(target)
        
        pred_emb = predictor(context_emb)
        loss = cosine_similarity_loss(pred_emb, target_emb)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(encoder, predictor, dataloader, device):
    """Evaluate and return mean cosine similarity."""
    encoder.eval()
    predictor.eval()
    
    all_sims = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', leave=False):
            context = batch['context'].to(device)
            target = batch['target'].to(device)
            
            context_emb = encoder.encode_context(context)
            target_emb = encoder.encode_single(target)
            pred_emb = predictor(context_emb)
            
            sims = cosine_similarity(pred_emb, target_emb).cpu().numpy()
            all_sims.extend(sims)
    
    return np.mean(all_sims)


def run_compositional_experiment(variant: str, device, args):
    """Run compositional generalization experiment for one variant."""
    print(f"\n{'='*60}")
    print(f"Running {variant.upper()}")
    print(f"{'='*60}")
    
    # Determine input mode
    mode = 'grayscale' if variant == 'v_jepa' else 'edge'
    
    # Create models
    encoder, predictor, emb_dim = get_clevr_models(variant)
    encoder = encoder.to(device)
    predictor = predictor.to(device)
    
    total_params = sum(p.numel() for p in encoder.parameters())
    total_params += sum(p.numel() for p in predictor.parameters())
    print(f"Total params: {total_params:,}")
    
    # Create dataloaders
    print(f"\nCreating datasets...")
    print(f"  Train combinations: {len(TRAIN_COMBINATIONS)}")
    print(f"  Novel combinations: {len(TEST_NOVEL_COMBINATIONS)}")
    
    train_loader = create_loader(mode, args.num_train, TRAIN_COMBINATIONS, args.batch_size, 42)
    test_id_loader = create_loader(mode, args.num_test, TRAIN_COMBINATIONS, args.batch_size, 123)
    test_ood_loader = create_loader(mode, args.num_test, TEST_NOVEL_COMBINATIONS, args.batch_size, 456)
    
    # Optimizer
    params = list(encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    
    # Training
    print(f"\nTraining on seen combinations for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        loss = train_epoch(encoder, predictor, train_loader, optimizer, device)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {loss:.4f}")
    
    # Evaluation
    print("\nEvaluating on seen combinations (ID)...")
    id_score = evaluate(encoder, predictor, test_id_loader, device)
    
    print("Evaluating on novel combinations (OOD)...")
    ood_score = evaluate(encoder, predictor, test_ood_loader, device)
    
    gen_gap = id_score - ood_score
    
    return {
        'variant': variant,
        'params': total_params,
        'id_score': id_score,
        'ood_score': ood_score,
        'gen_gap': gen_gap,
    }


def print_comparison(v_results: dict, a_results: dict):
    """Print side-by-side comparison."""
    print("\n")
    print("=" * 75)
    print("      Compositional Generalization: V-JEPA vs A-JEPA")
    print("=" * 75)
    
    print(f"\n{'Metric':<40} {'V-JEPA':>15} {'A-JEPA':>15}")
    print("-" * 75)
    
    print(f"{'Parameters':<40} {v_results['params']:>15,} {a_results['params']:>15,}")
    
    print("-" * 75)
    print("Cosine Similarity (higher = better):")
    
    print(f"{'  Seen Combinations (ID)':<40} {v_results['id_score']:>15.4f} {a_results['id_score']:>15.4f}")
    print(f"{'  Novel Combinations (OOD)':<40} {v_results['ood_score']:>15.4f} {a_results['ood_score']:>15.4f}")
    
    print("-" * 75)
    print(f"{'Generalization Gap (lower = better)':<40} {v_results['gen_gap']:>15.4f} {a_results['gen_gap']:>15.4f}")
    
    print("=" * 75)
    
    # Key findings
    print("\nKey Findings:")
    
    param_ratio = v_results['params'] / a_results['params']
    print(f"  - V-JEPA has {param_ratio:.1f}x more parameters than A-JEPA")
    
    # Generalization comparison
    if a_results['gen_gap'] < v_results['gen_gap']:
        improvement = ((v_results['gen_gap'] - a_results['gen_gap']) / max(v_results['gen_gap'], 0.001)) * 100
        print(f"  - A-JEPA generalizes {abs(improvement):.1f}% better to novel combinations")
    else:
        print(f"  - V-JEPA generalizes better to novel combinations")
    
    # OOD performance
    if a_results['ood_score'] > v_results['ood_score']:
        print(f"  - A-JEPA has higher OOD performance ({a_results['ood_score']:.4f} vs {v_results['ood_score']:.4f})")
    else:
        print(f"  - V-JEPA has higher OOD performance ({v_results['ood_score']:.4f} vs {a_results['ood_score']:.4f})")
    
    # Verdict
    print("\n" + "=" * 75)
    if a_results['gen_gap'] < v_results['gen_gap'] and a_results['ood_score'] > v_results['ood_score']:
        print("VERDICT: A-JEPA demonstrates superior compositional generalization!")
        print("         Shape-focused representations transfer better to novel color combos.")
    elif a_results['gen_gap'] < v_results['gen_gap']:
        print("VERDICT: A-JEPA has smaller generalization gap (better relative transfer).")
    elif a_results['ood_score'] > v_results['ood_score']:
        print("VERDICT: A-JEPA has better absolute OOD performance.")
    else:
        print("VERDICT: V-JEPA performs better on compositional generalization.")
    print("=" * 75)


def parse_args():
    parser = argparse.ArgumentParser(description='Compositional Generalization Test')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_train', type=int, default=500, help='Training videos')
    parser.add_argument('--num_test', type=int, default=100, help='Test videos per split')
    parser.add_argument('--num_workers', type=int, default=2, help='Dataloader workers')
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"\nCompositional Generalization Test (CLEVR-style)")
    print(f"  Train: {len(TRAIN_COMBINATIONS)} shape-color combinations")
    print(f"  Test OOD: {len(TEST_NOVEL_COMBINATIONS)} novel combinations")
    print(f"  Epochs: {args.epochs}")
    
    # Run both variants
    v_results = run_compositional_experiment('v_jepa', device, args)
    a_results = run_compositional_experiment('a_jepa', device, args)
    
    # Print comparison
    print_comparison(v_results, a_results)


if __name__ == '__main__':
    main()

