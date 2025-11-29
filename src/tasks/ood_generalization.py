"""
OOD Generalization Test: Novel Configuration

Tests whether A-JEPA generalizes better than V-JEPA to unseen configurations.
- Train on 2-ball scenarios
- Test on 2-ball (in-distribution) and 3-ball (out-of-distribution)

Hypothesis: A-JEPA learns abstract structure and generalizes better.
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

from src.datasets.bouncing_balls import (
    get_bouncing_balls_loaders,
    get_ood_test_loader,
    BouncingBallsDataset,
)
from src.tasks.predict_future import (
    TemporalEncoder,
    get_temporal_models,
    cosine_similarity_loss,
    cosine_similarity,
)


def train_epoch(temporal_encoder, predictor, dataloader, optimizer, device):
    """Train for one epoch."""
    temporal_encoder.train()
    predictor.train()
    
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc='Training', leave=False):
        context = batch['context'].to(device)
        target = batch['target'].to(device)
        
        context_emb = temporal_encoder.encode_context(context)
        
        with torch.no_grad():
            target_emb = temporal_encoder.encode_single(target)
        
        pred_emb = predictor(context_emb)
        loss = cosine_similarity_loss(pred_emb, target_emb)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(temporal_encoder, predictor, dataloader, device):
    """Evaluate and return mean cosine similarity."""
    temporal_encoder.eval()
    predictor.eval()
    
    all_sims = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', leave=False):
            context = batch['context'].to(device)
            target = batch['target'].to(device)
            
            context_emb = temporal_encoder.encode_context(context)
            target_emb = temporal_encoder.encode_single(target)
            pred_emb = predictor(context_emb)
            
            sims = cosine_similarity(pred_emb, target_emb).cpu().numpy()
            all_sims.extend(sims)
    
    return np.mean(all_sims)


def run_ood_experiment(variant: str, device, args):
    """
    Run the OOD generalization experiment for one variant.
    
    Returns dict with ID and OOD performance.
    """
    print(f"\n{'='*60}")
    print(f"Running {variant.upper()}")
    print(f"{'='*60}")
    
    mode = 'raw' if variant == 'v_jepa' else 'edge'
    
    # Training data: 2 balls
    print(f"\nLoading training data (2 balls)...")
    train_loader, id_test_loader = get_bouncing_balls_loaders(
        mode=mode,
        num_train=args.num_train,
        num_test=args.num_test,
        num_balls=2,  # Train on 2 balls
        context_length=args.context_length,
        max_horizon=args.max_horizon,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    # OOD test data: 3 balls
    print(f"Loading OOD test data (3 balls)...")
    ood_test_loader = get_ood_test_loader(
        mode=mode,
        num_test=args.num_test,
        num_balls=3,  # Test on 3 balls (OOD)
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
    total_params = sum(p.numel() for p in temporal_encoder.parameters())
    total_params += sum(p.numel() for p in predictor.parameters())
    print(f"Total params: {total_params:,}")
    
    # Optimizer
    params = list(temporal_encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    
    # Training on 2-ball data
    print(f"\nTraining on 2-ball scenarios for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        loss = train_epoch(temporal_encoder, predictor, train_loader, optimizer, device)
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {loss:.4f}")
    
    # Evaluate on ID (2 balls)
    print("\nEvaluating on 2-ball test set (ID)...")
    id_score = evaluate(temporal_encoder, predictor, id_test_loader, device)
    
    # Evaluate on OOD (3 balls)
    print("Evaluating on 3-ball test set (OOD)...")
    ood_score = evaluate(temporal_encoder, predictor, ood_test_loader, device)
    
    # Generalization gap
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
    print("=" * 70)
    print("        OOD Generalization: V-JEPA vs A-JEPA")
    print("=" * 70)
    
    print(f"\n{'Metric':<35} {'V-JEPA':>15} {'A-JEPA':>15}")
    print("-" * 70)
    
    # Parameters
    print(f"{'Parameters':<35} {v_results['params']:>15,} {a_results['params']:>15,}")
    
    print("-" * 70)
    print("Cosine Similarity (higher = better):")
    
    # ID performance
    print(f"{'  In-Distribution (2 balls)':<35} {v_results['id_score']:>15.4f} {a_results['id_score']:>15.4f}")
    
    # OOD performance
    print(f"{'  Out-of-Distribution (3 balls)':<35} {v_results['ood_score']:>15.4f} {a_results['ood_score']:>15.4f}")
    
    print("-" * 70)
    
    # Generalization gap
    print(f"{'Generalization Gap (lower = better)':<35} {v_results['gen_gap']:>15.4f} {a_results['gen_gap']:>15.4f}")
    
    print("=" * 70)
    
    # Key findings
    print("\nKey Findings:")
    
    # Parameter efficiency
    param_ratio = v_results['params'] / a_results['params']
    print(f"  - V-JEPA has {param_ratio:.1f}x more parameters than A-JEPA")
    
    # Generalization comparison
    if a_results['gen_gap'] < v_results['gen_gap']:
        improvement = ((v_results['gen_gap'] - a_results['gen_gap']) / v_results['gen_gap']) * 100
        print(f"  - A-JEPA generalizes {improvement:.1f}% better (smaller gap)")
        print(f"  - A-JEPA gen gap: {a_results['gen_gap']:.4f} vs V-JEPA: {v_results['gen_gap']:.4f}")
    else:
        print(f"  - V-JEPA generalizes better (gap: {v_results['gen_gap']:.4f} vs {a_results['gen_gap']:.4f})")
    
    # OOD performance
    if a_results['ood_score'] > v_results['ood_score']:
        print(f"  - A-JEPA maintains higher OOD performance ({a_results['ood_score']:.4f} vs {v_results['ood_score']:.4f})")
    else:
        print(f"  - V-JEPA maintains higher OOD performance ({v_results['ood_score']:.4f} vs {a_results['ood_score']:.4f})")
    
    # Verdict
    print("\n" + "=" * 70)
    if a_results['gen_gap'] < v_results['gen_gap'] and a_results['ood_score'] > v_results['ood_score']:
        print("VERDICT: A-JEPA demonstrates superior generalization to novel configurations!")
        print("         Abstract/relational representations transfer better to unseen scenarios.")
    elif v_results['gen_gap'] < a_results['gen_gap']:
        print("VERDICT: V-JEPA shows better generalization in this test.")
    else:
        print("VERDICT: Mixed results - further investigation needed.")
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(description='OOD Generalization Test')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_train', type=int, default=500, help='Training videos')
    parser.add_argument('--num_test', type=int, default=100, help='Test videos')
    parser.add_argument('--context_length', type=int, default=5, help='Context frames')
    parser.add_argument('--max_horizon', type=int, default=5, help='Max prediction horizon')
    parser.add_argument('--num_workers', type=int, default=2, help='Dataloader workers')
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"\nOOD Generalization Test")
    print(f"  Train: 2 balls")
    print(f"  Test ID: 2 balls")
    print(f"  Test OOD: 3 balls")
    print(f"  Epochs: {args.epochs}")
    print(f"  Training videos: {args.num_train}")
    
    # Run both variants
    v_results = run_ood_experiment('v_jepa', device, args)
    a_results = run_ood_experiment('a_jepa', device, args)
    
    # Print comparison
    print_comparison(v_results, a_results)


if __name__ == '__main__':
    main()

