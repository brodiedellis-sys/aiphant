"""
Interaction Dynamics Test: Ball-Ball Collisions

Tests whether A-JEPA handles multi-object interactions (collisions) better than V-JEPA.

Hypothesis: A-JEPA's relational reasoning block will help it model interactions
between objects, leading to better predictions when balls collide.

Comparison:
- Train/test on videos WITHOUT collisions (baseline)
- Train/test on videos WITH collisions (interaction test)
- Train WITHOUT collisions, test WITH collisions (transfer test)
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
    BouncingBallsDataset,
)
from src.tasks.predict_future import (
    TemporalEncoder,
    get_temporal_models,
    cosine_similarity_loss,
    cosine_similarity,
)
from torch.utils.data import DataLoader


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


def create_loader(mode, num_samples, num_balls, with_collisions, batch_size, seed):
    """Create a dataloader with specific settings."""
    dataset = BouncingBallsDataset(
        num_samples=num_samples,
        num_balls=num_balls,
        context_length=5,
        max_horizon=5,
        mode=mode,
        with_collisions=with_collisions,
        seed=seed,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=False)


def run_interaction_experiment(variant: str, device, args):
    """
    Run the full interaction dynamics experiment for one variant.
    
    Tests:
    1. No collisions (baseline)
    2. With collisions (interaction handling)
    3. Transfer: train no collision -> test with collision
    """
    print(f"\n{'='*60}")
    print(f"Running {variant.upper()}")
    print(f"{'='*60}")
    
    mode = 'raw' if variant == 'v_jepa' else 'edge'
    
    results = {}
    
    # Create models
    temporal_encoder, predictor, emb_dim = get_temporal_models(variant)
    temporal_encoder = temporal_encoder.to(device)
    predictor = predictor.to(device)
    
    total_params = sum(p.numel() for p in temporal_encoder.parameters())
    total_params += sum(p.numel() for p in predictor.parameters())
    print(f"Total params: {total_params:,}")
    results['params'] = total_params
    
    # =========================================
    # Test 1: No Collisions (Baseline)
    # =========================================
    print(f"\n--- Test 1: No Collisions (Baseline) ---")
    
    # Reset models
    temporal_encoder, predictor, _ = get_temporal_models(variant)
    temporal_encoder = temporal_encoder.to(device)
    predictor = predictor.to(device)
    
    train_loader = create_loader(mode, args.num_train, args.num_balls, False, args.batch_size, 42)
    test_loader = create_loader(mode, args.num_test, args.num_balls, False, args.batch_size, 123)
    
    params = list(temporal_encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    
    for epoch in range(args.epochs):
        loss = train_epoch(temporal_encoder, predictor, train_loader, optimizer, device)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {loss:.4f}")
    
    no_collision_score = evaluate(temporal_encoder, predictor, test_loader, device)
    results['no_collision'] = no_collision_score
    print(f"No Collision Score: {no_collision_score:.4f}")
    
    # =========================================
    # Test 2: With Collisions
    # =========================================
    print(f"\n--- Test 2: With Collisions ---")
    
    # Reset models
    temporal_encoder, predictor, _ = get_temporal_models(variant)
    temporal_encoder = temporal_encoder.to(device)
    predictor = predictor.to(device)
    
    train_loader = create_loader(mode, args.num_train, args.num_balls, True, args.batch_size, 42)
    test_loader = create_loader(mode, args.num_test, args.num_balls, True, args.batch_size, 123)
    
    params = list(temporal_encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    
    for epoch in range(args.epochs):
        loss = train_epoch(temporal_encoder, predictor, train_loader, optimizer, device)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {loss:.4f}")
    
    with_collision_score = evaluate(temporal_encoder, predictor, test_loader, device)
    results['with_collision'] = with_collision_score
    print(f"With Collision Score: {with_collision_score:.4f}")
    
    # =========================================
    # Test 3: Transfer (No Collision -> With Collision)
    # =========================================
    print(f"\n--- Test 3: Transfer (train no collision -> test with collision) ---")
    
    # Reset models
    temporal_encoder, predictor, _ = get_temporal_models(variant)
    temporal_encoder = temporal_encoder.to(device)
    predictor = predictor.to(device)
    
    # Train on NO collisions
    train_loader = create_loader(mode, args.num_train, args.num_balls, False, args.batch_size, 42)
    
    params = list(temporal_encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    
    for epoch in range(args.epochs):
        loss = train_epoch(temporal_encoder, predictor, train_loader, optimizer, device)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {loss:.4f}")
    
    # Test on WITH collisions
    test_loader_collision = create_loader(mode, args.num_test, args.num_balls, True, args.batch_size, 456)
    transfer_score = evaluate(temporal_encoder, predictor, test_loader_collision, device)
    results['transfer'] = transfer_score
    print(f"Transfer Score: {transfer_score:.4f}")
    
    # Calculate interaction gap (how much harder collisions are)
    results['interaction_gap'] = results['no_collision'] - results['with_collision']
    results['transfer_gap'] = results['no_collision'] - results['transfer']
    
    return results


def print_comparison(v_results: dict, a_results: dict):
    """Print side-by-side comparison."""
    print("\n")
    print("=" * 75)
    print("           Interaction Dynamics: V-JEPA vs A-JEPA")
    print("=" * 75)
    
    print(f"\n{'Metric':<40} {'V-JEPA':>15} {'A-JEPA':>15}")
    print("-" * 75)
    
    print(f"{'Parameters':<40} {v_results['params']:>15,} {a_results['params']:>15,}")
    
    print("-" * 75)
    print("Cosine Similarity (higher = better):")
    
    print(f"{'  No Collisions (baseline)':<40} {v_results['no_collision']:>15.4f} {a_results['no_collision']:>15.4f}")
    print(f"{'  With Collisions':<40} {v_results['with_collision']:>15.4f} {a_results['with_collision']:>15.4f}")
    print(f"{'  Transfer (no coll -> with coll)':<40} {v_results['transfer']:>15.4f} {a_results['transfer']:>15.4f}")
    
    print("-" * 75)
    print("Gaps (lower = better handling of complexity):")
    
    print(f"{'  Interaction Gap':<40} {v_results['interaction_gap']:>15.4f} {a_results['interaction_gap']:>15.4f}")
    print(f"{'  Transfer Gap':<40} {v_results['transfer_gap']:>15.4f} {a_results['transfer_gap']:>15.4f}")
    
    print("=" * 75)
    
    # Key findings
    print("\nKey Findings:")
    
    param_ratio = v_results['params'] / a_results['params']
    print(f"  - V-JEPA has {param_ratio:.1f}x more parameters than A-JEPA")
    
    # Collision handling
    if a_results['with_collision'] > v_results['with_collision']:
        print(f"  - A-JEPA handles collisions better ({a_results['with_collision']:.4f} vs {v_results['with_collision']:.4f})")
    else:
        print(f"  - V-JEPA handles collisions better ({v_results['with_collision']:.4f} vs {a_results['with_collision']:.4f})")
    
    # Interaction gap comparison
    if a_results['interaction_gap'] < v_results['interaction_gap']:
        improvement = ((v_results['interaction_gap'] - a_results['interaction_gap']) / max(v_results['interaction_gap'], 0.001)) * 100
        print(f"  - A-JEPA degrades {improvement:.1f}% less when collisions are added")
    else:
        print(f"  - V-JEPA degrades less when collisions are added")
    
    # Transfer comparison
    if a_results['transfer_gap'] < v_results['transfer_gap']:
        print(f"  - A-JEPA transfers better to unseen collision dynamics")
    else:
        print(f"  - V-JEPA transfers better to unseen collision dynamics")
    
    # Verdict
    print("\n" + "=" * 75)
    a_wins = 0
    if a_results['with_collision'] > v_results['with_collision']:
        a_wins += 1
    if a_results['interaction_gap'] < v_results['interaction_gap']:
        a_wins += 1
    if a_results['transfer_gap'] < v_results['transfer_gap']:
        a_wins += 1
    
    if a_wins >= 2:
        print("VERDICT: A-JEPA handles multi-object interactions better!")
        print("         Relational reasoning helps model collision dynamics.")
    elif a_wins == 0:
        print("VERDICT: V-JEPA handles interactions better in this test.")
    else:
        print("VERDICT: Mixed results - both models have strengths.")
    print("=" * 75)


def parse_args():
    parser = argparse.ArgumentParser(description='Interaction Dynamics Test')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs per test')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_train', type=int, default=500, help='Training videos')
    parser.add_argument('--num_test', type=int, default=100, help='Test videos')
    parser.add_argument('--num_balls', type=int, default=3, help='Number of balls')
    parser.add_argument('--num_workers', type=int, default=2, help='Dataloader workers')
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"\nInteraction Dynamics Test")
    print(f"  Number of balls: {args.num_balls}")
    print(f"  Epochs per test: {args.epochs}")
    print(f"  Training videos: {args.num_train}")
    
    # Run both variants
    v_results = run_interaction_experiment('v_jepa', device, args)
    a_results = run_interaction_experiment('a_jepa', device, args)
    
    # Print comparison
    print_comparison(v_results, a_results)


if __name__ == '__main__':
    main()

