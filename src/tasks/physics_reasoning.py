"""
Physics Reasoning Benchmark (Phyre-inspired)

Tests whether models can reason about physical outcomes:
1. Goal Prediction: Given initial frames, predict if ball reaches target
2. Temporal Stability: Maintain prediction accuracy over time

Hypothesis: A-JEPA's abstract representations should excel at physics
reasoning since it focuses on structure/motion, not visual details.
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.datasets.physics_reasoning import (
    PhysicsReasoningDataset,
    get_physics_reasoning_loaders,
)
from src.models import SmallEncoder, Predictor
from torch.utils.data import DataLoader


class PhysicsEncoder(nn.Module):
    """Encoder for physics reasoning with temporal aggregation."""
    
    def __init__(self, in_channels: int, emb_dim: int, width_mult: float, use_relational: bool):
        super().__init__()
        self.encoder = SmallEncoder(
            in_channels=in_channels,
            emb_dim=emb_dim,
            width_mult=width_mult,
            use_relational=use_relational,
        )
        self.emb_dim = emb_dim
    
    def encode_single(self, frame):
        """Encode single frame."""
        return self.encoder(frame)
    
    def encode_context(self, context):
        """Encode context frames and aggregate."""
        B, T, C, H, W = context.shape
        frames_flat = context.reshape(B * T, C, H, W)
        embeddings = self.encoder(frames_flat)
        D = embeddings.shape[-1]
        embeddings = embeddings.reshape(B, T, D)
        return embeddings.mean(dim=1)


class OutcomePredictor(nn.Module):
    """Predicts binary outcome (goal reached or not) from embedding."""
    
    def __init__(self, emb_dim: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(emb_dim // 2, 2),  # Binary classification
        )
    
    def forward(self, x):
        return self.classifier(x)


def get_physics_models(variant: str):
    """Create encoder and outcome predictor."""
    if variant == "v_jepa":
        encoder = PhysicsEncoder(
            in_channels=1,
            emb_dim=256,
            width_mult=1.0,
            use_relational=False,
        )
        emb_dim = 256
    elif variant == "a_jepa":
        encoder = PhysicsEncoder(
            in_channels=1,
            emb_dim=128,
            width_mult=0.5,
            use_relational=True,
        )
        emb_dim = 128
    else:
        raise ValueError(f"Unknown variant: {variant}")
    
    outcome_predictor = OutcomePredictor(emb_dim)
    temporal_predictor = Predictor(emb_dim=emb_dim, hidden_dim=emb_dim * 2)
    
    return encoder, outcome_predictor, temporal_predictor, emb_dim


def train_epoch_classification(encoder, predictor, dataloader, optimizer, criterion, device):
    """Train for one epoch on outcome classification."""
    encoder.train()
    predictor.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc='Training', leave=False):
        context = batch['context'].to(device)
        outcomes = batch['outcome'].to(device)
        
        # Encode context and predict outcome
        embedding = encoder.encode_context(context)
        logits = predictor(embedding)
        
        loss = criterion(logits, outcomes)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == outcomes).sum().item()
        total += outcomes.size(0)
    
    return total_loss / len(dataloader), 100 * correct / total


def evaluate_classification(encoder, predictor, dataloader, device):
    """Evaluate outcome prediction accuracy."""
    encoder.eval()
    predictor.eval()
    
    correct = 0
    total = 0
    
    # Track per-class accuracy
    correct_0, total_0 = 0, 0  # No goal
    correct_1, total_1 = 0, 0  # Goal reached
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', leave=False):
            context = batch['context'].to(device)
            outcomes = batch['outcome'].to(device)
            
            embedding = encoder.encode_context(context)
            logits = predictor(embedding)
            
            _, predicted = torch.max(logits, 1)
            
            correct += (predicted == outcomes).sum().item()
            total += outcomes.size(0)
            
            # Per-class
            for i in range(outcomes.size(0)):
                if outcomes[i] == 0:
                    total_0 += 1
                    if predicted[i] == 0:
                        correct_0 += 1
                else:
                    total_1 += 1
                    if predicted[i] == 1:
                        correct_1 += 1
    
    overall = 100 * correct / total
    acc_0 = 100 * correct_0 / max(total_0, 1)
    acc_1 = 100 * correct_1 / max(total_1, 1)
    
    return overall, acc_0, acc_1


def cosine_similarity(pred, target):
    pred = nn.functional.normalize(pred, dim=-1)
    target = nn.functional.normalize(target, dim=-1)
    return torch.sum(pred * target, dim=-1)


def train_temporal(encoder, predictor, dataloader, optimizer, device, num_epochs=5):
    """Train temporal prediction on physics videos."""
    encoder.train()
    predictor.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            video = batch['video'].to(device)  # (B, T, C, H, W)
            B, T, C, H, W = video.shape
            
            # Use first 5 frames as context, predict frame 10
            context = video[:, :5]
            target_frame = video[:, 10]  # Single frame
            
            context_emb = encoder.encode_context(context)
            
            with torch.no_grad():
                target_emb = encoder.encode_single(target_frame)
            
            pred_emb = predictor(context_emb)
            
            loss = -torch.mean(cosine_similarity(pred_emb, target_emb))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        if (epoch + 1) % 2 == 0:
            print(f"  Temporal Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/num_batches:.4f}")


def evaluate_temporal(encoder, predictor, dataloader, device, horizons=[5, 10, 15, 20]):
    """Evaluate temporal prediction at different horizons."""
    encoder.eval()
    predictor.eval()
    
    results = {h: [] for h in horizons}
    
    with torch.no_grad():
        for batch in dataloader:
            video = batch['video'].to(device)
            B, T, C, H, W = video.shape
            
            context = video[:, :5]
            context_emb = encoder.encode_context(context)
            pred_emb = predictor(context_emb)
            
            for h in horizons:
                if h < T:
                    target_frame = video[:, h]
                    target_emb = encoder.encode_single(target_frame)
                    sims = cosine_similarity(pred_emb, target_emb).cpu().numpy()
                    results[h].extend(sims)
    
    return {h: np.mean(results[h]) for h in horizons if results[h]}


def run_physics_experiment(variant: str, device, args):
    """Run full physics reasoning experiment."""
    print(f"\n{'='*60}")
    print(f"Running {variant.upper()}")
    print(f"{'='*60}")
    
    mode = 'raw' if variant == 'v_jepa' else 'edge'
    
    # Create models
    encoder, outcome_pred, temporal_pred, emb_dim = get_physics_models(variant)
    encoder = encoder.to(device)
    outcome_pred = outcome_pred.to(device)
    temporal_pred = temporal_pred.to(device)
    
    total_params = sum(p.numel() for p in encoder.parameters())
    total_params += sum(p.numel() for p in outcome_pred.parameters())
    total_params += sum(p.numel() for p in temporal_pred.parameters())
    print(f"Total params: {total_params:,}")
    
    # Create data loaders
    train_loader, test_loader = get_physics_reasoning_loaders(
        mode=mode,
        num_train=args.num_train,
        num_test=args.num_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    results = {'variant': variant, 'params': total_params}
    
    # ========================================
    # Task 1: Outcome Classification
    # ========================================
    print("\n--- Task 1: Goal Prediction ---")
    
    params = list(encoder.parameters()) + list(outcome_pred.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(args.epochs):
        loss, train_acc = train_epoch_classification(
            encoder, outcome_pred, train_loader, optimizer, criterion, device
        )
        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {loss:.4f}, Train Acc: {train_acc:.1f}%")
    
    overall_acc, acc_no_goal, acc_goal = evaluate_classification(
        encoder, outcome_pred, test_loader, device
    )
    
    results['goal_accuracy'] = overall_acc
    results['acc_no_goal'] = acc_no_goal
    results['acc_goal'] = acc_goal
    
    print(f"\nGoal Prediction Results:")
    print(f"  Overall Accuracy: {overall_acc:.1f}%")
    print(f"  'No Goal' Accuracy: {acc_no_goal:.1f}%")
    print(f"  'Goal Reached' Accuracy: {acc_goal:.1f}%")
    
    # ========================================
    # Task 2: Temporal Physics Prediction
    # ========================================
    print("\n--- Task 2: Temporal Physics Prediction ---")
    
    # Reset encoder weights and train for temporal prediction
    encoder, _, temporal_pred, _ = get_physics_models(variant)
    encoder = encoder.to(device)
    temporal_pred = temporal_pred.to(device)
    
    params = list(encoder.parameters()) + list(temporal_pred.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    
    train_temporal(encoder, temporal_pred, train_loader, optimizer, device, num_epochs=args.epochs)
    
    horizons = [5, 10, 15, 20]
    temporal_results = evaluate_temporal(encoder, temporal_pred, test_loader, device, horizons)
    
    results['temporal'] = temporal_results
    
    print(f"\nTemporal Prediction (Cosine Similarity):")
    for h, sim in temporal_results.items():
        print(f"  Horizon {h}: {sim:.4f}")
    
    # Calculate drift
    if 5 in temporal_results and 20 in temporal_results:
        drift = temporal_results[5] - temporal_results[20]
        results['temporal_drift'] = drift
        print(f"  Drift (5→20): {drift:.4f}")
    
    return results


def print_comparison(v_results: dict, a_results: dict):
    """Print side-by-side comparison."""
    print("\n")
    print("=" * 75)
    print("          Physics Reasoning: V-JEPA vs A-JEPA")
    print("=" * 75)
    
    print(f"\n{'Metric':<40} {'V-JEPA':>15} {'A-JEPA':>15}")
    print("-" * 75)
    
    print(f"{'Parameters':<40} {v_results['params']:>15,} {a_results['params']:>15,}")
    
    print("-" * 75)
    print("Goal Prediction Accuracy (%):")
    print(f"{'  Overall':<40} {v_results['goal_accuracy']:>15.1f} {a_results['goal_accuracy']:>15.1f}")
    print(f"{'  No Goal':<40} {v_results['acc_no_goal']:>15.1f} {a_results['acc_no_goal']:>15.1f}")
    print(f"{'  Goal Reached':<40} {v_results['acc_goal']:>15.1f} {a_results['acc_goal']:>15.1f}")
    
    print("-" * 75)
    print("Temporal Prediction (Cosine Similarity):")
    
    for h in [5, 10, 15, 20]:
        v_sim = v_results['temporal'].get(h, 0)
        a_sim = a_results['temporal'].get(h, 0)
        print(f"{'  Horizon ' + str(h):<40} {v_sim:>15.4f} {a_sim:>15.4f}")
    
    v_drift = v_results.get('temporal_drift', 0)
    a_drift = a_results.get('temporal_drift', 0)
    print(f"{'  Drift (lower = better)':<40} {v_drift:>15.4f} {a_drift:>15.4f}")
    
    print("=" * 75)
    
    # Key findings
    print("\nKey Findings:")
    
    param_ratio = v_results['params'] / a_results['params']
    print(f"  - V-JEPA has {param_ratio:.1f}x more parameters than A-JEPA")
    
    # Goal prediction
    if a_results['goal_accuracy'] > v_results['goal_accuracy']:
        print(f"  - A-JEPA predicts goals better ({a_results['goal_accuracy']:.1f}% vs {v_results['goal_accuracy']:.1f}%)")
    else:
        print(f"  - V-JEPA predicts goals better ({v_results['goal_accuracy']:.1f}% vs {a_results['goal_accuracy']:.1f}%)")
    
    # Drift
    if a_drift < v_drift:
        improvement = ((v_drift - a_drift) / max(v_drift, 0.001)) * 100
        print(f"  - A-JEPA has {abs(improvement):.1f}% lower temporal drift")
    else:
        print(f"  - V-JEPA has lower temporal drift")
    
    # Verdict
    print("\n" + "=" * 75)
    a_wins = 0
    if a_results['goal_accuracy'] > v_results['goal_accuracy']:
        a_wins += 1
    if a_drift < v_drift:
        a_wins += 1
    if a_results['temporal'].get(20, 0) > v_results['temporal'].get(20, 0):
        a_wins += 1
    
    if a_wins >= 2:
        print("VERDICT: A-JEPA excels at physics reasoning!")
        print("         Abstract representations capture physical dynamics better.")
    elif a_wins == 0:
        print("VERDICT: V-JEPA performs better on physics reasoning.")
    else:
        print("VERDICT: Mixed results on physics reasoning tasks.")
    print("=" * 75)


def parse_args():
    parser = argparse.ArgumentParser(description='Physics Reasoning Benchmark')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_train', type=int, default=500, help='Training samples')
    parser.add_argument('--num_test', type=int, default=100, help='Test samples')
    parser.add_argument('--num_workers', type=int, default=2, help='Dataloader workers')
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"\nPhysics Reasoning Benchmark (Phyre-inspired)")
    print(f"  Tasks: Goal Prediction + Temporal Physics")
    print(f"  Epochs: {args.epochs}")
    print(f"  Training samples: {args.num_train}")
    
    # Run both variants
    v_results = run_physics_experiment('v_jepa', device, args)
    a_results = run_physics_experiment('a_jepa', device, args)
    
    # Print comparison
    print_comparison(v_results, a_results)


if __name__ == '__main__':
    main()

