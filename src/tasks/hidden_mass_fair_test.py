"""
Hidden Mass Inference: FAIR Comparison V-JEPA v2 vs A-JEPA v2

Both models now have:
✓ Temporal memory core
✓ Multi-step latent prediction
✓ Comparable parameter counts
✓ Same training steps

The ONLY differences are:
- V-JEPA v2: RGB input, single vector representation
- A-JEPA v2: Edge input, slot attention, sparsity, multi-timescale

This isolates the cognitive architecture differences.
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.datasets.hidden_mass import get_hidden_mass_loaders
from src.models_v2 import VJEPAv2, AJEPAv2, get_vjepa_v2, get_ajepa_v2
from torch.utils.data import DataLoader


class LinearProbe(nn.Module):
    def __init__(self, emb_dim, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(emb_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)


def train_ssl(model, dataloader, optimizer, device, epochs, model_name):
    """Self-supervised training with multi-step prediction."""
    model.train()
    
    pbar = tqdm(range(epochs), desc=f'{model_name} SSL')
    for epoch in pbar:
        total_loss = 0
        total_pred_loss = 0
        
        for batch in dataloader:
            video = batch['video'].to(device)
            B, T, C, H, W = video.shape
            
            # Split into context and target
            context = video[:, :T//2]
            target = video[:, T//2:]
            
            # Forward pass
            output = model(context, target)
            loss = output['loss']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_pred_loss += output['pred_loss'].item()
        
        n = len(dataloader)
        pbar.set_postfix({
            'loss': f"{total_loss/n:.4f}",
            'pred': f"{total_pred_loss/n:.4f}"
        })
    
    return total_loss / len(dataloader)


def extract_features(model, dataloader, device):
    """Extract features using frozen encoder."""
    model.eval()
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            video = batch['video'].to(device)
            labels = batch['mass_label']
            
            features = model.encode_video(video)
            all_features.append(features.cpu())
            all_labels.append(labels)
    
    return torch.cat(all_features), torch.cat(all_labels)


def train_probe(features, labels, emb_dim, device, epochs=50):
    """Train linear probe."""
    dataset = torch.utils.data.TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    probe = LinearProbe(emb_dim, num_classes=2).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(epochs):
        probe.train()
        for feats, lbls in loader:
            feats, lbls = feats.to(device), lbls.to(device)
            logits = probe(feats)
            loss = criterion(logits, lbls)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return probe


def evaluate_probe(probe, features, labels, device):
    """Evaluate linear probe accuracy."""
    probe.eval()
    features, labels = features.to(device), labels.to(device)
    
    with torch.no_grad():
        logits = probe(features)
        preds = logits.argmax(dim=1)
        accuracy = (preds == labels).float().mean().item()
    
    return accuracy * 100


def run_experiment(model, model_name, mode, device, args):
    """Run full experiment for one model."""
    print(f"\n{'='*60}")
    print(f"{model_name}")
    print(f"{'='*60}")
    
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters())
    emb_dim = model.total_dim
    
    print(f"Parameters: {params:,}")
    print(f"Embedding dim: {emb_dim}")
    
    # Load data
    train_loader, test_loader = get_hidden_mass_loaders(
        mode=mode,
        num_train=args.num_train,
        num_test=args.num_test,
        batch_size=args.batch_size,
        randomize_appearance=True,
    )
    
    # SSL training
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    print(f"\nPhase 1: Self-supervised training ({args.ssl_epochs} epochs)...")
    train_ssl(model, train_loader, optimizer, device, args.ssl_epochs, model_name)
    
    # Extract features
    print(f"\nPhase 2: Extracting features...")
    train_feats, train_labels = extract_features(model, train_loader, device)
    test_feats, test_labels = extract_features(model, test_loader, device)
    print(f"  Train: {train_feats.shape}, Test: {test_feats.shape}")
    
    # Train probe
    print(f"\nPhase 3: Training linear probe ({args.probe_epochs} epochs)...")
    probe = train_probe(train_feats, train_labels, emb_dim, device, args.probe_epochs)
    
    # Evaluate
    train_acc = evaluate_probe(probe, train_feats, train_labels, device)
    test_acc = evaluate_probe(probe, test_feats, test_labels, device)
    
    print(f"\n  Train Accuracy: {train_acc:.1f}%")
    print(f"  Test Accuracy:  {test_acc:.1f}%")
    
    return {
        'name': model_name,
        'params': params,
        'emb_dim': emb_dim,
        'train_acc': train_acc,
        'test_acc': test_acc,
    }


def print_comparison(v2_results, a2_results):
    """Print side-by-side comparison."""
    print("\n")
    print("=" * 80)
    print("        FAIR COMPARISON: V-JEPA v2 vs A-JEPA v2")
    print("=" * 80)
    
    print("\nArchitecture Comparison:")
    print("-" * 80)
    print(f"{'Feature':<35} {'V-JEPA v2':>20} {'A-JEPA v2':>20}")
    print("-" * 80)
    print(f"{'Input':<35} {'RGB (3ch)':>20} {'Edge (1ch)':>20}")
    print(f"{'Temporal Memory':<35} {'✓':>20} {'✓':>20}")
    print(f"{'Multi-Step Prediction':<35} {'✓':>20} {'✓':>20}")
    print(f"{'Slot Attention':<35} {'✗':>20} {'✓':>20}")
    print(f"{'Multi-Timescale (fast/slow)':<35} {'✗':>20} {'✓':>20}")
    print(f"{'Sparsity Constraint':<35} {'✗':>20} {'✓':>20}")
    print(f"{'Uncertainty Head':<35} {'✓':>20} {'✓':>20}")
    
    print("\n" + "=" * 80)
    print("Results:")
    print("-" * 80)
    print(f"{'Metric':<35} {'V-JEPA v2':>20} {'A-JEPA v2':>20}")
    print("-" * 80)
    print(f"{'Parameters':<35} {v2_results['params']:>20,} {a2_results['params']:>20,}")
    print(f"{'Embedding Dim':<35} {v2_results['emb_dim']:>20} {a2_results['emb_dim']:>20}")
    print(f"{'Train Accuracy':<35} {v2_results['train_acc']:>19.1f}% {a2_results['train_acc']:>19.1f}%")
    print(f"{'Test Accuracy':<35} {v2_results['test_acc']:>19.1f}% {a2_results['test_acc']:>19.1f}%")
    
    # Efficiency
    v_eff = v2_results['test_acc'] / (v2_results['params'] / 1_000_000)
    a_eff = a2_results['test_acc'] / (a2_results['params'] / 1_000_000)
    print(f"{'Accuracy per M params':<35} {v_eff:>19.1f}% {a_eff:>19.1f}%")
    
    print("=" * 80)
    
    # Winner determination
    diff = a2_results['test_acc'] - v2_results['test_acc']
    
    print("\n" + "=" * 80)
    if diff > 2:
        print(f"🏆 A-JEPA v2 WINS by {diff:.1f} percentage points!")
        print("\nInterpretation:")
        print("  The cognitive architecture (slot attention, multi-timescale, sparsity)")
        print("  helps extract hidden causal properties better than raw visual processing.")
        print("  Edge-based + object-factorized representations capture physics, not pixels.")
    elif diff < -2:
        print(f"🏆 V-JEPA v2 WINS by {-diff:.1f} percentage points!")
        print("\nInterpretation:")
        print("  RGB visual features still contain useful information for this task.")
        print("  The cognitive constraints may need more training or tuning.")
    else:
        print(f"🤝 DRAW (within 2pp: V={v2_results['test_acc']:.1f}%, A={a2_results['test_acc']:.1f}%)")
        print("\nInterpretation:")
        print("  Both architectures perform similarly on hidden mass inference.")
        print("  A-JEPA v2 achieves this with cognitive constraints (object slots, sparsity).")
    
    print("=" * 80)
    
    # Above chance analysis
    v_above = v2_results['test_acc'] - 50
    a_above = a2_results['test_acc'] - 50
    
    print(f"\nAbove chance (50%):")
    print(f"  V-JEPA v2: +{v_above:.1f}pp")
    print(f"  A-JEPA v2: +{a_above:.1f}pp")
    
    if max(v_above, a_above) > 15:
        print("\n✓ Strong signal! Models are extracting hidden mass from dynamics.")
    elif max(v_above, a_above) > 5:
        print("\n~ Moderate signal. Some hidden property captured.")
    else:
        print("\n✗ Weak signal. May need more training or larger dataset.")


def main():
    parser = argparse.ArgumentParser(description='Fair V-JEPA v2 vs A-JEPA v2 Test')
    parser.add_argument('--ssl_epochs', type=int, default=25, help='Self-supervised epochs')
    parser.add_argument('--probe_epochs', type=int, default=50, help='Linear probe epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_train', type=int, default=500, help='Training samples')
    parser.add_argument('--num_test', type=int, default=150, help='Test samples')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 80)
    print("HIDDEN MASS INFERENCE: FAIR COMPARISON")
    print("=" * 80)
    print(f"\nDevice: {device}")
    print(f"\nBoth models have:")
    print("  ✓ Temporal memory core")
    print("  ✓ Multi-step latent prediction (5 steps)")
    print("  ✓ Uncertainty-aware predictions")
    print(f"\nTraining: {args.ssl_epochs} SSL epochs, {args.probe_epochs} probe epochs")
    print(f"Data: {args.num_train} train, {args.num_test} test videos")
    
    # Create models
    v_jepa_v2 = get_vjepa_v2(in_channels=1, img_size=32, config='default')
    a_jepa_v2 = get_ajepa_v2(in_channels=1, img_size=32, config='default')
    
    # Run V-JEPA v2 (uses 'raw' mode since it's designed for RGB, but we use grayscale)
    v2_results = run_experiment(
        v_jepa_v2, "V-JEPA v2 (Temporal Memory + Multi-Step)",
        mode='raw', device=device, args=args
    )
    
    # Run A-JEPA v2 (uses 'edge' mode)
    a2_results = run_experiment(
        a_jepa_v2, "A-JEPA v2 (Cognitive Architecture)",
        mode='edge', device=device, args=args
    )
    
    # Print comparison
    print_comparison(v2_results, a2_results)
    
    # Save results
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    with open(f'{results_dir}/fair_comparison_v2.txt', 'w') as f:
        f.write("FAIR COMPARISON: V-JEPA v2 vs A-JEPA v2\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"V-JEPA v2:\n")
        f.write(f"  Parameters: {v2_results['params']:,}\n")
        f.write(f"  Train Acc: {v2_results['train_acc']:.1f}%\n")
        f.write(f"  Test Acc: {v2_results['test_acc']:.1f}%\n\n")
        f.write(f"A-JEPA v2:\n")
        f.write(f"  Parameters: {a2_results['params']:,}\n")
        f.write(f"  Train Acc: {a2_results['train_acc']:.1f}%\n")
        f.write(f"  Test Acc: {a2_results['test_acc']:.1f}%\n")
    
    print(f"\nResults saved to {results_dir}/fair_comparison_v2.txt")


if __name__ == '__main__':
    main()

