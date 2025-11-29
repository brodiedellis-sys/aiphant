"""
A-JEPA v2 Fixed Test

Problem identified: SSL collapse (all embeddings become identical)
Solution: Add VICReg-style variance and covariance regularization

This prevents the trivial solution where predictor just outputs a constant.
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.datasets.hidden_mass import HiddenMassDataset
from src.models_v2 import get_ajepa_v2, get_vjepa_v2
from torch.utils.data import DataLoader


# =============================================================================
# VICReg-style Regularization (prevents collapse)
# =============================================================================

def variance_loss(z, gamma=1.0):
    """
    Variance regularization: prevent embeddings from collapsing to a point.
    From VICReg: https://arxiv.org/abs/2105.04906
    """
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return torch.mean(F.relu(gamma - std))


def covariance_loss(z):
    """
    Covariance regularization: decorrelate features.
    From VICReg: https://arxiv.org/abs/2105.04906
    """
    B, D = z.shape
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (B - 1)
    
    # Off-diagonal elements should be zero
    off_diag = cov.flatten()[:-1].view(D-1, D+1)[:, 1:].flatten()
    return (off_diag ** 2).mean()


def vicreg_loss(pred, target, sim_weight=25.0, var_weight=25.0, cov_weight=1.0):
    """
    Combined VICReg loss: similarity + variance + covariance
    """
    # Invariance (similarity)
    sim_loss = F.mse_loss(pred, target)
    
    # Variance (prevent collapse)
    var_loss = variance_loss(pred) + variance_loss(target)
    
    # Covariance (decorrelation)
    cov_loss = covariance_loss(pred) + covariance_loss(target)
    
    total = sim_weight * sim_loss + var_weight * var_loss + cov_weight * cov_loss
    
    return total, {
        'sim': sim_loss.item(),
        'var': var_loss.item(),
        'cov': cov_loss.item(),
    }


# =============================================================================
# Training with Anti-Collapse
# =============================================================================

def train_ssl_fixed(model, dataloader, optimizer, device, epochs, model_name):
    """SSL training with VICReg regularization to prevent collapse."""
    model.train()
    
    pbar = tqdm(range(epochs), desc=f'{model_name} SSL')
    for epoch in pbar:
        total_loss = 0
        total_var = 0
        
        for batch in dataloader:
            video = batch['video'].to(device)
            B, T, C, H, W = video.shape
            
            context = video[:, :T//2]
            target = video[:, T//2:]
            
            # Encode
            z_ctx = model.encode_video(context)
            z_tgt = model.encode_video(target)
            
            # Use VICReg loss instead of just cosine
            loss, loss_parts = vicreg_loss(z_ctx, z_tgt)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_var += loss_parts['var']
        
        n = len(dataloader)
        pbar.set_postfix({
            'loss': f"{total_loss/n:.3f}",
            'var': f"{total_var/n:.3f}"  # Monitor variance (should stay low, not spike)
        })
    
    return total_loss / len(dataloader)


class LinearProbe(nn.Module):
    def __init__(self, emb_dim, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(emb_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)


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
    """Run experiment with fixed SSL training."""
    print(f"\n{'='*60}")
    print(f"{model_name}")
    print(f"{'='*60}")
    
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters())
    emb_dim = model.total_dim
    
    print(f"Parameters: {params:,}")
    print(f"Embedding dim: {emb_dim}")
    
    # Load data
    train_dataset = HiddenMassDataset(
        num_samples=args.num_train, mode=mode,
        randomize_appearance=True, seed=42
    )
    test_dataset = HiddenMassDataset(
        num_samples=args.num_test, mode=mode,
        randomize_appearance=True, seed=123
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # SSL training with anti-collapse
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    print(f"\nSSL Training with VICReg ({args.ssl_epochs} epochs)...")
    train_ssl_fixed(model, train_loader, optimizer, device, args.ssl_epochs, model_name)
    
    # Check for collapse
    print("\nChecking feature quality...")
    train_feats, train_labels = extract_features(model, train_loader, device)
    feat_var = train_feats.var().item()
    feat_std = train_feats.std(dim=0).mean().item()
    print(f"  Feature variance: {feat_var:.4f}")
    print(f"  Avg feature std: {feat_std:.4f}")
    
    if feat_var < 0.01:
        print("  ⚠️  WARNING: Low variance - possible collapse!")
    else:
        print("  ✓ Features look healthy!")
    
    # Train probe
    print(f"\nTraining linear probe ({args.probe_epochs} epochs)...")
    test_feats, test_labels = extract_features(model, test_loader, device)
    probe = train_probe(train_feats, train_labels, emb_dim, device, args.probe_epochs)
    
    # Evaluate
    train_acc = evaluate_probe(probe, train_feats, train_labels, device)
    test_acc = evaluate_probe(probe, test_feats, test_labels, device)
    
    print(f"\nResults:")
    print(f"  Train Accuracy: {train_acc:.1f}%")
    print(f"  Test Accuracy:  {test_acc:.1f}%")
    
    return {
        'name': model_name,
        'params': params,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'feat_var': feat_var,
    }


def main():
    parser = argparse.ArgumentParser(description='A-JEPA Fixed Test')
    parser.add_argument('--ssl_epochs', type=int, default=20)
    parser.add_argument('--probe_epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_train', type=int, default=300)
    parser.add_argument('--num_test', type=int, default=100)
    parser.add_argument('--config', type=str, default='default')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("A-JEPA v2 FIXED TEST (with VICReg anti-collapse)")
    print("="*60)
    print(f"\nDevice: {device}")
    print(f"Config: {args.config}")
    print(f"\nFix: VICReg regularization prevents SSL collapse")
    print("  - Variance loss: keeps features spread out")
    print("  - Covariance loss: decorrelates features")
    
    # Create models
    v_jepa = get_vjepa_v2(in_channels=1, img_size=32, config='default')
    a_jepa = get_ajepa_v2(in_channels=1, img_size=32, config=args.config)
    
    # Run experiments
    v_results = run_experiment(v_jepa, "V-JEPA v2", 'raw', device, args)
    a_results = run_experiment(a_jepa, "A-JEPA v2", 'edge', device, args)
    
    # Comparison
    print("\n" + "="*60)
    print("COMPARISON (with VICReg fix)")
    print("="*60)
    
    print(f"\n{'Model':<20} {'Params':>12} {'Feat Var':>12} {'Test Acc':>12}")
    print("-"*60)
    print(f"{'V-JEPA v2':<20} {v_results['params']:>12,} {v_results['feat_var']:>12.4f} {v_results['test_acc']:>11.1f}%")
    print(f"{'A-JEPA v2':<20} {a_results['params']:>12,} {a_results['feat_var']:>12.4f} {a_results['test_acc']:>11.1f}%")
    
    print("\n" + "="*60)
    if a_results['test_acc'] > 55:
        print("✓ A-JEPA is now learning! VICReg fixed the collapse.")
    else:
        print("⚠️  A-JEPA still struggling. May need architecture simplification.")
    print("="*60)


if __name__ == '__main__':
    main()

