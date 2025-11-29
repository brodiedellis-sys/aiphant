"""
Hidden Mass Inference Test for A-JEPA v2

Compares:
- V-JEPA (standard visual model)
- A-JEPA v1 (edge-based + relational block)  
- A-JEPA v2 (full cognitive architecture)

Tests if the new cognitive features help extract hidden causal properties:
1. Slot attention for object factorization
2. Multi-timescale latents for separating fast (motion) vs slow (mass) properties
3. Uncertainty-aware predictions
4. Sparse bottleneck forcing abstract representations
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.datasets.hidden_mass import get_hidden_mass_loaders, HiddenMassDataset
from src.models import SmallEncoder, Predictor
from src.models_v2 import AJEPAv2, get_ajepa_v2
from torch.utils.data import DataLoader


class TemporalEncoderV1(nn.Module):
    """V1 encoder wrapper for video."""
    
    def __init__(self, in_channels, emb_dim, width_mult, use_relational):
        super().__init__()
        self.encoder = SmallEncoder(
            in_channels=in_channels,
            emb_dim=emb_dim,
            width_mult=width_mult,
            use_relational=use_relational,
        )
        self.emb_dim = emb_dim
    
    def encode_video(self, video):
        B, T, C, H, W = video.shape
        frames = video.reshape(B * T, C, H, W)
        embeddings = self.encoder(frames)
        embeddings = embeddings.view(B, T, -1)
        return embeddings.mean(dim=1)
    
    def forward(self, x):
        return self.encoder(x)


class LinearProbe(nn.Module):
    """Linear classifier for probing learned representations."""
    
    def __init__(self, emb_dim, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(emb_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)


def cosine_loss(pred, target):
    """Negative cosine similarity loss."""
    pred = nn.functional.normalize(pred, dim=-1)
    target = nn.functional.normalize(target, dim=-1)
    return -torch.mean(torch.sum(pred * target, dim=-1))


def train_ssl_v1(encoder, predictor, dataloader, optimizer, device, epochs):
    """Self-supervised training for V1 models."""
    encoder.train()
    predictor.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f'SSL Epoch {epoch+1}', leave=False):
            video = batch['video'].to(device)
            B, T, C, H, W = video.shape
            
            context = video[:, :T//2]
            target = video[:, T//2:]
            
            context_emb = encoder.encode_video(context)
            with torch.no_grad():
                target_emb = encoder.encode_video(target)
            
            pred_emb = predictor(context_emb)
            loss = cosine_loss(pred_emb, target_emb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.4f}")


def train_ssl_v2(model, dataloader, optimizer, device, epochs):
    """Self-supervised training for A-JEPA v2."""
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        total_pred_loss = 0
        total_aux_loss = 0
        
        for batch in tqdm(dataloader, desc=f'SSL Epoch {epoch+1}', leave=False):
            video = batch['video'].to(device)
            B, T, C, H, W = video.shape
            
            context = video[:, :T//2]
            target = video[:, T//2:]
            
            output = model(context, target)
            loss = output['loss']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_pred_loss += output['pred_loss'].item()
            total_aux_loss += output['aux_loss'].item() if isinstance(output['aux_loss'], torch.Tensor) else output['aux_loss']
        
        if (epoch + 1) % 5 == 0:
            n = len(dataloader)
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {total_loss/n:.4f} (pred: {total_pred_loss/n:.4f}, aux: {total_aux_loss/n:.4f})")


def extract_features(encoder, dataloader, device, is_v2=False):
    """Extract features using frozen encoder."""
    encoder.eval()
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Extracting', leave=False):
            video = batch['video'].to(device)
            labels = batch['mass_label']
            
            if is_v2:
                features = encoder.encode_video(video)
            else:
                features = encoder.encode_video(video)
            
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
    
    for epoch in range(epochs):
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


def run_v_jepa(device, args):
    """Run V-JEPA baseline."""
    print("\n" + "=" * 60)
    print("V-JEPA (Visual Baseline)")
    print("=" * 60)
    
    encoder = TemporalEncoderV1(
        in_channels=1, emb_dim=256, width_mult=1.0, use_relational=False
    ).to(device)
    predictor = Predictor(emb_dim=256, hidden_dim=512).to(device)
    
    params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in predictor.parameters())
    print(f"Parameters: {params:,}")
    
    train_loader, test_loader = get_hidden_mass_loaders(
        mode='raw', num_train=args.num_train, num_test=args.num_test,
        batch_size=args.batch_size, randomize_appearance=True,
    )
    
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    
    print(f"\nSSL Training ({args.ssl_epochs} epochs)...")
    train_ssl_v1(encoder, predictor, train_loader, optimizer, device, args.ssl_epochs)
    
    print("\nExtracting features...")
    train_feats, train_labels = extract_features(encoder, train_loader, device)
    test_feats, test_labels = extract_features(encoder, test_loader, device)
    
    print(f"\nTraining probe ({args.probe_epochs} epochs)...")
    probe = train_probe(train_feats, train_labels, 256, device, args.probe_epochs)
    
    train_acc = evaluate_probe(probe, train_feats, train_labels, device)
    test_acc = evaluate_probe(probe, test_feats, test_labels, device)
    
    print(f"\nResults: Train {train_acc:.1f}%, Test {test_acc:.1f}%")
    
    return {'params': params, 'train_acc': train_acc, 'test_acc': test_acc}


def run_a_jepa_v1(device, args):
    """Run A-JEPA v1 (edge + relational)."""
    print("\n" + "=" * 60)
    print("A-JEPA v1 (Edge + Relational)")
    print("=" * 60)
    
    encoder = TemporalEncoderV1(
        in_channels=1, emb_dim=128, width_mult=0.5, use_relational=True
    ).to(device)
    predictor = Predictor(emb_dim=128, hidden_dim=256).to(device)
    
    params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in predictor.parameters())
    print(f"Parameters: {params:,}")
    
    train_loader, test_loader = get_hidden_mass_loaders(
        mode='edge', num_train=args.num_train, num_test=args.num_test,
        batch_size=args.batch_size, randomize_appearance=True,
    )
    
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    
    print(f"\nSSL Training ({args.ssl_epochs} epochs)...")
    train_ssl_v1(encoder, predictor, train_loader, optimizer, device, args.ssl_epochs)
    
    print("\nExtracting features...")
    train_feats, train_labels = extract_features(encoder, train_loader, device)
    test_feats, test_labels = extract_features(encoder, test_loader, device)
    
    print(f"\nTraining probe ({args.probe_epochs} epochs)...")
    probe = train_probe(train_feats, train_labels, 128, device, args.probe_epochs)
    
    train_acc = evaluate_probe(probe, train_feats, train_labels, device)
    test_acc = evaluate_probe(probe, test_feats, test_labels, device)
    
    print(f"\nResults: Train {train_acc:.1f}%, Test {test_acc:.1f}%")
    
    return {'params': params, 'train_acc': train_acc, 'test_acc': test_acc}


def run_a_jepa_v2(device, args):
    """Run A-JEPA v2 (full cognitive architecture)."""
    print("\n" + "=" * 60)
    print("A-JEPA v2 (Cognitive Architecture)")
    print("=" * 60)
    print("  Features: Slot Attention, Multi-Timescale, Uncertainty, Sparsity")
    
    model = get_ajepa_v2(in_channels=1, img_size=32, config='default').to(device)
    
    params = sum(p.numel() for p in model.parameters())
    emb_dim = model.total_dim
    print(f"Parameters: {params:,}")
    print(f"Embedding dim: {emb_dim}")
    
    train_loader, test_loader = get_hidden_mass_loaders(
        mode='edge', num_train=args.num_train, num_test=args.num_test,
        batch_size=args.batch_size, randomize_appearance=True,
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    print(f"\nSSL Training ({args.ssl_epochs} epochs)...")
    train_ssl_v2(model, train_loader, optimizer, device, args.ssl_epochs)
    
    print("\nExtracting features...")
    train_feats, train_labels = extract_features(model.encoder, train_loader, device, is_v2=True)
    test_feats, test_labels = extract_features(model.encoder, test_loader, device, is_v2=True)
    
    print(f"\nTraining probe ({args.probe_epochs} epochs)...")
    probe = train_probe(train_feats, train_labels, emb_dim, device, args.probe_epochs)
    
    train_acc = evaluate_probe(probe, train_feats, train_labels, device)
    test_acc = evaluate_probe(probe, test_feats, test_labels, device)
    
    print(f"\nResults: Train {train_acc:.1f}%, Test {test_acc:.1f}%")
    
    return {'params': params, 'train_acc': train_acc, 'test_acc': test_acc}


def print_comparison(v_jepa, a_jepa_v1, a_jepa_v2):
    """Print comparison table."""
    print("\n")
    print("=" * 80)
    print("       Hidden Mass Inference: V-JEPA vs A-JEPA v1 vs A-JEPA v2")
    print("=" * 80)
    
    print(f"\n{'Metric':<30} {'V-JEPA':>15} {'A-JEPA v1':>15} {'A-JEPA v2':>15}")
    print("-" * 80)
    
    print(f"{'Parameters':<30} {v_jepa['params']:>15,} {a_jepa_v1['params']:>15,} {a_jepa_v2['params']:>15,}")
    print(f"{'Train Accuracy':<30} {v_jepa['train_acc']:>14.1f}% {a_jepa_v1['train_acc']:>14.1f}% {a_jepa_v2['train_acc']:>14.1f}%")
    print(f"{'Test Accuracy':<30} {v_jepa['test_acc']:>14.1f}% {a_jepa_v1['test_acc']:>14.1f}% {a_jepa_v2['test_acc']:>14.1f}%")
    
    print("-" * 80)
    
    # Efficiency comparison
    v_eff = v_jepa['test_acc'] / (v_jepa['params'] / 1000000)
    a1_eff = a_jepa_v1['test_acc'] / (a_jepa_v1['params'] / 1000000)
    a2_eff = a_jepa_v2['test_acc'] / (a_jepa_v2['params'] / 1000000)
    
    print(f"{'Acc/M params':<30} {v_eff:>14.1f}% {a1_eff:>14.1f}% {a2_eff:>14.1f}%")
    
    print("=" * 80)
    
    # Winner
    results = [
        ('V-JEPA', v_jepa['test_acc']),
        ('A-JEPA v1', a_jepa_v1['test_acc']),
        ('A-JEPA v2', a_jepa_v2['test_acc']),
    ]
    winner = max(results, key=lambda x: x[1])
    
    print(f"\n🏆 WINNER: {winner[0]} with {winner[1]:.1f}% test accuracy")
    
    # Analysis
    print("\nAnalysis:")
    if a_jepa_v2['test_acc'] > max(v_jepa['test_acc'], a_jepa_v1['test_acc']):
        print("  ✓ A-JEPA v2's cognitive features improve hidden property inference!")
        print("  ✓ Multi-timescale latents help separate fast (motion) from slow (mass) properties")
    elif a_jepa_v1['test_acc'] > v_jepa['test_acc']:
        print("  ✓ Edge-based representation helps, but v2 features didn't add benefit")
    else:
        print("  ✗ Visual features still dominate - may need more training or tuning")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Hidden Mass Test: V-JEPA vs A-JEPA v1 vs A-JEPA v2')
    parser.add_argument('--ssl_epochs', type=int, default=15)
    parser.add_argument('--probe_epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_train', type=int, default=300)
    parser.add_argument('--num_test', type=int, default=100)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\n" + "=" * 80)
    print("HIDDEN MASS INFERENCE: A-JEPA V2 TEST")
    print("=" * 80)
    print("\nComparing three architectures on inferring hidden mass from motion:")
    print("  • V-JEPA: Standard visual encoder")
    print("  • A-JEPA v1: Edge-based + relational reasoning")
    print("  • A-JEPA v2: Full cognitive architecture")
    print("      - Slot attention (object factorization)")
    print("      - Multi-timescale latents (fast/slow separation)")
    print("      - Uncertainty-aware predictions")
    print("      - Sparse bottleneck")
    
    # Run all three
    v_jepa = run_v_jepa(device, args)
    a_jepa_v1 = run_a_jepa_v1(device, args)
    a_jepa_v2 = run_a_jepa_v2(device, args)
    
    # Print comparison
    print_comparison(v_jepa, a_jepa_v1, a_jepa_v2)


if __name__ == '__main__':
    main()

