"""
Hidden Mass Inference Test (V2 - Appearance Blind)

Tests whether the model can extract an unseen causal property (mass) from
motion dynamics alone, with ZERO visual cues.

V2 Improvements (GPT-suggested):
- Randomized ball appearance per sample (texture, color, size)
- Appearance is UNCORRELATED with mass category
- Light balls: fast initial velocity, high bounce, get pushed in collisions
- Heavy balls: slow initial velocity, low bounce, dominate collisions
- Frame normalization to remove brightness artifacts

Protocol:
1. Self-supervised training: Train V-JEPA and A-JEPA on videos where ball
   appearance is RANDOMIZED but mass affects physics.
2. Linear probe: Freeze encoder, train linear classifier to predict mass
   category (light vs heavy) from latent embeddings.
3. Compare: Which model implicitly captured the hidden physics variable?

Hypothesis: A-JEPA will perform better because it encodes abstract relational
structure (motion patterns, collision dynamics) instead of pixel-level features.
V-JEPA cannot cheat via visual cues since appearance is randomized.
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.datasets.hidden_mass import get_hidden_mass_loaders, HiddenMassDataset
from src.models import SmallEncoder, Predictor
from torch.utils.data import DataLoader


class TemporalEncoder(nn.Module):
    """Encoder that processes video and aggregates across time."""
    
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
        """Encode full video and pool across time. video: (B, T, C, H, W)"""
        B, T, C, H, W = video.shape
        frames = video.contiguous().view(B * T, C, H, W)
        embeddings = self.encoder(frames)  # (B*T, D)
        embeddings = embeddings.view(B, T, -1)  # (B, T, D)
        return embeddings.mean(dim=1)  # (B, D) - temporal average


class LinearProbe(nn.Module):
    """Linear classifier for probing learned representations."""
    
    def __init__(self, emb_dim, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(emb_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)


def get_models(variant):
    """Create encoder and predictor for specified variant."""
    if variant == 'v_jepa':
        encoder = TemporalEncoder(
            in_channels=1,
            emb_dim=256,
            width_mult=1.0,
            use_relational=False,
        )
        emb_dim = 256
    else:  # a_jepa
        encoder = TemporalEncoder(
            in_channels=1,
            emb_dim=128,
            width_mult=0.5,
            use_relational=True,
        )
        emb_dim = 128
    
    predictor = Predictor(emb_dim=emb_dim, hidden_dim=emb_dim * 2)
    return encoder, predictor, emb_dim


def cosine_loss(pred, target):
    """Negative cosine similarity loss."""
    pred = nn.functional.normalize(pred, dim=-1)
    target = nn.functional.normalize(target, dim=-1)
    return -torch.mean(torch.sum(pred * target, dim=-1))


def self_supervised_train_epoch(encoder, predictor, dataloader, optimizer, device):
    """
    Self-supervised training: predict future frame embedding from past frames.
    No mass labels used - purely self-supervised.
    """
    encoder.train()
    predictor.train()
    
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc='SSL Training', leave=False):
        video = batch['video'].to(device)  # (B, T, 1, H, W)
        B, T, C, H, W = video.shape
        
        # Context: first half of video
        context = video[:, :T//2]
        # Target: second half
        target = video[:, T//2:]
        
        # Encode context
        context_emb = encoder.encode_video(context)
        
        # Encode target (stop gradient)
        with torch.no_grad():
            target_emb = encoder.encode_video(target)
        
        # Predict target from context
        pred_emb = predictor(context_emb)
        
        # Loss
        loss = cosine_loss(pred_emb, target_emb)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def extract_features(encoder, dataloader, device):
    """Extract features and labels from dataset using frozen encoder."""
    encoder.eval()
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Extracting', leave=False):
            video = batch['video'].to(device)
            labels = batch['mass_label']
            
            features = encoder.encode_video(video)
            all_features.append(features.cpu())
            all_labels.append(labels)
    
    return torch.cat(all_features), torch.cat(all_labels)


def train_linear_probe(features, labels, emb_dim, device, epochs=50):
    """Train linear probe on extracted features."""
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


def run_experiment(variant, device, args):
    """Run full hidden mass inference experiment for one variant."""
    print(f"\n{'='*60}")
    print(f"Running {variant.upper()}")
    print(f"{'='*60}")
    
    mode = 'raw' if variant == 'v_jepa' else 'edge'
    
    # Create models
    encoder, predictor, emb_dim = get_models(variant)
    encoder = encoder.to(device)
    predictor = predictor.to(device)
    
    total_params = sum(p.numel() for p in encoder.parameters())
    total_params += sum(p.numel() for p in predictor.parameters())
    print(f"Total params: {total_params:,}")
    
    # Load data (V2: randomized appearance, only physics differs)
    train_loader, test_loader = get_hidden_mass_loaders(
        mode=mode,
        num_train=args.num_train,
        num_test=args.num_test,
        num_balls=args.num_balls,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        randomize_appearance=True,  # V2: appearance uncorrelated with mass
    )
    
    # Phase 1: Self-supervised training (no labels!)
    print(f"\nPhase 1: Self-supervised training ({args.ssl_epochs} epochs)...")
    print("         (Learning to predict future states, no mass labels)")
    
    params = list(encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    
    for epoch in range(args.ssl_epochs):
        loss = self_supervised_train_epoch(encoder, predictor, train_loader, optimizer, device)
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{args.ssl_epochs} - SSL Loss: {loss:.4f}")
    
    # Phase 2: Extract features with frozen encoder
    print(f"\nPhase 2: Extracting features with frozen encoder...")
    
    train_features, train_labels = extract_features(encoder, train_loader, device)
    test_features, test_labels = extract_features(encoder, test_loader, device)
    
    print(f"  Train features: {train_features.shape}")
    print(f"  Test features: {test_features.shape}")
    print(f"  Train label balance: {train_labels.sum().item()}/{len(train_labels)} heavy")
    
    # Phase 3: Train linear probe
    print(f"\nPhase 3: Training linear probe ({args.probe_epochs} epochs)...")
    
    probe = train_linear_probe(train_features, train_labels, emb_dim, device, args.probe_epochs)
    
    # Phase 4: Evaluate
    train_acc = evaluate_probe(probe, train_features, train_labels, device)
    test_acc = evaluate_probe(probe, test_features, test_labels, device)
    
    print(f"\n  Train Accuracy: {train_acc:.2f}%")
    print(f"  Test Accuracy:  {test_acc:.2f}%")
    
    # Baseline: random chance is 50%
    print(f"  (Random baseline: 50.00%)")
    
    return {
        'variant': variant,
        'params': total_params,
        'train_acc': train_acc,
        'test_acc': test_acc,
    }


def print_comparison(v_results, a_results):
    """Print side-by-side comparison."""
    print("\n")
    print("=" * 75)
    print("       Hidden Mass Inference V2 (Appearance Blind): V-JEPA vs A-JEPA")
    print("=" * 75)
    
    print(f"\n{'Metric':<40} {'V-JEPA':>15} {'A-JEPA':>15}")
    print("-" * 75)
    
    print(f"{'Parameters':<40} {v_results['params']:>15,} {a_results['params']:>15,}")
    
    print("-" * 75)
    print("Linear Probe Accuracy (chance = 50%):")
    
    print(f"{'  Train Accuracy':<40} {v_results['train_acc']:>14.2f}% {a_results['train_acc']:>14.2f}%")
    print(f"{'  Test Accuracy':<40} {v_results['test_acc']:>14.2f}% {a_results['test_acc']:>14.2f}%")
    
    print("=" * 75)
    
    # Analysis
    print("\nKey Findings:")
    
    param_ratio = v_results['params'] / a_results['params']
    print(f"  - V-JEPA has {param_ratio:.1f}x more parameters than A-JEPA")
    
    # Who won?
    v_above_chance = v_results['test_acc'] - 50
    a_above_chance = a_results['test_acc'] - 50
    
    if a_results['test_acc'] > v_results['test_acc']:
        improvement = a_results['test_acc'] - v_results['test_acc']
        print(f"  - A-JEPA achieves {improvement:.1f}pp higher accuracy on hidden mass inference")
        print(f"  - A-JEPA encodes dynamics/physics better than V-JEPA")
    else:
        print(f"  - V-JEPA achieves higher accuracy on hidden mass inference")
    
    # Significance
    if max(v_above_chance, a_above_chance) > 10:
        print(f"  - Model successfully extracted hidden causal property from motion alone!")
    elif max(v_above_chance, a_above_chance) > 5:
        print(f"  - Weak signal detected - model partially captured mass dynamics")
    else:
        print(f"  - No significant signal - may need more training or harder task")
    
    print("\n" + "=" * 75)
    if a_results['test_acc'] > v_results['test_acc'] and a_above_chance > 5:
        print("VERDICT: A-JEPA better captures hidden physical variables from dynamics!")
        print("         Abstract representations encode causal structure, not pixels.")
    elif v_results['test_acc'] > a_results['test_acc'] and v_above_chance > 5:
        print("VERDICT: V-JEPA better captures hidden physical variables.")
    else:
        print("VERDICT: Inconclusive - both models show similar performance.")
    print("=" * 75)


def parse_args():
    parser = argparse.ArgumentParser(description='Hidden Mass Inference Test')
    parser.add_argument('--ssl_epochs', type=int, default=25, help='Self-supervised training epochs')
    parser.add_argument('--probe_epochs', type=int, default=50, help='Linear probe training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_train', type=int, default=500, help='Training videos')
    parser.add_argument('--num_test', type=int, default=100, help='Test videos')
    parser.add_argument('--num_balls', type=int, default=2, help='Number of balls')
    parser.add_argument('--num_workers', type=int, default=2, help='Dataloader workers')
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"\n" + "=" * 75)
    print("HIDDEN MASS INFERENCE TEST (V2 - APPEARANCE BLIND)")
    print("=" * 75)
    print(f"\nGoal: Can the model infer HIDDEN mass from motion dynamics alone?")
    print(f"      Ball appearance is RANDOMIZED and UNCORRELATED with mass.")
    print(f"      V-JEPA cannot cheat via visual cues!")
    print(f"\nV2 Improvements:")
    print(f"  ✓ Randomized texture/color/size per sample")
    print(f"  ✓ Light balls: fast, bouncy, pushed in collisions")
    print(f"  ✓ Heavy balls: slow, sticky, dominate collisions")
    print(f"  ✓ Frame normalization (no brightness bias)")
    print(f"\nProtocol:")
    print(f"  1. Self-supervised training (no labels)")
    print(f"  2. Freeze encoder")
    print(f"  3. Linear probe to classify: light vs heavy")
    print(f"\nSettings:")
    print(f"  SSL epochs: {args.ssl_epochs}")
    print(f"  Probe epochs: {args.probe_epochs}")
    print(f"  Training videos: {args.num_train}")
    
    # Run both variants
    v_results = run_experiment('v_jepa', device, args)
    a_results = run_experiment('a_jepa', device, args)
    
    # Print comparison
    print_comparison(v_results, a_results)


if __name__ == '__main__':
    main()

