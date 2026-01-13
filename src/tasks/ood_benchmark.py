"""
OOD Benchmark: Finding Where A-JEPA Excels

Three experiments to test the "aphantasia advantage":

1. OOD Generalization (Object Count)
   - Train on 2 balls → Test on 2, 3, 4 balls
   - Hypothesis: A-JEPA generalizes better to unseen configurations

2. Data Efficiency
   - Train with 100%, 50%, 25%, 10% of data
   - Hypothesis: A-JEPA learns faster due to edge preprocessing

3. Corruption Robustness
   - Test with noise, blur, brightness shifts
   - Hypothesis: Edge features are stable under corruptions

Usage:
    python src/tasks/ood_benchmark.py --experiment all
    python src/tasks/ood_benchmark.py --experiment ood_count
    python src/tasks/ood_benchmark.py --experiment data_efficiency
    python src/tasks/ood_benchmark.py --experiment corruption
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models_v2 import get_ajepa_v2, get_vjepa_v2

# Visualization
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# Dataset with Configurable Ball Count and Corruptions
# =============================================================================

class OODDataset(Dataset):
    """
    Dataset for OOD experiments with configurable:
    - Number of balls (for OOD generalization)
    - Corruptions (noise, blur, brightness)
    """
    
    def __init__(
        self,
        num_samples: int,
        num_frames: int = 30,
        img_size: int = 32,
        mode: str = 'edge',
        num_balls: int = 2,
        corruption: str = None,  # 'noise', 'blur', 'brightness', 'combined'
        corruption_strength: float = 0.3,
        seed: int = None,
    ):
        self.num_samples = num_samples
        self.mode = mode
        self.corruption = corruption
        self.corruption_strength = corruption_strength
        
        if seed is not None:
            np.random.seed(seed)
        
        self.data = []
        for i in range(num_samples):
            mass_cat = 'light' if i % 2 == 0 else 'heavy'
            video, label = self._generate_video(num_frames, img_size, mass_cat, num_balls)
            
            # Apply corruption before edge detection (for raw) or after (for visualization)
            if corruption and mode == 'raw':
                video = self._apply_corruption(video)
            
            if mode == 'edge':
                video = self._apply_edge(video)
            
            self.data.append({'video': video, 'mass_label': label})
    
    def _generate_video(self, num_frames, img_size, mass_cat, num_balls):
        """Generate bouncing balls video."""
        import cv2
        
        if mass_cat == 'light':
            mass = np.random.uniform(0.5, 0.8)
            label = 0
        else:
            mass = np.random.uniform(1.5, 2.0)
            label = 1
        
        balls = []
        for b in range(num_balls):
            radius = np.random.randint(4, 7)
            ball_mass = mass if b == 0 else np.random.uniform(0.7, 1.5)
            
            ball = {
                'x': np.random.uniform(radius + 2, img_size - radius - 2),
                'y': np.random.uniform(radius + 2, img_size * 0.4),
                'vx': np.random.uniform(-1.5, 1.5) / ball_mass,
                'vy': np.random.uniform(0, 1) / ball_mass,
                'radius': radius,
                'color': 200,
                'mass': ball_mass,
                'restitution': 0.95 - (ball_mass - 0.5) * 0.2,
            }
            balls.append(ball)
        
        gravity = 0.15
        friction = 0.99
        
        frames = []
        for _ in range(num_frames):
            frame = np.zeros((img_size, img_size), dtype=np.uint8)
            
            for ball in balls:
                cv2.circle(frame, (int(ball['x']), int(ball['y'])),
                          ball['radius'], ball['color'], -1)
            
            frames.append(frame)
            
            for ball in balls:
                ball['vy'] += gravity
                ball['vx'] *= friction
                ball['vy'] *= friction
                ball['x'] += ball['vx']
                ball['y'] += ball['vy']
                
                r = ball['radius']
                if ball['x'] - r < 0:
                    ball['x'] = r
                    ball['vx'] = -ball['vx'] * ball['restitution']
                elif ball['x'] + r > img_size:
                    ball['x'] = img_size - r
                    ball['vx'] = -ball['vx'] * ball['restitution']
                
                if ball['y'] - r < 0:
                    ball['y'] = r
                    ball['vy'] = -ball['vy'] * ball['restitution']
                elif ball['y'] + r > img_size:
                    ball['y'] = img_size - r
                    ball['vy'] = -ball['vy'] * ball['restitution']
                    ball['vy'] *= max(0.4, 1.0 - (ball['mass'] - 0.5) * 0.15)
            
            # Ball-ball collisions
            if num_balls > 1:
                for i in range(len(balls)):
                    for j in range(i + 1, len(balls)):
                        b1, b2 = balls[i], balls[j]
                        dx = b2['x'] - b1['x']
                        dy = b2['y'] - b1['y']
                        dist = np.sqrt(dx**2 + dy**2)
                        min_dist = b1['radius'] + b2['radius']
                        
                        if dist < min_dist and dist > 0:
                            nx, ny = dx / dist, dy / dist
                            dvx = b1['vx'] - b2['vx']
                            dvy = b1['vy'] - b2['vy']
                            dvn = dvx * nx + dvy * ny
                            
                            if dvn > 0:
                                m1, m2 = b1['mass'], b2['mass']
                                restitution = min(b1['restitution'], b2['restitution'])
                                impulse = (1 + restitution) * dvn / (1/m1 + 1/m2)
                                
                                b1['vx'] -= impulse / m1 * nx
                                b1['vy'] -= impulse / m1 * ny
                                b2['vx'] += impulse / m2 * nx
                                b2['vy'] += impulse / m2 * ny
                            
                            overlap = min_dist - dist
                            b1['x'] -= overlap/2 * nx
                            b1['y'] -= overlap/2 * ny
                            b2['x'] += overlap/2 * nx
                            b2['y'] += overlap/2 * ny
        
        video = np.stack(frames, axis=0)[:, np.newaxis, :, :]
        video = video.astype(np.float32) / 255.0
        
        return video, label
    
    def _apply_edge(self, video):
        """Convert to edge representation."""
        import cv2
        T, C, H, W = video.shape
        edges = []
        for t in range(T):
            frame = (video[t, 0] * 255).astype(np.uint8)
            edge = cv2.Canny(frame, 30, 100)
            edges.append(edge.astype(np.float32) / 255.0)
        return np.stack(edges, axis=0)[:, np.newaxis, :, :]
    
    def _apply_corruption(self, video):
        """Apply test-time corruption."""
        import cv2
        
        if self.corruption == 'noise':
            noise = np.random.randn(*video.shape) * self.corruption_strength
            video = np.clip(video + noise, 0, 1).astype(np.float32)
            
        elif self.corruption == 'blur':
            T, C, H, W = video.shape
            ksize = int(5 * self.corruption_strength) * 2 + 1
            blurred = []
            for t in range(T):
                frame = (video[t, 0] * 255).astype(np.uint8)
                frame = cv2.GaussianBlur(frame, (ksize, ksize), 0)
                blurred.append(frame.astype(np.float32) / 255.0)
            video = np.stack(blurred, axis=0)[:, np.newaxis, :, :]
            
        elif self.corruption == 'brightness':
            factor = 1.0 + self.corruption_strength * (np.random.rand() * 2 - 1)
            video = np.clip(video * factor, 0, 1).astype(np.float32)
            
        elif self.corruption == 'combined':
            # Apply all corruptions
            noise = np.random.randn(*video.shape) * (self.corruption_strength * 0.5)
            video = np.clip(video + noise, 0, 1).astype(np.float32)
            
            T, C, H, W = video.shape
            ksize = int(3 * self.corruption_strength) * 2 + 1
            if ksize > 1:
                blurred = []
                for t in range(T):
                    frame = (video[t, 0] * 255).astype(np.uint8)
                    frame = cv2.GaussianBlur(frame, (ksize, ksize), 0)
                    blurred.append(frame.astype(np.float32) / 255.0)
                video = np.stack(blurred, axis=0)[:, np.newaxis, :, :]
            
            factor = 1.0 + (self.corruption_strength * 0.5) * (np.random.rand() * 2 - 1)
            video = np.clip(video * factor, 0, 1).astype(np.float32)
        
        return video
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'video': torch.from_numpy(item['video'].copy()),
            'mass_label': item['mass_label'],
        }


# =============================================================================
# VICReg Loss
# =============================================================================

def variance_loss(z, gamma=1.0):
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return torch.mean(F.relu(gamma - std))

def covariance_loss(z):
    B, D = z.shape
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (B - 1)
    off_diag = cov.flatten()[:-1].view(D-1, D+1)[:, 1:].flatten()
    return (off_diag ** 2).mean()

def vicreg_loss(pred, target):
    sim_loss = F.mse_loss(pred, target)
    var_loss = variance_loss(pred) + variance_loss(target)
    cov_loss = covariance_loss(pred) + covariance_loss(target)
    return 25.0 * sim_loss + 25.0 * var_loss + 1.0 * cov_loss


# =============================================================================
# Training and Evaluation
# =============================================================================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_model(model, train_loader, device, epochs=60, lr=1e-3):
    """Train model with VICReg loss."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            video = batch['video'].to(device)
            B, T, C, H, W = video.shape
            
            context = video[:, :T//2]
            target = video[:, T//2:]
            
            z_ctx = model.encode_video(context)
            z_tgt = model.encode_video(target)
            
            loss = vicreg_loss(z_ctx, z_tgt)
            
            if hasattr(model, 'encoder') and hasattr(model.encoder, 'bottleneck'):
                aux = model.encoder.bottleneck.get_aux_loss()
                if isinstance(aux, torch.Tensor):
                    loss = loss + aux
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return model


class LinearProbe(nn.Module):
    def __init__(self, emb_dim, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(emb_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)


def extract_features(model, dataloader, device):
    model.eval()
    all_feats, all_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            video = batch['video'].to(device)
            labels = batch['mass_label']
            feats = model.encode_video(video)
            all_feats.append(feats.cpu())
            all_labels.append(labels)
    
    return torch.cat(all_feats), torch.cat(all_labels)


def train_and_eval_probe(train_feats, train_labels, test_feats, test_labels, emb_dim, device, epochs=50):
    dataset = torch.utils.data.TensorDataset(train_feats, train_labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    probe = LinearProbe(emb_dim).to(device)
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
    
    probe.eval()
    test_feats, test_labels = test_feats.to(device), test_labels.to(device)
    
    with torch.no_grad():
        logits = probe(test_feats)
        preds = logits.argmax(dim=1)
        acc = (preds == test_labels).float().mean().item() * 100
    
    return acc


# =============================================================================
# Experiment 1: OOD Generalization (Object Count)
# =============================================================================

def run_ood_count_experiment(device, epochs=60, num_train=300, num_test=100):
    """
    Train on 2 balls, test on 2, 3, 4 balls.
    Measures generalization to unseen object counts.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: OOD GENERALIZATION (Object Count)")
    print("=" * 70)
    print("Train: 2 balls | Test: 2, 3, 4 balls")
    
    results = {'ajepa': {}, 'vjepa': {}}
    
    for model_name in ['ajepa', 'vjepa']:
        print(f"\n[{model_name.upper()}]")
        
        mode = 'edge' if model_name == 'ajepa' else 'raw'
        
        # Create training data (2 balls only)
        train_dataset = OODDataset(
            num_samples=num_train,
            mode=mode,
            num_balls=2,
            seed=42,
        )
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        
        # Create model and train
        set_seed(42)
        if model_name == 'ajepa':
            model = get_ajepa_v2(in_channels=1, img_size=32, config='default')
        else:
            model = get_vjepa_v2(in_channels=1, img_size=32, config='capacity_matched')
        
        print(f"  Training on 2 balls ({epochs} epochs)...")
        model = train_model(model, train_loader, device, epochs=epochs)
        
        # Extract training features for probe
        train_feats, train_labels = extract_features(model, train_loader, device)
        
        # Test on different ball counts
        for num_balls in [2, 3, 4]:
            test_dataset = OODDataset(
                num_samples=num_test,
                mode=mode,
                num_balls=num_balls,
                seed=999 + num_balls,
            )
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
            
            test_feats, test_labels = extract_features(model, test_loader, device)
            
            acc = train_and_eval_probe(
                train_feats, train_labels,
                test_feats, test_labels,
                model.total_dim, device, epochs=50
            )
            
            results[model_name][num_balls] = acc
            tag = "(ID)" if num_balls == 2 else "(OOD)"
            print(f"  {num_balls} balls {tag}: {acc:.1f}%")
    
    # Compute generalization gap
    print("\n" + "-" * 50)
    print("Generalization Gap (2 balls → 3-4 balls):")
    for model_name in ['ajepa', 'vjepa']:
        id_acc = results[model_name][2]
        ood_acc = (results[model_name][3] + results[model_name][4]) / 2
        gap = id_acc - ood_acc
        print(f"  {model_name.upper()}: {gap:.1f}% drop (ID: {id_acc:.1f}% → OOD: {ood_acc:.1f}%)")
    
    return results


# =============================================================================
# Experiment 2: Data Efficiency
# =============================================================================

def run_data_efficiency_experiment(device, epochs=60, full_train=400, num_test=100):
    """
    Train with varying amounts of data: 100%, 50%, 25%, 10%.
    Measures sample efficiency.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: DATA EFFICIENCY")
    print("=" * 70)
    print("Training data fractions: 100%, 50%, 25%, 10%")
    
    fractions = [1.0, 0.5, 0.25, 0.1]
    results = {'ajepa': {}, 'vjepa': {}}
    
    for model_name in ['ajepa', 'vjepa']:
        print(f"\n[{model_name.upper()}]")
        
        mode = 'edge' if model_name == 'ajepa' else 'raw'
        
        # Full training dataset
        full_dataset = OODDataset(
            num_samples=full_train,
            mode=mode,
            num_balls=2,
            seed=42,
        )
        
        # Test dataset
        test_dataset = OODDataset(
            num_samples=num_test,
            mode=mode,
            num_balls=2,
            seed=999,
        )
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        for frac in fractions:
            n_samples = int(full_train * frac)
            indices = list(range(n_samples))
            subset = Subset(full_dataset, indices)
            train_loader = DataLoader(subset, batch_size=16, shuffle=True)
            
            # Create and train model
            set_seed(42)
            if model_name == 'ajepa':
                model = get_ajepa_v2(in_channels=1, img_size=32, config='default')
            else:
                model = get_vjepa_v2(in_channels=1, img_size=32, config='capacity_matched')
            
            model = train_model(model, train_loader, device, epochs=epochs)
            
            # Extract features and evaluate
            train_feats, train_labels = extract_features(model, train_loader, device)
            test_feats, test_labels = extract_features(model, test_loader, device)
            
            acc = train_and_eval_probe(
                train_feats, train_labels,
                test_feats, test_labels,
                model.total_dim, device, epochs=50
            )
            
            results[model_name][frac] = acc
            print(f"  {int(frac*100):3d}% data ({n_samples:3d} samples): {acc:.1f}%")
    
    # Compute efficiency ratio
    print("\n" + "-" * 50)
    print("Data Efficiency (accuracy drop from 100% → 10%):")
    for model_name in ['ajepa', 'vjepa']:
        full_acc = results[model_name][1.0]
        min_acc = results[model_name][0.1]
        drop = full_acc - min_acc
        print(f"  {model_name.upper()}: {drop:.1f}% drop (100%: {full_acc:.1f}% → 10%: {min_acc:.1f}%)")
    
    return results


# =============================================================================
# Experiment 3: Corruption Robustness
# =============================================================================

def run_corruption_experiment(device, epochs=60, num_train=300, num_test=100):
    """
    Train on clean data, test with corruptions.
    Measures robustness to distribution shift.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: CORRUPTION ROBUSTNESS")
    print("=" * 70)
    print("Corruptions: clean, noise, blur, brightness, combined")
    
    corruptions = [None, 'noise', 'blur', 'brightness', 'combined']
    results = {'ajepa': {}, 'vjepa': {}}
    
    for model_name in ['ajepa', 'vjepa']:
        print(f"\n[{model_name.upper()}]")
        
        mode = 'edge' if model_name == 'ajepa' else 'raw'
        
        # Clean training data
        train_dataset = OODDataset(
            num_samples=num_train,
            mode=mode,
            num_balls=2,
            corruption=None,
            seed=42,
        )
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        
        # Train model on clean data
        set_seed(42)
        if model_name == 'ajepa':
            model = get_ajepa_v2(in_channels=1, img_size=32, config='default')
        else:
            model = get_vjepa_v2(in_channels=1, img_size=32, config='capacity_matched')
        
        print(f"  Training on clean data ({epochs} epochs)...")
        model = train_model(model, train_loader, device, epochs=epochs)
        
        # Extract training features for probe
        train_feats, train_labels = extract_features(model, train_loader, device)
        
        # Test on different corruptions
        for corruption in corruptions:
            test_dataset = OODDataset(
                num_samples=num_test,
                mode=mode,
                num_balls=2,
                corruption=corruption,
                corruption_strength=0.3,
                seed=999,
            )
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
            
            test_feats, test_labels = extract_features(model, test_loader, device)
            
            acc = train_and_eval_probe(
                train_feats, train_labels,
                test_feats, test_labels,
                model.total_dim, device, epochs=50
            )
            
            corr_name = corruption or 'clean'
            results[model_name][corr_name] = acc
            print(f"  {corr_name:12s}: {acc:.1f}%")
    
    # Compute robustness
    print("\n" + "-" * 50)
    print("Corruption Robustness (clean → combined):")
    for model_name in ['ajepa', 'vjepa']:
        clean_acc = results[model_name]['clean']
        comb_acc = results[model_name]['combined']
        drop = clean_acc - comb_acc
        print(f"  {model_name.upper()}: {drop:.1f}% drop (clean: {clean_acc:.1f}% → combined: {comb_acc:.1f}%)")
    
    return results


# =============================================================================
# Plotting
# =============================================================================

def plot_all_results(ood_results, efficiency_results, corruption_results, save_path):
    """Create publication-quality plots for all experiments."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    colors = {'ajepa': '#2ecc71', 'vjepa': '#e74c3c'}
    
    # Plot 1: OOD Count
    ax1 = axes[0]
    for model_name in ['ajepa', 'vjepa']:
        x = [2, 3, 4]
        y = [ood_results[model_name][n] for n in x]
        ax1.plot(x, y, 'o-', color=colors[model_name], label=model_name.upper(), linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Balls', fontsize=11)
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_title('OOD Generalization\n(Train: 2 balls)', fontsize=12)
    ax1.axhline(50, color='gray', linestyle='--', alpha=0.5)
    ax1.legend()
    ax1.set_xticks([2, 3, 4])
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Data Efficiency
    ax2 = axes[1]
    for model_name in ['ajepa', 'vjepa']:
        x = [10, 25, 50, 100]
        y = [efficiency_results[model_name][f/100] for f in x]
        ax2.plot(x, y, 'o-', color=colors[model_name], label=model_name.upper(), linewidth=2, markersize=8)
    ax2.set_xlabel('Training Data (%)', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('Data Efficiency', fontsize=12)
    ax2.axhline(50, color='gray', linestyle='--', alpha=0.5)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Corruption Robustness
    ax3 = axes[2]
    corruptions = ['clean', 'noise', 'blur', 'brightness', 'combined']
    x = range(len(corruptions))
    width = 0.35
    
    for i, model_name in enumerate(['ajepa', 'vjepa']):
        y = [corruption_results[model_name][c] for c in corruptions]
        offset = -width/2 if i == 0 else width/2
        ax3.bar([xi + offset for xi in x], y, width, color=colors[model_name], label=model_name.upper(), alpha=0.8)
    
    ax3.set_xlabel('Corruption Type', fontsize=11)
    ax3.set_ylabel('Accuracy (%)', fontsize=11)
    ax3.set_title('Corruption Robustness\n(Train: clean)', fontsize=12)
    ax3.axhline(50, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(corruptions, rotation=30, ha='right')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='OOD Benchmark')
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', 'ood_count', 'data_efficiency', 'corruption'],
                        help='Which experiment to run')
    parser.add_argument('--epochs', type=int, default=60, help='Training epochs')
    parser.add_argument('--num_train', type=int, default=300, help='Training samples')
    parser.add_argument('--num_test', type=int, default=100, help='Test samples')
    parser.add_argument('--output_dir', type=str, default='results/ood_benchmark')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("OOD BENCHMARK: Finding Where A-JEPA Excels")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Train samples: {args.num_train}")
    print(f"Test samples: {args.num_test}")
    
    results = {}
    
    if args.experiment in ['all', 'ood_count']:
        results['ood_count'] = run_ood_count_experiment(
            device, args.epochs, args.num_train, args.num_test
        )
    
    if args.experiment in ['all', 'data_efficiency']:
        results['data_efficiency'] = run_data_efficiency_experiment(
            device, args.epochs, args.num_train, args.num_test
        )
    
    if args.experiment in ['all', 'corruption']:
        results['corruption'] = run_corruption_experiment(
            device, args.epochs, args.num_train, args.num_test
        )
    
    # Save results
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot if all experiments run
    if args.experiment == 'all' and HAS_MATPLOTLIB:
        plot_all_results(
            results['ood_count'],
            results['data_efficiency'],
            results['corruption'],
            os.path.join(args.output_dir, 'ood_benchmark_plot.png')
        )
        print(f"\nPlot saved to {args.output_dir}/ood_benchmark_plot.png")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Where Does A-JEPA Excel?")
    print("=" * 70)
    
    if 'ood_count' in results:
        a_gap = results['ood_count']['ajepa'][2] - (results['ood_count']['ajepa'][3] + results['ood_count']['ajepa'][4]) / 2
        v_gap = results['ood_count']['vjepa'][2] - (results['ood_count']['vjepa'][3] + results['ood_count']['vjepa'][4]) / 2
        winner = "A-JEPA" if a_gap < v_gap else "V-JEPA"
        print(f"\n1. OOD Generalization: {winner} wins (smaller gap is better)")
        print(f"   A-JEPA gap: {a_gap:.1f}% | V-JEPA gap: {v_gap:.1f}%")
    
    if 'data_efficiency' in results:
        a_drop = results['data_efficiency']['ajepa'][1.0] - results['data_efficiency']['ajepa'][0.1]
        v_drop = results['data_efficiency']['vjepa'][1.0] - results['data_efficiency']['vjepa'][0.1]
        winner = "A-JEPA" if a_drop < v_drop else "V-JEPA"
        print(f"\n2. Data Efficiency: {winner} wins (smaller drop is better)")
        print(f"   A-JEPA drop: {a_drop:.1f}% | V-JEPA drop: {v_drop:.1f}%")
    
    if 'corruption' in results:
        a_drop = results['corruption']['ajepa']['clean'] - results['corruption']['ajepa']['combined']
        v_drop = results['corruption']['vjepa']['clean'] - results['corruption']['vjepa']['combined']
        winner = "A-JEPA" if a_drop < v_drop else "V-JEPA"
        print(f"\n3. Corruption Robustness: {winner} wins (smaller drop is better)")
        print(f"   A-JEPA drop: {a_drop:.1f}% | V-JEPA drop: {v_drop:.1f}%")
    
    print("\n" + "=" * 70)
    print("OOD Benchmark complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()

