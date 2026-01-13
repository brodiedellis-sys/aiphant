"""
Rigorous Benchmark for A-JEPA v2 vs V-JEPA v2

Implements reviewer feedback for publishable results:
1. Multi-seed training (5 seeds) for mean ± std
2. Capacity-matched experiments
3. Evidence bundles (config, git commit, dataset hash, metrics, checkpoints)
4. Drift vs horizon plots with error bars
5. Ablation support

Usage:
    python src/tasks/rigorous_benchmark.py --seeds 5 --epochs 60
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
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
# Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """Full experiment configuration for reproducibility."""
    # Model
    model_type: str  # 'ajepa' or 'vjepa'
    model_config: str  # 'default', 'capacity_matched', etc.
    
    # Training
    num_epochs: int = 60
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Data
    num_train: int = 400
    num_test: int = 150
    num_frames: int = 30
    img_size: int = 32
    
    # Probe
    probe_epochs: int = 50
    
    # Seed
    seed: int = 42
    
    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# Evidence Bundle
# =============================================================================

def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        ).decode().strip()
    except:
        return "unknown"


def compute_dataset_hash(data: np.ndarray) -> str:
    """Compute SHA256 hash of dataset."""
    return hashlib.sha256(data.tobytes()).hexdigest()[:16]


class EvidenceBundle:
    """Saves all reproducibility artifacts for an experiment."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.metrics = {
            'train_losses': [],
            'probe_accuracies': [],
            'drift_curves': {},
            'timestamps': [],
        }
        
    def save_config(self, config: ExperimentConfig):
        """Save experiment configuration."""
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
    
    def save_git_commit(self):
        """Save git commit hash."""
        commit = get_git_commit()
        with open(os.path.join(self.output_dir, 'git_commit.txt'), 'w') as f:
            f.write(commit)
    
    def save_dataset_hash(self, data: np.ndarray):
        """Save dataset hash."""
        hash_val = compute_dataset_hash(data)
        with open(os.path.join(self.output_dir, 'dataset_hash.txt'), 'w') as f:
            f.write(hash_val)
    
    def log_train_loss(self, epoch: int, loss: float):
        """Log training loss."""
        self.metrics['train_losses'].append({'epoch': epoch, 'loss': loss})
        self.metrics['timestamps'].append(datetime.now().isoformat())
    
    def log_probe_accuracy(self, accuracy: float):
        """Log probe accuracy."""
        self.metrics['probe_accuracies'].append(accuracy)
    
    def log_drift_curve(self, model_name: str, curve: Dict[int, float]):
        """Log drift vs horizon curve."""
        self.metrics['drift_curves'][model_name] = curve
    
    def save_checkpoint(self, model: nn.Module, name: str = 'model'):
        """Save model checkpoint."""
        torch.save(model.state_dict(), os.path.join(self.output_dir, f'{name}.pt'))
    
    def save_metrics(self):
        """Save all metrics."""
        with open(os.path.join(self.output_dir, 'metrics.json'), 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def finalize(self):
        """Save all remaining data."""
        self.save_metrics()


# =============================================================================
# Dataset
# =============================================================================

class HiddenMassDataset(Dataset):
    """
    Bouncing balls dataset with hidden mass property.
    Same visual appearance, different physics.
    """
    
    def __init__(
        self,
        num_samples: int,
        num_frames: int = 30,
        img_size: int = 32,
        mode: str = 'edge',  # 'edge' or 'raw'
        num_balls: int = 2,
        seed: int = None,
    ):
        self.num_samples = num_samples
        self.mode = mode
        
        if seed is not None:
            np.random.seed(seed)
        
        self.data = []
        for i in range(num_samples):
            mass_cat = 'light' if i % 2 == 0 else 'heavy'
            video, label = self._generate_video(num_frames, img_size, mass_cat, num_balls)
            
            if mode == 'edge':
                video = self._apply_edge(video)
            
            self.data.append({'video': video, 'mass_label': label})
    
    def _generate_video(self, num_frames, img_size, mass_cat, num_balls):
        """Generate bouncing balls video."""
        import cv2
        
        # Mass assignment
        if mass_cat == 'light':
            mass = np.random.uniform(0.5, 0.8)
            label = 0
        else:
            mass = np.random.uniform(1.5, 2.0)
            label = 1
        
        # Create balls
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
            
            # Physics step
            for ball in balls:
                ball['vy'] += gravity
                ball['vx'] *= friction
                ball['vy'] *= friction
                ball['x'] += ball['vx']
                ball['y'] += ball['vy']
                
                # Wall bounces
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
        
        # Normalize
        video = (video - video.mean()) / (video.std() + 1e-6)
        video = np.clip(video, -3, 3)
        video = (video - video.min()) / (video.max() - video.min() + 1e-6)
        
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
    
    def get_raw_data(self) -> np.ndarray:
        """Return raw data for hashing."""
        return np.stack([d['video'] for d in self.data])
    
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
    """Variance regularization: prevent collapse."""
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return torch.mean(F.relu(gamma - std))


def covariance_loss(z):
    """Covariance regularization: decorrelate features."""
    B, D = z.shape
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (B - 1)
    off_diag = cov.flatten()[:-1].view(D-1, D+1)[:, 1:].flatten()
    return (off_diag ** 2).mean()


def vicreg_loss(pred, target, sim_weight=25.0, var_weight=25.0, cov_weight=1.0):
    """Combined VICReg loss."""
    sim_loss = F.mse_loss(pred, target)
    var_loss = variance_loss(pred) + variance_loss(target)
    cov_loss = covariance_loss(pred) + covariance_loss(target)
    total = sim_weight * sim_loss + var_weight * var_loss + cov_weight * cov_loss
    return total


# =============================================================================
# Training
# =============================================================================

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    bundle: EvidenceBundle,
) -> nn.Module:
    """Train model with VICReg loss."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    model.train()
    pbar = tqdm(range(config.num_epochs), desc='Training')
    
    for epoch in pbar:
        total_loss = 0
        
        for batch in train_loader:
            video = batch['video'].to(device)
            B, T, C, H, W = video.shape
            
            context = video[:, :T//2]
            target = video[:, T//2:]
            
            z_ctx = model.encode_video(context)
            z_tgt = model.encode_video(target)
            
            loss = vicreg_loss(z_ctx, z_tgt)
            
            # Add auxiliary loss (sparsity) if available
            if hasattr(model, 'encoder') and hasattr(model.encoder, 'bottleneck'):
                aux = model.encoder.bottleneck.get_aux_loss()
                if isinstance(aux, torch.Tensor):
                    loss = loss + aux
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        bundle.log_train_loss(epoch, avg_loss)
        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    return model


# =============================================================================
# Evaluation
# =============================================================================

class LinearProbe(nn.Module):
    def __init__(self, emb_dim, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(emb_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)


def extract_features(model, dataloader, device):
    """Extract features from frozen encoder."""
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


def train_and_eval_probe(
    train_feats, train_labels,
    test_feats, test_labels,
    emb_dim: int,
    device: torch.device,
    epochs: int = 50,
) -> float:
    """Train probe and evaluate."""
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
# Drift Computation
# =============================================================================

def compute_drift_curve(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    horizons: List[int] = [1, 2, 3, 5, 7, 10],
) -> Dict[int, float]:
    """
    Compute similarity vs prediction horizon.
    
    For each horizon h:
        - Encode frame t
        - Encode frame t+h
        - Compute cosine similarity
    """
    model.eval()
    
    results = {h: [] for h in horizons}
    
    with torch.no_grad():
        for item in dataset:
            video = item['video'].unsqueeze(0).to(device)  # (1, T, C, H, W)
            T = video.shape[1]
            
            # Encode all frames
            frames = video.squeeze(0)  # (T, C, H, W)
            z_frames = []
            for t in range(T):
                z = model.encoder.encode_frame(frames[t:t+1])
                z_frames.append(z)
            z_frames = torch.cat(z_frames, dim=0)  # (T, D)
            
            # Compute similarity for each horizon
            for h in horizons:
                if h >= T:
                    continue
                
                z_t = z_frames[:T-h]
                z_th = z_frames[h:]
                
                # Cosine similarity
                sim = F.cosine_similarity(z_t, z_th, dim=-1).mean().item()
                results[h].append(sim)
    
    # Aggregate
    return {h: np.mean(sims) if sims else 0 for h, sims in results.items()}


# =============================================================================
# Main Benchmark
# =============================================================================

class RigorousBenchmark:
    """
    Run rigorous multi-seed benchmark with capacity-matched controls.
    """
    
    def __init__(
        self,
        output_dir: str,
        seeds: List[int] = [42, 123, 456, 789, 1337],
        device: torch.device = None,
    ):
        self.output_dir = output_dir
        self.seeds = seeds
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Model configurations to test
        self.model_configs = [
            ('ajepa', 'default'),           # A-JEPA ~180K
            ('ajepa', 'capacity_matched'),  # A-JEPA ~1.6M
            ('vjepa', 'default'),           # V-JEPA ~1.6M
            ('vjepa', 'capacity_matched'),  # V-JEPA ~180K
        ]
    
    def create_model(self, model_type: str, model_config: str) -> nn.Module:
        """Create model based on type and config."""
        if model_type == 'ajepa':
            return get_ajepa_v2(in_channels=1, img_size=32, config=model_config)
        else:
            return get_vjepa_v2(in_channels=1, img_size=32, config=model_config)
    
    def run_single_experiment(
        self,
        config: ExperimentConfig,
    ) -> Dict:
        """Run a single experiment with one seed."""
        # Set seed
        set_seed(config.seed)
        
        # Create output directory
        exp_dir = os.path.join(
            self.output_dir,
            f"seed_{config.seed}",
            f"{config.model_type}_{config.model_config}",
        )
        bundle = EvidenceBundle(exp_dir)
        bundle.save_config(config)
        bundle.save_git_commit()
        
        # Create datasets
        mode = 'edge' if config.model_type == 'ajepa' else 'raw'
        
        train_dataset = HiddenMassDataset(
            num_samples=config.num_train,
            num_frames=config.num_frames,
            img_size=config.img_size,
            mode=mode,
            seed=config.seed,
        )
        
        test_dataset = HiddenMassDataset(
            num_samples=config.num_test,
            num_frames=config.num_frames,
            img_size=config.img_size,
            mode=mode,
            seed=config.seed + 1000,  # Different seed for test
        )
        
        bundle.save_dataset_hash(train_dataset.get_raw_data())
        
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        
        # Create model
        model = self.create_model(config.model_type, config.model_config)
        params = sum(p.numel() for p in model.parameters())
        
        # Train
        model = train_model(model, train_loader, config, self.device, bundle)
        bundle.save_checkpoint(model, 'encoder')
        
        # Extract features
        train_feats, train_labels = extract_features(model, train_loader, self.device)
        test_feats, test_labels = extract_features(model, test_loader, self.device)
        
        # Train probe and evaluate
        accuracy = train_and_eval_probe(
            train_feats, train_labels,
            test_feats, test_labels,
            model.total_dim,
            self.device,
            epochs=config.probe_epochs,
        )
        bundle.log_probe_accuracy(accuracy)
        
        # Compute drift curve
        drift_curve = compute_drift_curve(model, test_dataset, self.device)
        bundle.log_drift_curve(f"{config.model_type}_{config.model_config}", drift_curve)
        
        # Finalize
        bundle.finalize()
        
        return {
            'model_type': config.model_type,
            'model_config': config.model_config,
            'seed': config.seed,
            'params': params,
            'accuracy': accuracy,
            'drift_curve': drift_curve,
        }
    
    def run_all(
        self,
        num_epochs: int = 60,
        batch_size: int = 16,
        num_train: int = 400,
        num_test: int = 150,
    ) -> Dict:
        """Run all experiments across all seeds and model configs."""
        all_results = []
        
        total = len(self.seeds) * len(self.model_configs)
        pbar = tqdm(total=total, desc='All Experiments')
        
        for seed in self.seeds:
            for model_type, model_config in self.model_configs:
                config = ExperimentConfig(
                    model_type=model_type,
                    model_config=model_config,
                    seed=seed,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    num_train=num_train,
                    num_test=num_test,
                )
                
                result = self.run_single_experiment(config)
                all_results.append(result)
                pbar.update(1)
                
                print(f"\n  {model_type} ({model_config}), seed={seed}: {result['accuracy']:.1f}%")
        
        pbar.close()
        
        # Aggregate results
        summary = self.aggregate_results(all_results)
        
        # Save summary
        with open(os.path.join(self.output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def aggregate_results(self, all_results: List[Dict]) -> Dict:
        """Aggregate results across seeds."""
        summary = {}
        
        for model_type, model_config in self.model_configs:
            key = f"{model_type}_{model_config}"
            
            # Filter results for this config
            results = [r for r in all_results
                       if r['model_type'] == model_type and r['model_config'] == model_config]
            
            if not results:
                continue
            
            accuracies = [r['accuracy'] for r in results]
            
            # Aggregate drift curves
            all_horizons = set()
            for r in results:
                all_horizons.update(r['drift_curve'].keys())
            
            drift_mean = {}
            drift_std = {}
            for h in sorted(all_horizons):
                values = [r['drift_curve'].get(h, 0) for r in results]
                drift_mean[h] = np.mean(values)
                drift_std[h] = np.std(values)
            
            summary[key] = {
                'params': results[0]['params'],
                'accuracy_mean': np.mean(accuracies),
                'accuracy_std': np.std(accuracies),
                'accuracy_min': np.min(accuracies),
                'accuracy_max': np.max(accuracies),
                'num_seeds': len(results),
                'drift_mean': drift_mean,
                'drift_std': drift_std,
            }
        
        return summary
    
    def plot_results(self, summary: Dict, save_path: str = None):
        """Create publication-quality plots."""
        if not HAS_MATPLOTLIB:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Accuracy with error bars
        ax1 = axes[0]
        
        names = []
        means = []
        stds = []
        params = []
        
        for key, data in summary.items():
            names.append(key.replace('_', '\n'))
            means.append(data['accuracy_mean'])
            stds.append(data['accuracy_std'])
            params.append(data['params'])
        
        colors = ['#2ecc71', '#27ae60', '#e74c3c', '#c0392b']
        x = np.arange(len(names))
        
        bars = ax1.bar(x, means, yerr=stds, capsize=5, color=colors[:len(names)], alpha=0.8)
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, fontsize=9)
        ax1.set_title('Hidden Mass Inference\n(mean ± std across seeds)', fontsize=14)
        ax1.axhline(50, color='gray', linestyle='--', alpha=0.5, label='Chance')
        
        # Add param counts above bars
        for i, (bar, p) in enumerate(zip(bars, params)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[i] + 1,
                     f'{p/1e6:.2f}M', ha='center', fontsize=8)
        
        ax1.legend()
        
        # Plot 2: Drift curves with error bands
        ax2 = axes[1]
        
        colors = {'ajepa_default': '#2ecc71', 'ajepa_capacity_matched': '#27ae60',
                  'vjepa_default': '#e74c3c', 'vjepa_capacity_matched': '#c0392b'}
        labels = {'ajepa_default': 'A-JEPA (180K)', 'ajepa_capacity_matched': 'A-JEPA (1.6M)',
                  'vjepa_default': 'V-JEPA (1.6M)', 'vjepa_capacity_matched': 'V-JEPA (180K)'}
        
        for key, data in summary.items():
            if 'drift_mean' not in data:
                continue
            
            horizons = sorted(data['drift_mean'].keys())
            means = [data['drift_mean'][h] for h in horizons]
            stds = [data['drift_std'][h] for h in horizons]
            
            color = colors.get(key, '#333333')
            label = labels.get(key, key)
            
            ax2.plot(horizons, means, 'o-', color=color, label=label, linewidth=2)
            ax2.fill_between(horizons,
                             [m - s for m, s in zip(means, stds)],
                             [m + s for m, s in zip(means, stds)],
                             color=color, alpha=0.2)
        
        ax2.set_xlabel('Prediction Horizon (frames)', fontsize=12)
        ax2.set_ylabel('Cosine Similarity', fontsize=12)
        ax2.set_title('Representation Drift\n(mean ± std across seeds)', fontsize=14)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Rigorous Benchmark')
    parser.add_argument('--seeds', type=int, default=5, help='Number of seeds (uses [42, 123, 456, 789, 1337])')
    parser.add_argument('--epochs', type=int, default=60, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_train', type=int, default=400)
    parser.add_argument('--num_test', type=int, default=150)
    parser.add_argument('--output_dir', type=str, default='results/rigorous')
    args = parser.parse_args()
    
    # Select seeds based on argument
    all_seeds = [42, 123, 456, 789, 1337]
    seeds = all_seeds[:args.seeds]
    
    print("=" * 70)
    print("RIGOROUS BENCHMARK: Capacity-Matched Multi-Seed Experiments")
    print("=" * 70)
    print(f"\nSeeds: {seeds}")
    print(f"Epochs: {args.epochs}")
    print(f"Training samples: {args.num_train}")
    print(f"Test samples: {args.num_test}")
    print(f"Output: {args.output_dir}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Run benchmark
    benchmark = RigorousBenchmark(
        output_dir=args.output_dir,
        seeds=seeds,
        device=device,
    )
    
    summary = benchmark.run_all(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_train=args.num_train,
        num_test=args.num_test,
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS (mean ± std)")
    print("=" * 70)
    
    print(f"\n{'Model':<30} {'Params':>12} {'Accuracy':>18}")
    print("-" * 65)
    
    for key, data in summary.items():
        name = key.replace('_', ' ')
        params = f"{data['params']/1e6:.2f}M"
        acc = f"{data['accuracy_mean']:.1f} ± {data['accuracy_std']:.1f}%"
        print(f"{name:<30} {params:>12} {acc:>18}")
    
    # Create plots
    if HAS_MATPLOTLIB:
        benchmark.plot_results(summary, os.path.join(args.output_dir, 'results_plot.png'))
        print(f"\nPlot saved to {args.output_dir}/results_plot.png")
    
    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()

