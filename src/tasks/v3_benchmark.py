"""
V3 Benchmark: Compare A-JEPA v3 vs V-JEPA v3

Features:
1. Curriculum learning (Easy → Medium → Hard)
2. VICReg loss to prevent collapse
3. Multi-seed evaluation (10 seeds by default) for statistical rigor
4. Capacity-matched models for fair comparison
5. Proper statistics: p-values, effect sizes, 95% confidence intervals
6. Full reproducibility with deterministic seeding
7. Baselines: Random projection, SimpleCNN, V-JEPA-Tiny

Curriculum Phases:
- Easy (30 epochs): 1 ball, no sparsity - learn "what is an object"
- Medium (40 epochs): 2 balls, light sparsity - learn "what is an interaction"
- Hard (40 epochs): 2-3 balls, full sparsity - full complexity

Total: 110 epochs per model
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from scipy import stats


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int):
    """Full deterministic seeding for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # For bitwise CUDA reproducibility (optional, may slow down):
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    """Seed each DataLoader worker for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed: int) -> torch.Generator:
    """Get a seeded generator for DataLoader."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# =============================================================================
# BASELINE MODELS
# =============================================================================

class RandomBaseline(nn.Module):
    """
    Frozen random projection baseline - sanity check for chance performance.
    Expected accuracy: ~50% on binary classification.
    """
    def __init__(self, input_dim: int = 4*32*32, proj_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(input_dim, proj_dim)
        # Initialize with small weights to avoid overflow
        nn.init.normal_(self.proj.weight, std=0.01)
        nn.init.zeros_(self.proj.bias)
        # Freeze all parameters
        for p in self.parameters():
            p.requires_grad = False

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode single frame via random projection (normalized)."""
        z = self.proj(x.view(x.size(0), -1))
        return F.normalize(z, dim=-1)  # Normalize to prevent overflow

    def encode_video(self, video: torch.Tensor, return_all: bool = False) -> torch.Tensor:
        """Encode video (mean pool across time)."""
        B, T, C, H, W = video.shape
        frames = video.reshape(B * T, C, H, W)
        z = self.encode(frames).view(B, T, -1)
        if return_all:
            return z
        return z.mean(dim=1)

    def forward(self, context_video, target_video):
        """Dummy forward for compatibility."""
        return {'loss': torch.tensor(0.0), 'predictions': None, 'targets': None}


class SimpleCNN(nn.Module):
    """
    Single-frame CNN baseline - no temporal/relational reasoning.
    Tests whether temporal modeling helps.
    """
    def __init__(self, in_channels: int = 4, emb_dim: int = 128):
        super().__init__()
        self.emb_dim = emb_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.fc = nn.Linear(64 * 4 * 4, emb_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode single frame."""
        h = self.conv(x)
        return self.fc(h.view(h.size(0), -1))

    def encode_frame(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for encode."""
        return self.encode(x)

    def encode_video(self, video: torch.Tensor, return_all: bool = False) -> torch.Tensor:
        """Encode video by mean-pooling frame features."""
        B, T, C, H, W = video.shape
        frames = video.reshape(B * T, C, H, W)
        z = self.encode(frames).view(B, T, -1)
        if return_all:
            return z
        return z.mean(dim=1)

    def forward(self, context_video, target_video):
        """Training forward pass."""
        z_ctx = self.encode_video(context_video)
        z_tgt = self.encode_video(target_video)
        # Simple MSE loss
        loss = F.mse_loss(z_ctx, z_tgt.detach())
        return {'loss': loss, 'predictions': z_ctx, 'targets': z_tgt}

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models_v3 import AJEPAv3, VJEPAv3, get_ajepa_v3, get_vjepa_v3
from src.models_v2 import get_ajepa_v2, get_vjepa_v2
from src.datasets.bouncing_balls import (
    generate_video, preprocess_for_ajepa_v3, preprocess_for_vjepa_v3,
    BouncingBallsDataset,
)

try:
    from src.datasets.hidden_mass import generate_hidden_mass_video
except ImportError:
    generate_hidden_mass_video = None

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# CURRICULUM DATASET
# =============================================================================

class CurriculumDatasetV3(Dataset):
    """
    Dataset that supports curriculum learning phases and v3 preprocessing.
    """
    
    def __init__(
        self,
        num_samples: int = 200,
        num_frames: int = 30,
        img_size: int = 32,
        mode: str = 'ajepa_v3',  # 'ajepa_v3', 'vjepa_v3', 'edge', 'raw'
        phase: str = 'easy',     # 'easy', 'medium', 'hard'
        seed: int = None,
    ):
        self.num_samples = num_samples
        self.phase = phase
        self.mode = mode
        
        if seed is not None:
            np.random.seed(seed)
        
        # Determine complexity based on phase
        if phase == 'easy':
            num_balls = 1
        elif phase == 'medium':
            num_balls = 2
        else:  # hard
            num_balls = np.random.choice([2, 3])
        
        self.data = []
        for i in range(num_samples):
            # For hard phase, randomize ball count per sample
            if phase == 'hard':
                num_balls = np.random.choice([2, 3])
            
            # Generate video (grayscale)
            video, label = self._generate_video_with_label(num_frames, img_size, num_balls, i)
            
            # Apply preprocessing
            video = self._preprocess(video)
            
            self.data.append({
                'video': video,
                'mass_label': label,
            })
    
    def _generate_video_with_label(self, num_frames, img_size, num_balls, idx):
        """Generate video and assign mass label for probing."""
        import cv2
        
        # Simple mass categorization
        mass_cat = 'light' if idx % 2 == 0 else 'heavy'
        mass = 0.5 if mass_cat == 'light' else 2.0
        label = 0 if mass_cat == 'light' else 1
        
        frames = []
        # Simple ball simulation
        balls = []
        for b in range(num_balls):
            ball = {
                'x': np.random.uniform(5, img_size - 5),
                'y': np.random.uniform(5, img_size - 5),
                'vx': np.random.uniform(-2, 2),
                'vy': np.random.uniform(-2, 2),
                'r': 3,
                'mass': mass if b == 0 else 1.0,  # First ball has target mass
            }
            balls.append(ball)
        
        for t in range(num_frames):
            frame = np.zeros((img_size, img_size), dtype=np.uint8)
            
            for ball in balls:
                cv2.circle(frame, (int(ball['x']), int(ball['y'])), ball['r'], 255, -1)
                
                # Update position
                ball['x'] += ball['vx']
                ball['y'] += ball['vy']
                
                # Bounce off walls
                if ball['x'] < ball['r'] or ball['x'] > img_size - ball['r']:
                    ball['vx'] *= -1
                if ball['y'] < ball['r'] or ball['y'] > img_size - ball['r']:
                    ball['vy'] *= -1
                
                ball['x'] = np.clip(ball['x'], ball['r'], img_size - ball['r'])
                ball['y'] = np.clip(ball['y'], ball['r'], img_size - ball['r'])
            
            frames.append(frame)
        
        video = np.stack(frames, axis=0)[:, np.newaxis, :, :].astype(np.float32) / 255.0
        return video, label
    
    def _preprocess(self, video):
        """Apply v3 preprocessing."""
        if self.mode == 'ajepa_v3':
            return preprocess_for_ajepa_v3(video)
        elif self.mode == 'vjepa_v3':
            return preprocess_for_vjepa_v3(video)
        elif self.mode == 'edge':
            from src.datasets.bouncing_balls import apply_edge_transform
            return apply_edge_transform(video, method='canny')
        else:  # raw
            return video
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        item = self.data[idx]
        video = torch.from_numpy(item['video'])
        
        # Split into context and target
        T = video.shape[0]
        ctx_len = min(10, T // 2)
        
        context = video[:ctx_len]
        target = video[ctx_len:ctx_len + 5]
        
        return {
            'context': context,
            'target': target,
            'label': item['mass_label'],
        }


# =============================================================================
# VICREG LOSS
# =============================================================================

def vicreg_loss(z1, z2, sim_weight=25.0, var_weight=25.0, cov_weight=1.0):
    """
    VICReg loss for self-supervised learning.
    Prevents representation collapse by encouraging variance and decorrelation.
    """
    # Flatten if needed
    if z1.dim() > 2:
        z1 = z1.view(z1.shape[0], -1)
        z2 = z2.view(z2.shape[0], -1)
    
    # Similarity (MSE)
    sim_loss = F.mse_loss(z1, z2)
    
    # Variance (encourage spread)
    std1 = torch.sqrt(z1.var(dim=0) + 1e-4)
    std2 = torch.sqrt(z2.var(dim=0) + 1e-4)
    var_loss = torch.mean(F.relu(1 - std1)) + torch.mean(F.relu(1 - std2))
    
    # Covariance (decorrelation)
    z1_centered = z1 - z1.mean(dim=0)
    z2_centered = z2 - z2.mean(dim=0)
    
    cov1 = (z1_centered.T @ z1_centered) / (z1.shape[0] - 1)
    cov2 = (z2_centered.T @ z2_centered) / (z2.shape[0] - 1)
    
    # Off-diagonal covariance
    cov_loss = (cov1.pow(2).sum() - cov1.diag().pow(2).sum()) / z1.shape[1]
    cov_loss += (cov2.pow(2).sum() - cov2.diag().pow(2).sum()) / z2.shape[1]
    
    total = sim_weight * sim_loss + var_weight * var_loss + cov_weight * cov_loss
    return total, {'sim': sim_loss.item(), 'var': var_loss.item(), 'cov': cov_loss.item()}


# =============================================================================
# CURRICULUM TRAINER
# =============================================================================

class CurriculumTrainer:
    """
    Train with curriculum learning: Easy → Medium → Hard phases.
    """
    
    def __init__(
        self,
        model,
        device,
        lr: float = 1e-3,
        mode: str = 'ajepa_v3',
    ):
        self.model = model.to(device)
        self.device = device
        self.mode = mode
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Curriculum phases
        self.phases = [
            {'name': 'easy', 'epochs': 30, 'sparsity': 0.0},
            {'name': 'medium', 'epochs': 40, 'sparsity': 0.001},
            {'name': 'hard', 'epochs': 40, 'sparsity': 0.002},
        ]
    
    def set_sparsity(self, lambda_val):
        """Update sparsity lambda if model supports it."""
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'bottleneck'):
            if hasattr(self.model.encoder.bottleneck, 'sparsity_lambda'):
                self.model.encoder.bottleneck.sparsity_lambda = lambda_val
    
    def train_phase(
        self,
        phase: dict,
        num_samples: int = 200,
        batch_size: int = 16,
        verbose: bool = True,
        seed: int = None,
    ):
        """Train for one curriculum phase."""
        self.set_sparsity(phase['sparsity'])

        # Create dataset for this phase with deterministic seed
        phase_seed = seed if seed is not None else np.random.randint(10000)
        dataset = CurriculumDatasetV3(
            num_samples=num_samples,
            mode=self.mode,
            phase=phase['name'],
            seed=phase_seed,
        )
        # Use seeded DataLoader for reproducibility
        g = get_generator(phase_seed)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            worker_init_fn=seed_worker, generator=g
        )
        
        losses = []
        for epoch in range(phase['epochs']):
            epoch_loss = 0.0
            for batch in loader:
                context = batch['context'].to(self.device)
                target = batch['target'].to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                output = self.model(context, target)
                loss = output['loss']
                
                # Add VICReg if we have predictions/targets
                if 'predictions' in output and 'targets' in output:
                    pred = output['predictions']
                    tgt = output['targets']
                    # Flatten for VICReg
                    pred_flat = pred.reshape(pred.shape[0], -1)
                    tgt_flat = tgt.reshape(tgt.shape[0], -1)
                    vic_loss, _ = vicreg_loss(pred_flat, tgt_flat)
                    loss = loss + 0.1 * vic_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"    {phase['name']} Epoch {epoch+1}/{phase['epochs']}: Loss = {avg_loss:.4f}")
        
        return losses
    
    def train_full_curriculum(
        self,
        num_samples: int = 200,
        batch_size: int = 16,
        verbose: bool = True,
        seed: int = None,
    ):
        """Train through all curriculum phases."""
        all_losses = []

        for i, phase in enumerate(self.phases):
            if verbose:
                print(f"\n  Phase: {phase['name'].upper()} ({phase['epochs']} epochs)")

            # Each phase gets a deterministic but different seed
            phase_seed = (seed + i * 1000) if seed is not None else None
            phase_losses = self.train_phase(
                phase=phase,
                num_samples=num_samples,
                batch_size=batch_size,
                verbose=verbose,
                seed=phase_seed,
            )
            all_losses.extend(phase_losses)

        return all_losses


# =============================================================================
# LINEAR PROBE EVALUATION
# =============================================================================

def extract_temporal_features(model, context_frames: torch.Tensor, device) -> torch.Tensor:
    """
    Extract features with temporal pooling across all context frames.

    Args:
        model: Trained model with encode() or encode_video() method
        context_frames: (B, T, C, H, W) tensor
        device: torch device

    Returns:
        (B, D) pooled features
    """
    B, T, C, H, W = context_frames.shape

    with torch.no_grad():
        # Prefer encode_video if available (handles temporal internally)
        if hasattr(model, 'encode_video'):
            z = model.encode_video(context_frames.to(device))  # (B, D) or (B, K, D)
            # Flatten if needed
            if z.dim() > 2:
                z = z.view(z.shape[0], -1)
            return z

        # Fallback: encode each frame and mean-pool
        features = []
        for t in range(T):
            frame = context_frames[:, t].to(device)  # (B, C, H, W)
            if hasattr(model, 'encode'):
                feat = model.encode(frame)
            elif hasattr(model, 'encode_frame'):
                feat = model.encode_frame(frame)
            else:
                feat = model.encoder(frame)
            features.append(feat)

        # Mean pool across time
        stacked = torch.stack(features, dim=1)  # (B, T, D)
        pooled = stacked.mean(dim=1)  # (B, D)
        return pooled


def extract_features(model, dataset, device, mode='ajepa_v3', seed: int = None):
    """Extract features from a trained model with temporal pooling."""
    model.eval()
    features = []
    labels = []

    g = get_generator(seed) if seed is not None else None
    loader = DataLoader(
        dataset, batch_size=16, shuffle=False,
        worker_init_fn=seed_worker if seed else None,
        generator=g
    )

    with torch.no_grad():
        for batch in loader:
            context = batch['context'].to(device)

            # Use temporal pooling for proper evaluation
            z = extract_temporal_features(model, context, device)

            # Flatten for linear probe
            z_flat = z.view(z.shape[0], -1)
            features.append(z_flat.cpu())
            labels.append(batch['label'])

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)

    return features.numpy(), labels.numpy()


def train_linear_probe(features, labels, test_features, test_labels):
    """Train and evaluate a simple linear probe."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import warnings

    # Handle NaN/Inf values
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
    test_features = np.nan_to_num(test_features, nan=0.0, posinf=1.0, neginf=-1.0)

    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    test_features_scaled = scaler.transform(test_features)

    # Handle any remaining NaN/Inf after scaling
    features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
    test_features_scaled = np.nan_to_num(test_features_scaled, nan=0.0, posinf=1.0, neginf=-1.0)

    # Train logistic regression (suppress convergence warnings)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        clf = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
        clf.fit(features_scaled, labels)

    # Evaluate
    train_acc = clf.score(features_scaled, labels) * 100
    test_acc = clf.score(test_features_scaled, test_labels) * 100

    return train_acc, test_acc


# =============================================================================
# COLLISION DETECTION EVALUATION
# =============================================================================

class CollisionDataset(Dataset):
    """
    Dataset for collision detection evaluation.
    Each episode is one video, with per-frame collision labels.
    """

    def __init__(
        self,
        num_episodes: int = 100,
        num_frames: int = 30,
        num_balls: int = 2,
        img_size: int = 32,
        mode: str = 'ajepa_v3',
        seed: int = None,
    ):
        self.num_episodes = num_episodes
        self.mode = mode

        if seed is not None:
            set_seed(seed)

        self.episodes = []
        for _ in range(num_episodes):
            # Generate video with collision labels
            video, collision_labels = generate_video(
                num_frames=num_frames,
                num_balls=num_balls,
                img_size=img_size,
                with_collisions=True,
                return_collision_labels=True,
            )

            # Preprocess
            if mode == 'ajepa_v3':
                video = preprocess_for_ajepa_v3(video)
            elif mode == 'vjepa_v3':
                video = preprocess_for_vjepa_v3(video)

            self.episodes.append({
                'frames': torch.from_numpy(video),
                'collision_labels': torch.from_numpy(collision_labels),
            })

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, idx):
        return self.episodes[idx]


def evaluate_collision_detection(
    model,
    train_episodes: List[Dict],
    test_episodes: List[Dict],
    device,
) -> Dict[str, float]:
    """
    Collision detection with proper train/test split by episode.

    Uses balanced accuracy, F1, and AUROC due to class imbalance.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        f1_score, roc_auc_score, balanced_accuracy_score,
    )

    def extract_per_frame(episodes):
        features, labels = [], []
        model.eval()
        with torch.no_grad():
            for ep in episodes:
                frames = ep['frames']  # (T, C, H, W)
                collision_labels = ep['collision_labels']  # (T,)

                for t in range(frames.shape[0]):
                    frame = frames[t:t+1].to(device)  # (1, C, H, W)
                    if hasattr(model, 'encode'):
                        feat = model.encode(frame)
                    elif hasattr(model, 'encode_frame'):
                        feat = model.encode_frame(frame)
                    else:
                        feat = model.encoder(frame)
                    feat = feat.view(1, -1)
                    features.append(feat.cpu().numpy())
                    labels.append(collision_labels[t].item())

        return np.vstack(features), np.array(labels)

    X_train, y_train = extract_per_frame(train_episodes)
    X_test, y_test = extract_per_frame(test_episodes)

    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train with balanced class weights
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] if len(np.unique(y_train)) > 1 else None

    results = {
        'balanced_acc': balanced_accuracy_score(y_test, y_pred) * 100,
        'f1': f1_score(y_test, y_pred, zero_division=0),
    }

    # AUROC only if both classes present
    if y_prob is not None and len(np.unique(y_test)) > 1:
        results['auroc'] = roc_auc_score(y_test, y_prob)
    else:
        results['auroc'] = None

    # Class distribution info
    results['collision_rate_train'] = y_train.mean()
    results['collision_rate_test'] = y_test.mean()

    return results


def run_collision_benchmark(
    model,
    mode: str,
    device,
    num_train_episodes: int = 80,
    num_test_episodes: int = 20,
    seed: int = 42,
) -> Dict[str, float]:
    """Run collision detection benchmark for a single model."""
    set_seed(seed)

    # Create train and test episodes
    train_data = CollisionDataset(
        num_episodes=num_train_episodes,
        mode=mode,
        seed=seed,
    )
    test_data = CollisionDataset(
        num_episodes=num_test_episodes,
        mode=mode,
        seed=seed + 5000,
    )

    return evaluate_collision_detection(
        model=model,
        train_episodes=train_data.episodes,
        test_episodes=test_data.episodes,
        device=device,
    )


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def run_single_experiment(
    model_type: str,  # 'ajepa_v3', 'vjepa_v3', 'vjepa_tiny', 'simple_cnn', 'random', etc.
    seed: int,
    num_train: int = 200,
    num_test: int = 100,
    batch_size: int = 16,
    device: str = 'cpu',
    verbose: bool = True,
):
    """Run a single training + evaluation experiment."""
    # Full deterministic seeding
    set_seed(seed)

    # Create model and determine mode
    skip_training = False
    if model_type == 'ajepa_v3':
        model = get_ajepa_v3('default')
        mode = 'ajepa_v3'
    elif model_type == 'ajepa_v3_large':
        model = get_ajepa_v3('capacity_matched')
        mode = 'ajepa_v3'
    elif model_type == 'vjepa_v3':
        model = get_vjepa_v3('default')
        mode = 'vjepa_v3'
    elif model_type == 'vjepa_v3_small' or model_type == 'vjepa_tiny':
        model = get_vjepa_v3('capacity_matched')
        mode = 'vjepa_v3'
    elif model_type == 'ajepa_v2':
        model = get_ajepa_v2('default')
        mode = 'edge'
    elif model_type == 'vjepa_v2':
        model = get_vjepa_v2('default')
        mode = 'raw'
    elif model_type == 'simple_cnn':
        model = SimpleCNN(in_channels=4, emb_dim=128)
        mode = 'ajepa_v3'  # Use same 4-channel input
    elif model_type == 'random':
        model = RandomBaseline(input_dim=4*32*32, proj_dim=256)
        mode = 'ajepa_v3'  # Use same 4-channel input
        skip_training = True  # Random baseline doesn't train
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"\n{'='*60}")
        print(f"Model: {model_type.upper()}")
        print(f"Parameters: {params:,} (trainable: {trainable_params:,})")
        print(f"Seed: {seed}")
        print(f"{'='*60}")

    # Train with curriculum (skip for random baseline)
    losses = []
    if not skip_training:
        trainer = CurriculumTrainer(model, device, mode=mode)
        losses = trainer.train_full_curriculum(
            num_samples=num_train,
            batch_size=batch_size,
            verbose=verbose,
            seed=seed,
        )
    else:
        if verbose:
            print("  (Skipping training - frozen baseline)")
        model = model.to(device)

    # Create test dataset
    test_dataset = CurriculumDatasetV3(
        num_samples=num_test,
        mode=mode,
        phase='hard',  # Test on hard phase
        seed=seed + 1000,
    )

    # Extract features from training data for linear probe
    train_dataset = CurriculumDatasetV3(
        num_samples=num_train,
        mode=mode,
        phase='hard',
        seed=seed,
    )

    train_features, train_labels = extract_features(model, train_dataset, device, mode, seed=seed)
    test_features, test_labels = extract_features(model, test_dataset, device, mode, seed=seed+1000)

    # Train linear probe
    train_acc, test_acc = train_linear_probe(
        train_features, train_labels, test_features, test_labels
    )

    if verbose:
        print(f"\nResults:")
        print(f"  Train Accuracy: {train_acc:.1f}%")
        print(f"  Test Accuracy: {test_acc:.1f}%")

    return {
        'model': model_type,
        'seed': seed,
        'params': params,
        'trainable_params': trainable_params,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'final_loss': losses[-1] if losses else 0,
        'losses': losses,
    }


def compute_statistics(results_by_model: Dict[str, List[float]], baseline_model: str = 'ajepa_v3') -> Dict:
    """
    Compute comprehensive statistics for each model.
    
    Returns:
    - mean, std
    - 95% confidence interval
    - Cohen's d effect size vs baseline
    - p-value from t-test vs baseline
    """
    stats_dict = {}
    
    baseline_scores = results_by_model.get(baseline_model, [50.0])
    baseline_arr = np.array(baseline_scores)
    
    for model_name, scores in results_by_model.items():
        scores_arr = np.array(scores)
        n = len(scores_arr)
        mean = np.mean(scores_arr)
        std = np.std(scores_arr, ddof=1) if n > 1 else 0.0
        sem = std / np.sqrt(n) if n > 0 else 0.0
        
        # 95% confidence interval
        if n > 1:
            ci = stats.t.interval(0.95, n-1, loc=mean, scale=sem)
        else:
            ci = (mean, mean)
        
        # Effect size (Cohen's d) vs baseline
        if len(baseline_arr) > 1 and n > 1 and model_name != baseline_model:
            pooled_std = np.sqrt((np.var(scores_arr, ddof=1) + np.var(baseline_arr, ddof=1)) / 2)
            cohens_d = (mean - np.mean(baseline_arr)) / (pooled_std + 1e-8)
            
            # Welch's t-test
            t_stat, p_value = stats.ttest_ind(scores_arr, baseline_arr, equal_var=False)
        else:
            cohens_d = 0.0
            p_value = 1.0
        
        stats_dict[model_name] = {
            'mean': mean,
            'std': std,
            'sem': sem,
            'ci_low': ci[0],
            'ci_high': ci[1],
            'n': n,
            'cohens_d': cohens_d,
            'p_value': p_value,
        }
    
    return stats_dict


def run_benchmark(
    models: list = None,
    seeds: list = None,
    num_train: int = 200,
    num_test: int = 100,
    batch_size: int = 16,
    output_dir: str = 'results/v3_benchmark',
    device: str = None,
    include_capacity_matched: bool = False,
    with_collision: bool = False,
):
    """
    Run full benchmark comparing models with proper reproducibility.

    Models supported:
    - ajepa_v3: A-JEPA v3 default (~442K params)
    - vjepa_v3: V-JEPA v3 default (~2.7M params)
    - vjepa_tiny: V-JEPA v3 capacity-matched (~442K params)
    - simple_cnn: Single-frame CNN baseline (~100K params)
    - random: Random projection baseline (chance level)
    """
    if models is None:
        models = ['ajepa_v3', 'vjepa_v3']
    if seeds is None:
        seeds = [42, 123, 456]  # Default to 3 seeds
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("V3 BENCHMARK: A-JEPA v3 vs V-JEPA v3 (with Curriculum Learning)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Models: {models}")
    print(f"Seeds: {seeds} ({len(seeds)} seeds)")
    print(f"Train samples: {num_train}")
    print(f"Test samples: {num_test}")
    print(f"Collision benchmark: {with_collision}")
    
    # Run all experiments
    results = []
    for model_type in models:
        for seed in seeds:
            result = run_single_experiment(
                model_type=model_type,
                seed=seed,
                num_train=num_train,
                num_test=num_test,
                batch_size=batch_size,
                device=device,
                verbose=True,
            )
            results.append(result)
    
    # Aggregate results by model
    results_by_model = {}
    params_by_model = {}
    for model_type in models:
        model_results = [r for r in results if r['model'] == model_type]
        results_by_model[model_type] = [r['test_acc'] for r in model_results]
        params_by_model[model_type] = model_results[0]['params'] if model_results else 0
    
    # Compute statistics
    stats_dict = compute_statistics(results_by_model, baseline_model='ajepa_v3')
    
    # Print detailed results
    print("\n" + "=" * 70)
    print("STATISTICAL SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Model':<20} {'Acc ± Std':<15} {'95% CI':<20} {'Cohens d':<12} {'p-value':<10}")
    print("-" * 80)
    
    for model_type in models:
        s = stats_dict[model_type]
        params = params_by_model[model_type]
        ci_str = f"[{s['ci_low']:.1f}, {s['ci_high']:.1f}]"
        p_str = f"{s['p_value']:.4f}" if s['p_value'] < 1 else "baseline"
        d_str = f"{s['cohens_d']:+.2f}" if s['cohens_d'] != 0 else "baseline"
        
        print(f"{model_type:<20} {s['mean']:.1f} ± {s['std']:.1f}%    {ci_str:<20} {d_str:<12} {p_str:<10}")
        print(f"  └─ Params: {params:,}, n={s['n']}")
    
    # Statistical interpretation
    if 'ajepa_v3' in models and 'vjepa_v3' in models:
        ajepa_stats = stats_dict['ajepa_v3']
        vjepa_stats = stats_dict['vjepa_v3']
        
        print("\n" + "=" * 70)
        print("STATISTICAL INTERPRETATION")
        print("=" * 70)
        
        diff = ajepa_stats['mean'] - vjepa_stats['mean']
        p_val = vjepa_stats['p_value']
        d = -vjepa_stats['cohens_d']  # Flip sign since we're comparing to baseline
        
        print(f"\nA-JEPA v3 vs V-JEPA v3:")
        print(f"  Accuracy difference: {diff:+.1f}%")
        print(f"  Effect size (Cohen's d): {d:.2f} ", end="")
        if abs(d) < 0.2:
            print("(negligible)")
        elif abs(d) < 0.5:
            print("(small)")
        elif abs(d) < 0.8:
            print("(medium)")
        else:
            print("(large)")
        
        print(f"  p-value: {p_val:.4f} ", end="")
        if p_val < 0.01:
            print("(highly significant, p < 0.01)")
        elif p_val < 0.05:
            print("(significant, p < 0.05)")
        else:
            print("(not significant, p ≥ 0.05)")
        
        # Parameter efficiency
        ajepa_params = params_by_model['ajepa_v3']
        vjepa_params = params_by_model['vjepa_v3']
        param_ratio = vjepa_params / ajepa_params
        print(f"\n  Parameter ratio: V-JEPA uses {param_ratio:.1f}x more params")
        print(f"  A-JEPA efficiency: {ajepa_stats['mean']/ajepa_params*1e6:.1f} acc%/M-params")
        print(f"  V-JEPA efficiency: {vjepa_stats['mean']/vjepa_params*1e6:.1f} acc%/M-params")
    
    # Build summary dict
    summary = {}
    for model_type in models:
        summary[model_type] = {
            **stats_dict[model_type],
            'params': params_by_model[model_type],
            'accs': results_by_model[model_type],
        }
    
    # Save results
    output = {
        'config': {
            'models': models,
            'seeds': seeds,
            'num_train': num_train,
            'num_test': num_test,
            'batch_size': batch_size,
        },
        'summary': {k: {kk: vv for kk, vv in v.items() if kk != 'accs'} for k, v in summary.items()},
        'raw_results': [{k: v for k, v in r.items() if k != 'losses'} for r in results],
        'statistics': stats_dict,
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(output, f, indent=2)
    
    # Plot if available
    if HAS_MATPLOTLIB:
        plot_results(summary, output_dir)
    
    print(f"\nResults saved to: {output_dir}/results.json")
    
    return summary


def plot_results(summary, output_dir):
    """Create comparison plots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    models = list(summary.keys())
    accs = [summary[m]['mean'] for m in models]
    stds = [summary[m]['std'] for m in models]
    params = [summary[m]['params'] / 1e6 for m in models]  # In millions
    
    # Accuracy comparison
    ax1 = axes[0]
    colors = ['#2ecc71' if 'ajepa' in m else '#3498db' for m in models]
    bars = ax1.bar(models, accs, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylim([0, 100])
    for bar, acc in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                 f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Parameter efficiency
    ax2 = axes[1]
    efficiency = [accs[i] / params[i] for i in range(len(models))]
    bars = ax2.bar(models, efficiency, color=colors, alpha=0.8)
    ax2.set_ylabel('Accuracy per Million Params')
    ax2.set_title('Parameter Efficiency')
    for bar, eff in zip(bars, efficiency):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 f'{eff:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'v3_benchmark.png'), dpi=150)
    plt.close()
    print(f"\nPlot saved to: {output_dir}/v3_benchmark.png")


# =============================================================================
# MAIN
# =============================================================================

AVAILABLE_MODELS = [
    'ajepa_v3',       # A-JEPA v3 default (~442K params)
    'vjepa_v3',       # V-JEPA v3 default (~2.7M params)
    'vjepa_tiny',     # V-JEPA v3 capacity-matched (~442K params)
    'simple_cnn',     # SimpleCNN baseline (~100K params)
    'random',         # Random projection baseline (chance level)
    'ajepa_v3_large', # A-JEPA v3 scaled up (~2.7M params)
    'vjepa_v3_small', # V-JEPA v3 scaled down (alias for vjepa_tiny)
    'ajepa_v2',       # Legacy A-JEPA v2
    'vjepa_v2',       # Legacy V-JEPA v2
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='V3 Benchmark with Curriculum Learning and Statistical Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available models:
  ajepa_v3      - A-JEPA v3 default (~442K params) [main model]
  vjepa_v3      - V-JEPA v3 default (~2.7M params) [baseline]
  vjepa_tiny    - V-JEPA v3 capacity-matched (~442K params) [fair comparison]
  simple_cnn    - Single-frame CNN (~100K params) [no temporal reasoning]
  random        - Random projection [chance baseline ~50%]

Example:
  python v3_benchmark.py --models ajepa_v3 vjepa_tiny vjepa_v3 simple_cnn random --seeds 42 123 456
"""
    )
    parser.add_argument('--models', nargs='+', default=['ajepa_v3', 'vjepa_v3'],
                        choices=AVAILABLE_MODELS,
                        help='Models to benchmark')
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[42, 123, 456],
                        help='Random seeds for multi-run evaluation (default: 3 seeds)')
    parser.add_argument('--num_train', type=int, default=200,
                        help='Number of training samples per phase')
    parser.add_argument('--num_test', type=int, default=100,
                        help='Number of test samples')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--output_dir', type=str, default='results/v3_benchmark',
                        help='Output directory')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    parser.add_argument('--capacity_matched', action='store_true',
                        help='Include capacity-matched models for fair comparison')
    parser.add_argument('--with_collision', action='store_true',
                        help='Also run collision detection benchmark')
    parser.add_argument('--all_baselines', action='store_true',
                        help='Include all baseline models (random, simple_cnn, vjepa_tiny)')

    args = parser.parse_args()

    # Build model list
    models = args.models.copy() if hasattr(args.models, 'copy') else list(args.models)

    # Add all baselines if requested
    if args.all_baselines:
        for baseline in ['random', 'simple_cnn', 'vjepa_tiny']:
            if baseline not in models:
                models.append(baseline)

    # Add capacity-matched variants if requested
    if args.capacity_matched:
        if 'ajepa_v3' in models and 'ajepa_v3_large' not in models:
            models.append('ajepa_v3_large')
        if 'vjepa_v3' in models and 'vjepa_tiny' not in models:
            models.append('vjepa_tiny')

    run_benchmark(
        models=models,
        seeds=args.seeds,
        num_train=args.num_train,
        num_test=args.num_test,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        device=args.device,
        include_capacity_matched=args.capacity_matched,
        with_collision=args.with_collision,
    )

