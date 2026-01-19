"""
Ablation Benchmark: Isolate the contribution of each A-JEPA v3 component.

Tests which "v3 special" ingredients matter:
1. RelationalBlock - Does slot-to-slot reasoning help?
2. PerSlotBottleneck - Does preserving object identity help?
3. Motion Channel - Does explicit motion encoding help?
4. Multi-scale Edges - Does edge scale variety help?
5. 8 slots vs 4 slots - Does more slots help?

Each ablation removes ONE component while keeping others fixed.
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from scipy import stats

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models_v3 import (
    AJEPAv3, VJEPAv3, get_ajepa_v3, get_vjepa_v3,
    SlotAttention, RelationalBlock, PerSlotBottleneck, 
    SlotTemporalMemory, SlotPredictor, AJEPAv3Encoder
)
from datasets.bouncing_balls import (
    BouncingBallsDataset, 
    preprocess_for_ajepa_v3, 
    preprocess_for_vjepa_v3
)


# =============================================================================
# ABLATION VARIANTS
# =============================================================================

class AJEPAv3NoRelational(AJEPAv3):
    """A-JEPA v3 WITHOUT RelationalBlock (tests importance of slot-to-slot reasoning)."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Replace RelationalBlock with identity
        self.encoder.relational = nn.Identity()


class AJEPAv3SharedBottleneck(nn.Module):
    """A-JEPA v3 with SHARED bottleneck instead of per-slot (tests object identity preservation)."""
    
    def __init__(
        self,
        in_channels: int = 4,
        img_size: int = 32,
        num_slots: int = 8,
        slot_dim: int = 48,
        bottleneck_dim: int = 32,
        memory_dim: int = 64,
        num_pred_steps: int = 5,
        sparsity_lambda: float = 0.001,
    ):
        super().__init__()
        
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.bottleneck_dim = bottleneck_dim
        
        # Use standard A-JEPA v3 encoder
        base = get_ajepa_v3('default')
        self.encoder = base.encoder
        
        # Replace per-slot bottleneck with shared (flattened) bottleneck
        self.shared_bottleneck = nn.Sequential(
            nn.Linear(num_slots * slot_dim, bottleneck_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim * 4, num_slots * bottleneck_dim),
        )
        
        # Keep temporal and predictor
        self.temporal = base.temporal
        self.predictor = base.predictor
        
    def encode_video(self, video, return_all=False):
        B, T, C, H, W = video.shape
        
        # Encode frames through conv + slot attention + relational
        frames = video.reshape(B * T, C, H, W)
        h = self.encoder.conv(frames)
        h = h.view(B * T, self.encoder.conv_channels, -1).transpose(1, 2)
        slots, _ = self.encoder.slot_attention(h)
        slots = self.encoder.relational(slots)
        
        # SHARED bottleneck (flatten all slots)
        slots_flat = slots.view(B * T, -1)  # (B*T, K*slot_dim)
        z_flat = self.shared_bottleneck(slots_flat)  # (B*T, K*bottleneck_dim)
        z = z_flat.view(B * T, self.num_slots, self.bottleneck_dim)
        z = F.normalize(z, dim=-1)
        z = z.view(B, T, self.num_slots, self.bottleneck_dim)
        
        # Temporal
        z, _ = self.temporal(z)
        
        if return_all:
            return z
        return z.mean(dim=1)
    
    def forward(self, context_video, target_video):
        z_context = self.encode_video(context_video)
        z_targets = self.encode_video(target_video, return_all=True)
        
        pred_output = self.predictor(z_context)
        
        num_steps = min(pred_output['predictions'].shape[1], z_targets.shape[1])
        pred = pred_output['predictions'][:, :num_steps]
        target = z_targets[:, :num_steps]
        
        pred_norm = F.normalize(pred, dim=-1)
        target_norm = F.normalize(target.detach(), dim=-1)
        pred_loss = -torch.mean(torch.sum(pred_norm * target_norm, dim=-1))
        
        return {
            'loss': pred_loss,
            'pred_loss': pred_loss,
            'aux_loss': torch.tensor(0.0),
        }
    
    def encode(self, x):
        B = x.shape[0]
        h = self.encoder.conv(x)
        h = h.view(B, self.encoder.conv_channels, -1).transpose(1, 2)
        slots, _ = self.encoder.slot_attention(h)
        slots = self.encoder.relational(slots)
        slots_flat = slots.view(B, -1)
        z = self.shared_bottleneck(slots_flat)
        return z


class AJEPAv3NoMotion(AJEPAv3):
    """A-JEPA v3 trained without motion channel (3 channels: multi-scale edges only)."""
    
    def __init__(self, **kwargs):
        kwargs['in_channels'] = 3  # Only edges, no motion
        super().__init__(**kwargs)


class AJEPAv3SingleScaleEdge(AJEPAv3):
    """A-JEPA v3 with single-scale Sobel instead of multi-scale (1 channel + motion)."""
    
    def __init__(self, **kwargs):
        kwargs['in_channels'] = 2  # 1 edge + 1 motion
        super().__init__(**kwargs)


class AJEPAv3FourSlots(AJEPAv3):
    """A-JEPA v3 with 4 slots instead of 8 (like v2)."""
    
    def __init__(self, **kwargs):
        kwargs['num_slots'] = 4
        super().__init__(**kwargs)


# =============================================================================
# DATASET VARIANTS
# =============================================================================

def preprocess_no_motion(video):
    """Multi-scale edges without motion channel (3 channels)."""
    from datasets.bouncing_balls import multi_scale_sobel
    import cv2
    
    T, C, H, W = video.shape
    result = []
    for t in range(T):
        frame = video[t].permute(1, 2, 0).numpy()
        if C == 3:
            gray = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (frame[:, :, 0] * 255).astype(np.uint8)
        edges = multi_scale_sobel(gray)  # (3, H, W)
        result.append(torch.from_numpy(edges).float())
    return torch.stack(result)  # (T, 3, H, W)


def preprocess_single_edge_motion(video):
    """Single-scale edge + motion (2 channels)."""
    from datasets.bouncing_balls import compute_motion_features
    import cv2
    
    T, C, H, W = video.shape
    edges_list = []
    for t in range(T):
        frame = video[t].permute(1, 2, 0).numpy()
        if C == 3:
            gray = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (frame[:, :, 0] * 255).astype(np.uint8)
        
        # Single-scale Sobel ksize=3
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge = np.sqrt(sobelx**2 + sobely**2)
        edge = edge / (edge.max() + 1e-8)
        edges_list.append(torch.from_numpy(edge).float().unsqueeze(0))
    
    edges = torch.stack(edges_list)  # (T, 1, H, W)
    motion = compute_motion_features(video)  # (T, 1, H, W)
    
    return torch.cat([edges, motion], dim=1)  # (T, 2, H, W)


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    device: torch.device,
    lr: float = 1e-3,
) -> float:
    """Train model and return final loss."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    
    final_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            video = batch['video'].to(device)
            
            # Split into context and target
            T = video.shape[1]
            mid = T // 2
            context = video[:, :mid]
            target = video[:, mid:]
            
            optimizer.zero_grad()
            output = model(context, target)
            loss = output['loss']
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        final_loss = epoch_loss / len(train_loader)
    
    return final_loss


def extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features for linear probe."""
    model.eval()
    features, labels = [], []
    
    with torch.no_grad():
        for batch in loader:
            video = batch['video'].to(device)
            label = batch['num_balls'].numpy()
            
            # Use first frame
            x = video[:, 0]
            z = model.encode(x)
            
            features.append(z.cpu().numpy())
            labels.append(label)
    
    return np.concatenate(features), np.concatenate(labels)


def evaluate_linear_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> float:
    """Train and evaluate linear probe."""
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(train_features, train_labels)
    return clf.score(test_features, test_labels)


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def compute_statistics(results: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """Compute mean, std, 95% CI, and effect size vs baseline."""
    stats_dict = {}
    
    baseline_scores = results.get('ajepa_v3_full', [50.0])  # Use full model as baseline
    
    for name, scores in results.items():
        scores_arr = np.array(scores)
        mean = np.mean(scores_arr)
        std = np.std(scores_arr, ddof=1) if len(scores_arr) > 1 else 0.0
        n = len(scores_arr)
        
        # 95% confidence interval
        if n > 1:
            ci = stats.t.interval(0.95, n-1, loc=mean, scale=std/np.sqrt(n))
        else:
            ci = (mean, mean)
        
        # Effect size (Cohen's d) vs baseline
        baseline_arr = np.array(baseline_scores)
        if len(baseline_arr) > 1 and len(scores_arr) > 1:
            pooled_std = np.sqrt((np.var(scores_arr, ddof=1) + np.var(baseline_arr, ddof=1)) / 2)
            cohens_d = (mean - np.mean(baseline_arr)) / (pooled_std + 1e-8)
        else:
            cohens_d = 0.0
        
        stats_dict[name] = {
            'mean': mean,
            'std': std,
            'ci_low': ci[0],
            'ci_high': ci[1],
            'n': n,
            'cohens_d': cohens_d,
        }
    
    return stats_dict


# =============================================================================
# MAIN
# =============================================================================

@dataclass
class AblationConfig:
    seeds: List[int]
    epochs: int
    num_train: int
    num_test: int
    batch_size: int
    output_dir: str


def run_ablation_study(config: AblationConfig):
    """Run full ablation study."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Define ablations
    ablations = {
        'ajepa_v3_full': {
            'model_class': AJEPAv3,
            'model_kwargs': {'in_channels': 4, 'num_slots': 8},
            'preprocess': preprocess_for_ajepa_v3,
            'description': 'Full A-JEPA v3 (baseline)',
        },
        'no_relational': {
            'model_class': AJEPAv3NoRelational,
            'model_kwargs': {'in_channels': 4, 'num_slots': 8},
            'preprocess': preprocess_for_ajepa_v3,
            'description': 'A-JEPA v3 without RelationalBlock',
        },
        'shared_bottleneck': {
            'model_class': AJEPAv3SharedBottleneck,
            'model_kwargs': {},
            'preprocess': preprocess_for_ajepa_v3,
            'description': 'A-JEPA v3 with shared (flattened) bottleneck',
        },
        'no_motion': {
            'model_class': AJEPAv3NoMotion,
            'model_kwargs': {'num_slots': 8},
            'preprocess': preprocess_no_motion,
            'description': 'A-JEPA v3 without motion channel',
        },
        'single_edge': {
            'model_class': AJEPAv3SingleScaleEdge,
            'model_kwargs': {'num_slots': 8},
            'preprocess': preprocess_single_edge_motion,
            'description': 'A-JEPA v3 with single-scale edges',
        },
        '4_slots': {
            'model_class': AJEPAv3FourSlots,
            'model_kwargs': {'in_channels': 4},
            'preprocess': preprocess_for_ajepa_v3,
            'description': 'A-JEPA v3 with 4 slots (like v2)',
        },
        'vjepa_v3': {
            'model_class': VJEPAv3,
            'model_kwargs': {},
            'preprocess': preprocess_for_vjepa_v3,
            'description': 'V-JEPA v3 (no slots, no relational)',
        },
    }
    
    results = {name: [] for name in ablations.keys()}
    
    for seed in config.seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")
        
        set_seed(seed)
        
        for name, spec in ablations.items():
            print(f"\n[{name}] {spec['description']}")
            
            # Create model
            model = spec['model_class'](**spec['model_kwargs']).to(device)
            params = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {params:,}")
            
            # Create datasets with appropriate preprocessing
            # For ablations with different channel counts, we need custom datasets
            preprocess_fn = spec['preprocess']
            
            # Generate data
            train_data = BouncingBallsDataset(
                num_samples=config.num_train,
                num_frames=10,
                num_balls=(1, 3),
                img_size=32,
                mode='raw',  # We'll preprocess manually
                seed=seed,
            )
            
            test_data = BouncingBallsDataset(
                num_samples=config.num_test,
                num_frames=10,
                num_balls=(1, 3),
                img_size=32,
                mode='raw',
                seed=seed + 1000,
            )
            
            # Wrap with preprocessing
            class PreprocessedDataset(torch.utils.data.Dataset):
                def __init__(self, base_dataset, preprocess_fn):
                    self.base = base_dataset
                    self.preprocess_fn = preprocess_fn
                
                def __len__(self):
                    return len(self.base)
                
                def __getitem__(self, idx):
                    item = self.base[idx]
                    video = self.preprocess_fn(item['video'])
                    return {'video': video, 'num_balls': item['num_balls']}
            
            train_dataset = PreprocessedDataset(train_data, preprocess_fn)
            test_dataset = PreprocessedDataset(test_data, preprocess_fn)
            
            train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
            
            # Train
            final_loss = train_model(model, train_loader, config.epochs, device)
            print(f"  Final loss: {final_loss:.4f}")
            
            # Extract features
            train_features, train_labels = extract_features(model, train_loader, device)
            test_features, test_labels = extract_features(model, test_loader, device)
            
            # Linear probe
            accuracy = evaluate_linear_probe(train_features, train_labels, test_features, test_labels)
            print(f"  Accuracy: {accuracy*100:.1f}%")
            
            results[name].append(accuracy * 100)
    
    # Compute statistics
    print("\n" + "="*60)
    print("ABLATION RESULTS")
    print("="*60)
    
    stats_dict = compute_statistics(results)
    
    print(f"\n{'Ablation':<25} {'Acc ± Std':<15} {'95% CI':<20} {'Cohen\\'s d':<10}")
    print("-" * 70)
    
    for name, s in stats_dict.items():
        ci_str = f"[{s['ci_low']:.1f}, {s['ci_high']:.1f}]"
        print(f"{name:<25} {s['mean']:.1f} ± {s['std']:.1f}    {ci_str:<20} {s['cohens_d']:+.2f}")
    
    # Save results
    output_file = os.path.join(config.output_dir, 'ablation_results.json')
    with open(output_file, 'w') as f:
        json.dump({
            'config': asdict(config),
            'raw_results': results,
            'statistics': stats_dict,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2, default=lambda x: x if isinstance(x, (int, float, str, list, dict)) else str(x))
    
    print(f"\nResults saved to {output_file}")
    
    # Generate summary
    print("\n" + "="*60)
    print("COMPONENT CONTRIBUTION ANALYSIS")
    print("="*60)
    
    baseline = stats_dict['ajepa_v3_full']['mean']
    
    contributions = []
    for name in ['no_relational', 'shared_bottleneck', 'no_motion', 'single_edge', '4_slots']:
        ablated = stats_dict[name]['mean']
        contribution = baseline - ablated
        contributions.append((name, contribution))
    
    contributions.sort(key=lambda x: -x[1])
    
    print(f"\nComponent contributions (drop from {baseline:.1f}% baseline):")
    for name, contrib in contributions:
        component = name.replace('no_', '').replace('_', ' ').title()
        if contrib > 0:
            print(f"  {component}: {contrib:+.1f}% (removing hurts)")
        else:
            print(f"  {component}: {contrib:+.1f}% (removing helps)")
    
    return results, stats_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ablation Study for A-JEPA v3')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 1337])
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--num_train', type=int, default=300)
    parser.add_argument('--num_test', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_dir', type=str, default='results/ablation')
    
    args = parser.parse_args()
    
    config = AblationConfig(
        seeds=args.seeds,
        epochs=args.epochs,
        num_train=args.num_train,
        num_test=args.num_test,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
    
    print("="*60)
    print("A-JEPA v3 ABLATION STUDY")
    print("="*60)
    print(f"Seeds: {config.seeds}")
    print(f"Epochs: {config.epochs}")
    print(f"Training samples: {config.num_train}")
    print(f"Test samples: {config.num_test}")
    
    run_ablation_study(config)
