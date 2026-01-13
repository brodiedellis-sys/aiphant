"""
V3 Benchmark: Compare A-JEPA v3 vs V-JEPA v3

Features:
1. Curriculum learning (Easy → Medium → Hard)
2. VICReg loss to prevent collapse
3. Multi-seed evaluation for statistical rigor
4. Compare v2 vs v3 architectures

Curriculum Phases:
- Easy (30 epochs): 1 ball, no sparsity - learn "what is an object"
- Medium (40 epochs): 2 balls, light sparsity - learn "what is an interaction"
- Hard (40 epochs): 2-3 balls, full sparsity - full complexity

Total: 110 epochs per model
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models_v3 import AJEPAv3, VJEPAv3, get_ajepa_v3, get_vjepa_v3
from src.models_v2 import get_ajepa_v2, get_vjepa_v2
from src.datasets.bouncing_balls import generate_video, preprocess_for_ajepa_v3, preprocess_for_vjepa_v3
from src.datasets.hidden_mass import generate_hidden_mass_video

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
    ):
        """Train for one curriculum phase."""
        self.set_sparsity(phase['sparsity'])
        
        # Create dataset for this phase
        dataset = CurriculumDatasetV3(
            num_samples=num_samples,
            mode=self.mode,
            phase=phase['name'],
            seed=np.random.randint(10000),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
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
    ):
        """Train through all curriculum phases."""
        all_losses = []
        
        for phase in self.phases:
            if verbose:
                print(f"\n  Phase: {phase['name'].upper()} ({phase['epochs']} epochs)")
            
            phase_losses = self.train_phase(
                phase=phase,
                num_samples=num_samples,
                batch_size=batch_size,
                verbose=verbose,
            )
            all_losses.extend(phase_losses)
        
        return all_losses


# =============================================================================
# LINEAR PROBE EVALUATION
# =============================================================================

def extract_features(model, dataset, device, mode='ajepa_v3'):
    """Extract features from a trained model."""
    model.eval()
    features = []
    labels = []
    
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    with torch.no_grad():
        for batch in loader:
            context = batch['context'].to(device)
            
            # Encode
            if hasattr(model, 'encode_video'):
                z = model.encode_video(context)
            else:
                # Fallback for v2 models
                B, T, C, H, W = context.shape
                z = model.encoder.encode_video(context)
            
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
    
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    test_features_scaled = scaler.transform(test_features)
    
    # Train logistic regression
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(features_scaled, labels)
    
    # Evaluate
    train_acc = clf.score(features_scaled, labels) * 100
    test_acc = clf.score(test_features_scaled, test_labels) * 100
    
    return train_acc, test_acc


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def run_single_experiment(
    model_type: str,  # 'ajepa_v3', 'vjepa_v3', 'ajepa_v2', 'vjepa_v2'
    seed: int,
    num_train: int = 200,
    num_test: int = 100,
    batch_size: int = 16,
    device: str = 'cpu',
    verbose: bool = True,
):
    """Run a single training + evaluation experiment."""
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create model and determine mode
    if model_type == 'ajepa_v3':
        model = get_ajepa_v3('default')
        mode = 'ajepa_v3'
    elif model_type == 'vjepa_v3':
        model = get_vjepa_v3('default')
        mode = 'vjepa_v3'
    elif model_type == 'ajepa_v2':
        model = get_ajepa_v2('default')
        mode = 'edge'
    elif model_type == 'vjepa_v2':
        model = get_vjepa_v2('default')
        mode = 'raw'
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"\n{'='*60}")
        print(f"Model: {model_type.upper()}")
        print(f"Parameters: {params:,}")
        print(f"Seed: {seed}")
        print(f"{'='*60}")
    
    # Train with curriculum
    trainer = CurriculumTrainer(model, device, mode=mode)
    losses = trainer.train_full_curriculum(
        num_samples=num_train,
        batch_size=batch_size,
        verbose=verbose,
    )
    
    # Create test dataset
    test_dataset = CurriculumDatasetV3(
        num_samples=num_test,
        mode=mode,
        phase='hard',  # Test on hard phase
        seed=seed + 1000,
    )
    
    # Extract features
    train_dataset = CurriculumDatasetV3(
        num_samples=num_train,
        mode=mode,
        phase='hard',
        seed=seed,
    )
    
    train_features, train_labels = extract_features(model, train_dataset, device, mode)
    test_features, test_labels = extract_features(model, test_dataset, device, mode)
    
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
        'train_acc': train_acc,
        'test_acc': test_acc,
        'final_loss': losses[-1] if losses else 0,
        'losses': losses,
    }


def run_benchmark(
    models: list = None,
    seeds: list = None,
    num_train: int = 200,
    num_test: int = 100,
    batch_size: int = 16,
    output_dir: str = 'results/v3_benchmark',
    device: str = None,
):
    """Run full benchmark comparing v2 and v3 models."""
    if models is None:
        models = ['ajepa_v3', 'vjepa_v3', 'ajepa_v2', 'vjepa_v2']
    if seeds is None:
        seeds = [42, 123, 456]
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("V3 BENCHMARK: A-JEPA v3 vs V-JEPA v3 (with Curriculum Learning)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Models: {models}")
    print(f"Seeds: {seeds}")
    print(f"Train samples: {num_train}")
    print(f"Test samples: {num_test}")
    
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
    
    # Aggregate results
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    summary = {}
    for model_type in models:
        model_results = [r for r in results if r['model'] == model_type]
        accs = [r['test_acc'] for r in model_results]
        
        summary[model_type] = {
            'mean_acc': np.mean(accs),
            'std_acc': np.std(accs),
            'params': model_results[0]['params'],
            'accs': accs,
        }
        
        print(f"\n{model_type.upper()}:")
        print(f"  Parameters: {summary[model_type]['params']:,}")
        print(f"  Accuracy: {summary[model_type]['mean_acc']:.1f} ± {summary[model_type]['std_acc']:.1f}%")
    
    # Save results
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump({
            'summary': {k: {kk: vv for kk, vv in v.items() if kk != 'accs'} for k, v in summary.items()},
            'raw_results': [{k: v for k, v in r.items() if k != 'losses'} for r in results],
        }, f, indent=2)
    
    # Plot if available
    if HAS_MATPLOTLIB:
        plot_results(summary, output_dir)
    
    return summary


def plot_results(summary, output_dir):
    """Create comparison plots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    models = list(summary.keys())
    accs = [summary[m]['mean_acc'] for m in models]
    stds = [summary[m]['std_acc'] for m in models]
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='V3 Benchmark with Curriculum Learning')
    parser.add_argument('--models', nargs='+', default=['ajepa_v3', 'vjepa_v3', 'ajepa_v2', 'vjepa_v2'],
                        help='Models to benchmark')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456],
                        help='Random seeds for multi-run evaluation')
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
    
    args = parser.parse_args()
    
    run_benchmark(
        models=args.models,
        seeds=args.seeds,
        num_train=args.num_train,
        num_test=args.num_test,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        device=args.device,
    )

