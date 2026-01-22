"""
V4 Benchmark: Test A-JEPA v4 with all 6 Cognitive Principles

Tests:
1. Full v4 (all phases enabled)
2. Ablation studies (disable each phase individually)
3. Comparison with v3 baseline
4. TransformerPlanner evaluation

Cognitive Principles:
- Phase 1: Precision-Weighted Top-Down Prediction
- Phase 2: Dual Pathway (Spatial vs Object)
- Phase 3: Symbolic Bottleneck (VQ-VAE)
- Phase 4: Top-Down Gating
- Phase 5: Structured Temporal Memory
- Phase 6: TransformerPlanner (optional imagination module)

Uses same curriculum learning as v3:
- Easy (30 epochs): 1 ball
- Medium (40 epochs): 2 balls
- Hard (40 epochs): 2-3 balls

For TransformerPlanner models, includes phased training:
- Phase A (0-15% epochs): Single-step teacher forcing
- Phase B (15-30% epochs): Multi-step teacher forcing
- Phase C (30%+ epochs): Scheduled sampling (prob 0→0.5)

Total: 110 epochs per model
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from scipy import stats

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models_v4 import AJEPAv4, get_ajepa_v4
from src.models_v3 import get_ajepa_v3
from src.datasets.bouncing_balls import preprocess_for_ajepa_v3

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# CURRICULUM DATASET (same as v3)
# =============================================================================

class CurriculumDatasetV4(Dataset):
    """
    Dataset that supports curriculum learning phases for v4.
    Uses same preprocessing as v3 (edge + motion).
    """

    def __init__(
        self,
        num_samples: int = 200,
        num_frames: int = 30,
        img_size: int = 32,
        phase: str = 'easy',
        seed: int = None,
    ):
        self.num_samples = num_samples
        self.phase = phase

        if seed is not None:
            np.random.seed(seed)

        self.data = []
        for i in range(num_samples):
            # Determine ball count based on phase
            if phase == 'easy':
                num_balls = 1
            elif phase == 'medium':
                num_balls = 2
            else:  # hard
                num_balls = np.random.choice([2, 3])

            video, label = self._generate_video_with_label(num_frames, img_size, num_balls, i)
            video = preprocess_for_ajepa_v3(video)  # 4-channel: edge + motion

            self.data.append({
                'video': video,
                'mass_label': label,
            })

    def _generate_video_with_label(self, num_frames, img_size, num_balls, idx):
        """Generate video and assign mass label for probing."""
        import cv2

        mass_cat = 'light' if idx % 2 == 0 else 'heavy'
        mass = 0.5 if mass_cat == 'light' else 2.0
        label = 0 if mass_cat == 'light' else 1

        frames = []
        balls = []
        for b in range(num_balls):
            ball = {
                'x': np.random.uniform(5, img_size - 5),
                'y': np.random.uniform(5, img_size - 5),
                'vx': np.random.uniform(-2, 2),
                'vy': np.random.uniform(-2, 2),
                'r': 3,
                'mass': mass if b == 0 else 1.0,
            }
            balls.append(ball)

        for t in range(num_frames):
            frame = np.zeros((img_size, img_size), dtype=np.uint8)

            for ball in balls:
                cv2.circle(frame, (int(ball['x']), int(ball['y'])), ball['r'], 255, -1)

                ball['x'] += ball['vx']
                ball['y'] += ball['vy']

                if ball['x'] < ball['r'] or ball['x'] > img_size - ball['r']:
                    ball['vx'] *= -1
                if ball['y'] < ball['r'] or ball['y'] > img_size - ball['r']:
                    ball['vy'] *= -1

                ball['x'] = np.clip(ball['x'], ball['r'], img_size - ball['r'])
                ball['y'] = np.clip(ball['y'], ball['r'], img_size - ball['r'])

            frames.append(frame)

        video = np.stack(frames, axis=0)[:, np.newaxis, :, :].astype(np.float32) / 255.0
        return video, label

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        item = self.data[idx]
        video = torch.from_numpy(item['video'])

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
# VICREG LOSS (same as v3)
# =============================================================================

def vicreg_loss(z1, z2, sim_weight=25.0, var_weight=25.0, cov_weight=1.0):
    """VICReg loss for self-supervised learning."""
    if z1.dim() > 2:
        z1 = z1.view(z1.shape[0], -1)
        z2 = z2.view(z2.shape[0], -1)

    sim_loss = F.mse_loss(z1, z2)

    std1 = torch.sqrt(z1.var(dim=0) + 1e-4)
    std2 = torch.sqrt(z2.var(dim=0) + 1e-4)
    var_loss = torch.mean(F.relu(1 - std1)) + torch.mean(F.relu(1 - std2))

    z1_centered = z1 - z1.mean(dim=0)
    z2_centered = z2 - z2.mean(dim=0)

    cov1 = (z1_centered.T @ z1_centered) / (z1.shape[0] - 1)
    cov2 = (z2_centered.T @ z2_centered) / (z2.shape[0] - 1)

    cov_loss = (cov1.pow(2).sum() - cov1.diag().pow(2).sum()) / z1.shape[1]
    cov_loss += (cov2.pow(2).sum() - cov2.diag().pow(2).sum()) / z2.shape[1]

    total = sim_weight * sim_loss + var_weight * var_loss + cov_weight * cov_loss
    return total, {'sim': sim_loss.item(), 'var': var_loss.item(), 'cov': cov_loss.item()}


# =============================================================================
# PLANNER TRAINING PHASE SCHEDULER
# =============================================================================

def get_planner_phase(epoch: int, total_epochs: int = 110) -> Tuple[str, float]:
    """
    Get TransformerPlanner training phase and sampling probability.

    Phase A (epochs 1-N1): Single-step teacher forcing
    Phase B (epochs N1-N2): Multi-step teacher forcing
    Phase C (epochs N2+): Scheduled sampling with increasing probability

    Args:
        epoch: Current epoch (0-indexed)
        total_epochs: Total training epochs

    Returns:
        phase: 'A', 'B', or 'C'
        sampling_prob: Probability of using own predictions (phase C only)
    """
    # Phase boundaries (roughly: 15% Phase A, 15% Phase B, 70% Phase C)
    phase_a_end = int(total_epochs * 0.15)  # ~16 epochs
    phase_b_end = int(total_epochs * 0.30)  # ~33 epochs

    if epoch < phase_a_end:
        return 'A', 0.0
    elif epoch < phase_b_end:
        return 'B', 0.0
    else:
        # Phase C: scheduled sampling
        # Linearly increase sampling prob from 0 to 0.5 over remaining epochs
        phase_c_progress = (epoch - phase_b_end) / (total_epochs - phase_b_end)
        sampling_prob = min(0.5, phase_c_progress * 0.5)
        return 'C', sampling_prob


# =============================================================================
# CURRICULUM TRAINER FOR V4
# =============================================================================

class CurriculumTrainerV4:
    """
    Train v4 with curriculum learning.
    Same structure as v3 trainer.

    For models with TransformerPlanner, includes phased training:
    - Phase A: Single-step teacher forcing
    - Phase B: Multi-step teacher forcing
    - Phase C: Scheduled sampling
    """

    def __init__(
        self,
        model,
        device,
        lr: float = 1e-3,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        # Check if model has planner
        self.has_planner = hasattr(model, 'use_planner') and model.use_planner

        self.phases = [
            {'name': 'easy', 'epochs': 30, 'sparsity': 0.0},
            {'name': 'medium', 'epochs': 40, 'sparsity': 0.001},
            {'name': 'hard', 'epochs': 40, 'sparsity': 0.002},
        ]

        # Track total epochs for planner phase scheduling
        self.total_epochs = sum(p['epochs'] for p in self.phases)
        self.current_epoch = 0

    def set_sparsity(self, lambda_val):
        """Update sparsity lambda if model supports it."""
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'bottleneck'):
            if hasattr(self.model.encoder.bottleneck, 'sparsity_lambda'):
                self.model.encoder.bottleneck.sparsity_lambda = lambda_val

    def _update_planner_phase(self):
        """Update TransformerPlanner training phase based on current epoch."""
        if self.has_planner and hasattr(self.model, 'set_planner_phase'):
            phase, sampling_prob = get_planner_phase(self.current_epoch, self.total_epochs)
            self.model.set_planner_phase(phase, sampling_prob)
            return phase, sampling_prob
        return None, 0.0

    def train_phase(
        self,
        phase: dict,
        num_samples: int = 200,
        batch_size: int = 16,
        verbose: bool = True,
    ):
        """Train for one curriculum phase."""
        self.set_sparsity(phase['sparsity'])

        dataset = CurriculumDatasetV4(
            num_samples=num_samples,
            phase=phase['name'],
            seed=np.random.randint(10000),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        losses = []
        for epoch in range(phase['epochs']):
            # Update planner phase for TransformerPlanner models
            planner_phase, sampling_prob = self._update_planner_phase()

            epoch_loss = 0.0
            ce_loss_total = 0.0
            planner_loss_total = 0.0
            for batch in loader:
                context = batch['context'].to(self.device)
                target = batch['target'].to(self.device)

                self.optimizer.zero_grad()

                output = self.model(context, target)
                loss = output['loss']

                # Track CE loss for planner replace mode
                if 'ce_loss' in output:
                    ce_loss_total += output['ce_loss'].item()
                # Track planner loss for auxiliary mode
                if 'planner_loss' in output:
                    planner_loss_total += output['planner_loss'].item()

                # Add VICReg
                if 'predictions' in output and 'targets' in output:
                    pred = output['predictions']
                    tgt = output['targets']
                    pred_flat = pred.reshape(pred.shape[0], -1)
                    tgt_flat = tgt.reshape(tgt.shape[0], -1)
                    vic_loss, _ = vicreg_loss(pred_flat, tgt_flat)
                    loss = loss + 0.1 * vic_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            avg_ce_loss = ce_loss_total / len(loader) if ce_loss_total > 0 else 0.0
            avg_planner_loss = planner_loss_total / len(loader) if planner_loss_total > 0 else 0.0
            losses.append(avg_loss)

            if verbose and (epoch + 1) % 10 == 0:
                phase_str = f" [Planner: {planner_phase}, p={sampling_prob:.2f}]" if planner_phase else ""
                # Show CE loss for replace mode, planner loss for aux mode
                ce_str = f", CE={avg_ce_loss:.4f}" if avg_ce_loss > 0 else ""
                planner_str = f", PL={avg_planner_loss:.4f}" if avg_planner_loss > 0 else ""
                print(f"    {phase['name']} Epoch {epoch+1}/{phase['epochs']}: Loss = {avg_loss:.4f}{ce_str}{planner_str}{phase_str}")

            self.current_epoch += 1

        return losses

    def train_full_curriculum(
        self,
        num_samples: int = 200,
        batch_size: int = 16,
        verbose: bool = True,
    ):
        """Train through all curriculum phases."""
        all_losses = []
        self.current_epoch = 0  # Reset epoch counter

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

def extract_features(model, dataset, device):
    """Extract features from a trained model."""
    model.eval()
    features = []
    labels = []

    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            context = batch['context'].to(device)

            if hasattr(model, 'encode_video'):
                z = model.encode_video(context)
            else:
                B, T, C, H, W = context.shape
                z = model.encoder.encode_video(context)

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

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    test_features_scaled = scaler.transform(test_features)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(features_scaled, labels)

    train_acc = clf.score(features_scaled, labels) * 100
    test_acc = clf.score(test_features_scaled, test_labels) * 100

    return train_acc, test_acc


# =============================================================================
# HORIZON DRIFT EVALUATION
# =============================================================================

def evaluate_horizon_drift(
    model,
    dataset,
    device: str,
    num_steps: int = 5,
) -> Dict[str, List[float]]:
    """
    Evaluate prediction error vs horizon step.

    Computes per-step cosine similarity between predictions and targets
    to measure how error accumulates over prediction horizon.

    Args:
        model: Trained AJEPAv4 model
        dataset: CurriculumDatasetV4
        device: 'cpu' or 'cuda'
        num_steps: Number of prediction steps

    Returns:
        dict with:
            'steps': [1, 2, 3, 4, 5]
            'step_similarities': List[float] - mean cosine sim per step
            'step_errors': List[float] - mean error (1 - sim) per step
            'drift': float - error increase from step 1 to step 5
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # Accumulate per-step similarities
    step_sims = [[] for _ in range(num_steps)]

    with torch.no_grad():
        for batch in loader:
            context = batch['context'].to(device)
            target = batch['target'].to(device)

            output = model(context, target)

            # Get predictions and targets
            pred = output['predictions']  # (B, S, K, D)
            tgt = output['targets']       # (B, S, K, D)

            # Normalize for cosine similarity
            pred_norm = F.normalize(pred, dim=-1)
            tgt_norm = F.normalize(tgt, dim=-1)

            # Per-step, per-slot similarity
            similarity = torch.sum(pred_norm * tgt_norm, dim=-1)  # (B, S, K)

            # Average over batch and slots for each step
            actual_steps = min(pred.shape[1], num_steps)
            for step in range(actual_steps):
                step_sim = similarity[:, step, :].mean().item()
                step_sims[step].append(step_sim)

    # Compute means
    step_similarities = [np.mean(sims) if sims else 0.0 for sims in step_sims]
    step_errors = [1.0 - sim for sim in step_similarities]

    # Compute drift: increase in error from step 1 to final step
    if len(step_errors) >= 2:
        drift = step_errors[-1] - step_errors[0]
    else:
        drift = 0.0

    return {
        'steps': list(range(1, num_steps + 1)),
        'step_similarities': step_similarities,
        'step_errors': step_errors,
        'drift': drift,
    }


# =============================================================================
# SINGLE EXPERIMENT
# =============================================================================

def run_single_experiment(
    config: str,  # 'default', 'no_dual', 'no_symbolic', 'no_gating', 'no_structured_mem', 'continuous', 'with_planner_aux', 'with_planner_replace', 'planner_only', 'v3'
    seed: int,
    num_train: int = 200,
    num_test: int = 100,
    batch_size: int = 16,
    device: str = 'cpu',
    verbose: bool = True,
) -> Dict:
    """Run a single training experiment."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Config: {config}, Seed: {seed}")
        print(f"{'='*60}")

    # Create model
    if config == 'v3':
        model = get_ajepa_v3('default')
        model_type = 'ajepa_v3'
    else:
        model = get_ajepa_v4(config)
        model_type = 'ajepa_v4'

    params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"Parameters: {params:,}")

    # Train with curriculum
    trainer = CurriculumTrainerV4(model, device)
    start_time = time.time()
    losses = trainer.train_full_curriculum(
        num_samples=num_train,
        batch_size=batch_size,
        verbose=verbose,
    )
    train_time = time.time() - start_time

    # Evaluate with linear probe
    if verbose:
        print("\nEvaluating with linear probe...")

    train_dataset = CurriculumDatasetV4(
        num_samples=num_train,
        phase='hard',
        seed=seed,
    )
    test_dataset = CurriculumDatasetV4(
        num_samples=num_test,
        phase='hard',
        seed=seed + 1000,
    )

    train_features, train_labels = extract_features(model, train_dataset, device)
    test_features, test_labels = extract_features(model, test_dataset, device)

    train_acc, test_acc = train_linear_probe(
        train_features, train_labels,
        test_features, test_labels,
    )

    # Evaluate horizon drift
    if verbose:
        print("\nEvaluating horizon drift...")

    horizon_drift_results = evaluate_horizon_drift(
        model=model,
        dataset=test_dataset,
        device=device,
        num_steps=5,
    )

    if verbose:
        print(f"\nResults:")
        print(f"  Train accuracy: {train_acc:.1f}%")
        print(f"  Test accuracy: {test_acc:.1f}%")
        print(f"  Training time: {train_time:.1f}s")
        print(f"  Horizon drift: {horizon_drift_results['drift']:.4f}")
        print(f"  Step errors: {[f'{e:.3f}' for e in horizon_drift_results['step_errors']]}")

    return {
        'config': config,
        'seed': seed,
        'params': params,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_time': train_time,
        'final_loss': losses[-1] if losses else None,
        'horizon_drift': horizon_drift_results,
    }


# =============================================================================
# MULTI-SEED BENCHMARK
# =============================================================================

def run_benchmark(
    configs: List[str],
    seeds: List[int],
    num_train: int = 200,
    num_test: int = 100,
    batch_size: int = 16,
    device: str = 'cpu',
    output_dir: str = 'results/v4_benchmark',
):
    """Run full benchmark across multiple configs and seeds."""

    os.makedirs(output_dir, exist_ok=True)

    results = {config: [] for config in configs}

    for config in configs:
        print(f"\n{'#'*60}")
        print(f"# BENCHMARKING: {config}")
        print(f"{'#'*60}")

        for seed in seeds:
            result = run_single_experiment(
                config=config,
                seed=seed,
                num_train=num_train,
                num_test=num_test,
                batch_size=batch_size,
                device=device,
                verbose=True,
            )
            results[config].append(result)

    # Compute statistics
    stats_results = {}
    for config in configs:
        accs = [r['test_acc'] for r in results[config]]
        drifts = [r['horizon_drift']['drift'] for r in results[config]]

        # Compute mean step errors across seeds
        num_steps = len(results[config][0]['horizon_drift']['step_errors'])
        mean_step_errors = []
        for step in range(num_steps):
            step_errs = [r['horizon_drift']['step_errors'][step] for r in results[config]]
            mean_step_errors.append(np.mean(step_errs))

        stats_results[config] = {
            'mean': np.mean(accs),
            'std': np.std(accs),
            'ci_low': np.mean(accs) - 1.96 * np.std(accs) / np.sqrt(len(accs)),
            'ci_high': np.mean(accs) + 1.96 * np.std(accs) / np.sqrt(len(accs)),
            'n': len(accs),
            'raw_results': accs,
            'params': results[config][0]['params'],
            'mean_drift': np.mean(drifts),
            'std_drift': np.std(drifts),
            'mean_step_errors': mean_step_errors,
        }

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"{'Config':<20} {'Mean':>10} {'Std':>8} {'95% CI':>15} {'Drift':>10}")
    print("-" * 60)

    for config in configs:
        s = stats_results[config]
        ci_str = f"[{s['ci_low']:.1f}, {s['ci_high']:.1f}]"
        print(f"{config:<20} {s['mean']:>10.1f}% {s['std']:>7.1f}% {ci_str:>15} {s['mean_drift']:>10.4f}")

    # Print horizon drift details
    print("\n" + "=" * 60)
    print("HORIZON DRIFT ANALYSIS (Step Errors)")
    print("=" * 60)
    print(f"{'Config':<20} {'Step 1':>8} {'Step 2':>8} {'Step 3':>8} {'Step 4':>8} {'Step 5':>8}")
    print("-" * 60)
    for config in configs:
        s = stats_results[config]
        step_strs = [f"{e:.3f}" for e in s['mean_step_errors']]
        print(f"{config:<20} {step_strs[0]:>8} {step_strs[1]:>8} {step_strs[2]:>8} {step_strs[3]:>8} {step_strs[4]:>8}")

    # Save results
    output = {
        'config': {
            'configs': configs,
            'seeds': seeds,
            'num_train': num_train,
            'num_test': num_test,
        },
        'raw_results': {config: [r for r in results[config]] for config in configs},
        'statistics': stats_results,
        'timestamp': datetime.now().isoformat(),
    }

    output_path = os.path.join(output_dir, 'v4_benchmark_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))

    print(f"\nResults saved to: {output_path}")

    # Create comparison plot
    if HAS_MATPLOTLIB:
        create_comparison_plot(stats_results, output_dir)

    return stats_results


def create_comparison_plot(stats_results, output_dir):
    """Create bar plot comparing different configs."""
    configs = list(stats_results.keys())
    means = [stats_results[c]['mean'] for c in configs]
    stds = [stats_results[c]['std'] for c in configs]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(configs))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color='steelblue', alpha=0.8)

    ax.set_ylabel('Test Accuracy (%)')
    ax.set_xlabel('Configuration')
    ax.set_title('A-JEPA v4 Benchmark: Ablation Study')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Chance')

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                f'{mean:.1f}%', ha='center', va='bottom', fontsize=9)

    ax.legend()
    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'v4_benchmark_plot.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"Plot saved to: {plot_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='A-JEPA v4 Benchmark')
    parser.add_argument('--configs', nargs='+',
                        default=['default', 'no_dual', 'no_symbolic', 'no_gating', 'no_structured_mem', 'continuous', 'with_planner_aux', 'with_planner_replace', 'planner_only'],
                        help='Configs to benchmark')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456],
                        help='Random seeds for experiments')
    parser.add_argument('--num_train', type=int, default=200,
                        help='Number of training samples')
    parser.add_argument('--num_test', type=int, default=100,
                        help='Number of test samples')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--output_dir', type=str, default='results/v4_benchmark',
                        help='Output directory')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with fewer samples and epochs')
    parser.add_argument('--planner_only', action='store_true',
                        help='Only benchmark planner configurations')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    if args.quick:
        print("Running quick test mode...")
        args.num_train = 50
        args.num_test = 25
        args.seeds = [42]

    if args.planner_only:
        print("Running planner-only benchmark...")
        args.configs = ['default', 'with_planner_aux', 'planner_only']

    run_benchmark(
        configs=args.configs,
        seeds=args.seeds,
        num_train=args.num_train,
        num_test=args.num_test,
        batch_size=args.batch_size,
        device=device,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
