"""
Diagnostic Benchmark: Deep Analysis of A-JEPA and V-JEPA

This benchmark goes beyond accuracy to understand:
1. LEARNING DYNAMICS - How does the model learn over time?
2. SLOT UTILIZATION - Are all slots being used? (A-JEPA specific)
3. REPRESENTATION QUALITY - Is the latent space well-structured?
4. FAILURE ANALYSIS - Which samples fail and why?
5. CURRICULUM EFFECTS - Does phased learning help?
6. COMPONENT INTERACTIONS - How do parts work together?

Output: Detailed JSON + visualizations for actionable insights.
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict, field
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models_v3 import AJEPAv3, VJEPAv3, get_ajepa_v3, get_vjepa_v3
from src.datasets.bouncing_balls import (
    BouncingBallsDataset,
    preprocess_for_ajepa_v3,
    preprocess_for_vjepa_v3,
)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE
    from scipy import stats
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# =============================================================================
# DATA CLASSES FOR STRUCTURED OUTPUT
# =============================================================================

@dataclass
class LearningCurve:
    """Track learning dynamics over training."""
    epoch: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    pred_loss: List[float] = field(default_factory=list)
    aux_loss: List[float] = field(default_factory=list)
    phase: List[str] = field(default_factory=list)


@dataclass
class SlotAnalysis:
    """Analyze slot attention behavior (A-JEPA specific)."""
    slot_utilization: List[float] = field(default_factory=list)  # Per-slot usage
    attention_entropy: float = 0.0  # Higher = more distributed attention
    slot_diversity: float = 0.0  # How different are slots from each other
    active_slots: int = 0  # Slots with utilization > threshold
    collapsed_slots: int = 0  # Slots that are nearly identical


@dataclass 
class RepresentationQuality:
    """Assess the quality of learned representations."""
    variance: float = 0.0  # Feature variance (low = collapse)
    effective_rank: float = 0.0  # Dimensionality of representation
    class_separability: float = 0.0  # How separable are classes
    within_class_var: float = 0.0
    between_class_var: float = 0.0


@dataclass
class FailureAnalysis:
    """Understand failure modes."""
    total_samples: int = 0
    correct: int = 0
    incorrect: int = 0
    accuracy: float = 0.0
    
    # Per-class breakdown
    per_class_accuracy: Dict[str, float] = field(default_factory=dict)
    
    # Confusion patterns
    confusion_matrix: List[List[int]] = field(default_factory=list)
    
    # Confidence analysis
    mean_confidence_correct: float = 0.0
    mean_confidence_incorrect: float = 0.0
    
    # Sample-level errors (for later analysis)
    error_indices: List[int] = field(default_factory=list)
    error_predictions: List[int] = field(default_factory=list)
    error_labels: List[int] = field(default_factory=list)


@dataclass
class PhaseResults:
    """Results for a single curriculum phase."""
    phase_name: str = ""
    epochs_trained: int = 0
    final_loss: float = 0.0
    accuracy_after_phase: float = 0.0
    representation_quality: Optional[RepresentationQuality] = None


@dataclass
class DiagnosticResults:
    """Complete diagnostic output for a model."""
    model_name: str = ""
    model_params: int = 0
    seed: int = 0
    
    # Learning dynamics
    learning_curve: LearningCurve = field(default_factory=LearningCurve)
    
    # Slot analysis (A-JEPA only)
    slot_analysis: Optional[SlotAnalysis] = None
    
    # Representation quality
    final_representation: RepresentationQuality = field(default_factory=RepresentationQuality)
    
    # Failure analysis
    failure_analysis: FailureAnalysis = field(default_factory=FailureAnalysis)
    
    # Per-phase breakdown
    phase_results: List[PhaseResults] = field(default_factory=list)
    
    # Timing
    total_training_time: float = 0.0
    
    # Metadata
    timestamp: str = ""


# =============================================================================
# CURRICULUM DATASET
# =============================================================================

class DiagnosticDataset(Dataset):
    """Dataset with extra metadata for diagnostic analysis."""
    
    def __init__(
        self,
        num_samples: int,
        num_frames: int = 30,
        img_size: int = 32,
        num_balls: int = 2,
        mode: str = 'ajepa_v3',
        seed: int = None,
    ):
        self.num_samples = num_samples
        self.mode = mode
        self.num_balls = num_balls
        
        if seed is not None:
            np.random.seed(seed)
        
        self.data = []
        for i in range(num_samples):
            video, label, metadata = self._generate_video(num_frames, img_size, num_balls, i)
            video = self._preprocess(video)
            
            self.data.append({
                'video': video,
                'label': label,
                'metadata': metadata,
                'index': i,
            })
    
    def _generate_video(self, num_frames, img_size, num_balls, idx):
        """Generate video with rich metadata."""
        import cv2
        
        # Mass category
        mass_cat = 'light' if idx % 2 == 0 else 'heavy'
        mass = 0.5 if mass_cat == 'light' else 2.0
        label = 0 if mass_cat == 'light' else 1
        
        # Initialize balls with varying properties
        balls = []
        for b in range(num_balls):
            ball = {
                'x': np.random.uniform(8, img_size - 8),
                'y': np.random.uniform(8, img_size - 8),
                'vx': np.random.uniform(-3, 3),
                'vy': np.random.uniform(-3, 3),
                'r': 3 + np.random.randint(0, 2),  # Slight size variation
                'mass': mass if b == 0 else 1.0,
            }
            balls.append(ball)
        
        # Track metadata
        metadata = {
            'mass_category': mass_cat,
            'num_balls': num_balls,
            'initial_speed': np.sqrt(balls[0]['vx']**2 + balls[0]['vy']**2),
            'collisions': 0,
        }
        
        frames = []
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
                    metadata['collisions'] += 1
                if ball['y'] < ball['r'] or ball['y'] > img_size - ball['r']:
                    ball['vy'] *= -1
                    metadata['collisions'] += 1
                
                ball['x'] = np.clip(ball['x'], ball['r'], img_size - ball['r'])
                ball['y'] = np.clip(ball['y'], ball['r'], img_size - ball['r'])
            
            frames.append(frame)
        
        video = np.stack(frames, axis=0)[:, np.newaxis, :, :].astype(np.float32) / 255.0
        return video, label, metadata
    
    def _preprocess(self, video):
        """Apply v3 preprocessing."""
        # preprocess functions expect numpy arrays
        if self.mode == 'ajepa_v3':
            result = preprocess_for_ajepa_v3(video)
            return result.numpy() if isinstance(result, torch.Tensor) else result
        elif self.mode == 'vjepa_v3':
            result = preprocess_for_vjepa_v3(video)
            return result.numpy() if isinstance(result, torch.Tensor) else result
        else:
            return video
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        item = self.data[idx]
        video = torch.from_numpy(item['video'])
        
        T = video.shape[0]
        ctx_len = min(10, T // 2)
        
        return {
            'context': video[:ctx_len],
            'target': video[ctx_len:ctx_len + 5],
            'full_video': video,
            'label': item['label'],
            'metadata': item['metadata'],
            'index': item['index'],
        }


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_slots(model: AJEPAv3, loader: DataLoader, device: torch.device) -> SlotAnalysis:
    """Analyze slot attention behavior."""
    model.eval()
    
    all_slot_norms = []
    all_attentions = []
    all_slot_features = []
    
    with torch.no_grad():
        for batch in loader:
            context = batch['context'].to(device)
            B, T, C, H, W = context.shape
            
            # Get first frame
            x = context[:, 0]  # (B, C, H, W)
            
            # Forward through encoder components
            h = model.encoder.conv(x)
            h = h.view(B, model.encoder.conv_channels, -1).transpose(1, 2)
            
            # Slot attention with attention weights
            slots, attn = model.encoder.slot_attention(h)  # slots: (B, K, D), attn: (B, K, N)
            
            # Track slot norms (utilization proxy)
            slot_norms = torch.norm(slots, dim=-1)  # (B, K)
            all_slot_norms.append(slot_norms.cpu())
            
            # Track attention patterns
            all_attentions.append(attn.cpu())
            
            # Track slot features
            all_slot_features.append(slots.cpu())
    
    # Aggregate analysis
    slot_norms = torch.cat(all_slot_norms, dim=0)  # (N_samples, K)
    attentions = torch.cat(all_attentions, dim=0)  # (N_samples, K, spatial)
    slot_features = torch.cat(all_slot_features, dim=0)  # (N_samples, K, D)
    
    # Slot utilization: average norm per slot
    mean_norms = slot_norms.mean(dim=0).numpy()  # (K,)
    utilization = mean_norms / (mean_norms.max() + 1e-8)  # Normalize to [0, 1]
    
    # Attention entropy: higher = more distributed
    attn_probs = attentions.mean(dim=0)  # (K, spatial)
    attn_probs = attn_probs / (attn_probs.sum(dim=-1, keepdim=True) + 1e-8)
    entropy = -(attn_probs * (attn_probs + 1e-8).log()).sum(dim=-1).mean().item()
    
    # Slot diversity: average pairwise cosine distance between slots
    slots_flat = slot_features.mean(dim=0)  # (K, D)
    slots_norm = F.normalize(slots_flat, dim=-1)
    similarity = torch.mm(slots_norm, slots_norm.T)
    # Off-diagonal similarity
    K = slots_flat.shape[0]
    mask = 1 - torch.eye(K)
    off_diag_sim = (similarity * mask).sum() / (K * (K - 1))
    diversity = 1 - off_diag_sim.item()  # Higher = more diverse
    
    # Count active/collapsed slots
    active_threshold = 0.3
    collapse_threshold = 0.95
    active_slots = (utilization > active_threshold).sum()
    
    # Check for collapsed (identical) slots
    collapsed = 0
    for i in range(K):
        for j in range(i + 1, K):
            if similarity[i, j] > collapse_threshold:
                collapsed += 1
    
    return SlotAnalysis(
        slot_utilization=utilization.tolist(),
        attention_entropy=entropy,
        slot_diversity=diversity,
        active_slots=int(active_slots),
        collapsed_slots=collapsed,
    )


def analyze_representations(
    features: np.ndarray, 
    labels: np.ndarray,
) -> RepresentationQuality:
    """Analyze representation quality."""
    
    # Variance: feature-wise variance (low = collapse)
    variance = np.var(features, axis=0).mean()
    
    # Effective rank: how many dimensions are actually used
    # Via singular value analysis
    centered = features - features.mean(axis=0)
    try:
        u, s, vh = np.linalg.svd(centered, full_matrices=False)
        # Effective rank = exp(entropy of normalized singular values)
        s_norm = s / (s.sum() + 1e-8)
        entropy = -np.sum(s_norm * np.log(s_norm + 1e-8))
        effective_rank = np.exp(entropy)
    except:
        effective_rank = features.shape[1]
    
    # Class separability: ratio of between-class to within-class variance
    unique_labels = np.unique(labels)
    class_means = []
    within_var = 0
    
    for label in unique_labels:
        mask = labels == label
        class_features = features[mask]
        class_mean = class_features.mean(axis=0)
        class_means.append(class_mean)
        within_var += np.var(class_features, axis=0).mean()
    
    within_var /= len(unique_labels)
    
    global_mean = features.mean(axis=0)
    between_var = np.var(np.array(class_means), axis=0).mean()
    
    separability = between_var / (within_var + 1e-8)
    
    return RepresentationQuality(
        variance=float(variance),
        effective_rank=float(effective_rank),
        class_separability=float(separability),
        within_class_var=float(within_var),
        between_class_var=float(between_var),
    )


def analyze_failures(
    predictions: np.ndarray,
    labels: np.ndarray,
    confidences: Optional[np.ndarray] = None,
    metadata_list: Optional[List[Dict]] = None,
) -> FailureAnalysis:
    """Detailed failure analysis."""
    
    correct_mask = predictions == labels
    
    # Basic stats
    total = len(labels)
    n_correct = correct_mask.sum()
    n_incorrect = total - n_correct
    accuracy = n_correct / total
    
    # Per-class accuracy
    unique_labels = np.unique(labels)
    per_class = {}
    for label in unique_labels:
        mask = labels == label
        class_acc = correct_mask[mask].mean()
        per_class[f"class_{label}"] = float(class_acc)
    
    # Confusion matrix
    n_classes = len(unique_labels)
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for pred, true in zip(predictions, labels):
        confusion[true, pred] += 1
    
    # Confidence analysis
    if confidences is not None:
        conf_correct = confidences[correct_mask].mean() if correct_mask.any() else 0
        conf_incorrect = confidences[~correct_mask].mean() if (~correct_mask).any() else 0
    else:
        conf_correct = 0
        conf_incorrect = 0
    
    # Error indices
    error_idx = np.where(~correct_mask)[0].tolist()
    error_preds = predictions[~correct_mask].tolist()
    error_labs = labels[~correct_mask].tolist()
    
    return FailureAnalysis(
        total_samples=total,
        correct=int(n_correct),
        incorrect=int(n_incorrect),
        accuracy=float(accuracy),
        per_class_accuracy=per_class,
        confusion_matrix=confusion.tolist(),
        mean_confidence_correct=float(conf_correct),
        mean_confidence_incorrect=float(conf_incorrect),
        error_indices=error_idx[:50],  # Limit size
        error_predictions=error_preds[:50],
        error_labels=error_labs[:50],
    )


# =============================================================================
# TRAINING WITH DIAGNOSTICS
# =============================================================================

class DiagnosticTrainer:
    """Train with comprehensive diagnostic tracking."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        mode: str = 'ajepa_v3',
        lr: float = 1e-3,
    ):
        self.model = model.to(device)
        self.device = device
        self.mode = mode
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        
        self.learning_curve = LearningCurve()
        self.epoch_counter = 0
    
    def train_epoch(self, loader: DataLoader, phase_name: str) -> Dict[str, float]:
        """Train one epoch with detailed tracking."""
        self.model.train()
        
        total_loss = 0
        total_pred_loss = 0
        total_aux_loss = 0
        n_batches = 0
        
        for batch in loader:
            context = batch['context'].to(self.device)
            target = batch['target'].to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(context, target)
            
            loss = output['loss']
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_pred_loss += output['pred_loss'].item()
            total_aux_loss += output.get('aux_loss', torch.tensor(0.0)).item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        avg_pred = total_pred_loss / n_batches
        avg_aux = total_aux_loss / n_batches
        
        # Record
        self.learning_curve.epoch.append(self.epoch_counter)
        self.learning_curve.train_loss.append(avg_loss)
        self.learning_curve.pred_loss.append(avg_pred)
        self.learning_curve.aux_loss.append(avg_aux)
        self.learning_curve.phase.append(phase_name)
        
        self.epoch_counter += 1
        
        return {'loss': avg_loss, 'pred_loss': avg_pred, 'aux_loss': avg_aux}
    
    def train_phase(
        self,
        phase_name: str,
        num_balls: int,
        epochs: int,
        num_samples: int,
        batch_size: int,
        test_loader: DataLoader,
    ) -> PhaseResults:
        """Train a curriculum phase with evaluation."""
        
        # Create phase-specific dataset
        train_dataset = DiagnosticDataset(
            num_samples=num_samples,
            num_balls=num_balls,
            mode=self.mode,
            seed=np.random.randint(10000),
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        print(f"\n  Phase: {phase_name} ({epochs} epochs, {num_balls} ball(s))")
        
        final_loss = 0
        for epoch in range(epochs):
            metrics = self.train_epoch(train_loader, phase_name)
            final_loss = metrics['loss']
            
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{epochs}: loss={final_loss:.4f}")
        
        # Evaluate after phase
        accuracy, _ = self.evaluate(test_loader)
        
        # Get representation quality
        features, labels = self.extract_features(test_loader)
        rep_quality = analyze_representations(features, labels)
        
        return PhaseResults(
            phase_name=phase_name,
            epochs_trained=epochs,
            final_loss=final_loss,
            accuracy_after_phase=accuracy,
            representation_quality=rep_quality,
        )
    
    def extract_features(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features for analysis."""
        self.model.eval()
        features, labels = [], []
        
        with torch.no_grad():
            for batch in loader:
                context = batch['context'].to(self.device)
                label = batch['label']
                
                # Get representation
                if hasattr(self.model, 'encode_video'):
                    z = self.model.encode_video(context)
                else:
                    z = self.model.encode(context[:, 0])
                
                z_flat = z.view(z.shape[0], -1)
                features.append(z_flat.cpu().numpy())
                labels.append(label.numpy())
        
        return np.concatenate(features), np.concatenate(labels)
    
    def evaluate(self, loader: DataLoader) -> Tuple[float, FailureAnalysis]:
        """Full evaluation with failure analysis."""
        features, labels = self.extract_features(loader)
        
        # Handle NaN/Inf values
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        features = np.clip(features, -1e6, 1e6)  # Prevent overflow
        
        # Linear probe
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(features_scaled, labels)
        
        predictions = clf.predict(features_scaled)
        confidences = clf.predict_proba(features_scaled).max(axis=1)
        
        accuracy = (predictions == labels).mean()
        
        failure = analyze_failures(predictions, labels, confidences)
        
        return accuracy, failure


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_diagnostic_plots(
    results: DiagnosticResults,
    output_dir: str,
):
    """Create comprehensive diagnostic visualizations."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plots")
        return
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Learning curve
    ax1 = fig.add_subplot(gs[0, 0])
    lc = results.learning_curve
    ax1.plot(lc.epoch, lc.train_loss, 'b-', label='Total Loss', linewidth=2)
    ax1.plot(lc.epoch, lc.pred_loss, 'g--', label='Pred Loss', alpha=0.7)
    
    # Mark phase transitions
    phases = np.array(lc.phase)
    for i, phase in enumerate(['easy', 'medium', 'hard']):
        phase_epochs = np.where(phases == phase)[0]
        if len(phase_epochs) > 0:
            ax1.axvspan(phase_epochs[0], phase_epochs[-1], alpha=0.1, 
                       color=['green', 'yellow', 'red'][i],
                       label=phase)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Learning Curve')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Phase breakdown
    ax2 = fig.add_subplot(gs[0, 1])
    phase_names = [p.phase_name for p in results.phase_results]
    phase_accs = [p.accuracy_after_phase * 100 for p in results.phase_results]
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']
    bars = ax2.bar(phase_names, phase_accs, color=colors[:len(phase_names)])
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy After Each Phase')
    ax2.set_ylim([0, 100])
    for bar, acc in zip(bars, phase_accs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{acc:.1f}%', ha='center', fontsize=10)
    
    # 3. Slot utilization (if available)
    ax3 = fig.add_subplot(gs[0, 2])
    if results.slot_analysis:
        util = results.slot_analysis.slot_utilization
        slots = list(range(len(util)))
        colors = ['#27ae60' if u > 0.3 else '#e74c3c' for u in util]
        ax3.bar(slots, util, color=colors)
        ax3.axhline(y=0.3, color='r', linestyle='--', label='Active threshold')
        ax3.set_xlabel('Slot Index')
        ax3.set_ylabel('Utilization')
        ax3.set_title(f'Slot Utilization (Active: {results.slot_analysis.active_slots}/{len(util)})')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'N/A (V-JEPA)', ha='center', va='center', fontsize=14)
        ax3.set_title('Slot Utilization')
    
    # 4. Representation quality metrics
    ax4 = fig.add_subplot(gs[1, 0])
    rep = results.final_representation
    metrics = ['Variance', 'Eff. Rank', 'Separability']
    values = [rep.variance, rep.effective_rank / 100, rep.class_separability]  # Normalize
    ax4.bar(metrics, values, color=['#3498db', '#9b59b6', '#1abc9c'])
    ax4.set_title('Representation Quality')
    ax4.set_ylabel('Value (normalized)')
    
    # 5. Confusion matrix
    ax5 = fig.add_subplot(gs[1, 1])
    cm = np.array(results.failure_analysis.confusion_matrix)
    im = ax5.imshow(cm, cmap='Blues')
    ax5.set_xlabel('Predicted')
    ax5.set_ylabel('True')
    ax5.set_title(f'Confusion Matrix (Acc: {results.failure_analysis.accuracy*100:.1f}%)')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax5.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=12)
    
    # 6. Confidence distribution
    ax6 = fig.add_subplot(gs[1, 2])
    fa = results.failure_analysis
    conf_data = [fa.mean_confidence_correct, fa.mean_confidence_incorrect]
    ax6.bar(['Correct', 'Incorrect'], conf_data, color=['#27ae60', '#e74c3c'])
    ax6.set_ylabel('Mean Confidence')
    ax6.set_title('Confidence vs Correctness')
    ax6.set_ylim([0, 1])
    
    # 7. Summary text
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    summary_text = f"""
    MODEL: {results.model_name}  |  PARAMS: {results.model_params:,}  |  SEED: {results.seed}
    
    FINAL ACCURACY: {results.failure_analysis.accuracy*100:.1f}%
    
    LEARNING: Loss dropped from {results.learning_curve.train_loss[0]:.3f} to {results.learning_curve.train_loss[-1]:.3f}
    
    REPRESENTATION: Variance={rep.variance:.3f}, Eff.Rank={rep.effective_rank:.1f}, Separability={rep.class_separability:.3f}
    """
    
    if results.slot_analysis:
        sa = results.slot_analysis
        summary_text += f"""
    SLOTS: {sa.active_slots}/{len(sa.slot_utilization)} active, Diversity={sa.slot_diversity:.3f}, Entropy={sa.attention_entropy:.3f}
        """
    
    ax7.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=11,
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Diagnostic Report: {results.model_name}', fontsize=14, fontweight='bold')
    
    # Save
    output_path = os.path.join(output_dir, f'{results.model_name}_diagnostic.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {output_path}")


# =============================================================================
# MAIN BENCHMARK
# =============================================================================

def run_diagnostic(
    model_name: str,
    seed: int,
    num_train: int,
    num_test: int,
    batch_size: int,
    device: torch.device,
    output_dir: str,
) -> DiagnosticResults:
    """Run full diagnostic for one model."""
    
    import time
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC: {model_name} (seed={seed})")
    print(f"{'='*60}")
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create model
    if model_name == 'ajepa_v3':
        model = get_ajepa_v3('default')
        mode = 'ajepa_v3'
    elif model_name == 'vjepa_v3':
        model = get_vjepa_v3('default')
        mode = 'vjepa_v3'
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    
    # Create test dataset (fixed across phases)
    test_dataset = DiagnosticDataset(
        num_samples=num_test,
        num_balls=2,
        mode=mode,
        seed=seed + 1000,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize trainer
    trainer = DiagnosticTrainer(model, device, mode=mode)
    
    # Curriculum training
    curriculum = [
        ('easy', 1, 30),    # (phase_name, num_balls, epochs)
        ('medium', 2, 40),
        ('hard', 2, 40),     # Note: Could randomize 2-3 balls
    ]
    
    phase_results = []
    for phase_name, num_balls, epochs in curriculum:
        result = trainer.train_phase(
            phase_name=phase_name,
            num_balls=num_balls,
            epochs=epochs,
            num_samples=num_train,
            batch_size=batch_size,
            test_loader=test_loader,
        )
        phase_results.append(result)
        print(f"    → Accuracy after {phase_name}: {result.accuracy_after_phase*100:.1f}%")
    
    # Final evaluation
    print("\n  Final Evaluation...")
    features, labels = trainer.extract_features(test_loader)
    final_rep = analyze_representations(features, labels)
    
    _, failure = trainer.evaluate(test_loader)
    
    # Slot analysis (A-JEPA only)
    slot_analysis = None
    if 'ajepa' in model_name:
        print("  Analyzing slots...")
        slot_analysis = analyze_slots(model, test_loader, device)
        print(f"    Active slots: {slot_analysis.active_slots}/{len(slot_analysis.slot_utilization)}")
        print(f"    Slot diversity: {slot_analysis.slot_diversity:.3f}")
    
    total_time = time.time() - start_time
    
    # Compile results
    results = DiagnosticResults(
        model_name=model_name,
        model_params=params,
        seed=seed,
        learning_curve=trainer.learning_curve,
        slot_analysis=slot_analysis,
        final_representation=final_rep,
        failure_analysis=failure,
        phase_results=phase_results,
        total_training_time=total_time,
        timestamp=datetime.now().isoformat(),
    )
    
    print(f"\n  FINAL ACCURACY: {failure.accuracy*100:.1f}%")
    print(f"  Training time: {total_time/60:.1f} minutes")
    
    # Create visualizations
    create_diagnostic_plots(results, output_dir)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Diagnostic Benchmark')
    parser.add_argument('--models', nargs='+', default=['ajepa_v3', 'vjepa_v3'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--num_train', type=int, default=200)
    parser.add_argument('--num_test', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_dir', type=str, default='results/diagnostic')
    parser.add_argument('--device', type=str, default=None)
    
    args = parser.parse_args()
    
    device = torch.device(args.device if args.device else 
                         ('cuda' if torch.cuda.is_available() else 'cpu'))
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("DIAGNOSTIC BENCHMARK")
    print("="*70)
    print(f"Models: {args.models}")
    print(f"Seeds: {args.seeds}")
    print(f"Device: {device}")
    
    all_results = []
    
    for model_name in args.models:
        for seed in args.seeds:
            results = run_diagnostic(
                model_name=model_name,
                seed=seed,
                num_train=args.num_train,
                num_test=args.num_test,
                batch_size=args.batch_size,
                device=device,
                output_dir=args.output_dir,
            )
            all_results.append(results)
    
    # Save all results
    output_file = os.path.join(args.output_dir, 'diagnostic_results.json')
    
    # Convert to serializable format
    def to_dict(obj):
        if hasattr(obj, '__dict__'):
            return {k: to_dict(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
        elif isinstance(obj, list):
            return [to_dict(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj
    
    with open(output_file, 'w') as f:
        json.dump([to_dict(r) for r in all_results], f, indent=2)
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    for r in all_results:
        print(f"\n{r.model_name} (seed={r.seed}):")
        print(f"  Accuracy: {r.failure_analysis.accuracy*100:.1f}%")
        print(f"  Rep Quality: var={r.final_representation.variance:.3f}, "
              f"rank={r.final_representation.effective_rank:.1f}")
        if r.slot_analysis:
            print(f"  Slots: {r.slot_analysis.active_slots} active, "
                  f"diversity={r.slot_analysis.slot_diversity:.3f}")
    
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
