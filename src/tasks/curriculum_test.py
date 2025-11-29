"""
Curriculum Learning Test for A-JEPA v2

Strategy:
1. Phase 1 (Easy): 1 ball, no collisions - learn basic physics
2. Phase 2 (Medium): 2 balls, collisions - learn interactions
3. Sparsity annealing: Start with 0 L1 penalty, ramp up slowly
4. VICReg regularization to prevent collapse

Hypothesis: Curriculum helps A-JEPA's slot attention organize properly
before tackling complex multi-object physics.
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models_v2 import VJEPAv2, AJEPAv2, get_vjepa_v2, get_ajepa_v2
from src.datasets.hidden_mass import generate_hidden_mass_video
from torch.utils.data import DataLoader, Dataset

# Try to import visualization
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False


# =============================================================================
# Curriculum Dataset
# =============================================================================

class CurriculumDataset(Dataset):
    """
    Dataset that adjusts difficulty based on curriculum phase.
    
    Phases:
    - 'easy': 1 ball, simple bouncing
    - 'medium': 2 balls, interactions
    - 'hard': 2+ balls, varying masses
    """
    
    def __init__(
        self,
        num_samples: int = 200,
        num_frames: int = 30,
        img_size: int = 32,
        mode: str = 'edge',  # 'raw' or 'edge'
        phase: str = 'easy',  # 'easy', 'medium', 'hard'
        seed: int = None,
    ):
        self.num_samples = num_samples
        self.phase = phase
        self.mode = mode
        
        if seed is not None:
            np.random.seed(seed)
        
        self.data = []
        for i in range(num_samples):
            # Mass category (for probing later)
            mass_cat = 'light' if i % 2 == 0 else 'heavy'
            
            video, label = self._generate_phase_video(
                num_frames, img_size, mass_cat, phase
            )
            
            if mode == 'edge':
                video = self._apply_edge(video)
            
            self.data.append({
                'video': video,
                'mass_label': label,
            })
    
    def _generate_phase_video(self, num_frames, img_size, mass_cat, phase):
        """Generate video based on curriculum phase."""
        import cv2
        
        # Mass assignment
        if mass_cat == 'light':
            mass = np.random.uniform(0.5, 0.8)
            label = 0
        else:
            mass = np.random.uniform(1.5, 2.0)
            label = 1
        
        # Phase-specific settings
        if phase == 'easy':
            num_balls = 1  # Single ball - learn basic physics
        elif phase == 'medium':
            num_balls = 2  # Two balls - learn interactions
        else:  # hard
            num_balls = np.random.randint(2, 4)  # 2-3 balls
        
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
                'color': 200,  # Same color for all
                'mass': ball_mass,
                'restitution': 0.95 - (ball_mass - 0.5) * 0.2,
            }
            balls.append(ball)
        
        # Physics
        gravity = 0.15
        friction = 0.99
        
        frames = []
        for _ in range(num_frames):
            # Render
            frame = np.zeros((img_size, img_size), dtype=np.uint8)
            
            for ball in balls:
                cv2.circle(frame, (int(ball['x']), int(ball['y'])), 
                           ball['radius'], ball['color'], -1)
            
            frames.append(frame)
            
            # Physics step with collisions (for medium/hard)
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
            
            # Ball-ball collisions (for medium/hard phases)
            if phase in ['medium', 'hard'] and len(balls) > 1:
                for i in range(len(balls)):
                    for j in range(i + 1, len(balls)):
                        b1, b2 = balls[i], balls[j]
                        dx = b2['x'] - b1['x']
                        dy = b2['y'] - b1['y']
                        dist = np.sqrt(dx**2 + dy**2)
                        min_dist = b1['radius'] + b2['radius']
                        
                        if dist < min_dist and dist > 0:
                            # Collision response
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
                            
                            # Separate
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
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'video': torch.from_numpy(item['video'].copy()),
            'mass_label': item['mass_label'],
        }


# =============================================================================
# VICReg Loss (from ood_emergence_test)
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
    return total, {'sim': sim_loss.item(), 'var': var_loss.item(), 'cov': cov_loss.item()}


# =============================================================================
# Curriculum Training
# =============================================================================

class CurriculumTrainer:
    """
    Curriculum learning trainer for A-JEPA v2.
    
    Schedule:
    - Phase 1: Easy (1 ball) for N epochs, sparsity=0
    - Phase 2: Medium (2 balls) for N epochs, sparsity ramping
    - Phase 3: Hard (2-3 balls) for N epochs, full sparsity
    """
    
    def __init__(
        self,
        model,
        device,
        mode='edge',
        num_samples=200,
        batch_size=16,
        lr=1e-3,
    ):
        self.model = model.to(device)
        self.device = device
        self.mode = mode
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Track sparsity weight (starts at 0)
        self.sparsity_weight = 0.0
        
    def get_sparsity_weight(self, epoch, total_epochs, phase):
        """Anneal sparsity weight based on phase and epoch."""
        if phase == 'easy':
            return 0.0  # No sparsity in easy phase
        elif phase == 'medium':
            # Ramp from 0 to 0.5 during medium phase
            progress = epoch / total_epochs
            return 0.5 * progress
        else:  # hard
            # Full sparsity in hard phase
            return 1.0
    
    def train_phase(self, phase, epochs, phase_num, total_phases):
        """Train one curriculum phase."""
        print(f"\n{'='*60}")
        print(f"Phase {phase_num}/{total_phases}: {phase.upper()}")
        print(f"{'='*60}")
        
        # Create dataset for this phase
        dataset = CurriculumDataset(
            num_samples=self.num_samples,
            mode=self.mode,
            phase=phase,
            seed=42 + phase_num,
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        
        pbar = tqdm(range(epochs), desc=f'{phase} phase')
        losses = []
        
        for epoch in pbar:
            # Update sparsity weight
            self.sparsity_weight = self.get_sparsity_weight(epoch, epochs, phase)
            
            total_loss = 0
            total_var = 0
            
            for batch in loader:
                video = batch['video'].to(self.device)
                B, T, C, H, W = video.shape
                
                context = video[:, :T//2]
                target = video[:, T//2:]
                
                # Encode
                z_ctx = self.model.encode_video(context)
                z_tgt = self.model.encode_video(target)
                
                # VICReg loss
                loss, loss_parts = vicreg_loss(z_ctx, z_tgt)
                
                # Add sparsity penalty (scaled by current weight)
                if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'bottleneck'):
                    aux = self.model.encoder.bottleneck.get_aux_loss()
                    if isinstance(aux, torch.Tensor):
                        loss = loss + self.sparsity_weight * aux
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_var += loss_parts['var']
            
            n = len(loader)
            avg_loss = total_loss / n
            losses.append(avg_loss)
            
            pbar.set_postfix({
                'loss': f"{avg_loss:.3f}",
                'var': f"{total_var/n:.3f}",
                'sparse': f"{self.sparsity_weight:.2f}"
            })
        
        return losses
    
    def train_curriculum(self, easy_epochs=15, medium_epochs=20, hard_epochs=15):
        """Full curriculum training."""
        all_losses = []
        
        # Phase 1: Easy
        losses = self.train_phase('easy', easy_epochs, 1, 3)
        all_losses.extend(losses)
        
        # Phase 2: Medium
        losses = self.train_phase('medium', medium_epochs, 2, 3)
        all_losses.extend(losses)
        
        # Phase 3: Hard
        losses = self.train_phase('hard', hard_epochs, 3, 3)
        all_losses.extend(losses)
        
        return all_losses


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


def train_and_eval_probe(train_feats, train_labels, test_feats, test_labels, emb_dim, device, epochs=50):
    """Train probe and evaluate."""
    # Train
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
    
    # Evaluate
    probe.eval()
    test_feats, test_labels = test_feats.to(device), test_labels.to(device)
    
    with torch.no_grad():
        logits = probe(test_feats)
        preds = logits.argmax(dim=1)
        acc = (preds == test_labels).float().mean().item() * 100
    
    return acc


def visualize_latent_space(features, labels, title, save_path):
    """Create t-SNE visualization."""
    if not HAS_MATPLOTLIB or not HAS_TSNE:
        return None
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
    embedding = tsne.fit_transform(features)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#2ecc71', '#e74c3c']  # Green=light, Red=heavy
    for label_val in [0, 1]:
        mask = labels == label_val
        name = 'Light' if label_val == 0 else 'Heavy'
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=colors[label_val], label=name, alpha=0.7, s=50)
    
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=12)
    
    # Separation metric
    c0 = embedding[labels == 0].mean(axis=0)
    c1 = embedding[labels == 1].mean(axis=0)
    sep = np.linalg.norm(c0 - c1)
    spread = (np.std(embedding[labels == 0]) + np.std(embedding[labels == 1])) / 2
    ratio = sep / (spread + 1e-6)
    
    ax.text(0.02, 0.98, f'Separation: {ratio:.2f}', transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    return ratio


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Curriculum Learning Test')
    parser.add_argument('--easy_epochs', type=int, default=15, help='Epochs for easy phase')
    parser.add_argument('--medium_epochs', type=int, default=25, help='Epochs for medium phase')
    parser.add_argument('--hard_epochs', type=int, default=20, help='Epochs for hard phase')
    parser.add_argument('--probe_epochs', type=int, default=60, help='Epochs for linear probe')
    parser.add_argument('--num_train', type=int, default=400, help='Training samples per phase')
    parser.add_argument('--num_test', type=int, default=150, help='Test samples')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--config', type=str, default='default', help='Model config')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("CURRICULUM LEARNING TEST: A-JEPA v2")
    print("=" * 70)
    print(f"\nDevice: {device}")
    print(f"\nCurriculum Schedule:")
    print(f"  Phase 1 (Easy):   {args.easy_epochs} epochs - 1 ball, no sparsity")
    print(f"  Phase 2 (Medium): {args.medium_epochs} epochs - 2 balls, sparsity ramps 0→0.5")
    print(f"  Phase 3 (Hard):   {args.hard_epochs} epochs - 2-3 balls, full sparsity")
    total_epochs = args.easy_epochs + args.medium_epochs + args.hard_epochs
    print(f"  Total: {total_epochs} SSL epochs")
    
    # Create models
    print(f"\nCreating models (config: {args.config})...")
    
    # A-JEPA v2 with curriculum
    a_jepa_curriculum = get_ajepa_v2(in_channels=1, img_size=32, config=args.config)
    
    # A-JEPA v2 baseline (standard training, no curriculum)
    a_jepa_baseline = get_ajepa_v2(in_channels=1, img_size=32, config=args.config)
    
    # V-JEPA v2 for comparison
    v_jepa = get_vjepa_v2(in_channels=1, img_size=32, config='default')
    
    a_params = sum(p.numel() for p in a_jepa_curriculum.parameters())
    v_params = sum(p.numel() for p in v_jepa.parameters())
    
    print(f"A-JEPA v2 params: {a_params:,}")
    print(f"V-JEPA v2 params: {v_params:,}")
    
    # ==========================================================================
    # Train A-JEPA with Curriculum
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TRAINING A-JEPA v2 WITH CURRICULUM")
    print("=" * 70)
    
    curriculum_trainer = CurriculumTrainer(
        model=a_jepa_curriculum,
        device=device,
        mode='edge',
        num_samples=args.num_train,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    
    curriculum_losses = curriculum_trainer.train_curriculum(
        easy_epochs=args.easy_epochs,
        medium_epochs=args.medium_epochs,
        hard_epochs=args.hard_epochs,
    )
    
    # ==========================================================================
    # Train A-JEPA Baseline (No Curriculum - just 'hard' for same epochs)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TRAINING A-JEPA v2 BASELINE (NO CURRICULUM)")
    print("=" * 70)
    
    baseline_trainer = CurriculumTrainer(
        model=a_jepa_baseline,
        device=device,
        mode='edge',
        num_samples=args.num_train,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    
    # Train on hard only (same total epochs)
    baseline_losses = baseline_trainer.train_phase('hard', total_epochs, 1, 1)
    
    # ==========================================================================
    # Train V-JEPA (same epochs on 'hard')
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TRAINING V-JEPA v2 (COMPARISON)")
    print("=" * 70)
    
    v_trainer = CurriculumTrainer(
        model=v_jepa,
        device=device,
        mode='raw',  # RGB for V-JEPA
        num_samples=args.num_train,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    
    v_losses = v_trainer.train_phase('hard', total_epochs, 1, 1)
    
    # ==========================================================================
    # Evaluation
    # ==========================================================================
    print("\n" + "=" * 70)
    print("EVALUATION: Hidden Mass Inference")
    print("=" * 70)
    
    # Create test datasets
    test_dataset_edge = CurriculumDataset(
        num_samples=args.num_test,
        mode='edge',
        phase='hard',
        seed=999,
    )
    test_loader_edge = DataLoader(test_dataset_edge, batch_size=args.batch_size, shuffle=False)
    
    test_dataset_raw = CurriculumDataset(
        num_samples=args.num_test,
        mode='raw',
        phase='hard',
        seed=999,
    )
    test_loader_raw = DataLoader(test_dataset_raw, batch_size=args.batch_size, shuffle=False)
    
    # Training data for probes
    train_dataset_edge = CurriculumDataset(
        num_samples=args.num_train,
        mode='edge',
        phase='hard',
        seed=42,
    )
    train_loader_edge = DataLoader(train_dataset_edge, batch_size=args.batch_size, shuffle=False)
    
    train_dataset_raw = CurriculumDataset(
        num_samples=args.num_train,
        mode='raw',
        phase='hard',
        seed=42,
    )
    train_loader_raw = DataLoader(train_dataset_raw, batch_size=args.batch_size, shuffle=False)
    
    results = {}
    
    # A-JEPA Curriculum
    print("\n[A-JEPA v2 + Curriculum]")
    train_f, train_l = extract_features(a_jepa_curriculum, train_loader_edge, device)
    test_f, test_l = extract_features(a_jepa_curriculum, test_loader_edge, device)
    acc = train_and_eval_probe(train_f, train_l, test_f, test_l, 
                                a_jepa_curriculum.total_dim, device, args.probe_epochs)
    results['A-JEPA Curriculum'] = acc
    print(f"  Accuracy: {acc:.1f}%")
    
    # Visualization
    os.makedirs('results/curriculum', exist_ok=True)
    sep = visualize_latent_space(test_f.numpy(), test_l.numpy(),
                                  'A-JEPA v2 + Curriculum', 'results/curriculum/ajepa_curriculum.png')
    if sep:
        print(f"  Latent separation: {sep:.2f}")
    
    # A-JEPA Baseline
    print("\n[A-JEPA v2 Baseline (No Curriculum)]")
    train_f, train_l = extract_features(a_jepa_baseline, train_loader_edge, device)
    test_f, test_l = extract_features(a_jepa_baseline, test_loader_edge, device)
    acc = train_and_eval_probe(train_f, train_l, test_f, test_l,
                                a_jepa_baseline.total_dim, device, args.probe_epochs)
    results['A-JEPA Baseline'] = acc
    print(f"  Accuracy: {acc:.1f}%")
    
    sep = visualize_latent_space(test_f.numpy(), test_l.numpy(),
                                  'A-JEPA v2 Baseline', 'results/curriculum/ajepa_baseline.png')
    if sep:
        print(f"  Latent separation: {sep:.2f}")
    
    # V-JEPA
    print("\n[V-JEPA v2]")
    train_f, train_l = extract_features(v_jepa, train_loader_raw, device)
    test_f, test_l = extract_features(v_jepa, test_loader_raw, device)
    acc = train_and_eval_probe(train_f, train_l, test_f, test_l,
                                v_jepa.total_dim, device, args.probe_epochs)
    results['V-JEPA'] = acc
    print(f"  Accuracy: {acc:.1f}%")
    
    sep = visualize_latent_space(test_f.numpy(), test_l.numpy(),
                                  'V-JEPA v2', 'results/curriculum/vjepa.png')
    if sep:
        print(f"  Latent separation: {sep:.2f}")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("CURRICULUM LEARNING RESULTS")
    print("=" * 70)
    
    print(f"\n{'Model':<30} {'Params':>12} {'Accuracy':>12}")
    print("-" * 60)
    print(f"{'A-JEPA v2 + Curriculum':<30} {a_params:>10,} {results['A-JEPA Curriculum']:>10.1f}%")
    print(f"{'A-JEPA v2 Baseline':<30} {a_params:>10,} {results['A-JEPA Baseline']:>10.1f}%")
    print(f"{'V-JEPA v2':<30} {v_params:>10,} {results['V-JEPA']:>10.1f}%")
    print("-" * 60)
    
    # Curriculum benefit
    curriculum_benefit = results['A-JEPA Curriculum'] - results['A-JEPA Baseline']
    
    print(f"\nCurriculum Benefit: {curriculum_benefit:+.1f}%")
    
    if curriculum_benefit > 5:
        print("\n✓ CURRICULUM HELPS! Gradual complexity improves A-JEPA learning.")
    elif curriculum_benefit > 0:
        print("\n~ Curriculum provides marginal benefit.")
    else:
        print("\n✗ Curriculum didn't help in this run.")
    
    # Compare to V-JEPA
    a_best = max(results['A-JEPA Curriculum'], results['A-JEPA Baseline'])
    gap = a_best - results['V-JEPA']
    
    if gap > 0:
        print(f"\n🏆 A-JEPA beats V-JEPA by {gap:.1f}%!")
    elif gap > -5:
        print(f"\n≈ A-JEPA competitive with V-JEPA (gap: {gap:.1f}%)")
    else:
        print(f"\n  V-JEPA still ahead by {-gap:.1f}%")
    
    # Save results
    with open('results/curriculum/results.txt', 'w') as f:
        f.write("CURRICULUM LEARNING RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Schedule:\n")
        f.write(f"  Easy: {args.easy_epochs} epochs (1 ball)\n")
        f.write(f"  Medium: {args.medium_epochs} epochs (2 balls)\n")
        f.write(f"  Hard: {args.hard_epochs} epochs (2-3 balls)\n\n")
        for name, acc in results.items():
            f.write(f"{name}: {acc:.1f}%\n")
        f.write(f"\nCurriculum benefit: {curriculum_benefit:+.1f}%\n")
    
    print(f"\nResults saved to results/curriculum/")
    print("=" * 70)


if __name__ == '__main__':
    main()

