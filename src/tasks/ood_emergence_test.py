"""
OOD Generalization & Emergence Test

Tests:
1. In-distribution (ID) accuracy
2. Out-of-distribution (OOD) robustness:
   - Different textures
   - Different lighting/brightness
   - Different object counts
   - Different ball sizes
3. Latent space visualization (UMAP/t-SNE)
   - Does abstraction emerge before probe succeeds?
   
Hypothesis: V-JEPA accuracy collapses on OOD, A-JEPA survives.
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

from src.datasets.hidden_mass import HiddenMassDataset, generate_hidden_mass_video
from src.models_v2 import VJEPAv2, AJEPAv2, get_vjepa_v2, get_ajepa_v2
from torch.utils.data import DataLoader, Dataset

# Try to import visualization libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found, skipping visualizations")

try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: umap-learn not found, using t-SNE only")


# =============================================================================
# OOD Dataset Variants
# =============================================================================

class OODHiddenMassDataset(Dataset):
    """
    Hidden mass dataset with OOD perturbations.
    
    OOD modes:
    - 'texture': Different ball rendering (solid/hollow/gradient mix)
    - 'brightness': Extreme brightness variations
    - 'count': Different number of balls (3-4 instead of 2)
    - 'size': Different ball sizes (tiny or large)
    - 'combined': All perturbations at once
    """
    
    def __init__(
        self,
        num_samples: int = 100,
        num_frames: int = 30,
        img_size: int = 32,
        mode: str = 'raw',
        ood_type: str = 'none',  # 'none', 'texture', 'brightness', 'count', 'size', 'combined'
        seed: int = None,
    ):
        self.num_samples = num_samples
        self.ood_type = ood_type
        
        if seed is not None:
            np.random.seed(seed)
        
        self.data = []
        for i in range(num_samples):
            mass_cat = 'light' if i % 2 == 0 else 'heavy'
            
            # Generate with OOD perturbations
            video, label = self._generate_ood_video(
                num_frames, img_size, mass_cat, ood_type
            )
            
            if mode == 'edge':
                video = self._apply_edge(video)
            
            self.data.append({
                'video': video,
                'mass_label': label,
            })
    
    def _generate_ood_video(self, num_frames, img_size, mass_cat, ood_type):
        """Generate video with OOD perturbations."""
        import cv2
        
        # Mass assignment
        if mass_cat == 'light':
            mass = np.random.uniform(0.5, 0.8)
            label = 0
        else:
            mass = np.random.uniform(1.5, 2.0)
            label = 1
        
        # OOD parameters
        if ood_type == 'count':
            num_balls = np.random.randint(3, 5)  # 3-4 balls instead of 2
        else:
            num_balls = 2
        
        if ood_type == 'size':
            # Extreme sizes
            radius = np.random.choice([2, 7])  # Very small or very large
        else:
            radius = np.random.randint(3, 6)
        
        if ood_type in ['brightness', 'combined']:
            brightness_mult = np.random.choice([0.3, 2.0])  # Very dark or very bright
        else:
            brightness_mult = 1.0
        
        # Generate balls
        balls = []
        for _ in range(num_balls):
            ball = {
                'x': np.random.uniform(radius + 2, img_size - radius - 2),
                'y': np.random.uniform(radius + 2, img_size * 0.4),
                'vx': np.random.uniform(-1.5, 1.5) / mass,
                'vy': np.random.uniform(0, 1) / mass,
                'radius': radius if ood_type != 'combined' else np.random.randint(2, 8),
                'color': np.random.randint(150, 255),
                'mass': mass,
                'restitution': 0.95 - (mass - 0.5) * 0.2,
            }
            balls.append(ball)
        
        # Physics constants
        gravity = 0.15
        friction = 0.99
        
        frames = []
        for _ in range(num_frames):
            # Render frame
            frame = np.zeros((img_size, img_size), dtype=np.uint8)
            
            for ball in balls:
                color = int(ball['color'] * brightness_mult)
                color = max(0, min(255, color))
                
                # Different textures for OOD
                if ood_type in ['texture', 'combined']:
                    texture = np.random.choice(['solid', 'hollow', 'cross'])
                    if texture == 'solid':
                        cv2.circle(frame, (int(ball['x']), int(ball['y'])), 
                                   ball['radius'], color, -1)
                    elif texture == 'hollow':
                        cv2.circle(frame, (int(ball['x']), int(ball['y'])), 
                                   ball['radius'], color, 2)
                    else:  # cross
                        cv2.circle(frame, (int(ball['x']), int(ball['y'])), 
                                   ball['radius'], color, 1)
                        cv2.line(frame, 
                                 (int(ball['x'] - ball['radius']), int(ball['y'])),
                                 (int(ball['x'] + ball['radius']), int(ball['y'])),
                                 color, 1)
                else:
                    cv2.circle(frame, (int(ball['x']), int(ball['y'])), 
                               ball['radius'], color, -1)
            
            frames.append(frame)
            
            # Physics step
            for ball in balls:
                ball['vy'] += gravity
                ball['vx'] *= friction
                ball['vy'] *= friction
                ball['x'] += ball['vx']
                ball['y'] += ball['vy']
                
                # Bounce
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
# Latent Space Visualization
# =============================================================================

def visualize_latent_space(
    features: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    save_path: str,
    method: str = 'tsne'
):
    """
    Create UMAP/t-SNE visualization of latent space.
    
    Checks if mass classes separate even when linear probe accuracy is low.
    """
    if not HAS_MATPLOTLIB:
        print("Skipping visualization (matplotlib not available)")
        return
    
    # Reduce dimensionality
    if method == 'umap' and HAS_UMAP:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
        embedding = reducer.fit_transform(features)
    elif HAS_TSNE:
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
        embedding = reducer.fit_transform(features)
    else:
        print("No dimensionality reduction available")
        return
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#2ecc71', '#e74c3c']  # Green for light, red for heavy
    markers = ['o', 's']
    
    for label_val in [0, 1]:
        mask = labels == label_val
        label_name = 'Light' if label_val == 0 else 'Heavy'
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=colors[label_val],
            marker=markers[label_val],
            label=label_name,
            alpha=0.7,
            s=50,
        )
    
    ax.set_title(f'{model_name} Latent Space ({method.upper()})', fontsize=14)
    ax.legend(fontsize=12)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    
    # Add separation metric
    light_center = embedding[labels == 0].mean(axis=0)
    heavy_center = embedding[labels == 1].mean(axis=0)
    separation = np.linalg.norm(light_center - heavy_center)
    
    light_spread = np.std(embedding[labels == 0])
    heavy_spread = np.std(embedding[labels == 1])
    avg_spread = (light_spread + heavy_spread) / 2
    
    separation_ratio = separation / (avg_spread + 1e-6)
    
    ax.text(0.02, 0.98, f'Separation ratio: {separation_ratio:.2f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return separation_ratio


# =============================================================================
# VICReg-style Regularization (prevents collapse)
# =============================================================================

def variance_loss(z, gamma=1.0):
    """
    Variance regularization: prevent embeddings from collapsing to a point.
    From VICReg: https://arxiv.org/abs/2105.04906
    """
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return torch.mean(F.relu(gamma - std))


def covariance_loss(z):
    """
    Covariance regularization: decorrelate features.
    From VICReg: https://arxiv.org/abs/2105.04906
    """
    B, D = z.shape
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (B - 1)
    
    # Off-diagonal elements should be zero
    off_diag = cov.flatten()[:-1].view(D-1, D+1)[:, 1:].flatten()
    return (off_diag ** 2).mean()


def vicreg_loss(pred, target, sim_weight=25.0, var_weight=25.0, cov_weight=1.0):
    """
    Combined VICReg loss: similarity + variance + covariance
    """
    # Invariance (similarity)
    sim_loss = F.mse_loss(pred, target)
    
    # Variance (prevent collapse)
    var_loss = variance_loss(pred) + variance_loss(target)
    
    # Covariance (decorrelation)
    cov_loss = covariance_loss(pred) + covariance_loss(target)
    
    total = sim_weight * sim_loss + var_weight * var_loss + cov_weight * cov_loss
    
    return total, {
        'sim': sim_loss.item(),
        'var': var_loss.item(),
        'cov': cov_loss.item(),
    }


# =============================================================================
# Training and Evaluation
# =============================================================================

class LinearProbe(nn.Module):
    def __init__(self, emb_dim, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(emb_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)


def train_ssl(model, dataloader, optimizer, device, epochs, model_name):
    """Self-supervised training with VICReg anti-collapse."""
    model.train()
    
    pbar = tqdm(range(epochs), desc=f'{model_name} SSL')
    for epoch in pbar:
        total_loss = 0
        total_var = 0
        
        for batch in dataloader:
            video = batch['video'].to(device)
            B, T, C, H, W = video.shape
            
            context = video[:, :T//2]
            target = video[:, T//2:]
            
            # Encode
            z_ctx = model.encode_video(context)
            z_tgt = model.encode_video(target)
            
            # Use VICReg loss instead of just cosine
            loss, loss_parts = vicreg_loss(z_ctx, z_tgt)
            
            # Add auxiliary losses from model (if any)
            if hasattr(model, 'get_aux_loss') and callable(model.get_aux_loss):
                aux_loss = model.get_aux_loss()
                if isinstance(aux_loss, torch.Tensor) and aux_loss.numel() > 0:
                    loss += aux_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_var += loss_parts['var']
        
        n = len(dataloader)
        pbar.set_postfix({
            'loss': f"{total_loss/n:.3f}",
            'var': f"{total_var/n:.3f}"
        })
    
    return total_loss / len(dataloader)


def extract_features(model, dataloader, device):
    """Extract features using frozen encoder."""
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            video = batch['video'].to(device)
            labels = batch['mass_label']
            features = model.encode_video(video)
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
    
    for _ in range(epochs):
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


def run_ood_experiment(model, model_name, mode, device, args):
    """Run full OOD experiment for one model."""
    print(f"\n{'='*70}")
    print(f"{model_name}")
    print(f"{'='*70}")
    
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters())
    emb_dim = model.total_dim
    
    print(f"Parameters: {params:,}")
    print(f"Embedding dim: {emb_dim}")
    
    # Create ID training data
    train_dataset = HiddenMassDataset(
        num_samples=args.num_train,
        mode=mode,
        randomize_appearance=True,
        seed=42,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # SSL training
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f"\nPhase 1: Self-supervised training ({args.ssl_epochs} epochs)...")
    train_ssl(model, train_loader, optimizer, device, args.ssl_epochs, model_name)
    
    # Extract features and train probe on ID data
    print(f"\nPhase 2: Training linear probe...")
    train_feats, train_labels = extract_features(model, train_loader, device)
    probe = train_probe(train_feats, train_labels, emb_dim, device, args.probe_epochs)
    
    # Evaluate on ID and OOD
    results = {'name': model_name, 'params': params}
    ood_types = ['none', 'texture', 'brightness', 'count', 'size', 'combined']
    
    print(f"\nPhase 3: Evaluating ID and OOD...")
    
    for ood_type in ood_types:
        test_dataset = OODHiddenMassDataset(
            num_samples=args.num_test,
            mode=mode,
            ood_type=ood_type,
            seed=123 + hash(ood_type) % 1000,
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        test_feats, test_labels = extract_features(model, test_loader, device)
        acc = evaluate_probe(probe, test_feats, test_labels, device)
        
        key = 'ID' if ood_type == 'none' else f'OOD_{ood_type}'
        results[key] = acc
        
        # Visualize latent space
        if args.visualize and HAS_MATPLOTLIB:
            os.makedirs('results/latent_viz', exist_ok=True)
            safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
            viz_path = f'results/latent_viz/{safe_name}_{ood_type}.png'
            sep_ratio = visualize_latent_space(
                test_feats.numpy(),
                test_labels.numpy(),
                f'{model_name} ({ood_type})',
                viz_path,
                method='tsne'
            )
            results[f'{key}_separation'] = sep_ratio
            print(f"  {key}: {acc:.1f}% (separation: {sep_ratio:.2f})")
        else:
            print(f"  {key}: {acc:.1f}%")
    
    return results


def print_ood_comparison(v_results, a_results):
    """Print OOD comparison table."""
    print("\n")
    print("=" * 90)
    print("        OOD GENERALIZATION: V-JEPA v2 vs A-JEPA v2 (LARGE)")
    print("=" * 90)
    
    print(f"\n{'Condition':<20} {'V-JEPA v2':>15} {'A-JEPA v2':>15} {'Δ (A-V)':>15} {'Winner':>15}")
    print("-" * 90)
    
    conditions = ['ID', 'OOD_texture', 'OOD_brightness', 'OOD_count', 'OOD_size', 'OOD_combined']
    
    a_wins = 0
    v_wins = 0
    
    for cond in conditions:
        v_acc = v_results.get(cond, 0)
        a_acc = a_results.get(cond, 0)
        diff = a_acc - v_acc
        
        if diff > 2:
            winner = "A-JEPA ✓"
            a_wins += 1
        elif diff < -2:
            winner = "V-JEPA ✓"
            v_wins += 1
        else:
            winner = "Tie"
        
        print(f"{cond:<20} {v_acc:>14.1f}% {a_acc:>14.1f}% {diff:>+14.1f}% {winner:>15}")
    
    print("-" * 90)
    
    # Average OOD
    ood_conds = [c for c in conditions if c.startswith('OOD')]
    v_ood_avg = np.mean([v_results.get(c, 0) for c in ood_conds])
    a_ood_avg = np.mean([a_results.get(c, 0) for c in ood_conds])
    
    print(f"{'OOD Average':<20} {v_ood_avg:>14.1f}% {a_ood_avg:>14.1f}% {a_ood_avg-v_ood_avg:>+14.1f}%")
    
    # Robustness gap (ID - OOD)
    v_gap = v_results.get('ID', 0) - v_ood_avg
    a_gap = a_results.get('ID', 0) - a_ood_avg
    
    print(f"{'Robustness Gap':<20} {v_gap:>14.1f}% {a_gap:>14.1f}% {a_gap-v_gap:>+14.1f}%")
    
    print("=" * 90)
    
    # Verdict
    print("\n" + "=" * 90)
    if a_wins > v_wins:
        print(f"🏆 A-JEPA v2 WINS {a_wins}/{len(conditions)} conditions!")
        print("\n✓ Hypothesis confirmed: A-JEPA survives OOD perturbations better")
        print("  Edge-based + cognitive architecture doesn't rely on raw visual patterns")
    elif v_wins > a_wins:
        print(f"🏆 V-JEPA v2 WINS {v_wins}/{len(conditions)} conditions!")
        print("\n✗ Hypothesis not confirmed on this run")
    else:
        print(f"🤝 TIE: Both models win {a_wins} conditions each")
    
    if a_gap < v_gap:
        print(f"\n✓ A-JEPA is MORE ROBUST: smaller gap ({a_gap:.1f}% vs {v_gap:.1f}%)")
    
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description='OOD Generalization & Emergence Test')
    parser.add_argument('--ssl_epochs', type=int, default=25)
    parser.add_argument('--probe_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_train', type=int, default=500)
    parser.add_argument('--num_test', type=int, default=100)
    parser.add_argument('--visualize', action='store_true', default=True)
    parser.add_argument('--config', type=str, default='large', 
                        help='Model config: default, large')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 90)
    print("OOD GENERALIZATION & EMERGENCE TEST")
    print("=" * 90)
    print(f"\nDevice: {device}")
    print(f"Config: {args.config}")
    print(f"\nOOD Perturbations:")
    print("  • texture: Different ball rendering styles")
    print("  • brightness: Extreme brightness variations")
    print("  • count: 3-4 balls instead of 2")
    print("  • size: Very small or very large balls")
    print("  • combined: All perturbations at once")
    print(f"\nVisualization: {'Enabled' if args.visualize else 'Disabled'}")
    
    # Create models with matching config
    print(f"\nCreating models with '{args.config}' config...")
    
    # V-JEPA v2 (always uses default config but we can scale it)
    if args.config == 'large':
        from src.models_v2 import VJEPAv2
        v_jepa = VJEPAv2(
            in_channels=1, img_size=32, 
            emb_dim=192, memory_dim=192,
            num_pred_steps=5, use_uncertainty=True
        )
    else:
        v_jepa = get_vjepa_v2(in_channels=1, img_size=32, config='default')
    
    a_jepa = get_ajepa_v2(in_channels=1, img_size=32, config=args.config)
    
    v_params = sum(p.numel() for p in v_jepa.parameters())
    a_params = sum(p.numel() for p in a_jepa.parameters())
    print(f"V-JEPA v2 params: {v_params:,}")
    print(f"A-JEPA v2 params: {a_params:,}")
    print(f"Ratio: {v_params/a_params:.2f}x")
    
    # Run experiments
    v_results = run_ood_experiment(v_jepa, "V-JEPA v2 Large", 'raw', device, args)
    a_results = run_ood_experiment(a_jepa, "A-JEPA v2 Large", 'edge', device, args)
    
    # Print comparison
    print_ood_comparison(v_results, a_results)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/ood_emergence_test.txt', 'w') as f:
        f.write("OOD GENERALIZATION & EMERGENCE TEST\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"V-JEPA v2 Large:\n")
        for k, v in v_results.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\nA-JEPA v2 Large:\n")
        for k, v in a_results.items():
            f.write(f"  {k}: {v}\n")
    
    print(f"\nResults saved to results/ood_emergence_test.txt")
    if args.visualize:
        print(f"Visualizations saved to results/latent_viz/")


if __name__ == '__main__':
    main()

