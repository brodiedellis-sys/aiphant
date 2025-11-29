"""
Diagnostic Script for A-JEPA v2 Collapse

Step 1: Visualize what A-JEPA actually sees (edge inputs)
Step 2: Test baseline encoder (no cognitive features) 
Step 3: Identify the culprit

If baseline fails → input/edge data problem
If baseline works → architecture problem (add features back one by one)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.datasets.hidden_mass import HiddenMassDataset, generate_hidden_mass_video
from torch.utils.data import DataLoader

# Try matplotlib for visualization
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False
    print("Warning: matplotlib not available")


def visualize_edge_inputs(num_samples=5, save_dir='results/diagnostics'):
    """Visualize what A-JEPA actually sees."""
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("STEP 1: Visualizing A-JEPA Edge Inputs")
    print("="*60)
    
    # Generate samples
    for mode in ['raw', 'edge']:
        dataset = HiddenMassDataset(
            num_samples=num_samples,
            mode=mode,
            randomize_appearance=True,
            seed=42,
        )
        
        print(f"\n{mode.upper()} mode:")
        
        for i in range(min(num_samples, 3)):
            sample = dataset[i]
            video = sample['video'].numpy()
            label = sample['mass_label']
            
            print(f"  Sample {i}: shape={video.shape}, label={'heavy' if label else 'light'}")
            print(f"    Value range: [{video.min():.3f}, {video.max():.3f}]")
            print(f"    Mean: {video.mean():.3f}, Std: {video.std():.3f}")
            
            # Check if mostly zeros (empty/black)
            nonzero_ratio = (np.abs(video) > 0.01).mean()
            print(f"    Non-zero pixels: {nonzero_ratio*100:.1f}%")
            
            if nonzero_ratio < 0.1:
                print(f"    ⚠️  WARNING: Very sparse! Mostly empty frames.")
            
            if HAS_PLT:
                # Save first and last frame comparison
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                
                frames_to_show = [0, video.shape[0]//3, 2*video.shape[0]//3, -1]
                for ax, frame_idx in zip(axes, frames_to_show):
                    frame = video[frame_idx, 0]
                    ax.imshow(frame, cmap='gray', vmin=0, vmax=1)
                    ax.set_title(f'Frame {frame_idx}')
                    ax.axis('off')
                
                fig.suptitle(f'{mode.upper()} - Sample {i} - {"Heavy" if label else "Light"}')
                plt.tight_layout()
                plt.savefig(f'{save_dir}/{mode}_sample_{i}.png', dpi=100)
                plt.close()
                print(f"    Saved: {save_dir}/{mode}_sample_{i}.png")
    
    return True


class SimpleEncoder(nn.Module):
    """
    Minimal encoder - NO cognitive features.
    Just conv + pool + fc. This is our baseline.
    """
    def __init__(self, in_channels=1, emb_dim=64):
        super().__init__()
        self.emb_dim = emb_dim
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # 32 -> 4 after 3 stride-2 convs
        self.fc = nn.Linear(128 * 4 * 4, emb_dim)
    
    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        z = self.fc(h)
        return F.normalize(z, dim=-1)
    
    def encode_video(self, video):
        B, T, C, H, W = video.shape
        frames = video.reshape(B * T, C, H, W)
        z = self.forward(frames)
        z = z.view(B, T, -1)
        return z.mean(dim=1)


class SimplePredictor(nn.Module):
    def __init__(self, emb_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim * 2, emb_dim),
        )
    
    def forward(self, x):
        return self.net(x)


class LinearProbe(nn.Module):
    def __init__(self, emb_dim, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(emb_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)


def cosine_loss(pred, target):
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return -torch.mean(torch.sum(pred * target, dim=-1))


def test_baseline_encoder(mode='edge', num_train=200, num_test=50, 
                          ssl_epochs=15, probe_epochs=30, device='cpu'):
    """Test if the edge data is even learnable with a simple encoder."""
    
    print("\n" + "="*60)
    print(f"STEP 2: Testing Baseline Encoder on '{mode}' data")
    print("="*60)
    
    # Create model
    encoder = SimpleEncoder(in_channels=1, emb_dim=64).to(device)
    predictor = SimplePredictor(emb_dim=64).to(device)
    
    params = sum(p.numel() for p in encoder.parameters())
    params += sum(p.numel() for p in predictor.parameters())
    print(f"Baseline params: {params:,} (tiny!)")
    
    # Load data
    train_dataset = HiddenMassDataset(num_samples=num_train, mode=mode, 
                                       randomize_appearance=True, seed=42)
    test_dataset = HiddenMassDataset(num_samples=num_test, mode=mode,
                                      randomize_appearance=True, seed=123)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # SSL Training
    print(f"\nSSL Training ({ssl_epochs} epochs)...")
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=1e-3
    )
    
    for epoch in range(ssl_epochs):
        encoder.train()
        predictor.train()
        total_loss = 0
        
        for batch in train_loader:
            video = batch['video'].to(device)
            B, T, C, H, W = video.shape
            
            context = video[:, :T//2]
            target = video[:, T//2:]
            
            ctx_emb = encoder.encode_video(context)
            with torch.no_grad():
                tgt_emb = encoder.encode_video(target)
            
            pred = predictor(ctx_emb)
            loss = cosine_loss(pred, tgt_emb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{ssl_epochs} - Loss: {total_loss/len(train_loader):.4f}")
    
    # Extract features
    print("\nExtracting features...")
    encoder.eval()
    
    def extract(loader):
        feats, labels = [], []
        with torch.no_grad():
            for batch in loader:
                video = batch['video'].to(device)
                z = encoder.encode_video(video)
                feats.append(z.cpu())
                labels.append(batch['mass_label'])
        return torch.cat(feats), torch.cat(labels)
    
    train_feats, train_labels = extract(train_loader)
    test_feats, test_labels = extract(test_loader)
    
    print(f"  Train features: {train_feats.shape}")
    print(f"  Feature variance: {train_feats.var().item():.4f}")
    
    # Check for collapse
    if train_feats.var().item() < 0.01:
        print("  ⚠️  WARNING: Features have very low variance - possible collapse!")
    
    # Train probe
    print(f"\nTraining linear probe ({probe_epochs} epochs)...")
    probe = LinearProbe(64, 2).to(device)
    probe_opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    probe_dataset = torch.utils.data.TensorDataset(train_feats, train_labels)
    probe_loader = DataLoader(probe_dataset, batch_size=32, shuffle=True)
    
    for epoch in range(probe_epochs):
        probe.train()
        for feats, lbls in probe_loader:
            feats, lbls = feats.to(device), lbls.to(device)
            logits = probe(feats)
            loss = criterion(logits, lbls)
            probe_opt.zero_grad()
            loss.backward()
            probe_opt.step()
    
    # Evaluate
    probe.eval()
    with torch.no_grad():
        train_acc = (probe(train_feats.to(device)).argmax(1) == train_labels.to(device)).float().mean().item() * 100
        test_acc = (probe(test_feats.to(device)).argmax(1) == test_labels.to(device)).float().mean().item() * 100
    
    print(f"\nResults:")
    print(f"  Train Accuracy: {train_acc:.1f}%")
    print(f"  Test Accuracy:  {test_acc:.1f}%")
    print(f"  (Chance = 50%)")
    
    return test_acc


def main():
    print("="*60)
    print("A-JEPA v2 DIAGNOSTIC")
    print("="*60)
    print("\nGoal: Find why A-JEPA collapsed to 50% (chance)")
    print("Strategy: Check inputs → Test baseline → Identify culprit")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Step 1: Visualize inputs
    visualize_edge_inputs()
    
    # Step 2: Test baseline on both raw and edge
    print("\n" + "="*60)
    print("BASELINE COMPARISON")
    print("="*60)
    
    raw_acc = test_baseline_encoder(mode='raw', device=device)
    edge_acc = test_baseline_encoder(mode='edge', device=device)
    
    # Step 3: Diagnosis
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)
    
    print(f"\nBaseline Results:")
    print(f"  Raw mode:  {raw_acc:.1f}%")
    print(f"  Edge mode: {edge_acc:.1f}%")
    
    if edge_acc < 55:
        print("\n❌ EDGE DATA IS THE PROBLEM!")
        print("   The edge transform is destroying information.")
        print("   Fix: Adjust Canny thresholds or use softer edge detection.")
    elif raw_acc > 70 and edge_acc > 60:
        print("\n✓ Data is fine. Problem is A-JEPA v2 architecture.")
        print("   Fix: Simplify cognitive features, add residuals, warmup.")
    else:
        print("\n⚠️  Both modes struggling. May need more training or data.")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    
    if edge_acc < 55:
        print("1. Fix edge detection (lower thresholds, use gradient magnitude)")
        print("2. Consider using raw grayscale instead of edges")
        print("3. Or use soft edges (Sobel) instead of binary Canny")
    else:
        print("1. Start with SimpleEncoder + cognitive features ONE AT A TIME")
        print("2. Add slot attention first (with residual connection)")
        print("3. Then add multi-timescale (with warmup)")
        print("4. Finally add sparsity (very low weight)")


if __name__ == '__main__':
    main()

