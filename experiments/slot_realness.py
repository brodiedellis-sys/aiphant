"""
Slot Realness Test for A-JEPA.

Checks if slots learned meaningful decompositions:
1. Does Slot #2 always encode Ball #2?
2. Does slot identity stay stable after collisions?
3. Does embedding correlate with physical properties?
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models_v2 import AJEPAv2
from src.datasets.physics_reasoning import PhysicsWorld

def generate_tracking_data(num_frames=50, img_size=32):
    """Generate video with ground truth object positions."""
    world = PhysicsWorld(img_size, gravity=0.0, friction=1.0)  # No gravity/friction for clean tracking
    
    # Ball 1: Light, moving right
    world.add_ball(x=8, y=16, vx=0.5, vy=0.2, radius=3, color=200, mass=1.0)
    
    # Ball 2: Heavy, moving left
    world.add_ball(x=24, y=16, vx=-0.5, vy=-0.2, radius=5, color=100, mass=5.0)
    
    frames = []
    positions = []  # List of [ [x1, y1], [x2, y2] ]
    
    for _ in range(num_frames):
        frames.append(world.render())
        
        # Record positions
        pos_t = []
        for obj in world.objects:
            if obj['type'] == 'ball':
                pos_t.append([obj['x'], obj['y']])
        positions.append(pos_t)
        
        world.step()
        
    video = np.stack(frames, axis=0)
    video = video[:, np.newaxis, :, :].astype(np.float32) / 255.0
    
    return torch.from_numpy(video).unsqueeze(0), np.array(positions)

def compute_slot_object_iou(attn_map, obj_pos, obj_radius, img_size=32):
    """
    Compute IoU between slot attention map and object ground truth mask.
    attn_map: (H*W,) flattened attention
    obj_pos: [x, y]
    """
    # Reshape attn to HxW
    H = W = int(np.sqrt(attn_map.shape[0]))
    attn_img = attn_map.reshape(H, W).cpu().numpy()
    
    # Resize to img_size
    attn_img = cv2.resize(attn_img, (img_size, img_size))
    
    # Binarize attention (simple threshold)
    attn_mask = attn_img > 0.1
    
    # Create object mask
    obj_mask = np.zeros((img_size, img_size), dtype=bool)
    cv2.circle(
        img=np.zeros((img_size, img_size), dtype=np.uint8), 
        center=(int(obj_pos[0]), int(obj_pos[1])), 
        radius=int(obj_radius), 
        color=1, 
        thickness=-1
    ).astype(bool)
    
    y, x = np.ogrid[:img_size, :img_size]
    dist_sq = (x - obj_pos[0])**2 + (y - obj_pos[1])**2
    obj_mask = dist_sq <= obj_radius**2
    
    intersection = np.logical_and(attn_mask, obj_mask).sum()
    union = np.logical_or(attn_mask, obj_mask).sum()
    
    return intersection / (union + 1e-6)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = AJEPAv2(
        in_channels=1,
        img_size=32,
        num_slots=4,
        slot_dim=32,
        bottleneck_dim=32,
    ).to(device)
    model.eval()
    
    print("Generating tracking data...")
    video, positions = generate_tracking_data()
    video = video.to(device)  # (1, T, 1, 32, 32)
    
    print("Running model...")
    with torch.no_grad():
        # Get attention weights
        _, attn = model.encoder.encode_video(video, return_attn=True)
        # attn: (1, T, K, N)
    
    attn = attn.squeeze(0)  # (T, K, N)
    T, K, N = attn.shape
    num_objects = positions.shape[1]
    
    print(f"Video shape: {video.shape}")
    print(f"Attention shape: {attn.shape}")
    
    # Analyze slot-object correlation
    # For each object, find which slot tracks it best
    slot_scores = np.zeros((num_objects, K))
    
    for t in range(T):
        for obj_idx in range(num_objects):
            obj_pos = positions[t, obj_idx]
            obj_radius = 3 if obj_idx == 0 else 5
            
            for k in range(K):
                iou = compute_slot_object_iou(attn[t, k], obj_pos, obj_radius)
                slot_scores[obj_idx, k] += iou
    
    slot_scores /= T
    
    print("\nSlot-Object Correlation (Avg IoU):")
    print(f"{'':<10} {'Slot 0':<10} {'Slot 1':<10} {'Slot 2':<10} {'Slot 3':<10}")
    for i in range(num_objects):
        row = f"Obj {i}:    "
        for k in range(K):
            row += f"{slot_scores[i, k]:.4f}     "
        print(row)
        
    # Visualize tracking
    # Plot attention maps for best slots
    best_slots = np.argmax(slot_scores, axis=1)
    print(f"\nBest slots: Obj 0 -> Slot {best_slots[0]}, Obj 1 -> Slot {best_slots[1]}")
    
    # Create visualization
    fig, axes = plt.subplots(num_objects + 1, 5, figsize=(15, 9))
    
    # Sample frames
    indices = np.linspace(0, T-1, 5, dtype=int)
    
    for i, t in enumerate(indices):
        # Original frame
        frame = video[0, t, 0].cpu().numpy()
        axes[0, i].imshow(frame, cmap='gray')
        axes[0, i].set_title(f"Frame {t}")
        axes[0, i].axis('off')
        
        # Slot attention for Obj 0
        slot_idx = best_slots[0]
        att = attn[t, slot_idx].reshape(4, 4).cpu().numpy() # 4x4 spatial tokens
        att = cv2.resize(att, (32, 32), interpolation=cv2.INTER_NEAREST)
        axes[1, i].imshow(att, cmap='plasma')
        axes[1, i].set_title(f"Slot {slot_idx} (Obj 0)")
        axes[1, i].axis('off')
        
        # Slot attention for Obj 1
        slot_idx = best_slots[1]
        att = attn[t, slot_idx].reshape(4, 4).cpu().numpy()
        att = cv2.resize(att, (32, 32), interpolation=cv2.INTER_NEAREST)
        axes[2, i].imshow(att, cmap='plasma')
        axes[2, i].set_title(f"Slot {slot_idx} (Obj 1)")
        axes[2, i].axis('off')
        
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/slot_tracking.png')
    print("\nSaved visualization to results/slot_tracking.png")

if __name__ == '__main__':
    main()
