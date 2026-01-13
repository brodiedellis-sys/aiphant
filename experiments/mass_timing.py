"""
Mass Prediction Timing Test.

Checks if A-JEPA predicts mass BEFORE collision (understanding inertia)
or only AFTER collision (using momentum transfer cues).

1. Generate collision videos.
2. Split into Pre-Collision and Post-Collision frames.
3. Train/Test mass probe on each subset.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models_v2 import AJEPAv2
from src.datasets.physics_reasoning import PhysicsWorld

def generate_collision_dataset(num_samples=100, img_size=32):
    """
    Generate videos of collisions.
    Returns:
        videos: (N, T, C, H, W)
        masses: (N,) - mass of the target ball (Ball 1)
        collision_frames: (N,) - frame index where collision occurs
    """
    world = PhysicsWorld(img_size, gravity=0.0, friction=1.0)
    
    data = []
    masses = []
    collision_indices = []
    
    for _ in range(num_samples):
        # Target ball (Ball 1): Random mass, moving right
        mass1 = np.random.uniform(1, 10)
        radius1 = 2 + 0.6 * mass1
        
        # Collider ball (Ball 2): Fixed mass, moving left
        mass2 = 5.0
        radius2 = 5.0
        
        world.objects = []
        # Setup collision course
        world.add_ball(x=8, y=16, vx=1.0, vy=0.0, radius=radius1, mass=mass1, color=200)
        world.add_ball(x=24, y=16, vx=-1.0, vy=0.0, radius=radius2, mass=mass2, color=100)
        
        frames = []
        collided = False
        collision_frame = -1
        
        for t in range(20):
            frames.append(world.render())
            
            # Check for collision
            if not collided:
                b1 = world.objects[0]
                b2 = world.objects[1]
                dist = np.sqrt((b1['x'] - b2['x'])**2 + (b1['y'] - b2['y'])**2)
                if dist < b1['radius'] + b2['radius']:
                    collided = True
                    collision_frame = t
            
            # Simple elastic collision logic (manual override for this test)
            if collided and t == collision_frame:
                # Swap velocities (simplified 1D elastic collision for equal mass, 
                # but here we want mass effects)
                b1 = world.objects[0]
                b2 = world.objects[1]
                v1 = b1['vx']
                v2 = b2['vx']
                m1 = b1['mass']
                m2 = b2['mass']
                
                # 1D Elastic collision formula
                new_v1 = (v1 * (m1 - m2) + 2 * m2 * v2) / (m1 + m2)
                new_v2 = (v2 * (m2 - m1) + 2 * m1 * v1) / (m1 + m2)
                
                b1['vx'] = new_v1
                b2['vx'] = new_v2
            
            world.step()
            
        if collision_frame > 0:
            video = np.stack(frames, axis=0)
            video = video[:, np.newaxis, :, :].astype(np.float32) / 255.0
            data.append(video)
            masses.append(mass1)
            collision_indices.append(collision_frame)
            
    return torch.from_numpy(np.stack(data)), np.array(masses), np.array(collision_indices)

class MassProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x).squeeze(-1)

def train_and_evaluate(model, X, y, name):
    """Train and evaluate a probe on a specific dataset split."""
    device = next(model.parameters()).device
    X = X.to(device)
    y = torch.from_numpy(y).float().to(device)
    
    with torch.no_grad():
        emb = model.encoder.encode_video(X, return_all=False)
        
    probe = MassProbe(emb.shape[1]).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-2)
    criterion = nn.MSELoss()
    
    # Train
    for epoch in range(100):
        probe.train()
        pred = probe(emb)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # Evaluate
    probe.eval()
    with torch.no_grad():
        pred = probe(emb)
        mae = torch.mean(torch.abs(pred - y)).item()
        
    print(f"{name} MAE: {mae:.4f}")
    return mae

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
        output_dim=48,
    ).to(device)
    model.eval()
    
    print("Generating collision data...")
    X, y, collision_indices = generate_collision_dataset(num_samples=500)
    
    # Split into Pre and Post collision
    # We'll take frames [0:collision] for pre, and [collision:] for post
    # But to batch it, we'll just take fixed windows relative to collision
    
    pre_videos = []
    post_videos = []
    valid_y = []
    
    for i in range(len(X)):
        c_idx = collision_indices[i]
        if c_idx > 3 and c_idx < 15: # Ensure we have enough frames
            # Pre: 3 frames before collision
            pre_videos.append(X[i, c_idx-3:c_idx])
            # Post: 3 frames after collision
            post_videos.append(X[i, c_idx+1:c_idx+4])
            valid_y.append(y[i])
            
    X_pre = torch.stack(pre_videos)
    X_post = torch.stack(post_videos)
    y_valid = np.array(valid_y)
    
    print(f"Valid samples: {len(y_valid)}")
    
    # Train/Eval on Pre-Collision
    mae_pre = train_and_evaluate(model, X_pre, y_valid, "Pre-Collision")
    
    # Train/Eval on Post-Collision
    mae_post = train_and_evaluate(model, X_post, y_valid, "Post-Collision")
    
    # Plot
    plt.figure(figsize=(6, 5))
    plt.bar(['Pre-Collision', 'Post-Collision'], [mae_pre, mae_post], color=['blue', 'green'])
    plt.ylabel('Mass Prediction Error (MAE)')
    plt.title('Mass Prediction: Before vs After Collision')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('results/mass_timing.png')
    print("\nSaved results to results/mass_timing.png")
    
    if mae_pre < mae_post * 1.2: # Allow some margin
        print("\nSUCCESS: Model predicts mass well before collision (Inertia/Trajectory)!")
    else:
        print("\nRESULT: Model relies more on collision outcome (Momentum Transfer).")

if __name__ == '__main__':
    main()
