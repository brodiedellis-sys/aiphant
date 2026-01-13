"""
Zero-Shot Generalization Test for A-JEPA.

Trains mass probe on standard environment, then tests on:
1. New Friction (Ice/Sand)
2. Zero Gravity
3. Invisible Objects
4. New Shapes (Squares)
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

def generate_dataset(num_samples=100, img_size=32, **kwargs):
    """Generate dataset with specific physics parameters."""
    world = PhysicsWorld(img_size, 
                         gravity=kwargs.get('gravity', 0.3),
                         friction=kwargs.get('friction', 0.98))
    
    data = []
    masses = []
    
    visible = kwargs.get('visible', True)
    shape = kwargs.get('shape', 'circle')
    
    for _ in range(num_samples):
        mass = np.random.uniform(1, 10)
        radius = 2 + 0.6 * mass
        
        world.objects = []
        world.add_ball(
            x=np.random.uniform(8, 24),
            y=np.random.uniform(8, 24),
            vx=np.random.uniform(-1, 1),
            vy=np.random.uniform(-1, 1),
            radius=radius,
            mass=mass,
            visible=visible,
            shape=shape,
            affected_by_gravity=(kwargs.get('gravity', 0.3) > 0)
        )
        
        frames = []
        for _ in range(5):
            frames.append(world.render())
            world.step()
            
        video = np.stack(frames, axis=0)
        video = video[:, np.newaxis, :, :].astype(np.float32) / 255.0
        
        data.append(video)
        masses.append(mass)
        
    return torch.from_numpy(np.stack(data)), np.array(masses)

class MassProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x).squeeze(-1)

def evaluate_probe(model, probe, dataset_name, **kwargs):
    """Evaluate probe on a specific environment."""
    print(f"Generating {dataset_name} data...")
    X, y = generate_dataset(num_samples=200, **kwargs)
    device = next(model.parameters()).device
    X = X.to(device)
    y = torch.from_numpy(y).float().to(device)
    
    with torch.no_grad():
        emb = model.encoder.encode_video(X, return_all=False)
        pred = probe(emb)
        error = torch.mean(torch.abs(pred - y)).item()
        
    print(f"{dataset_name} MAE: {error:.4f}")
    return error

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
    
    # 1. Train on Standard Environment
    print("\nTraining Mass Probe on Standard Environment...")
    X_train, y_train = generate_dataset(num_samples=1000)
    X_train = X_train.to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    
    with torch.no_grad():
        emb_train = model.encoder.encode_video(X_train, return_all=False)
        
    probe = MassProbe(emb_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-2)
    criterion = nn.MSELoss()
    
    for epoch in range(100):
        probe.train()
        pred = probe(emb_train)
        loss = criterion(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Training Final Loss: {loss.item():.4f}")
    
    # 2. Evaluate on New Environments
    print("\nEvaluating Zero-Shot Generalization...")
    results = {}
    
    # Standard (Baseline)
    results['Standard'] = evaluate_probe(model, probe, "Standard Test")
    
    # Friction Changes
    results['Ice (No Friction)'] = evaluate_probe(model, probe, "Ice", friction=1.0)
    results['Sand (High Friction)'] = evaluate_probe(model, probe, "Sand", friction=0.8)
    
    # Gravity Changes
    results['Zero Gravity'] = evaluate_probe(model, probe, "Zero Gravity", gravity=0.0)
    
    # Appearance Changes
    results['Invisible Ball'] = evaluate_probe(model, probe, "Invisible", visible=False)
    results['Square Shape'] = evaluate_probe(model, probe, "Square", shape='square')
    
    # Plot Results
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    values = list(results.values())
    
    plt.bar(names, values, color=['gray', 'skyblue', 'orange', 'purple', 'red', 'green'])
    plt.ylabel('Mean Absolute Error (Mass)')
    plt.title('Zero-Shot Generalization: Mass Prediction')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/zero_shot_generalization.png')
    print("\nSaved results to results/zero_shot_generalization.png")

if __name__ == '__main__':
    main()
