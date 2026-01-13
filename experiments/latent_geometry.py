"""
Latent Geometry Test for A-JEPA.

Tests if mass is encoded linearly in the latent space:
1. Interpolation: z_interp = a*z_light + (1-a)*z_heavy -> Probe(z_interp) should be linear.
2. PCA/UMAP: Do embeddings cluster by mass naturally?
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models_v2 import AJEPAv2
from src.datasets.physics_reasoning import PhysicsWorld

def generate_mass_dataset(num_samples=500, img_size=32):
    """Generate dataset of single balls with varying mass/radius."""
    world = PhysicsWorld(img_size)
    
    data = []
    masses = []
    
    for _ in range(num_samples):
        # Random mass between 1 and 10
        mass = np.random.uniform(1, 10)
        # Radius correlated with mass (r ~ m^0.5 for constant density 2D, or just linear for simplicity)
        # Let's use linear for clear visual distinction: r = 2 + 0.6 * mass
        radius = 2 + 0.6 * mass
        
        world.objects = []
        world.add_ball(
            x=np.random.uniform(8, 24),
            y=np.random.uniform(8, 24),
            vx=np.random.uniform(-1, 1),
            vy=np.random.uniform(-1, 1),
            radius=radius,
            mass=mass
        )
        
        # Render a few frames to get motion info
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
    
    # 1. Train Mass Probe
    print("\nGenerating training data for mass probe...")
    X_train, y_train = generate_mass_dataset(num_samples=1000)
    X_train = X_train.to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    
    print("Extracting embeddings...")
    with torch.no_grad():
        # Get mean embedding over time
        emb_train = model.encoder.encode_video(X_train, return_all=False)
        
    print(f"Embedding shape: {emb_train.shape}")
    
    probe = MassProbe(emb_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-2)
    criterion = nn.MSELoss()
    
    print("Training probe...")
    for epoch in range(100):
        probe.train()
        pred = probe(emb_train)
        loss = criterion(pred, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
            
    # 2. Interpolation Test
    print("\nRunning Interpolation Test...")
    # Generate light (mass=2) and heavy (mass=8) samples
    light_video, _ = generate_mass_dataset(num_samples=10) # Get a batch to average
    heavy_video, _ = generate_mass_dataset(num_samples=10)
    
    # Force masses for these specific samples
    # (Re-generate with fixed mass to be sure)
    def get_fixed_mass_video(mass):
        world = PhysicsWorld(32)
        radius = 2 + 0.6 * mass
        world.add_ball(16, 16, 0.5, 0.5, radius=radius, mass=mass)
        frames = []
        for _ in range(5):
            frames.append(world.render())
            world.step()
        video = np.stack(frames, axis=0)
        return torch.from_numpy(video).unsqueeze(0).unsqueeze(2).float() / 255.0
        
    # Get embeddings for specific light and heavy instances
    # We'll generate 10 pairs and average the results
    alphas = np.linspace(0, 1, 11)
    predicted_masses = []
    
    with torch.no_grad():
        z_light = model.encoder.encode_video(X_train[y_train < 3][:10], return_all=False).mean(0)
        z_heavy = model.encoder.encode_video(X_train[y_train > 8][:10], return_all=False).mean(0)
        
        for alpha in alphas:
            z_interp = alpha * z_light + (1 - alpha) * z_heavy
            pred = probe(z_interp.unsqueeze(0))
            predicted_masses.append(pred.item())
            
    # Plot interpolation
    plt.figure(figsize=(8, 5))
    plt.plot(alphas, predicted_masses, 'o-', label='Predicted Mass')
    # Ideal linear interpolation
    ideal = alphas * predicted_masses[0] + (1 - alphas) * predicted_masses[-1]
    plt.plot(alphas, ideal, 'r--', label='Linear Ideal')
    plt.xlabel('Interpolation Alpha (Light -> Heavy)')
    plt.ylabel('Predicted Mass')
    plt.title('Latent Space Interpolation: Mass')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/latent_interpolation.png')
    print("Saved interpolation plot to results/latent_interpolation.png")
    
    # 3. PCA Visualization
    print("\nRunning PCA on embeddings...")
    # Use the training embeddings
    emb_np = emb_train.cpu().numpy()
    mass_np = y_train.cpu().numpy()
    
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(emb_np)
    
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(z_pca[:, 0], z_pca[:, 1], c=mass_np, cmap='viridis', alpha=0.7)
    plt.colorbar(sc, label='Mass')
    plt.title('PCA of Latent Space (Colored by Mass)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, alpha=0.3)
    plt.savefig('results/latent_pca.png')
    print("Saved PCA plot to results/latent_pca.png")

if __name__ == '__main__':
    main()
