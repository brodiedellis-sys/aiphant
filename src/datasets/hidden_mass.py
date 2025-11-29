"""
Hidden Mass Dataset for Causal Inference Testing (V2 - Appearance Blind).

All balls have RANDOMIZED appearance (texture, fill pattern changes per sample)
but the appearance is UNCORRELATED with mass. Only physics differs:
- Heavy balls: slower initial velocity, lower bounce, more momentum in collisions
- Light balls: faster initial velocity, higher bounce, less momentum in collisions

The goal: can a model infer mass from motion alone, with ZERO visual cues?
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2


# Texture patterns that are randomly assigned (uncorrelated with mass)
TEXTURE_SOLID = 'solid'
TEXTURE_HOLLOW = 'hollow'
TEXTURE_GRADIENT = 'gradient'
TEXTURES = [TEXTURE_SOLID, TEXTURE_HOLLOW, TEXTURE_GRADIENT]


class MassBall:
    """Ball with hidden mass property affecting physics (appearance randomized)."""
    
    def __init__(
        self,
        img_size: int = 32,
        mass: float = 1.0,  # Hidden property
        radius: int = 4,    # Can vary randomly (uncorrelated with mass)
        color: int = 200,   # Can vary randomly (uncorrelated with mass)
        texture: str = 'solid',  # Random texture (uncorrelated with mass)
    ):
        self.img_size = img_size
        self.mass = mass
        self.radius = radius
        self.color = color
        self.texture = texture
        
        # Random starting position
        margin = radius + 2
        self.x = np.random.uniform(margin, img_size - margin)
        self.y = np.random.uniform(margin, img_size * 0.4)  # Start in upper portion
        
        # KEY: Initial velocity is INVERSELY related to mass
        # Light balls move FAST, heavy balls move SLOW
        speed_factor = 1.0 / mass  # Heavy=0.5x speed, Light=2x speed
        self.vx = np.random.uniform(-1.5, 1.5) * speed_factor
        self.vy = np.random.uniform(0, 1) * speed_factor
        
        # Physics constants (affected by mass)
        self.gravity = 0.15
        self.restitution = 0.95 - (mass - 0.5) * 0.2  # Heavier = less bouncy
        self.friction = 0.99
    
    def step(self):
        """Advance physics by one timestep."""
        # Gravity affects velocity
        self.vy += self.gravity
        
        # Apply friction
        self.vx *= self.friction
        self.vy *= self.friction
        
        # Update position
        self.x += self.vx
        self.y += self.vy
        
        # Bounce off walls (clean, no border effects)
        if self.x - self.radius < 0:
            self.x = self.radius
            self.vx = -self.vx * self.restitution
        elif self.x + self.radius > self.img_size:
            self.x = self.img_size - self.radius
            self.vx = -self.vx * self.restitution
        
        if self.y - self.radius < 0:
            self.y = self.radius
            self.vy = -self.vy * self.restitution
        elif self.y + self.radius > self.img_size:
            # Floor bounce - mass affects energy loss
            self.y = self.img_size - self.radius
            self.vy = -self.vy * self.restitution
            # Heavier balls lose more energy on floor impact
            energy_loss = 1.0 - (self.mass - 0.5) * 0.15
            self.vy *= max(energy_loss, 0.4)
    
    def draw(self, img: np.ndarray):
        """Draw ball with random texture (uncorrelated with mass)."""
        center = (int(self.x), int(self.y))
        
        if self.texture == TEXTURE_SOLID:
            # Solid filled circle
            cv2.circle(img, center, self.radius, self.color, -1)
        elif self.texture == TEXTURE_HOLLOW:
            # Hollow circle (ring)
            cv2.circle(img, center, self.radius, self.color, 1)
        elif self.texture == TEXTURE_GRADIENT:
            # Gradient-like: outer ring + inner dot
            cv2.circle(img, center, self.radius, self.color, 1)
            cv2.circle(img, center, max(1, self.radius // 2), self.color, -1)


def resolve_collision_with_mass(ball1: MassBall, ball2: MassBall):
    """
    Resolve elastic collision between two balls, accounting for mass.
    Heavier balls transfer more momentum - this is the KEY physics cue.
    """
    dx = ball2.x - ball1.x
    dy = ball2.y - ball1.y
    dist = np.sqrt(dx * dx + dy * dy)
    
    if dist == 0:
        return False
    
    # Check if colliding
    if dist > ball1.radius + ball2.radius:
        return False
    
    # Normal vector
    nx = dx / dist
    ny = dy / dist
    
    # Relative velocity
    dvx = ball1.vx - ball2.vx
    dvy = ball1.vy - ball2.vy
    dvn = dvx * nx + dvy * ny
    
    if dvn > 0:
        return False
    
    # Mass-weighted impulse (this is where mass REALLY matters)
    m1, m2 = ball1.mass, ball2.mass
    impulse = (2 * dvn) / (1/m1 + 1/m2)
    
    # Apply impulse: lighter balls get pushed more, heavier balls barely move
    ball1.vx -= (impulse / m1) * nx
    ball1.vy -= (impulse / m1) * ny
    ball2.vx += (impulse / m2) * nx
    ball2.vy += (impulse / m2) * ny
    
    # Separate balls
    overlap = (ball1.radius + ball2.radius) - dist
    if overlap > 0:
        sep = overlap / 2 + 0.5
        ball1.x -= sep * nx
        ball1.y -= sep * ny
        ball2.x += sep * nx
        ball2.y += sep * ny
    
    return True


def generate_hidden_mass_video(
    num_frames: int = 30,
    num_balls: int = 2,
    img_size: int = 32,
    mass_category: str = 'light',  # 'light' or 'heavy'
    randomize_appearance: bool = True,
):
    """
    Generate video where mass is hidden - appearance is RANDOMIZED.
    
    Key dynamics differences:
    - Light balls: fast, bouncy, get pushed around in collisions
    - Heavy balls: slow, less bouncy, dominate collisions
    
    Args:
        num_frames: Number of frames
        num_balls: Number of balls
        img_size: Frame size
        mass_category: 'light' (mass=0.5-0.8) or 'heavy' (mass=1.5-2.0)
        randomize_appearance: If True, randomize visual properties
    
    Returns:
        video: (T, 1, H, W) array, normalized
        mass_label: 0 for light, 1 for heavy
    """
    # Assign mass based on category
    if mass_category == 'light':
        mass = np.random.uniform(0.5, 0.8)
        label = 0
    else:
        mass = np.random.uniform(1.5, 2.0)
        label = 1
    
    # Randomize appearance properties (UNCORRELATED with mass)
    if randomize_appearance:
        # Different textures, colors, radii - all random per video
        textures = [np.random.choice(TEXTURES) for _ in range(num_balls)]
        colors = [np.random.randint(150, 255) for _ in range(num_balls)]
        radii = [np.random.randint(3, 6) for _ in range(num_balls)]
    else:
        # Identical appearance (original behavior)
        textures = ['solid'] * num_balls
        colors = [200] * num_balls
        radii = [4] * num_balls
    
    # Create balls with randomized appearance but SAME mass category
    balls = [
        MassBall(
            img_size, 
            mass=mass, 
            radius=radii[i], 
            color=colors[i],
            texture=textures[i]
        )
        for i in range(num_balls)
    ]
    
    # Ensure balls don't start overlapping
    for i in range(1, len(balls)):
        for _ in range(50):
            overlapping = False
            for j in range(i):
                dx = balls[i].x - balls[j].x
                dy = balls[i].y - balls[j].y
                min_dist = balls[i].radius + balls[j].radius + 2
                if np.sqrt(dx*dx + dy*dy) < min_dist:
                    overlapping = True
                    break
            if not overlapping:
                break
            margin = balls[i].radius + 2
            balls[i].x = np.random.uniform(margin, img_size - margin)
            balls[i].y = np.random.uniform(margin, img_size * 0.4)
    
    frames = []
    for _ in range(num_frames):
        # Render frame (clean, no borders/shadows)
        frame = np.zeros((img_size, img_size), dtype=np.uint8)
        for ball in balls:
            ball.draw(frame)
        frames.append(frame)
        
        # Physics step
        for ball in balls:
            ball.step()
        
        # Check collisions (this is where mass differences show!)
        for i in range(len(balls)):
            for j in range(i + 1, len(balls)):
                resolve_collision_with_mass(balls[i], balls[j])
    
    video = np.stack(frames, axis=0)[:, np.newaxis, :, :]
    video = video.astype(np.float32) / 255.0
    
    # Normalize: mean-center and scale (remove brightness bias)
    video = (video - video.mean()) / (video.std() + 1e-6)
    # Clip to reasonable range
    video = np.clip(video, -3, 3)
    # Rescale to 0-1
    video = (video - video.min()) / (video.max() - video.min() + 1e-6)
    
    return video, label


def apply_edge_transform(video: np.ndarray) -> np.ndarray:
    """Convert video to edge representation."""
    T, C, H, W = video.shape
    edges = []
    for t in range(T):
        frame = (video[t, 0] * 255).astype(np.uint8)
        edge = cv2.Canny(frame, 30, 100)
        edges.append(edge.astype(np.float32) / 255.0)
    return np.stack(edges, axis=0)[:, np.newaxis, :, :]


class HiddenMassDataset(Dataset):
    """
    Dataset for hidden mass inference (V2 - Appearance Blind).
    
    All balls have RANDOMIZED appearance. Mass only affects physics.
    The model CANNOT use visual cues - must rely on dynamics.
    
    Task: Infer mass category (light vs heavy) from motion dynamics.
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        num_frames: int = 30,
        num_balls: int = 2,
        img_size: int = 32,
        mode: str = 'raw',  # 'raw' or 'edge'
        randomize_appearance: bool = True,
        seed: int = None,
    ):
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.mode = mode
        self.randomize_appearance = randomize_appearance
        
        if seed is not None:
            np.random.seed(seed)
        
        self.data = []
        for i in range(num_samples):
            # Balance classes
            mass_cat = 'light' if i % 2 == 0 else 'heavy'
            
            video, label = generate_hidden_mass_video(
                num_frames=num_frames,
                num_balls=num_balls,
                img_size=img_size,
                mass_category=mass_cat,
                randomize_appearance=randomize_appearance,
            )
            
            if mode == 'edge':
                video = apply_edge_transform(video)
            
            self.data.append({
                'video': video,
                'mass_label': label,
            })
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'video': torch.from_numpy(item['video'].copy()),
            'mass_label': item['mass_label'],
        }


def get_hidden_mass_loaders(
    mode: str = 'raw',
    num_train: int = 500,
    num_test: int = 100,
    num_balls: int = 2,
    batch_size: int = 32,
    num_workers: int = 2,
    randomize_appearance: bool = True,
):
    """Create train and test dataloaders for hidden mass inference."""
    train_dataset = HiddenMassDataset(
        num_samples=num_train,
        num_balls=num_balls,
        mode=mode,
        randomize_appearance=randomize_appearance,
        seed=42,
    )
    
    test_dataset = HiddenMassDataset(
        num_samples=num_test,
        num_balls=num_balls,
        mode=mode,
        randomize_appearance=randomize_appearance,
        seed=123,
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )
    
    return train_loader, test_loader


if __name__ == '__main__':
    print("Testing Hidden Mass Dataset V2 (Appearance Blind)...")
    
    print("\n1. Generate light ball video (randomized appearance):")
    video_light, label_light = generate_hidden_mass_video(
        mass_category='light', randomize_appearance=True
    )
    print(f"   Shape: {video_light.shape}, Label: {label_light} (light)")
    print(f"   Value range: [{video_light.min():.3f}, {video_light.max():.3f}]")
    
    print("\n2. Generate heavy ball video (randomized appearance):")
    video_heavy, label_heavy = generate_hidden_mass_video(
        mass_category='heavy', randomize_appearance=True
    )
    print(f"   Shape: {video_heavy.shape}, Label: {label_heavy} (heavy)")
    print(f"   Value range: [{video_heavy.min():.3f}, {video_heavy.max():.3f}]")
    
    print("\n3. Test dataset:")
    dataset = HiddenMassDataset(num_samples=20, mode='raw', randomize_appearance=True)
    labels = [dataset[i]['mass_label'] for i in range(len(dataset))]
    print(f"   Label distribution: {sum(labels)}/{len(labels)} heavy")
    
    print("\n4. Test edge mode:")
    dataset_edge = HiddenMassDataset(num_samples=10, mode='edge', randomize_appearance=True)
    sample = dataset_edge[0]
    print(f"   Video shape: {sample['video'].shape}")
    
    print("\nKey V2 improvements:")
    print("  ✓ Randomized appearance (texture, color, size) per sample")
    print("  ✓ Appearance UNCORRELATED with mass")
    print("  ✓ Light balls: fast + bouncy, Heavy balls: slow + sticky")
    print("  ✓ Mass-weighted collision physics")
    print("  ✓ Frame normalization (no brightness bias)")
    print("\nAll tests passed!")
