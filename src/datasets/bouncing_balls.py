"""
Synthetic bouncing balls dataset for temporal prediction tasks.

Generates 32x32 grayscale videos of 1-3 balls bouncing elastically.
Supports both wall bounces and ball-ball collisions.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2


class Ball:
    """A single bouncing ball with position, velocity, and radius."""
    
    def __init__(self, img_size: int = 32, min_radius: int = 2, max_radius: int = 5):
        self.img_size = img_size
        self.radius = np.random.randint(min_radius, max_radius + 1)
        self.mass = self.radius ** 2  # Mass proportional to area
        
        # Random starting position (within bounds)
        margin = self.radius + 1
        self.x = np.random.uniform(margin, img_size - margin)
        self.y = np.random.uniform(margin, img_size - margin)
        
        # Random velocity (pixels per frame)
        speed = np.random.uniform(1.0, 3.0)
        angle = np.random.uniform(0, 2 * np.pi)
        self.vx = speed * np.cos(angle)
        self.vy = speed * np.sin(angle)
    
    def step(self):
        """Move ball one timestep and bounce off walls."""
        self.x += self.vx
        self.y += self.vy
        
        # Bounce off walls (elastic collision)
        if self.x - self.radius < 0:
            self.x = self.radius
            self.vx = -self.vx
        elif self.x + self.radius > self.img_size:
            self.x = self.img_size - self.radius
            self.vx = -self.vx
        
        if self.y - self.radius < 0:
            self.y = self.radius
            self.vy = -self.vy
        elif self.y + self.radius > self.img_size:
            self.y = self.img_size - self.radius
            self.vy = -self.vy
    
    def draw(self, img: np.ndarray):
        """Draw ball on image."""
        cv2.circle(
            img,
            (int(self.x), int(self.y)),
            self.radius,
            255,  # White ball
            -1   # Filled
        )
    
    def distance_to(self, other: 'Ball') -> float:
        """Calculate distance between centers of two balls."""
        dx = self.x - other.x
        dy = self.y - other.y
        return np.sqrt(dx * dx + dy * dy)
    
    def is_colliding(self, other: 'Ball') -> bool:
        """Check if this ball is colliding with another."""
        return self.distance_to(other) < (self.radius + other.radius)


def resolve_collision(ball1: Ball, ball2: Ball):
    """
    Resolve elastic collision between two balls.
    Uses 2D elastic collision physics with conservation of momentum and energy.
    """
    # Vector from ball1 to ball2
    dx = ball2.x - ball1.x
    dy = ball2.y - ball1.y
    dist = np.sqrt(dx * dx + dy * dy)
    
    if dist == 0:
        # Balls at same position, push apart randomly
        angle = np.random.uniform(0, 2 * np.pi)
        dx, dy = np.cos(angle), np.sin(angle)
        dist = 1.0
    
    # Normal vector
    nx = dx / dist
    ny = dy / dist
    
    # Relative velocity
    dvx = ball1.vx - ball2.vx
    dvy = ball1.vy - ball2.vy
    
    # Relative velocity along collision normal
    dvn = dvx * nx + dvy * ny
    
    # Don't resolve if balls are moving apart
    if dvn > 0:
        return
    
    # Elastic collision formula (equal mass simplification for now)
    m1, m2 = ball1.mass, ball2.mass
    
    # Impulse scalar
    impulse = (2 * dvn) / (1/m1 + 1/m2)
    
    # Update velocities
    ball1.vx -= (impulse / m1) * nx
    ball1.vy -= (impulse / m1) * ny
    ball2.vx += (impulse / m2) * nx
    ball2.vy += (impulse / m2) * ny
    
    # Separate balls to prevent overlap
    overlap = (ball1.radius + ball2.radius) - dist
    if overlap > 0:
        separation = overlap / 2 + 0.5
        ball1.x -= separation * nx
        ball1.y -= separation * ny
        ball2.x += separation * nx
        ball2.y += separation * ny


def check_all_collisions(balls: list):
    """Check and resolve all ball-ball collisions."""
    n = len(balls)
    for i in range(n):
        for j in range(i + 1, n):
            if balls[i].is_colliding(balls[j]):
                resolve_collision(balls[i], balls[j])


def generate_video(
    num_frames: int = 20,
    num_balls: int = 2,
    img_size: int = 32,
    with_collisions: bool = True,
) -> np.ndarray:
    """
    Generate a single bouncing balls video.
    
    Args:
        num_frames: Number of frames to generate
        num_balls: Number of balls (1-3)
        img_size: Image size (32x32)
        with_collisions: Whether to enable ball-ball collisions
    
    Returns:
        Video tensor of shape (num_frames, 1, img_size, img_size)
    """
    balls = [Ball(img_size) for _ in range(num_balls)]
    
    # Ensure balls don't start overlapping
    max_attempts = 100
    for i in range(1, len(balls)):
        for attempt in range(max_attempts):
            overlapping = False
            for j in range(i):
                if balls[i].is_colliding(balls[j]):
                    overlapping = True
                    break
            if not overlapping:
                break
            # Reinitialize ball position
            margin = balls[i].radius + 1
            balls[i].x = np.random.uniform(margin, img_size - margin)
            balls[i].y = np.random.uniform(margin, img_size - margin)
    
    frames = []
    
    for _ in range(num_frames):
        # Create blank frame
        frame = np.zeros((img_size, img_size), dtype=np.uint8)
        
        # Draw all balls
        for ball in balls:
            ball.draw(frame)
        
        frames.append(frame)
        
        # Move balls for next frame
        for ball in balls:
            ball.step()
        
        # Check and resolve ball-ball collisions
        if with_collisions and num_balls > 1:
            check_all_collisions(balls)
    
    # Stack frames: (T, H, W) -> (T, 1, H, W)
    video = np.stack(frames, axis=0)
    video = video[:, np.newaxis, :, :]
    
    # Normalize to [0, 1]
    video = video.astype(np.float32) / 255.0
    
    return video


def apply_edge_transform(video: np.ndarray) -> np.ndarray:
    """
    Apply Canny edge detection to video frames (for A-JEPA).
    
    Args:
        video: (T, 1, H, W) float32 array in [0, 1]
    
    Returns:
        Edge video of same shape
    """
    T, C, H, W = video.shape
    edge_frames = []
    
    for t in range(T):
        frame = (video[t, 0] * 255).astype(np.uint8)
        edges = cv2.Canny(frame, 50, 150)
        edges = edges.astype(np.float32) / 255.0
        edge_frames.append(edges)
    
    edge_video = np.stack(edge_frames, axis=0)[:, np.newaxis, :, :]
    return edge_video


class BouncingBallsDataset(Dataset):
    """
    Dataset of synthetic bouncing balls videos for temporal prediction.
    
    Each sample returns:
        - context_frames: First N frames (for encoding context)
        - target_frame: A future frame (for prediction target)
        - horizon: How many steps ahead the target is
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        num_frames: int = 20,
        context_length: int = 5,
        max_horizon: int = 10,
        num_balls: int = 2,
        img_size: int = 32,
        mode: str = 'raw',  # 'raw' for V-JEPA, 'edge' for A-JEPA
        with_collisions: bool = True,  # Enable ball-ball collisions
        seed: int = None,
    ):
        """
        Args:
            num_samples: Number of videos to generate
            num_frames: Total frames per video
            context_length: Number of initial frames as context
            max_horizon: Maximum prediction horizon (target at context_length + horizon)
            num_balls: Number of balls per video (1-3)
            img_size: Frame size
            mode: 'raw' for grayscale, 'edge' for edge-detected (A-JEPA)
            with_collisions: Whether balls collide with each other
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.context_length = context_length
        self.max_horizon = max_horizon
        self.num_balls = num_balls
        self.img_size = img_size
        self.mode = mode
        self.with_collisions = with_collisions
        
        if seed is not None:
            np.random.seed(seed)
        
        # Pre-generate all videos
        self.videos = []
        for _ in range(num_samples):
            video = generate_video(num_frames, num_balls, img_size, with_collisions)
            if mode == 'edge':
                video = apply_edge_transform(video)
            self.videos.append(video)
    
    def __len__(self):
        return self.num_samples * self.max_horizon  # Each video tested at multiple horizons
    
    def __getitem__(self, idx):
        # Map index to video and horizon
        video_idx = idx // self.max_horizon
        horizon = (idx % self.max_horizon) + 1  # 1 to max_horizon
        
        video = self.videos[video_idx]
        
        # Context: first N frames
        context_frames = video[:self.context_length]  # (N, 1, H, W)
        
        # Target: frame at context_length + horizon - 1
        target_idx = self.context_length + horizon - 1
        if target_idx >= self.num_frames:
            target_idx = self.num_frames - 1
        
        target_frame = video[target_idx]  # (1, H, W)
        
        return {
            'context': torch.from_numpy(context_frames),      # (N, 1, H, W)
            'target': torch.from_numpy(target_frame),         # (1, H, W)
            'horizon': horizon,
            'video_idx': video_idx,
        }


def get_bouncing_balls_loaders(
    mode: str = 'raw',
    num_train: int = 500,
    num_test: int = 100,
    num_balls: int = 2,
    context_length: int = 5,
    max_horizon: int = 10,
    batch_size: int = 32,
    num_workers: int = 2,
    with_collisions: bool = True,
):
    """
    Create train and test dataloaders for bouncing balls.
    
    Args:
        mode: 'raw' for V-JEPA (grayscale), 'edge' for A-JEPA
        num_train: Number of training videos
        num_test: Number of test videos
        num_balls: Fixed number of balls per video
        context_length: Frames of context
        max_horizon: Max prediction steps ahead
        batch_size: Batch size
        num_workers: Dataloader workers
        with_collisions: Whether balls collide with each other
    
    Returns:
        train_loader, test_loader
    """
    train_dataset = BouncingBallsDataset(
        num_samples=num_train,
        num_balls=num_balls,
        context_length=context_length,
        max_horizon=max_horizon,
        mode=mode,
        with_collisions=with_collisions,
        seed=42,
    )
    
    test_dataset = BouncingBallsDataset(
        num_samples=num_test,
        num_balls=num_balls,
        context_length=context_length,
        max_horizon=max_horizon,
        mode=mode,
        with_collisions=with_collisions,
        seed=123,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    
    return train_loader, test_loader


def get_ood_test_loader(
    mode: str = 'raw',
    num_test: int = 100,
    num_balls: int = 3,
    context_length: int = 5,
    max_horizon: int = 10,
    batch_size: int = 32,
    num_workers: int = 2,
    with_collisions: bool = True,
):
    """
    Create a test dataloader with a different number of balls (for OOD evaluation).
    
    Args:
        mode: 'raw' for V-JEPA (grayscale), 'edge' for A-JEPA
        num_test: Number of test videos
        num_balls: Number of balls (different from training for OOD)
        context_length: Frames of context
        max_horizon: Max prediction steps ahead
        batch_size: Batch size
        num_workers: Dataloader workers
        with_collisions: Whether balls collide with each other
    
    Returns:
        test_loader
    """
    test_dataset = BouncingBallsDataset(
        num_samples=num_test,
        num_balls=num_balls,
        context_length=context_length,
        max_horizon=max_horizon,
        mode=mode,
        with_collisions=with_collisions,
        seed=456,  # Different seed for OOD test
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    
    return test_loader


if __name__ == '__main__':
    # Quick test
    print("Generating test video (no collisions)...")
    video = generate_video(num_frames=10, num_balls=2, with_collisions=False)
    print(f"Video shape: {video.shape}")  # (10, 1, 32, 32)
    
    print("\nGenerating test video (with collisions)...")
    video_coll = generate_video(num_frames=20, num_balls=3, with_collisions=True)
    print(f"Video with collisions shape: {video_coll.shape}")
    
    print("\nTesting dataset (with collisions)...")
    dataset = BouncingBallsDataset(num_samples=10, mode='raw', with_collisions=True)
    sample = dataset[0]
    print(f"Context shape: {sample['context'].shape}")
    print(f"Target shape: {sample['target'].shape}")
    print(f"Horizon: {sample['horizon']}")
    
    print("\nTesting edge mode (with collisions)...")
    dataset_edge = BouncingBallsDataset(num_samples=10, mode='edge', with_collisions=True)
    sample_edge = dataset_edge[0]
    print(f"Edge context shape: {sample_edge['context'].shape}")
    
    print("\nAll tests passed!")

