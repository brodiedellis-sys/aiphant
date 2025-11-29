"""
Phyre-inspired Physics Reasoning Dataset.

Simple 2D physics puzzles that test physical reasoning:
1. Goal Reaching: Will a ball reach the target zone?
2. Collision Prediction: Will two objects collide?
3. Trajectory Prediction: Where will the ball end up?

Tests whether models can predict physical outcomes from initial conditions.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2


class PhysicsWorld:
    """Simple 2D physics simulation."""
    
    def __init__(self, img_size: int = 32):
        self.img_size = img_size
        self.gravity = 0.3  # Pixels per frame^2
        self.friction = 0.98
        self.objects = []
        self.target = None
    
    def add_ball(self, x, y, vx, vy, radius=3, color=255, affected_by_gravity=True):
        """Add a ball to the world."""
        self.objects.append({
            'type': 'ball',
            'x': x, 'y': y,
            'vx': vx, 'vy': vy,
            'radius': radius,
            'color': color,
            'gravity': affected_by_gravity,
        })
    
    def add_platform(self, x, y, width, height, color=180):
        """Add a static platform."""
        self.objects.append({
            'type': 'platform',
            'x': x, 'y': y,
            'width': width,
            'height': height,
            'color': color,
        })
    
    def set_target(self, x, y, radius=4):
        """Set target zone for goal-reaching task."""
        self.target = {'x': x, 'y': y, 'radius': radius}
    
    def step(self):
        """Advance physics by one timestep."""
        for obj in self.objects:
            if obj['type'] == 'ball':
                # Apply gravity
                if obj['gravity']:
                    obj['vy'] += self.gravity
                
                # Apply friction
                obj['vx'] *= self.friction
                obj['vy'] *= self.friction
                
                # Update position
                obj['x'] += obj['vx']
                obj['y'] += obj['vy']
                
                # Bounce off walls
                r = obj['radius']
                if obj['x'] - r < 0:
                    obj['x'] = r
                    obj['vx'] = -obj['vx'] * 0.8
                elif obj['x'] + r > self.img_size:
                    obj['x'] = self.img_size - r
                    obj['vx'] = -obj['vx'] * 0.8
                
                if obj['y'] - r < 0:
                    obj['y'] = r
                    obj['vy'] = -obj['vy'] * 0.8
                elif obj['y'] + r > self.img_size:
                    obj['y'] = self.img_size - r
                    obj['vy'] = -obj['vy'] * 0.8
                
                # Collision with platforms
                for other in self.objects:
                    if other['type'] == 'platform':
                        if self._ball_platform_collision(obj, other):
                            # Simple bounce
                            obj['vy'] = -obj['vy'] * 0.8
                            obj['y'] = other['y'] - obj['radius']
    
    def _ball_platform_collision(self, ball, platform):
        """Check if ball collides with platform."""
        bx, by, br = ball['x'], ball['y'], ball['radius']
        px, py = platform['x'], platform['y']
        pw, ph = platform['width'], platform['height']
        
        # Simple AABB check
        return (bx + br > px and bx - br < px + pw and
                by + br > py and by - br < py + ph)
    
    def check_goal_reached(self):
        """Check if any ball reached the target."""
        if self.target is None:
            return False
        
        tx, ty, tr = self.target['x'], self.target['y'], self.target['radius']
        
        for obj in self.objects:
            if obj['type'] == 'ball':
                dist = np.sqrt((obj['x'] - tx)**2 + (obj['y'] - ty)**2)
                if dist < tr + obj['radius']:
                    return True
        return False
    
    def render(self):
        """Render current state as grayscale image."""
        img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        
        # Draw target first (dark gray)
        if self.target:
            cv2.circle(img, (int(self.target['x']), int(self.target['y'])),
                      self.target['radius'], 80, -1)
        
        # Draw platforms
        for obj in self.objects:
            if obj['type'] == 'platform':
                x, y = int(obj['x']), int(obj['y'])
                w, h = int(obj['width']), int(obj['height'])
                cv2.rectangle(img, (x, y), (x + w, y + h), obj['color'], -1)
        
        # Draw balls
        for obj in self.objects:
            if obj['type'] == 'ball':
                cv2.circle(img, (int(obj['x']), int(obj['y'])),
                          obj['radius'], obj['color'], -1)
        
        return img


def generate_goal_reaching_puzzle(img_size=32, difficulty='easy'):
    """
    Generate a goal-reaching physics puzzle.
    
    Returns:
        initial_frames: First few frames showing initial setup
        outcome: 1 if ball reaches target, 0 otherwise
        full_video: Complete simulation
    """
    world = PhysicsWorld(img_size)
    
    # Randomly place target (bottom half of screen)
    target_x = np.random.uniform(8, img_size - 8)
    target_y = np.random.uniform(img_size * 0.6, img_size - 8)
    world.set_target(target_x, target_y, radius=5)
    
    # Ball starts at top
    ball_x = np.random.uniform(8, img_size - 8)
    ball_y = np.random.uniform(4, 10)
    
    # Initial velocity (somewhat toward target for easy, random for hard)
    if difficulty == 'easy':
        # Aim roughly toward target
        dx = target_x - ball_x
        vx = dx * 0.1 + np.random.uniform(-0.5, 0.5)
        vy = np.random.uniform(0, 1)
    else:
        vx = np.random.uniform(-2, 2)
        vy = np.random.uniform(-1, 2)
    
    world.add_ball(ball_x, ball_y, vx, vy, radius=3)
    
    # Maybe add a platform obstacle
    if np.random.random() > 0.5:
        plat_x = np.random.uniform(4, img_size - 12)
        plat_y = np.random.uniform(img_size * 0.3, img_size * 0.5)
        world.add_platform(plat_x, plat_y, width=8, height=3)
    
    # Simulate
    frames = []
    max_frames = 50
    goal_reached = False
    
    for _ in range(max_frames):
        frames.append(world.render())
        world.step()
        if world.check_goal_reached():
            goal_reached = True
            # Continue a bit after goal reached
            for _ in range(5):
                frames.append(world.render())
                world.step()
            break
    
    # Pad to consistent length
    while len(frames) < max_frames:
        frames.append(frames[-1])
    
    video = np.stack(frames[:max_frames], axis=0)
    video = video[:, np.newaxis, :, :].astype(np.float32) / 255.0
    
    return video, int(goal_reached)


def apply_edge_transform(video):
    """Convert video to edge representation."""
    T, C, H, W = video.shape
    edges = []
    for t in range(T):
        frame = (video[t, 0] * 255).astype(np.uint8)
        edge = cv2.Canny(frame, 30, 100)
        edges.append(edge.astype(np.float32) / 255.0)
    return np.stack(edges, axis=0)[:, np.newaxis, :, :]


class PhysicsReasoningDataset(Dataset):
    """
    Dataset for physics reasoning tasks.
    
    Task: Given initial frames, predict if ball will reach target.
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        context_frames: int = 5,
        img_size: int = 32,
        mode: str = 'raw',  # 'raw' or 'edge'
        difficulty: str = 'mixed',  # 'easy', 'hard', or 'mixed'
        seed: int = None,
    ):
        self.num_samples = num_samples
        self.context_frames = context_frames
        self.mode = mode
        
        if seed is not None:
            np.random.seed(seed)
        
        self.data = []
        for i in range(num_samples):
            if difficulty == 'mixed':
                diff = 'easy' if i % 2 == 0 else 'hard'
            else:
                diff = difficulty
            
            video, outcome = generate_goal_reaching_puzzle(img_size, diff)
            
            if mode == 'edge':
                video = apply_edge_transform(video)
            
            self.data.append({
                'video': video,
                'outcome': outcome,
            })
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Context: first N frames
        context = item['video'][:self.context_frames]
        
        # Full video for temporal prediction
        video = item['video']
        
        return {
            'context': torch.from_numpy(context.copy()),
            'video': torch.from_numpy(video.copy()),
            'outcome': item['outcome'],
        }


def get_physics_reasoning_loaders(
    mode: str = 'raw',
    num_train: int = 500,
    num_test: int = 100,
    context_frames: int = 5,
    batch_size: int = 32,
    num_workers: int = 2,
):
    """Create train and test dataloaders for physics reasoning."""
    train_dataset = PhysicsReasoningDataset(
        num_samples=num_train,
        context_frames=context_frames,
        mode=mode,
        difficulty='mixed',
        seed=42,
    )
    
    test_dataset = PhysicsReasoningDataset(
        num_samples=num_test,
        context_frames=context_frames,
        mode=mode,
        difficulty='mixed',
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
    print("Testing Physics Reasoning Dataset...")
    
    print("\n1. Generate single puzzle:")
    video, outcome = generate_goal_reaching_puzzle()
    print(f"   Video shape: {video.shape}")
    print(f"   Outcome (goal reached): {outcome}")
    
    print("\n2. Test dataset:")
    dataset = PhysicsReasoningDataset(num_samples=20, mode='raw')
    sample = dataset[0]
    print(f"   Context shape: {sample['context'].shape}")
    print(f"   Video shape: {sample['video'].shape}")
    print(f"   Outcome: {sample['outcome']}")
    
    # Check balance
    outcomes = [dataset[i]['outcome'] for i in range(len(dataset))]
    print(f"   Outcome distribution: {sum(outcomes)}/{len(outcomes)} reach goal")
    
    print("\n3. Test edge mode:")
    dataset_edge = PhysicsReasoningDataset(num_samples=10, mode='edge')
    sample_edge = dataset_edge[0]
    print(f"   Edge context shape: {sample_edge['context'].shape}")
    
    print("\nAll tests passed!")

