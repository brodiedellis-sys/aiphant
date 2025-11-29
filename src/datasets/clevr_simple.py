"""
Simple CLEVR-style dataset for compositional generalization testing.

Generates 32x32 images with 2D shapes (circle, square, triangle) of different
colors and sizes. Tests whether models can generalize to novel shape-color
combinations not seen during training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2


# Define shapes and colors
SHAPES = ['circle', 'square', 'triangle']
COLORS = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'cyan': (0, 255, 255),
    'magenta': (255, 0, 255),
}
SIZES = ['small', 'medium', 'large']
SIZE_RADII = {'small': 4, 'medium': 6, 'large': 8}


def draw_shape(img, shape, color_name, size, x, y):
    """Draw a shape on the image at position (x, y)."""
    color = COLORS[color_name]
    radius = SIZE_RADII[size]
    
    if shape == 'circle':
        cv2.circle(img, (int(x), int(y)), radius, color, -1)
    
    elif shape == 'square':
        half = radius
        pts = np.array([
            [x - half, y - half],
            [x + half, y - half],
            [x + half, y + half],
            [x - half, y + half]
        ], dtype=np.int32)
        cv2.fillPoly(img, [pts], color)
    
    elif shape == 'triangle':
        half = radius
        pts = np.array([
            [x, y - half],           # top
            [x - half, y + half],    # bottom-left
            [x + half, y + half]     # bottom-right
        ], dtype=np.int32)
        cv2.fillPoly(img, [pts], color)


def generate_scene(
    num_objects: int = 2,
    img_size: int = 32,
    allowed_combinations: list = None,
    include_motion: bool = False,
    num_frames: int = 1,
):
    """
    Generate a scene with multiple objects.
    
    Args:
        num_objects: Number of objects in scene
        img_size: Image size
        allowed_combinations: List of (shape, color) tuples to use. If None, use all.
        include_motion: Whether to generate video with moving objects
        num_frames: Number of frames if include_motion=True
    
    Returns:
        If include_motion: video (T, 3, H, W) or (T, 1, H, W) for grayscale
        Otherwise: image (3, H, W)
        Also returns: list of object properties
    """
    # Generate object properties
    objects = []
    for _ in range(num_objects):
        if allowed_combinations:
            shape, color = allowed_combinations[np.random.randint(len(allowed_combinations))]
        else:
            shape = SHAPES[np.random.randint(len(SHAPES))]
            color = list(COLORS.keys())[np.random.randint(len(COLORS))]
        
        size = SIZES[np.random.randint(len(SIZES))]
        radius = SIZE_RADII[size]
        margin = radius + 2
        
        x = np.random.uniform(margin, img_size - margin)
        y = np.random.uniform(margin, img_size - margin)
        
        # Random velocity for motion
        vx = np.random.uniform(-2, 2)
        vy = np.random.uniform(-2, 2)
        
        objects.append({
            'shape': shape,
            'color': color,
            'size': size,
            'x': x,
            'y': y,
            'vx': vx,
            'vy': vy,
            'radius': radius,
        })
    
    if include_motion:
        frames = []
        for t in range(num_frames):
            img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            
            for obj in objects:
                draw_shape(img, obj['shape'], obj['color'], obj['size'], obj['x'], obj['y'])
                
                # Update position
                obj['x'] += obj['vx']
                obj['y'] += obj['vy']
                
                # Bounce off walls
                if obj['x'] - obj['radius'] < 0 or obj['x'] + obj['radius'] > img_size:
                    obj['vx'] = -obj['vx']
                    obj['x'] = np.clip(obj['x'], obj['radius'], img_size - obj['radius'])
                if obj['y'] - obj['radius'] < 0 or obj['y'] + obj['radius'] > img_size:
                    obj['vy'] = -obj['vy']
                    obj['y'] = np.clip(obj['y'], obj['radius'], img_size - obj['radius'])
            
            frames.append(img)
        
        video = np.stack(frames, axis=0)  # (T, H, W, 3)
        video = video.transpose(0, 3, 1, 2)  # (T, 3, H, W)
        video = video.astype(np.float32) / 255.0
        return video, objects
    
    else:
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        for obj in objects:
            draw_shape(img, obj['shape'], obj['color'], obj['size'], obj['x'], obj['y'])
        
        img = img.transpose(2, 0, 1)  # (3, H, W)
        img = img.astype(np.float32) / 255.0
        return img, objects


def to_grayscale(img):
    """Convert RGB image/video to grayscale."""
    if len(img.shape) == 4:  # Video: (T, 3, H, W)
        # Simple average
        return img.mean(axis=1, keepdims=True)
    else:  # Image: (3, H, W)
        return img.mean(axis=0, keepdims=True)


def to_edge(img):
    """Convert to edge map."""
    if len(img.shape) == 4:  # Video
        T = img.shape[0]
        edges = []
        for t in range(T):
            frame = (img[t].transpose(1, 2, 0) * 255).astype(np.uint8)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            edge = cv2.Canny(gray, 50, 150)
            edges.append(edge.astype(np.float32) / 255.0)
        return np.stack(edges, axis=0)[:, np.newaxis, :, :]
    else:  # Image
        frame = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edge = cv2.Canny(gray, 50, 150)
        return edge.astype(np.float32)[np.newaxis, :, :] / 255.0


# Predefined train/test splits for compositional generalization
# Train on some shape-color combos, test on held-out combos

TRAIN_COMBINATIONS = [
    ('circle', 'red'),
    ('circle', 'green'),
    ('circle', 'blue'),
    ('square', 'red'),
    ('square', 'yellow'),
    ('square', 'cyan'),
    ('triangle', 'green'),
    ('triangle', 'blue'),
    ('triangle', 'magenta'),
]

# Novel combinations not seen during training
TEST_NOVEL_COMBINATIONS = [
    ('circle', 'yellow'),    # circle with new color
    ('circle', 'cyan'),
    ('square', 'green'),     # square with new color
    ('square', 'blue'),
    ('triangle', 'red'),     # triangle with new color
    ('triangle', 'yellow'),
]


class CLEVRSimpleDataset(Dataset):
    """
    Simple CLEVR-style dataset for compositional generalization.
    
    Can generate:
    - Static images (for classification/recognition)
    - Videos (for temporal prediction)
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        num_objects: int = 2,
        img_size: int = 32,
        mode: str = 'rgb',  # 'rgb', 'grayscale', or 'edge'
        combinations: list = None,  # Shape-color combinations to use
        include_motion: bool = True,
        num_frames: int = 20,
        context_length: int = 5,
        max_horizon: int = 10,
        seed: int = None,
    ):
        self.num_samples = num_samples
        self.num_objects = num_objects
        self.img_size = img_size
        self.mode = mode
        self.combinations = combinations
        self.include_motion = include_motion
        self.num_frames = num_frames
        self.context_length = context_length
        self.max_horizon = max_horizon
        
        if seed is not None:
            np.random.seed(seed)
        
        # Pre-generate all scenes
        self.data = []
        for _ in range(num_samples):
            if include_motion:
                video, objects = generate_scene(
                    num_objects=num_objects,
                    img_size=img_size,
                    allowed_combinations=combinations,
                    include_motion=True,
                    num_frames=num_frames,
                )
                
                if mode == 'grayscale':
                    video = to_grayscale(video)
                elif mode == 'edge':
                    video = to_edge(video)
                
                self.data.append({'video': video, 'objects': objects})
            else:
                img, objects = generate_scene(
                    num_objects=num_objects,
                    img_size=img_size,
                    allowed_combinations=combinations,
                    include_motion=False,
                )
                
                if mode == 'grayscale':
                    img = to_grayscale(img)
                elif mode == 'edge':
                    img = to_edge(img)
                
                self.data.append({'image': img, 'objects': objects})
    
    def __len__(self):
        if self.include_motion:
            return self.num_samples * self.max_horizon
        return self.num_samples
    
    def __getitem__(self, idx):
        if self.include_motion:
            # Video mode - return context and target frames
            video_idx = idx // self.max_horizon
            horizon = (idx % self.max_horizon) + 1
            
            video = self.data[video_idx]['video']
            
            context = video[:self.context_length]
            target_idx = min(self.context_length + horizon - 1, self.num_frames - 1)
            target = video[target_idx]
            
            return {
                'context': torch.from_numpy(context.copy()),
                'target': torch.from_numpy(target.copy()),
                'horizon': horizon,
            }
        else:
            # Image mode
            img = self.data[idx]['image']
            objects = self.data[idx]['objects']
            
            # Create label from object properties (e.g., count shapes)
            label = len(objects)
            
            return {
                'image': torch.from_numpy(img.copy()),
                'label': label,
            }


def get_clevr_loaders(
    mode: str = 'rgb',
    num_train: int = 500,
    num_test_id: int = 100,
    num_test_ood: int = 100,
    num_objects: int = 2,
    include_motion: bool = True,
    batch_size: int = 32,
    num_workers: int = 2,
):
    """
    Create train, in-distribution test, and out-of-distribution test loaders.
    
    Train: Uses TRAIN_COMBINATIONS
    Test ID: Uses same TRAIN_COMBINATIONS
    Test OOD: Uses TEST_NOVEL_COMBINATIONS (unseen shape-color combos)
    """
    # For A-JEPA, use edge mode
    a_jepa_mode = 'edge'
    v_jepa_mode = 'grayscale' if mode != 'rgb' else 'rgb'
    
    # Training data
    train_dataset = CLEVRSimpleDataset(
        num_samples=num_train,
        num_objects=num_objects,
        mode=mode,
        combinations=TRAIN_COMBINATIONS,
        include_motion=include_motion,
        seed=42,
    )
    
    # In-distribution test (same combinations as training)
    test_id_dataset = CLEVRSimpleDataset(
        num_samples=num_test_id,
        num_objects=num_objects,
        mode=mode,
        combinations=TRAIN_COMBINATIONS,
        include_motion=include_motion,
        seed=123,
    )
    
    # Out-of-distribution test (novel combinations)
    test_ood_dataset = CLEVRSimpleDataset(
        num_samples=num_test_ood,
        num_objects=num_objects,
        mode=mode,
        combinations=TEST_NOVEL_COMBINATIONS,
        include_motion=include_motion,
        seed=456,
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False
    )
    
    test_id_loader = DataLoader(
        test_id_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )
    
    test_ood_loader = DataLoader(
        test_ood_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )
    
    return train_loader, test_id_loader, test_ood_loader


if __name__ == '__main__':
    print("Testing CLEVR-style dataset...")
    
    # Test static images
    print("\n1. Static image generation:")
    img, objects = generate_scene(num_objects=3, include_motion=False)
    print(f"   Image shape: {img.shape}")
    print(f"   Objects: {[(o['shape'], o['color']) for o in objects]}")
    
    # Test video generation
    print("\n2. Video generation:")
    video, objects = generate_scene(num_objects=2, include_motion=True, num_frames=10)
    print(f"   Video shape: {video.shape}")
    
    # Test edge conversion
    print("\n3. Edge conversion:")
    edge_video = to_edge(video)
    print(f"   Edge video shape: {edge_video.shape}")
    
    # Test dataset
    print("\n4. Dataset (video mode, edge):")
    dataset = CLEVRSimpleDataset(
        num_samples=10,
        mode='edge',
        combinations=TRAIN_COMBINATIONS,
        include_motion=True,
    )
    sample = dataset[0]
    print(f"   Context shape: {sample['context'].shape}")
    print(f"   Target shape: {sample['target'].shape}")
    
    # Test loaders
    print("\n5. Data loaders:")
    train_loader, test_id, test_ood = get_clevr_loaders(
        mode='edge', num_train=20, num_test_id=10, num_test_ood=10
    )
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Test ID batches: {len(test_id)}")
    print(f"   Test OOD batches: {len(test_ood)}")
    
    print("\nAll tests passed!")

