"""
JEPA Training Script for V-JEPA and A-JEPA variants on CIFAR-10.
"""

import argparse
import os

import torch
import torch.nn as nn
from tqdm import tqdm

from src.dataset import get_cifar10_loaders
from src.models import get_jepa_models
from src.transforms import RandomMask
from src.utils import cosine_similarity_loss, save_checkpoint, AverageMeter


def parse_args():
    parser = argparse.ArgumentParser(description='Train JEPA on CIFAR-10')
    parser.add_argument('--variant', type=str, default='v_jepa', 
                        choices=['v_jepa', 'a_jepa'],
                        help='JEPA variant: v_jepa (RGB) or a_jepa (Edge)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--mask_ratio', type=float, default=0.5,
                        help='Ratio of patches to mask')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory for CIFAR-10 data')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory for saving checkpoints')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    return parser.parse_args()


def train_epoch(encoder, predictor, dataloader, optimizer, masker, device):
    """Train for one epoch."""
    encoder.train()
    predictor.train()
    
    loss_meter = AverageMeter()
    
    pbar = tqdm(dataloader, desc='Training')
    for images, _ in pbar:
        images = images.to(device)
        batch_size = images.size(0)
        
        # Target: encode full image (no gradient for target)
        with torch.no_grad():
            target = encoder(images)
        
        # Context: encode masked image
        masked_images = torch.stack([masker(img) for img in images])
        masked_images = masked_images.to(device)
        context = encoder(masked_images)
        
        # Predict target from context
        pred = predictor(context)
        
        # Compute loss
        loss = cosine_similarity_loss(pred, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_meter.update(loss.item(), batch_size)
        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
    
    return loss_meter.avg


def main():
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Training {args.variant} for {args.epochs} epochs')
    
    # Determine mode based on variant
    mode = 'rgb' if args.variant == 'v_jepa' else 'edge'
    
    # Load data
    train_loader, _ = get_cifar10_loaders(
        mode=mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir
    )
    
    # Create models
    encoder, predictor = get_jepa_models(args.variant)
    encoder = encoder.to(device)
    predictor = predictor.to(device)
    
    # Create masker
    masker = RandomMask(mask_ratio=args.mask_ratio, patch_size=4)
    
    # Optimizer
    params = list(encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        
        loss = train_epoch(encoder, predictor, train_loader, optimizer, masker, device)
        scheduler.step()
        
        print(f'Epoch {epoch+1} - Loss: {loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.checkpoint_dir, f'{args.variant}_encoder.pth')
        save_checkpoint(encoder, predictor, optimizer, epoch, checkpoint_path)
        
        if loss < best_loss:
            best_loss = loss
            best_path = os.path.join(args.checkpoint_dir, f'{args.variant}_encoder_best.pth')
            save_checkpoint(encoder, predictor, optimizer, epoch, best_path)
            print(f'New best loss: {best_loss:.4f}')
    
    print(f'\nTraining complete. Best loss: {best_loss:.4f}')
    print(f'Checkpoint saved to: {checkpoint_path}')


if __name__ == '__main__':
    main()

