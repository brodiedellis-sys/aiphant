"""
Utility functions for training and evaluation.
"""

import os
import torch


def cosine_similarity_loss(pred, target):
    """
    Compute negative cosine similarity loss.
    
    Args:
        pred: Predicted embeddings (B, D)
        target: Target embeddings (B, D) - should have stop_gradient applied before calling
    
    Returns:
        Scalar loss value
    """
    pred = torch.nn.functional.normalize(pred, dim=-1)
    target = torch.nn.functional.normalize(target, dim=-1)
    
    # Negative cosine similarity
    loss = -torch.mean(torch.sum(pred * target, dim=-1))
    return loss


def save_checkpoint(encoder, predictor, optimizer, epoch, path):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'predictor_state_dict': predictor.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }, path)


def load_encoder(path, encoder):
    """Load encoder from checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    return encoder


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

