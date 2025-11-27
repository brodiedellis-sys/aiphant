"""
Linear Evaluation Script for V-JEPA and A-JEPA on CIFAR-10.
"""

import argparse
import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from src.dataset import get_cifar10_loaders, get_perturbed_test_loader
from src.models import get_jepa_models, LinearProbe
from src.utils import load_encoder, AverageMeter


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_time(encoder, input_shape, device, num_runs=100):
    """Measure average inference time per batch."""
    encoder.eval()
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = encoder(dummy_input)
    
    # Measure
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = encoder(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    return (elapsed / num_runs) * 1000  # ms per batch


def print_model_stats(encoder, variant, device, batch_size=256):
    """Print model efficiency statistics."""
    in_channels = 3 if variant == 'v_jepa' else 1
    input_shape = (batch_size, in_channels, 32, 32)
    
    param_count = count_parameters(encoder)
    inference_time = measure_inference_time(encoder, input_shape, device)
    
    print('\n' + '-'*50)
    print('Model Efficiency Stats')
    print('-'*50)
    print(f'Parameters:      {param_count:,} ({param_count/1e6:.2f}M)')
    print(f'Inference Time:  {inference_time:.2f} ms/batch (batch_size={batch_size})')
    print(f'Throughput:      {batch_size / (inference_time/1000):.0f} samples/sec')
    print('-'*50)


def parse_args():
    parser = argparse.ArgumentParser(description='Linear evaluation of JEPA on CIFAR-10')
    parser.add_argument('--variant', type=str, default='v_jepa',
                        choices=['v_jepa', 'a_jepa'],
                        help='JEPA variant: v_jepa (RGB) or a_jepa (Edge)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to encoder checkpoint (default: checkpoints/{variant}_encoder.pth)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of linear probe training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate for linear probe')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory for CIFAR-10 data')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    return parser.parse_args()


def extract_features(encoder, dataloader, device):
    """Extract features using frozen encoder."""
    encoder.eval()
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Extracting features'):
            images = images.to(device)
            features = encoder(images)
            all_features.append(features.cpu())
            all_labels.append(labels)
    
    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


def train_linear_probe(features, labels, emb_dim, device, epochs=100, lr=0.1):
    """Train linear classifier on extracted features."""
    # Create dataset from features
    dataset = torch.utils.data.TensorDataset(features, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    
    # Linear probe
    probe = LinearProbe(emb_dim, num_classes=10).to(device)
    
    # Optimizer and scheduler
    optimizer = torch.optim.SGD(probe.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(epochs):
        probe.train()
        loss_meter = AverageMeter()
        
        for feats, lbls in loader:
            feats, lbls = feats.to(device), lbls.to(device)
            
            logits = probe(feats)
            loss = criterion(logits, lbls)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_meter.update(loss.item(), feats.size(0))
        
        scheduler.step()
        
        if (epoch + 1) % 20 == 0:
            print(f'Probe Epoch {epoch+1}/{epochs} - Loss: {loss_meter.avg:.4f}')
    
    return probe


def evaluate(encoder, probe, dataloader, device):
    """Evaluate accuracy on a dataloader."""
    encoder.eval()
    probe.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            
            features = encoder(images)
            logits = probe(features)
            
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total


def main():
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Evaluating {args.variant}')
    
    # Determine mode and embedding dim based on variant
    mode = 'rgb' if args.variant == 'v_jepa' else 'edge'
    emb_dim = 512 if args.variant == 'v_jepa' else 128
    
    # Load data
    train_loader, test_loader = get_cifar10_loaders(
        mode=mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir
    )
    
    perturbed_loader = get_perturbed_test_loader(
        mode=mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir
    )
    
    # Load encoder
    encoder, _ = get_jepa_models(args.variant)
    
    checkpoint_path = args.checkpoint or f'./checkpoints/{args.variant}_encoder.pth'
    if not os.path.exists(checkpoint_path):
        print(f'Error: Checkpoint not found at {checkpoint_path}')
        print('Please train the model first using train_jepa.py')
        return
    
    encoder = load_encoder(checkpoint_path, encoder)
    encoder = encoder.to(device)
    encoder.eval()
    
    print(f'Loaded encoder from {checkpoint_path}')
    
    # Print model efficiency stats
    print_model_stats(encoder, args.variant, device, args.batch_size)
    
    # Extract training features
    print('\nExtracting training features...')
    train_features, train_labels = extract_features(encoder, train_loader, device)
    
    # Train linear probe
    print('\nTraining linear probe...')
    probe = train_linear_probe(
        train_features, train_labels, emb_dim, device,
        epochs=args.epochs, lr=args.lr
    )
    
    # Evaluate on clean test set
    print('\nEvaluating on clean test set...')
    clean_acc = evaluate(encoder, probe, test_loader, device)
    
    # Evaluate on perturbed test set
    print('\nEvaluating on perturbed test set...')
    perturbed_acc = evaluate(encoder, probe, perturbed_loader, device)
    
    # Print results
    print('\n' + '='*50)
    print(f'Results for {args.variant.upper()}')
    print('='*50)
    print(f'Clean Accuracy:     {clean_acc:.2f}%')
    print(f'Perturbed Accuracy: {perturbed_acc:.2f}%')
    print(f'Robustness Gap:     {clean_acc - perturbed_acc:.2f}%')
    print('='*50)


if __name__ == '__main__':
    main()

