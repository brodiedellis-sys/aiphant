"""
Compare V-JEPA vs A-JEPA results side-by-side.
"""

import argparse
import os
import time

import torch
import torch.nn as nn

from src.dataset import get_cifar10_loaders, get_perturbed_test_loader
from src.models import get_jepa_models, LinearProbe
from src.utils import load_encoder, AverageMeter
from linear_eval import (
    count_parameters,
    measure_inference_time,
    extract_features,
    train_linear_probe,
    evaluate,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Compare V-JEPA vs A-JEPA')
    parser.add_argument('--probe_epochs', type=int, default=100,
                        help='Number of linear probe training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory for CIFAR-10 data')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    return parser.parse_args()


def evaluate_variant(variant, device, args):
    """Evaluate a single variant and return metrics."""
    mode = 'rgb' if variant == 'v_jepa' else 'edge'
    emb_dim = 512 if variant == 'v_jepa' else 128
    in_channels = 3 if variant == 'v_jepa' else 1
    
    # Check checkpoint exists
    checkpoint_path = f'./checkpoints/{variant}_encoder.pth'
    if not os.path.exists(checkpoint_path):
        return None
    
    # Load encoder
    encoder, _ = get_jepa_models(variant)
    encoder = load_encoder(checkpoint_path, encoder)
    encoder = encoder.to(device)
    encoder.eval()
    
    # Efficiency metrics
    param_count = count_parameters(encoder)
    input_shape = (args.batch_size, in_channels, 32, 32)
    inference_time = measure_inference_time(encoder, input_shape, device)
    throughput = args.batch_size / (inference_time / 1000)
    
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
    
    # Extract features and train probe
    print(f'\n  Extracting features for {variant}...')
    train_features, train_labels = extract_features(encoder, train_loader, device)
    
    print(f'  Training linear probe for {variant}...')
    probe = train_linear_probe(
        train_features, train_labels, emb_dim, device,
        epochs=args.probe_epochs, lr=0.1
    )
    
    # Evaluate
    print(f'  Evaluating {variant}...')
    clean_acc = evaluate(encoder, probe, test_loader, device)
    perturbed_acc = evaluate(encoder, probe, perturbed_loader, device)
    
    return {
        'variant': variant,
        'params': param_count,
        'inference_ms': inference_time,
        'throughput': throughput,
        'clean_acc': clean_acc,
        'perturbed_acc': perturbed_acc,
        'robustness_gap': clean_acc - perturbed_acc,
    }


def print_comparison_table(results):
    """Print side-by-side comparison table."""
    print('\n')
    print('=' * 70)
    print('                    V-JEPA vs A-JEPA Comparison')
    print('=' * 70)
    print(f'{"Metric":<25} {"V-JEPA":>20} {"A-JEPA":>20}')
    print('-' * 70)
    
    v = results.get('v_jepa', {})
    a = results.get('a_jepa', {})
    
    def fmt(val, fmt_str='.2f'):
        if val is None:
            return 'N/A'
        return f'{val:{fmt_str}}'
    
    # Parameters
    v_params = f"{v.get('params', 0)/1e6:.2f}M" if v else 'N/A'
    a_params = f"{a.get('params', 0)/1e6:.2f}M" if a else 'N/A'
    print(f'{"Parameters":<25} {v_params:>20} {a_params:>20}')
    
    # Inference time
    print(f'{"Inference (ms/batch)":<25} {fmt(v.get("inference_ms")):>20} {fmt(a.get("inference_ms")):>20}')
    
    # Throughput
    print(f'{"Throughput (samples/s)":<25} {fmt(v.get("throughput"), ".0f"):>20} {fmt(a.get("throughput"), ".0f"):>20}')
    
    print('-' * 70)
    
    # Accuracy metrics
    print(f'{"Clean Accuracy (%)":<25} {fmt(v.get("clean_acc")):>20} {fmt(a.get("clean_acc")):>20}')
    print(f'{"Perturbed Accuracy (%)":<25} {fmt(v.get("perturbed_acc")):>20} {fmt(a.get("perturbed_acc")):>20}')
    print(f'{"Robustness Gap (%)":<25} {fmt(v.get("robustness_gap")):>20} {fmt(a.get("robustness_gap")):>20}')
    
    print('=' * 70)
    
    # Summary
    if v and a:
        print('\nKey Findings:')
        
        # Parameter efficiency
        param_ratio = v['params'] / a['params']
        print(f'  - V-JEPA has {param_ratio:.1f}x more parameters than A-JEPA')
        
        # Speed comparison
        speed_ratio = a['throughput'] / v['throughput']
        print(f'  - A-JEPA is {speed_ratio:.1f}x faster in throughput')
        
        # Robustness comparison
        if a['robustness_gap'] < v['robustness_gap']:
            print(f'  - A-JEPA is more robust (smaller gap: {a["robustness_gap"]:.1f}% vs {v["robustness_gap"]:.1f}%)')
        else:
            print(f'  - V-JEPA is more robust (smaller gap: {v["robustness_gap"]:.1f}% vs {a["robustness_gap"]:.1f}%)')


def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    results = {}
    
    # Evaluate V-JEPA
    print('\n' + '='*50)
    print('Evaluating V-JEPA (RGB, 512-dim)')
    print('='*50)
    v_result = evaluate_variant('v_jepa', device, args)
    if v_result:
        results['v_jepa'] = v_result
    else:
        print('  Checkpoint not found. Skipping V-JEPA.')
    
    # Evaluate A-JEPA
    print('\n' + '='*50)
    print('Evaluating A-JEPA (Edge, 128-dim)')
    print('='*50)
    a_result = evaluate_variant('a_jepa', device, args)
    if a_result:
        results['a_jepa'] = a_result
    else:
        print('  Checkpoint not found. Skipping A-JEPA.')
    
    # Print comparison
    if results:
        print_comparison_table(results)
    else:
        print('\nNo checkpoints found. Please train models first:')
        print('  python train_jepa.py --variant v_jepa --epochs 20')
        print('  python train_jepa.py --variant a_jepa --epochs 20')


if __name__ == '__main__':
    main()

