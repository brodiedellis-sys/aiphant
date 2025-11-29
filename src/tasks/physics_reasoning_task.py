"""
Physics Reasoning Test (Phase 2.3)

Goal: determine whether A-JEPA (edge-based, relational) can predict physical
outcomes (ball reaches target) better than V-JEPA (visual/grayscale) when
trained on synthetic goal-reaching puzzles inspired by PHYRE.
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from tqdm import tqdm

# Ensure repo root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.datasets.physics_reasoning import get_physics_reasoning_loaders
from src.models import SmallEncoder, Predictor


class PhysicsReasoningModel(nn.Module):
    """Encode context frames and predict binary outcome."""

    def __init__(self, in_channels: int, emb_dim: int, width_mult: float, use_relational: bool):
        super().__init__()
        self.encoder = SmallEncoder(
            in_channels=in_channels,
            emb_dim=emb_dim,
            width_mult=width_mult,
            use_relational=use_relational,
        )
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim * 2, 2),
        )

    def encode_context(self, context: torch.Tensor) -> torch.Tensor:
        """
        Encode context frames and mean-pool over time.
        Args:
            context: (B, T, C, H, W)
        Returns:
            context embedding: (B, D)
        """
        B, T, C, H, W = context.shape
        frames = context.view(B * T, C, H, W)
        embeddings = self.encoder(frames)
        embeddings = embeddings.view(B, T, -1)
        context_emb = embeddings.mean(dim=1)
        return context_emb

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        emb = self.encode_context(context)
        logits = self.classifier(emb)
        return logits


def get_models(variant: str):
    """Configure model per variant."""
    if variant == "v_jepa":
        # Grayscale input, larger emb dim
        model = PhysicsReasoningModel(
            in_channels=1,
            emb_dim=256,
            width_mult=1.0,
            use_relational=False,
        )
    elif variant == "a_jepa":
        # Edge input, smaller emb dim with relational block
        model = PhysicsReasoningModel(
            in_channels=1,
            emb_dim=128,
            width_mult=0.5,
            use_relational=True,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return model


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0
    total = 0
    correct = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        context = batch["context"].to(device)  # (B, T, C, H, W)
        labels = batch["outcome"].to(device)

        logits = model(context)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * context.size(0)
        total += context.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()

    return running_loss / total, correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0
    total = 0
    correct = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            context = batch["context"].to(device)
            labels = batch["outcome"].to(device)

            logits = model(context)
            loss = criterion(logits, labels)

            running_loss += loss.item() * context.size(0)
            total += context.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()

    return running_loss / total, correct / total


def run_experiment(variant: str, device, args):
    print(f"\n{'='*60}")
    print(f"Running {variant.upper()}")
    print(f"{'='*60}")

    mode = 'grayscale' if variant == 'v_jepa' else 'edge'
    train_loader, test_loader = get_physics_reasoning_loaders(
        mode='raw' if variant == 'v_jepa' else 'edge',
        num_train=args.num_train,
        num_test=args.num_test,
        context_frames=args.context_frames,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if variant == 'v_jepa':
        input_channels = 1
    else:
        input_channels = 1

    model = get_models(variant).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}/{args.epochs} "
              f"- Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.2f}% "
              f"- Test Acc: {test_acc*100:.2f}%")

    return {
        "variant": variant,
        "params": params,
        "test_acc": test_acc,
    }


def print_results(v_results, a_results):
    print("\n")
    print("="*70)
    print("        Physics Reasoning: V-JEPA vs A-JEPA")
    print("="*70)
    print(f"{'Metric':<35} {'V-JEPA':>15} {'A-JEPA':>15}")
    print("-"*70)
    print(f"{'Parameters':<35} {v_results['params']:>15,} {a_results['params']:>15,}")
    print(f"{'Test Accuracy':<35} {v_results['test_acc']*100:>14.2f}% {a_results['test_acc']*100:>14.2f}%")
    print("="*70)

    param_ratio = v_results['params'] / a_results['params']
    print("\nKey Findings:")
    print(f"  - V-JEPA has {param_ratio:.1f}x more parameters than A-JEPA")
    if a_results['test_acc'] >= v_results['test_acc']:
        improvement = (a_results['test_acc'] - v_results['test_acc']) * 100
        print(f"  - A-JEPA achieves higher accuracy (+{improvement:.2f} percentage points)")
        print("  - Relational, edge-based representation transfers better to physics outcomes.")
    else:
        drop = (v_results['test_acc'] - a_results['test_acc']) * 100
        print(f"  - V-JEPA outperforms by {drop:.2f} percentage points.")
    print("="*70)


def parse_args():
    parser = argparse.ArgumentParser(description="Physics Reasoning Test")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_train', type=int, default=500)
    parser.add_argument('--num_test', type=int, default=100)
    parser.add_argument('--context_frames', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("Physics Reasoning Task: goal-reaching puzzles")

    v_results = run_experiment('v_jepa', device, args)
    a_results = run_experiment('a_jepa', device, args)

    print_results(v_results, a_results)


if __name__ == "__main__":
    main()

