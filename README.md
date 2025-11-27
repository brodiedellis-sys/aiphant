# AIphant: V-JEPA vs A-JEPA Comparative Study

A comparative study between **V-JEPA** (Visual JEPA with RGB inputs, high-dimensional latent) and **A-JEPA** (Abstract JEPA with Edge-only inputs, low-dimensional latent) on CIFAR-10.

## Overview

This project implements Joint-Embedding Predictive Architecture (JEPA) in two variants:

| Variant | Input | Channels | Embedding Dim |
|---------|-------|----------|---------------|
| V-JEPA  | RGB   | 3        | 512           |
| A-JEPA  | Edge  | 1        | 128           |

The goal is to compare:
1. **Representation quality** via linear probe accuracy
2. **Robustness** to texture perturbations

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train V-JEPA (RGB variant)

```bash
python train_jepa.py --variant v_jepa --epochs 100 --batch_size 256
```

### Train A-JEPA (Edge variant)

```bash
python train_jepa.py --variant a_jepa --epochs 100 --batch_size 256
```

### Evaluate Models

```bash
# Evaluate V-JEPA
python linear_eval.py --variant v_jepa

# Evaluate A-JEPA
python linear_eval.py --variant a_jepa
```

## Project Structure

```
AIphant/
├── src/
│   ├── __init__.py
│   ├── dataset.py      # CIFAR-10 data loading
│   ├── models.py       # Encoder, Predictor, LinearProbe
│   ├── transforms.py   # CannyEdge, TexturePerturbation, RandomMask
│   └── utils.py        # Training utilities
├── train_jepa.py       # JEPA training script
├── linear_eval.py      # Linear evaluation script
├── requirements.txt    # Dependencies
└── README.md
```

## Method

### JEPA Training

1. **Target**: Encode full image → `z = encoder(x)`
2. **Context**: Encode masked image → `z' = encoder(masked_x)`
3. **Prediction**: Predict target from context → `p = predictor(z')`
4. **Loss**: Negative cosine similarity between `p` and `z` (stop-gradient on `z`)

### Edge Detection (A-JEPA)

Uses Canny edge detection to convert RGB images to single-channel edge maps, forcing the model to learn from structural/shape information only.

### Robustness Evaluation

Applies strong texture perturbations (ColorJitter + Gaussian noise) to test images to evaluate robustness to texture changes.

## Expected Results

- **V-JEPA**: Higher clean accuracy due to richer RGB features, but potentially lower robustness
- **A-JEPA**: Potentially lower clean accuracy but better robustness to texture perturbations (shape-focused representations)

