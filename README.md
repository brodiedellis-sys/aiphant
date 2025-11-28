# AIphant: Aphantasic Joint-Embedding Predictive Architecture

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17755792.svg)](https://doi.org/10.5281/zenodo.17755792)

A research project exploring whether AI systems can benefit from "aphantasic" processing — learning abstract, structural representations instead of rich visual imagery.

**Key Finding**: A-JEPA (Aphantasic JEPA) achieves **23x lower temporal drift** and **92% better generalization** to novel configurations compared to standard visual JEPA, while using **3.8x fewer parameters**.

---

## Overview

This project implements and compares two variants of Joint-Embedding Predictive Architecture (JEPA):

| Variant | Input | Parameters | Approach |
|---------|-------|------------|----------|
| **V-JEPA** | RGB images | 1.49M | Standard visual processing |
| **A-JEPA** | Edge maps | 0.39M | Abstract/structural processing with relational reasoning |

Inspired by [aphantasia](https://en.wikipedia.org/wiki/Aphantasia) — the condition where people lack visual imagery but often excel at abstract reasoning — we test whether stripping away visual details and focusing on structure leads to more robust, generalizable representations.

---

## Key Results

### Experiment 1: Temporal Coherence (Bouncing Balls)
*Can the model maintain stable predictions over long time horizons?*

| Metric | V-JEPA | A-JEPA |
|--------|--------|--------|
| Similarity @ Horizon 1 | 0.973 | **0.999** |
| Similarity @ Horizon 10 | 0.959 | **0.999** |
| Drift | 0.014 | **0.0006** |

**A-JEPA shows 23x lower drift** — near-perfect state coherence over time.

### Experiment 2: OOD Generalization (2→3 Balls)
*Can the model generalize to unseen configurations?*

| Metric | V-JEPA | A-JEPA |
|--------|--------|--------|
| In-Distribution (2 balls) | 0.55 | **0.66** |
| Out-of-Distribution (3 balls) | 0.53 | **0.65** |
| Generalization Gap | 0.024 | **0.002** |

**A-JEPA generalizes 92% better** to novel configurations.

### Experiment 3: Efficiency

| Metric | V-JEPA | A-JEPA |
|--------|--------|--------|
| Parameters | 1.49M | **0.39M** (3.8x smaller) |
| Throughput | 1,678/s | **4,535/s** (2.7x faster) |

---

## Installation

```bash
git clone https://github.com/brodiedellis-sys/aiphant.git
cd aiphant
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Quick Start

### Train Models (CIFAR-10)
```bash
python train_jepa.py --variant v_jepa --epochs 20
python train_jepa.py --variant a_jepa --epochs 20
```

### Run Experiments
```bash
# Temporal coherence test
python src/tasks/predict_future.py --epochs 10

# OOD generalization test  
python src/tasks/ood_generalization.py --epochs 10

# CIFAR-10 robustness comparison
python compare_results.py
```

See [EXPERIMENTS.md](EXPERIMENTS.md) for detailed commands and expected outputs.

---

## Project Structure

```
aiphant/
├── src/
│   ├── models.py          # V-JEPA/A-JEPA architectures, RelationalBlock
│   ├── transforms.py      # Edge detection, perturbations
│   ├── dataset.py         # CIFAR-10 loaders
│   ├── datasets/
│   │   └── bouncing_balls.py   # Synthetic physics dataset
│   └── tasks/
│       ├── predict_future.py      # Temporal prediction experiment
│       └── ood_generalization.py  # OOD generalization experiment
├── train_jepa.py          # JEPA training script
├── linear_eval.py         # Linear probe evaluation
├── compare_results.py     # Side-by-side comparison
├── CONCEPT.md             # Theoretical motivation
├── EXPERIMENTS.md         # Reproducible experiment guide
└── ROADMAP.md             # Future research directions
```

---

## Citation

If you use this work, please cite:

```
@misc{aiphant2024,
  author = {Brodie Ellis},
  title = {AIphant: Aphantasic Joint-Embedding Predictive Architecture},
  year = {2024},
  url = {https://github.com/brodiedellis-sys/aiphant}
}
```

---

## Acknowledgments

This work builds on Yann LeCun's [JEPA framework](https://openreview.net/forum?id=BZ5a1r-kVsf) and is inspired by research on the [texture vs. shape bias](https://arxiv.org/abs/1811.12231) in neural networks.

---

## License

MIT License
