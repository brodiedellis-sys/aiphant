# AIphant: Aphantasic Joint-Embedding Predictive Architecture

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17755792.svg)](https://doi.org/10.5281/zenodo.17755792)

A research project exploring whether AI systems can benefit from "aphantasic" processing — learning abstract, structural representations instead of rich visual imagery.

**Key Finding**: A-JEPA v3 **outperforms V-JEPA v3 with 6x fewer parameters** (50.7% vs 48.3% accuracy), while being **2x more data-efficient** and **robust to corruptions**.

---

## Overview

This project implements and compares variants of Joint-Embedding Predictive Architecture (JEPA):

| Model | Input | Parameters | Accuracy | Key Features |
|-------|-------|------------|----------|--------------|
| **A-JEPA v3** | Edge+Motion (4ch) | **442K** | **50.7 ± 1.9%** | 8 slots, RelationalBlock, per-slot GRU |
| V-JEPA v3 | RGB+Motion (4ch) | 2.72M | 48.3 ± 3.3% | Dense embedding |
| A-JEPA v2 | Edge (1ch) | 180K | 47.3 ± 3.6% | 4 slots, shared bottleneck |
| V-JEPA v2 | RGB (3ch) | 1.64M | 50.3 ± 4.1% | Dense embedding |

Inspired by [aphantasia](https://en.wikipedia.org/wiki/Aphantasia) — the condition where people lack visual imagery but often excel at abstract reasoning — we test whether stripping away visual details and focusing on structure leads to more robust, generalizable representations.

---

## Key Results

### Main Result: A-JEPA v3 vs V-JEPA v3

| Model | Parameters | Accuracy | Variance |
|-------|------------|----------|----------|
| **A-JEPA v3** | **442K** | **50.7%** | **±1.9%** |
| V-JEPA v3 | 2.72M | 48.3% | ±3.3% |

**A-JEPA v3 wins with 6x fewer parameters and lower training variance!**

### Data Efficiency

*Train with 100%, 50%, 25%, 10% of data*

| Model | 100% | 50% | 25% | 10% | Drop |
|-------|------|-----|-----|-----|------|
| **A-JEPA** | 57% | 55% | 57% | **54%** | **3%** |
| V-JEPA | 55% | 45% | 50% | 48% | 7% |

**A-JEPA is 2x more data-efficient** — only 3% accuracy drop at 10% data vs 7% for V-JEPA.

### Corruption Robustness

*Train on clean data, test with noise/blur/brightness*

| Model | Clean | Noise | Blur | Combined |
|-------|-------|-------|------|----------|
| **A-JEPA** | 57% | 56% | 58% | **60%** |
| V-JEPA | 55% | 50% | 50% | 48% |

**A-JEPA improves under corruption** (+3%) while V-JEPA drops (-7%). Edge preprocessing provides natural invariance.

---

## What Makes A-JEPA v3 Special?

### Cognitive Architecture (Inspired by Aphantasia)

1. **RelationalBlock** — Explicit slot-to-slot reasoning for physics understanding
2. **Per-Slot Processing** — Each object tracked independently (no flattening)
3. **SlotTemporalMemory** — Per-object GRU maintains separate histories
4. **Multi-scale Edge + Motion** — Structure and velocity without pixels

### Training Innovations

1. **Curriculum Learning** — Easy (1 ball) → Medium (2 balls) → Hard (2-3 balls)
2. **VICReg Loss** — Prevents representation collapse
3. **Sparsity Annealing** — L1 penalty ramps up through phases

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

### Run the V3 Benchmark (Recommended)
```bash
# Full benchmark with curriculum learning (110 epochs × 2 models × 3 seeds)
python src/tasks/v3_benchmark.py \
  --models ajepa_v3 vjepa_v3 \
  --seeds 42 123 456 \
  --num_train 200 \
  --num_test 100 \
  --output_dir results/v3_benchmark
```

### Run OOD Benchmark
```bash
# Test data efficiency and corruption robustness
python src/tasks/ood_benchmark.py \
  --experiment all \
  --epochs 60 \
  --num_train 300 \
  --num_test 100 \
  --output_dir results/ood_benchmark
```

### Quick Test (5 minutes)
```bash
# Sanity check that models train
python src/models_v3.py  # Tests architecture
python src/datasets/bouncing_balls.py  # Tests preprocessing
```

---

## Project Structure

```
aiphant/
├── src/
│   ├── models_v2.py           # A-JEPA/V-JEPA v2 architectures
│   ├── models_v3.py           # A-JEPA/V-JEPA v3 with RelationalBlock
│   ├── transforms.py          # Edge detection, perturbations
│   ├── datasets/
│   │   ├── bouncing_balls.py  # Synthetic physics dataset
│   │   └── hidden_mass.py     # Mass inference task
│   └── tasks/
│       ├── v3_benchmark.py    # V3 curriculum training benchmark
│       ├── ood_benchmark.py   # OOD/data efficiency tests
│       ├── rigorous_benchmark.py  # Multi-seed evaluation
│       └── predict_future.py  # Temporal prediction
├── results/
│   ├── v3_benchmark/          # V3 results and plots
│   ├── ood_benchmark/         # OOD test results
│   └── rigorous/              # Multi-seed experiment bundles
├── PAPER.md                   # Full research paper with methodology
├── CONCEPT.md                 # Theoretical motivation
└── requirements.txt
```

---

## Results Visualization

After running benchmarks, find plots in:
- `results/v3_benchmark/v3_benchmark.png` — V3 comparison
- `results/ood_benchmark/ood_benchmark_plot.png` — OOD/efficiency/corruption tests

---

## Citation

If you use this work, please cite:

```bibtex
@misc{aiphant2026,
  author = {Brodie Ellis},
  title = {AIphant: Aphantasic Joint-Embedding Predictive Architecture},
  year = {2026},
  url = {https://github.com/brodiedellis-sys/aiphant}
}
```

---

## Acknowledgments

This work builds on Yann LeCun's [JEPA framework](https://openreview.net/forum?id=BZ5a1r-kVsf) and is inspired by research on the [texture vs. shape bias](https://arxiv.org/abs/1811.12231) in neural networks.

---

## License

MIT License
