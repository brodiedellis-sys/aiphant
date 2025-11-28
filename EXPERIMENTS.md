# Experiments

This document provides exact commands to reproduce all experiments and expected results.

---

## Setup

```bash
cd aiphant
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
mkdir -p results
```

---

## Experiment 1: CIFAR-10 Robustness

**Goal**: Compare V-JEPA and A-JEPA on image classification and texture robustness.

### Training

```bash
# Train V-JEPA (RGB input, 512-dim embedding)
python train_jepa.py --variant v_jepa --epochs 20 --batch_size 256

# Train A-JEPA (Edge input, 128-dim embedding, relational block)
python train_jepa.py --variant a_jepa --epochs 20 --batch_size 256
```

### Evaluation

```bash
python compare_results.py | tee results/cifar10_comparison.txt
```

### Expected Output

```
======================================================================
                    V-JEPA vs A-JEPA Comparison
======================================================================
Metric                                V-JEPA              A-JEPA
----------------------------------------------------------------------
Parameters                             1.49M               0.32M
Inference (ms/batch)                  ~150                ~55
Throughput (samples/s)                ~1700               ~4500
----------------------------------------------------------------------
Clean Accuracy (%)                    ~25                 ~17
Perturbed Accuracy (%)                ~21                 ~10
Robustness Gap (%)                    ~4                  ~7
======================================================================
```

**Note**: V-JEPA shows lower robustness gap on this task. This may be because the texture perturbation (noise + color jitter) degrades edge detection quality, unfairly penalizing A-JEPA.

---

## Experiment 2: Temporal Coherence (Bouncing Balls)

**Goal**: Test whether A-JEPA maintains stable predictions over long time horizons.

### Run Experiment

```bash
python src/tasks/predict_future.py \
    --epochs 10 \
    --num_train 500 \
    --num_test 100 \
    | tee results/temporal_prediction.txt
```

### Expected Output

```
======================================================================
        Temporal Prediction: V-JEPA vs A-JEPA
======================================================================
Metric                                     V-JEPA             A-JEPA
----------------------------------------------------------------------
Parameters                              1,487,488            389,968
----------------------------------------------------------------------
Cosine Similarity by Horizon (higher = better):
  Horizon 1   steps                       ~0.97               ~0.999
  Horizon 3   steps                       ~0.97               ~0.999
  Horizon 5   steps                       ~0.96               ~0.999
  Horizon 10  steps                       ~0.96               ~0.999
----------------------------------------------------------------------
Drift (lower = more stable)               ~0.014              ~0.0006
======================================================================

Key Findings:
  - A-JEPA has ~23x lower drift (more temporally stable)
  - A-JEPA maintains near-perfect coherence at all horizons
```

**Key Result**: A-JEPA shows **23x lower drift**, maintaining ~0.999 similarity even at horizon 10.

---

## Experiment 3: OOD Generalization (2→3 Balls)

**Goal**: Test whether A-JEPA generalizes better to unseen configurations.

### Run Experiment

```bash
python src/tasks/ood_generalization.py \
    --epochs 10 \
    --num_train 500 \
    --num_test 100 \
    | tee results/ood_generalization.txt
```

### Expected Output

```
======================================================================
        OOD Generalization: V-JEPA vs A-JEPA
======================================================================
Metric                                       V-JEPA          A-JEPA
----------------------------------------------------------------------
Parameters                                1,487,488         389,968
----------------------------------------------------------------------
Cosine Similarity (higher = better):
  In-Distribution (2 balls)                  ~0.55           ~0.66
  Out-of-Distribution (3 balls)              ~0.53           ~0.65
----------------------------------------------------------------------
Generalization Gap (lower = better)          ~0.024          ~0.002
======================================================================

VERDICT: A-JEPA demonstrates superior generalization to novel configurations!
```

**Key Result**: A-JEPA shows **92% smaller generalization gap** when tested on 3 balls after training on 2 balls.

---

## Summary Table

| Experiment | Metric | V-JEPA | A-JEPA | Winner |
|------------|--------|--------|--------|--------|
| CIFAR-10 | Robustness Gap | 4% | 7% | V-JEPA |
| Temporal | Drift | 0.014 | 0.0006 | **A-JEPA** (23x better) |
| OOD | Gen. Gap | 0.024 | 0.002 | **A-JEPA** (92% better) |
| Efficiency | Parameters | 1.49M | 0.39M | **A-JEPA** (3.8x smaller) |
| Efficiency | Throughput | 1.7K/s | 4.5K/s | **A-JEPA** (2.7x faster) |

---

## Reproducing with Different Seeds

For statistical rigor, run experiments multiple times:

```bash
# Temporal prediction with different seeds
for seed in 42 123 456; do
    python src/tasks/predict_future.py --epochs 10 --num_train 500 --seed $seed
done
```

*(Note: Seed argument not yet implemented — add to scripts if needed.)*

---

## Hardware Notes

All experiments can run on:
- **CPU**: Apple M1/M2/M3, Intel i5+
- **GPU**: Any CUDA-capable GPU (optional, speeds up training ~3x)

Typical runtime on M1 MacBook:
- CIFAR-10 training (20 epochs): ~30 min per variant
- Temporal prediction (10 epochs): ~5 min
- OOD generalization (10 epochs): ~5 min

