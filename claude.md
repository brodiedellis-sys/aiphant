# Claude Context for AIphant

## Project Overview

**AIphant** is a research project exploring "aphantasic" AI - whether removing visual richness and focusing on abstract/structural representations leads to better generalization and robustness.

**Key Finding**: A-JEPA v3 outperforms V-JEPA v3 with **6x fewer parameters** (442K vs 2.72M) while being more data-efficient and corruption-robust.

## Core Concept

Inspired by **aphantasia** (the condition where people lack mental imagery but often excel at abstract reasoning), this project tests:
- **A-JEPA**: Uses edge maps + motion (structural skeleton only)
- **V-JEPA**: Uses full RGB images (standard approach)

The hypothesis: stripping away visual detail forces the model to learn invariant, structural features that generalize better.

## Architecture Highlights (V3)

1. **RelationalBlock** - Explicit slot-to-slot reasoning
2. **Per-Slot Processing** - Each object tracked independently (8 slots)
3. **SlotTemporalMemory** - Per-object GRU maintains separate histories
4. **Multi-scale Edge + Motion** - 4-channel input (structure + velocity)
5. **Curriculum Learning** - Easy (1 ball) → Medium (2) → Hard (2-3)
6. **VICReg Loss** - Prevents representation collapse

## A-JEPA V4 (NEW - In Development)

V4 implements 6 key computational principles from aphantasia cognitive science research:

1. **Precision-Weighted Predictions** (Phase 1) - Confidence modulates prediction strength
2. **Dual Pathway** (Phase 2) - Separate WHERE (spatial grid) and WHAT (object slots) streams
3. **Symbolic Bottleneck** (Phase 3) - VQ-VAE style discrete codes for verbal-like compression
4. **Top-Down Gating** (Phase 4) - Predictions gated by consistency checks
5. **Structured Temporal Memory** (Phase 5) - Multi-scale memory (fast/slow/working)
6. **TransformerPlanner** (Phase 6, Optional) - Imagination module for multi-step prediction

### V4 Architecture Flow
```
Input (edges + motion: 4ch)
    ↓
Conv Encoder
    ↓
┌─────────────┬──────────────┐
│ Spatial     │ Object       │
│ Pathway     │ Pathway      │
│ (WHERE)     │ (WHAT)       │
└─────────────┴──────────────┘
        ↓
Cross-Pathway Integration
        ↓
Relational Reasoning
        ↓
Symbolic Bottleneck (VQ-VAE)
        ↓
┌─────────────────────────────┐
│ Top-Down Gated Predictor    │  ← Default
│         OR                  │
│ TransformerPlanner          │  ← Optional (Phase 6)
└─────────────────────────────┘
        ↓
Structured Temporal Memory
```

### TransformerPlanner (Phase 6)

Transformer-based "imagination" module that operates on symbolic bottleneck codes:
- **Residual Hybrid Fusion**: Combines discrete code embeddings + continuous vectors
- **Time-Causal, Slot-Full Attention**: Full attention within timestep, causal across time
- **Continuous-Only Rollout**: No argmax feedback to preserve gradients
- **Auxiliary Discrete Loss**: Cross-entropy on codebook indices for symbolic grounding

**Phased Training Curriculum:**
- Phase A (0-15%): Single-step teacher forcing
- Phase B (15-30%): Multi-step teacher forcing
- Phase C (30%+): Scheduled sampling (prob 0→0.5)

### V4 Configs Available
- `default`: Full v4 with all 5 phases (~560K params)
- `no_dual`: Disable dual pathway
- `no_symbolic`: Use continuous bottleneck
- `no_gating`: Disable top-down gating
- `no_structured_mem`: Use simple GRU
- `continuous`: Like v3 + precision only (~357K params)
- `with_planner`: Full v4 + TransformerPlanner (~546K params)
- `planner_only`: Minimal v4 with planner (~407K params)

## Key Results

| Metric | A-JEPA | V-JEPA |
|--------|--------|--------|
| Accuracy | 50.7% | 48.3% |
| Parameters | 442K | 2.72M |
| Drift | 23x lower | baseline |
| Data efficiency | 3% drop at 10% data | 7% drop |
| Corruption | +3% (improves!) | -7% (degrades) |

## Project Structure

```
src/
├── models_v4.py          # A-JEPA V4 with 5 cognitive principles (NEW)
├── models_v3.py          # V3 architectures (A-JEPA, V-JEPA)
├── models_v2.py          # Earlier V2 versions
├── transforms.py         # Edge detection, perturbations
├── datasets/
│   ├── bouncing_balls.py # Synthetic physics dataset
│   └── hidden_mass.py    # Mass inference task
└── tasks/
    ├── v4_benchmark.py   # V4 benchmark with ablations (NEW)
    ├── v3_benchmark.py   # V3 benchmark script
    ├── ood_benchmark.py  # Out-of-distribution tests
    └── rigorous_benchmark.py
```

## Important Commands

```bash
# Test V4 architecture (all 6 phases including TransformerPlanner)
python src/models_v4.py

# Run V4 benchmark (ablation study)
python src/tasks/v4_benchmark.py --configs default no_dual no_symbolic --seeds 42 123 456

# Quick V4 test
python src/tasks/v4_benchmark.py --quick

# Benchmark planner configs only
python src/tasks/v4_benchmark.py --configs with_planner planner_only --seeds 42 123 456

# Run V3 benchmark (baseline)
python src/tasks/v3_benchmark.py --models ajepa_v3 --seeds 42 123 456

# Run OOD benchmark
python src/tasks/ood_benchmark.py --experiment all
```

## Current Status

**V3 - Complete:**
- [x] V-JEPA and A-JEPA v3 implemented
- [x] RelationalBlock added
- [x] Bouncing balls temporal prediction
- [x] OOD generalization tests
- [x] Multi-seed benchmarks
- [x] V3 achieves ~52% test accuracy with curriculum+VICReg

**V4 - Complete (Architecture):**
- [x] Phase 1: Precision Estimator
- [x] Phase 2: Dual Pathway (Spatial + Object)
- [x] Phase 3: Symbolic Bottleneck (VQ-VAE)
- [x] Phase 4: Top-Down Gating
- [x] Phase 5: Structured Temporal Memory
- [x] Phase 6: TransformerPlanner (imagination module)
- [x] All phases integrated into AJEPAv4 class
- [x] v4_benchmark.py with phased training curriculum
- [ ] Full benchmark pending (need to run with more seeds/data)

**Next Steps:**
- Run full V4 benchmark to compare with V3 baseline
- Ablation study to measure contribution of each cognitive principle
- Compare with_planner vs default configs
- Phase 2 from ROADMAP: Richer physics (collisions, CLEVR, Phyre)

## Files to Know

- `src/models_v4.py` - A-JEPA V4 with 6 cognitive principles (includes TransformerPlanner)
- `src/models_v3.py` - V3 architectures (AJEPA_V3, VJEPA_V3 classes)
- `src/datasets/bouncing_balls.py` - Dataset with edge/motion preprocessing
- `src/tasks/v4_benchmark.py` - NEW: V4 benchmark with ablation configs
- `src/tasks/v3_benchmark.py` - V3 training with curriculum learning
- `PAPER.md` - Full methodology and results
- `CONCEPT.md` - Theoretical motivation

## Notes

- Dataset: Synthetic bouncing balls (configurable 1-3 balls)
- Task: Predict future frame embeddings from context frames
- Evaluation: Linear probe on learned representations
- Environment: Python venv in `./venv`
