# The Aphantasic AI Hypothesis

## Motivation

**Aphantasia** is a neurological condition in which individuals lack the ability to form voluntary mental images. When asked to "picture a sunset," people with aphantasia see nothing — yet they can still describe a sunset, reason about it, and navigate the world effectively. Some studies suggest aphantasics may even excel at certain abstract reasoning tasks.

This raises a question for AI: **What if the inability to "visualize" is not a deficit, but an alternative cognitive strategy?**

Current vision models learn rich, high-dimensional representations that encode textures, colors, lighting, and fine-grained visual details. While powerful for recognition tasks, these representations may be:

1. **Fragile** — Sensitive to surface-level changes (texture, style, lighting)
2. **Inefficient** — Storing redundant visual information
3. **Prone to drift** — Accumulating errors in temporal/predictive tasks

We propose **A-JEPA (Aphantasic JEPA)**: a model that deliberately discards visual richness and operates on abstract, structural representations — analogous to how an aphantasic mind might process the world.

---

## The Core Idea

### Standard Visual Models (V-JEPA)
- Input: Full RGB images
- Learn: Textures, colors, fine details
- Representation: High-dimensional, visually grounded

### Aphantasic Models (A-JEPA)
- Input: Edge maps (structural skeleton only)
- Learn: Shapes, relationships, geometry
- Representation: Low-dimensional, abstract
- Enhancement: Relational reasoning block for global context

The key insight: **By removing visual detail, we force the model to learn invariant, structural features that may generalize better and remain stable over time.**

---

## Theoretical Grounding

### 1. Texture vs. Shape Bias
Geirhos et al. (2019) showed that ImageNet-trained CNNs are heavily biased toward texture, while humans rely more on shape. When texture and shape conflict, CNNs follow texture; humans follow shape. A-JEPA, by using edge-only inputs, eliminates texture entirely and forces shape-based reasoning.

### 2. World Models and Latent Prediction
Yann LeCun's JEPA framework argues that intelligence requires learning predictive models of the world in latent space — not pixel space. Predicting abstract states rather than reconstructing images is more efficient and robust. A-JEPA takes this further: it operates on already-abstracted inputs.

### 3. Object-Centric Representations
Research in object-centric learning suggests that decomposing scenes into objects and their relations leads to better generalization. A-JEPA's RelationalBlock explicitly aggregates global context, encouraging relational reasoning over raw feature matching.

---

## Hypotheses

We test three hypotheses:

### H1: Robustness to Visual Perturbations
> A-JEPA will be more robust to texture/color changes because it never learned to rely on them.

**Result**: Mixed. V-JEPA actually showed lower robustness gap on CIFAR-10 texture perturbations. This may be because our perturbation (noise + color jitter) degrades edge quality.

### H2: Temporal Coherence
> A-JEPA will maintain stable state predictions over longer time horizons because it tracks abstract structure, not visual appearance.

**Result**: Confirmed. A-JEPA shows **23x lower drift** on bouncing ball prediction, maintaining ~0.999 similarity even at horizon 10.

### H3: Out-of-Distribution Generalization
> A-JEPA will generalize better to novel configurations because it learns structural relationships, not visual templates.

**Result**: Confirmed. When trained on 2-ball scenarios and tested on 3-ball scenarios, A-JEPA shows **92% smaller generalization gap**.

---

## Implications

If these findings hold at scale, they suggest:

1. **Efficiency**: Abstract representations can achieve strong performance with far fewer parameters.

2. **Stability**: Removing visual detail may be essential for long-horizon prediction and planning tasks.

3. **Generalization**: Structure-first learning may be key to out-of-distribution robustness.

4. **Architecture Design**: The RelationalBlock — aggregating global context before prediction — may be a simple but powerful addition for abstract reasoning.

---

## Limitations and Open Questions

- **Scale**: Current experiments use small models (0.4M params) on simple datasets. Will findings hold at scale?
- **Input Modality**: Edge detection is a crude abstraction. What other "aphantasic" inputs might work better? (Depth maps, segmentation masks, symbolic representations)
- **Task Diversity**: Results are strong on physics prediction but weaker on static image classification. When does aphantasic processing help vs. hurt?
- **Hybrid Approaches**: Can we combine visual richness with abstract structure dynamically?

---

## References

- LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence.
- Geirhos, R., et al. (2019). ImageNet-trained CNNs are biased towards texture.
- Assran, M., et al. (2023). Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.
- Zeman, A., et al. (2015). Lives without imagery – Congenital aphantasia.

