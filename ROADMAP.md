# Research Roadmap

This document outlines the future direction of the AIphant / A-JEPA research program.

---

## Current Status (Phase 1: Proof of Concept) ✓

**Completed:**
- [x] Implemented V-JEPA and A-JEPA architectures
- [x] Added RelationalBlock for abstract reasoning
- [x] CIFAR-10 robustness evaluation
- [x] Bouncing balls temporal prediction
- [x] OOD generalization (2→3 balls)
- [x] Demonstrated 23x lower drift, 92% better generalization

**Key Insight**: Abstract/structural representations excel at temporal coherence and generalization, even if they underperform on texture-dependent static recognition.

---

## Phase 2: Richer Physics Environments

**Goal**: Validate findings on more complex, realistic physics scenarios.

### 2.1 Multi-Object Interactions
- Add ball-ball collisions (not just wall bounces)
- Test if A-JEPA handles interaction dynamics better

### 2.2 CLEVR-Style Scenes
- 3D rendered scenes with multiple objects
- Test compositional generalization (novel object combinations)

### 2.3 Phyre Benchmark
- Standard physics reasoning benchmark
- Compare A-JEPA to published baselines

### 2.4 Real Video (Optional)
- Simple real-world physics videos (rolling balls, pendulums)
- Test sim-to-real transfer of abstract representations

---

## Phase 3: Alternative "Aphantasic" Inputs

**Goal**: Explore what abstractions work best — edges may not be optimal.

### 3.1 Depth Maps
- Single-channel geometric structure
- May preserve 3D information better than edges

### 3.2 Segmentation Masks
- Object-level abstraction
- Each object as a solid region, no texture

### 3.3 Optical Flow
- Motion-only representation
- Test if pure dynamics (no appearance) is sufficient

### 3.4 Symbolic/Structured Input
- Object positions + velocities as vectors
- Extreme abstraction — no pixels at all

---

## Phase 4: Scaling Up

**Goal**: Test if findings hold at larger scale.

### 4.1 Larger Models
- Scale A-JEPA to 10M, 100M parameters
- Compare scaling curves: does A-JEPA scale more efficiently?

### 4.2 Larger Datasets
- ImageNet / Kinetics for video
- Does the efficiency advantage persist?

### 4.3 Longer Horizons
- Test prediction at 50, 100, 500 steps
- Does drift remain low at extreme horizons?

---

## Phase 5: Multimodal Integration

**Goal**: Connect A-JEPA to language and action.

### 5.1 Language Grounding
- Add text encoder (e.g., CLIP-style)
- Can A-JEPA representations be described in language?

### 5.2 Decision/Action Head
- Add policy network on top of A-JEPA states
- Test on simple control tasks (e.g., CartPole, MuJoCo)

### 5.3 World Model for Planning
- Use A-JEPA as latent dynamics model
- Plan in abstract state space, not pixel space

---

## Phase 6: Theoretical Understanding

**Goal**: Understand *why* aphantasic processing works.

### 6.1 Information Bottleneck Analysis
- What information does A-JEPA discard?
- Is there a principled compression story?

### 6.2 Invariance Analysis
- What transformations is A-JEPA invariant to?
- Compare to V-JEPA invariances

### 6.3 Mechanistic Interpretability
- What does the RelationalBlock learn?
- Can we visualize "abstract features"?

---

## Publication Milestones

| Phase | Target Venue | Timeline |
|-------|--------------|----------|
| 1 | Blog post / arXiv preprint | Now |
| 2 | Workshop paper (NeurIPS, ICLR) | 3-6 months |
| 3-4 | Full conference paper | 6-12 months |
| 5-6 | Journal / major venue | 12-18 months |

---

## Collaborations & Community

- Share reproducible code + results on GitHub
- Engage with:
  - World models community (LeCun, Ha, Hafner)
  - Object-centric learning (Locatello, Greff)
  - Shape bias research (Geirhos, Brendel)
- Consider: Open-source model weights, leaderboards, challenges

---

## Open Questions

1. **When does aphantasic processing help vs. hurt?**
   - Clearly helps: temporal prediction, OOD generalization
   - Unclear: static recognition, fine-grained discrimination

2. **What's the right level of abstraction?**
   - Edges? Depth? Segmentation? Symbolic?
   - Likely task-dependent

3. **Can we learn the abstraction?**
   - Instead of hand-crafted edge detection, learn to extract task-relevant structure

4. **Is RelationalBlock the right inductive bias?**
   - Other options: Transformers, GNNs, slot attention

---

## Contact

For collaboration inquiries or questions:
- GitHub: [brodiedellis-sys](https://github.com/brodiedellis-sys)
- Project: [AIphant](https://github.com/brodiedellis-sys/aiphant)

