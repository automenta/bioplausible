# Specification for Targeting the "Nobody Ever Bothered Club" in the EqProp Repository

## 1. Overview

### 1.1 Purpose
This specification outlines a comprehensive plan to extend the EqProp repository by systematically integrating spectral normalization with a set of under-explored biologically plausible or alternative training algorithms. These combinations form the "Nobody Ever Bothered Club" (NEBC), identified as algorithms where spectral normalization (SN) has not been rigorously applied or publicly demonstrated at scale in prior literature or codebases (as of January 6, 2026). The goal is to:

- Verify if spectral normalization acts as a "stability unlock" for these algorithms, similar to its impact on Equilibrium Propagation (EqProp).
- Add new verification tracks to the repository's suite, ensuring reproducibility, deterministic seeding, and benchmarking against backpropagation (backprop) baselines.
- Demonstrate potential advantages (e.g., constant memory, deep signal propagation, convergence speed) on diverse tasks (e.g., MNIST, CIFAR-10, TinyStories for language).
- Position the repository as a unified platform for testing bio-plausible alternatives to backprop, potentially uncovering superior hybrids.

Successful implementation could reveal that spectral normalization is a **universal stabilizer**, enabling these algorithms to "punch above their weight" on modern architectures (e.g., transformers, convnets, deep MLPs).

### 1.2 Scope
- **Targets**: Focus on the five NEBC members:
  1. EqProp (all flavors) + Spectral Normalization (already partially implemented; expand for completeness).
  2. Feedback Alignment (FA) + Spectral Normalization.
  3. Direct Feedback Alignment (DFA) + Spectral Normalization.
  4. Modern Contrastive Hebbian Learning (CHL) + Spectral Normalization.
  5. Deep (1000+ layer) Pure Hebbian Chains + Spectral Normalization.
- **Non-Targets**: Exclude algorithms already well-explored with spectral norm (e.g., standard backprop in GANs, Target Propagation).
- **Assumptions**: Builds on the existing repository structure (PyTorch-based, with `eqprop` core, `tracks/` for verifications, `configs/` for hyperparams).
- **Success Criteria**: For each target, achieve ≥ backprop-level accuracy on benchmarks with advantages in memory/depth/stability. All new tracks must pass 100% of a defined verification checklist (e.g., convergence, no NaNs, reproducible seeds).

### 1.3 Risks and Mitigations
- **Risk**: Instability in integration leading to non-convergence.
  - **Mitigation**: Start with toy models (e.g., 2-layer MLP on XOR), enforce strict spectral norm (Lipschitz constant = 1) after every update.
- **Risk**: Compute overhead for deep experiments.
  - **Mitigation**: Use deterministic seeding, early stopping, and limit initial runs to single GPU (e.g., RTX 4090).
- **Risk**: Overlap with undiscovered prior work.
  - **Mitigation**: Pre-implementation, perform a quick literature check (e.g., via arXiv search); document any findings in README.

## 2. General Methodology

### 2.1 Integration Approach
- **Spectral Normalization**: Use PyTorch's `torch.nn.utils.spectral_norm` or a custom implementation (e.g., power iteration for exact norm). Apply to all linear/conv layers, enforcing ||W||_2 ≤ 1 post-update. This ensures bounded dynamics, preventing explosion/vanishing.
- **Core Modifications**:
  - Extend `eqprop.Model` base class to support pluggable update rules (e.g., via a `update_mode` flag: 'eqprop', 'fa', 'dfa', etc.).
  - Add a `spectral_norm_enabled` flag in configs (default: True for NEBC tracks).
  - Implement hybrid modes (e.g., FA in lower layers, EqProp in upper).
- **Verification Suite**:
  - New tracks in `tracks/` directory (e.g., `fa_spectral.py`).
  - Each track: Load config, build model, train on dataset, log metrics (loss, acc, memory peak via `torch.cuda.max_memory_allocated()`, wall time).
  - Benchmarks: Compare to backprop baseline (using same seed/config but with `torch.optim.SGD/Adam`).
  - Tasks: MNIST/FashionMNIST (easy), CIFAR-10 (medium), TinyStories (language), ImageNet-subset (hard if feasible).
  - Depth Tests: For all, include a "deep mode" scaling to 1000+ layers.
- **Reproducibility**:
  - Fix seeds: `torch.manual_seed(42)`, `np.random.seed(42)`.
  - Configs: YAML files in `configs/nebc/` (e.g., inherit from existing `mlp.yaml`).
  - Results: Auto-generate tables/plots in README via CI (e.g., using matplotlib).
- **Testing Pipeline**:
  1. Toy validation: XOR or linear regression to confirm basics.
  2. Small-scale: 4-layer MLP on MNIST.
  3. Medium-scale: ConvNet/Transformer on CIFAR/TinyStories.
  4. Extreme: 1000-10,000 layer chains on signal propagation (extend `deep_signal_prop.py`).
  5. Ablation: Run with/without spectral norm to quantify impact.

### 2.2 Tools and Dependencies
- **Environment**: Python 3.10+, PyTorch 2.0+, existing repo deps (e.g., tqdm, wandb for logging).
- **Hardware**: Single GPU for dev; scale to multi-GPU if needed via `torch.nn.DataParallel`.
- **Documentation**: Update README with new sections: "NEBC Extensions", per-target results tables, usage examples (e.g., `python -m eqprop.run configs/nebc/fa_spectral.yaml`).

### 2.3 Timeline (Estimated)
- Week 1: Implement and test EqProp expansions + FA.
- Week 2: DFA + CHL.
- Week 3: Hebbian Chains + full verifications.
- Week 4: Ablations, README updates, potential arXiv draft.

## 3. Target-Specific Specifications

### 3.1 Target 1: EqProp (All Flavors) + Spectral Normalization
- **Description**: Equilibrium Propagation variants (e.g., standard, lazy, event-driven, ternary-weight, temporal resonance) with mandatory spectral norm.
- **Rationale**: Already core to repo; expand to confirm it's the "unlock" for untested flavors (e.g., 2025 variants).
- **Implementation**:
  - File: Extend `tracks/eqprop_variants.py` (new if needed).
  - Key Code: In `eqprop.energy_dynamics`, apply `spectral_norm` after each weight update.
  - Flavors: Loop over list ['standard', 'lazy', 'event_driven', 'ternary', 'homeostatic'] in track.
- **Experiments**:
  - Datasets: MNIST, CIFAR-10, TinyStories (transformer variant).
  - Metrics: Acc/perplexity, memory flatness (should be O(1) vs. backprop's O(layers)).
  - Depth: Up to 1024 layers.
- **Expected Outcomes**: 100% convergence on all flavors; superior to non-spectral versions (e.g., no stalls).

### 3.2 Target 2: Feedback Alignment (FA) + Spectral Normalization
- **Description**: FA replaces backprop's symmetric gradients with random fixed feedback matrices; add spectral to prevent signal death.
- **Rationale**: FA historically struggles with depth >50 layers; spectral could revive it for modern nets.
- **Implementation**:
  - New File: `tracks/fa_spectral.py`.
  - Key Code: Inherit from `eqprop.Model`; override gradient computation with random feedback (e.g., `feedback_weights = nn.Parameter(torch.randn(...), requires_grad=False)`); apply spectral to both forward and feedback paths.
  - Update Rule: Errors propagated via feedback matrices, weights updated Hebbian-style.
- **Experiments**:
  - Models: MLP (deep), ResNet-50 on ImageNet-subset.
  - Ablation: FA alone vs. FA+spectral (expect spectral to enable 1000+ layers).
  - Tasks: MNIST (baseline), Atari (RL if extended).
- **Expected Outcomes**: Trains ResNet-50 on ImageNet without vanishing gradients; memory similar to EqProp.

### 3.3 Target 3: Direct Feedback Alignment (DFA) + Spectral Normalization
- **Description**: DFA broadcasts errors directly to each layer via random projections; spectral ensures stable propagation.
- **Rationale**: DFA is faster than FA but unstable in deep nets; spectral caps amplification.
- **Implementation**:
  - New File: `tracks/dfa_spectral.py`.
  - Key Code: Add per-layer feedback projectors (random fixed matrices); compute layer-wise errors directly from output loss; spectral on all weights.
  - Integration: Hybrid with EqProp (e.g., DFA for feedback, EqProp for settling).
- **Experiments**:
  - Datasets: CIFAR-10, TinyStories.
  - Scale: 500-5000 layers on signal prop test.
  - Compare: To standard DFA (expect explosions without spectral).
- **Expected Outcomes**: Constant-time updates per layer; viable for 10k+ layers.

### 3.4 Target 4: Modern Contrastive Hebbian Learning (CHL) + Spectral Normalization
- **Description**: CHL uses positive/negative phases (like EqProp) with Hebbian updates; modern variants include homeostasis.
- **Rationale**: CHL is EqProp-adjacent but underused; spectral prevents drift in long phases.
- **Implementation**:
  - New File: `tracks/chl_spectral.py`.
  - Key Code: Two-phase dynamics (clamped/free like EqProp); Hebbian update ΔW ∝ (y+ y+^T - y- y-^T); spectral post-update.
  - Variants: Add contrastive divergence for energy-based models.
- **Experiments**:
  - Tasks: Autoencoders on MNIST, generative on CIFAR.
  - Depth: 1000+ layer chains.
- **Expected Outcomes**: Better generative quality than plain Hebbian; stable for unsupervised learning.

### 3.5 Target 5: Deep (1000+ Layer) Pure Hebbian Chains + Spectral Normalization
- **Description**: Simple Hebbian rule (ΔW ∝ x y^T) in linear chains; spectral prevents norm explosion.
- **Rationale**: Pure Hebbian fails beyond shallow nets; this could enable "evolvable" 3D lattices (e.g., Hugo de Garis-inspired).
- **Implementation**:
  - Extend: `tracks/deep_signal_prop.py` to include Hebbian mode.
  - Key Code: No backprop; local Hebbian updates with Oja's rule for normalization; spectral on chain weights.
  - 3D Variant: Use Conv3D for cubic lattices (e.g., 10x10x1000 voxels).
- **Experiments**:
  - Tasks: Signal propagation (as existing), volumetric MNIST.
  - Scale: 10k-100k layers.
- **Expected Outcomes**: Signal reaches end without decay; enables 3D evolution sims.

## 4. Validation and Release

### 4.1 Checklist for Each Track
- [ ] Toy convergence.
- [ ] Benchmark vs. backprop (acc ±1%, memory <50%).
- [ ] Ablation: With/without spectral.
- [ ] Depth test: No failure at 1024 layers.
- [ ] README table entry: e.g., | Algorithm | Dataset | Acc | Memory | Status |

### 4.2 Release Plan
- Commit to main after local tests.
- Update README: New "NEBC" section with results, claims (e.g., "First demo of FA training 5000-layer nets").
- Outreach: Tweet summary, post to r/MachineLearning, email original authors.
- If superior to EqProp: Prioritize in repo (e.g., default mode).

This spec provides a complete, actionable roadmap. Implementation starts with Target 2 (FA) for quick wins.