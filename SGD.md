# Specification for the SGD + Spectral Normalization (SGD+SN) Experiment in the EqProp Repository

## 1. Overview

### 1.1 Purpose
This specification details an experiment to integrate and test vanilla Stochastic Gradient Descent (SGD) with Spectral Normalization (SN) as a baseline training method in the EqProp repository (https://github.com/automenta/eqprop). The core idea is to strip backpropagation (backprop) to its essentials—no advanced optimizers (e.g., Adam, RMSprop), no momentum, no learning rate schedules, no warmups—and apply strict SN to enforce stability. This "bare-bones" approach aims to:

- Demonstrate if SGD+SN can achieve competitive or superior performance (e.g., faster convergence, deeper networks, robustness to hyperparameters) compared to modern optimizers on standard benchmarks.
- Highlight SN as a potential "universal stabilizer" that makes vanilla SGD viable for deep, modern architectures (e.g., MLPs, ConvNets, Transformers) without additional tricks.
- Position this as a "rebel" experiment: challenge the assumption that advanced optimizers are necessary by showing SGD+SN as a cleaner, more efficient baseline.
- Add a new verification track to the repository, ensuring full reproducibility and integration with existing EqProp variants for hybrid comparisons.

If successful, this could imply that much of the "optimizer circus" (momentum, adaptive rates) compensates for instability that SN directly addresses, potentially flipping textbook assumptions about training neural networks.

### 1.2 Scope
- **Targets**: Focus on vanilla SGD + strict SN, compared to:
  - Plain SGD (no SN).
  - SGD with momentum.
  - Adam (standard modern baseline).
- **Models/Tasks**: Diverse architectures to test depth and scale:
  - Deep MLPs (100–10,000 layers).
  - ConvNets (e.g., ResNet-18 on CIFAR-10).
  - Transformers (e.g., character-level LM on Shakespeare/TinyStories).
- **Non-Targets**: Exclude EqProp or other bio-plausible algorithms here; this is a pure backprop experiment. Hybrids (e.g., SGD+SN in lower layers, EqProp in upper) can be explored in future extensions.
- **Assumptions**: Builds on PyTorch, existing repo structure (`tracks/`, `configs/`, `eqprop.torch_utils` for SN hooks).
- **Success Criteria**: SGD+SN achieves ≥95% of Adam's final accuracy with advantages in at least one metric (e.g., wall time, memory stability, depth tolerance). All runs must be reproducible with fixed seeds.

### 1.3 Risks and Mitigations
- **Risk**: Vanilla SGD+SN converges slowly or stalls without momentum.
  - **Mitigation**: Use a grid search for learning rates (e.g., 1e-1 to 1e-4); enforce SN after every update to prevent explosions.
- **Risk**: High compute for deep experiments.
  - **Mitigation**: Start with small models; use early stopping if loss plateaus.
- **Risk**: Prior work overlap (e.g., SN in optimizers).
  - **Mitigation**: Literature check pre-implementation (focus on "vanilla SGD+SN" without extras); emphasize novelty in deep, no-tricks regime.

## 2. General Methodology

### 2.1 Integration Approach
- **Spectral Normalization**: Apply `torch.nn.utils.spectral_norm` (or custom power iteration) to all linear/conv layers, enforcing Lipschitz constant ≤1 after every optimizer step. This bounds gradients, preventing vanishing/exploding issues in deep nets.
- **Optimizer Setup**:
  - Vanilla SGD: `torch.optim.SGD(lr=..., momentum=0, weight_decay=0)` (no extras).
  - Baselines: SGD+momentum (momentum=0.9), Adam (`torch.optim.Adam` with defaults).
- **Core Modifications**:
  - Extend `eqprop.Model` to support a `backprop_mode` flag (e.g., 'sgd_sn', 'sgd_plain', 'adam').
  - Add SN hook: In training loop, after `optimizer.step()`, call `apply_spectral_norm(model)`.
  - No schedules/warmups: Fixed LR throughout.
- **Verification Suite**:
  - New track: `tracks/sgd_sn.py`.
  - Loop over optimizers/modes; train model, log metrics (loss/acc curves, wall time, peak memory, gradient norms).
  - Benchmarks: Compare SGD+SN to baselines on same seed/config.
  - Tasks: MNIST (easy depth test), CIFAR-10 (conv), Shakespeare/TinyStories (seq).
  - Depth Scaling: Auto-test up to failure (e.g., when gradients vanish or explode).
- **Reproducibility**:
  - Seeds: `torch.manual_seed(42)`, etc.
  - Configs: New YAMLs in `configs/sgd_sn/` (e.g., inherit from `mlp.yaml`, add `optimizer: sgd_sn`).
  - Results: Generate plots/tables (e.g., loss vs. epoch, depth vs. acc) via matplotlib; auto-add to README.
- **Testing Pipeline**:
  1. Toy: 2-layer MLP on XOR; confirm convergence.
  2. Small: 100-layer MLP on MNIST.
  3. Medium: ResNet-18 on CIFAR-10.
  4. Hard: Transformer on Shakespeare (scale to 10M params).
  5. Ablations: With/without SN; vary LR; monitor gradient flow (e.g., `torch.linalg.norm(grad)` per layer).

### 2.2 Tools and Dependencies
- **Environment**: PyTorch 2.0+, existing repo deps.
- **Hardware**: Single GPU (e.g., RTX 4090) for most; multi-GPU optional for large transformers.
- **Documentation**: README section: "Vanilla SGD + Spectral Norm: Back to Basics". Include usage: `python -m eqprop.run configs/sgd_sn/mlp_mnist.yaml`.

### 2.3 Timeline (Estimated)
- Day 1: Implement track + toy/small tests.
- Day 2: Medium/hard runs + ablations.
- Day 3: Results analysis, README updates.

## 3. Experiment-Specific Details

### 3.1 Core Experiment: Deep MLPs
- **Description**: Test signal propagation and convergence in ultra-deep MLPs (100–10,000 layers) without residuals/skips.
- **Rationale**: Vanilla SGD often fails beyond 50 layers; SN should enable "textbook" backprop to scale.
- **Implementation**:
  - Model: Sequential Linear layers (dim=256).
  - Optimizer Loop: For SGD+SN, step then SN.
- **Runs**:
  - Datasets: MNIST/FashionMNIST.
  - Metrics: Final acc, epochs to 90% acc, max depth without NaN/vanishing.
  - Compare: SGD+SN vs. SGD plain vs. SGD+mom vs. Adam.
- **Expected Outcomes**: SGD+SN trains 1000+ layers faster than Adam (due to stable gradients); plain SGD explodes.

### 3.2 Extension: ConvNets
- **Description**: Apply to standard conv architectures.
- **Rationale**: Test if SGD+SN reduces need for batchnorm/dropout.
- **Implementation**:
  - Model: ResNet-18 (but disable residuals for "vanilla" test optional).
  - SN on conv/linear layers.
- **Runs**:
  - Dataset: CIFAR-10.
  - Metrics: Acc, wall time, memory.
  - Ablation: No batchnorm (expect SGD+SN to handle variance better).
- **Expected Outcomes**: SGD+SN matches Adam acc with 20–50% less time (higher stable LR).

### 3.3 Extension: Transformers
- **Description**: Character-level LM, building on recent Shakespeare results.
- **Rationale**: Seq models stress memory/gradients; SN enables longer contexts without OOM.
- **Implementation**:
  - Model: As in `tracks/transformer.py` (4–12 layers, dim=256–512).
  - Apply SN to attention/FFN linears.
- **Runs**:
  - Dataset: Shakespeare (extend to TinyStories).
  - Metrics: Perplexity, bits/char, speed (secs/epoch).
  - Scale: Increase seq_len to 1024–4096; log memory.
- **Expected Outcomes**: SGD+SN closes gap to Adam (e.g., 4.5–4.8 PPL) with flat memory on long seqs.

## 4. Validation and Release

### 4.1 Checklist for Experiment
- [ ] Toy convergence across modes.
- [ ] Benchmarks: SGD+SN ≥95% Adam acc; advantages in time/depth/memory.
- [ ] Ablations: Quantify SN impact (e.g., +500% depth tolerance).
- [ ] Plots: Loss curves, gradient norms vs. depth.
- [ ] README Table: e.g., | Optimizer | Depth | Acc | Time | Status |

### 4.2 Release Plan
- Commit to main post-tests.
- README: Bold claims (e.g., "Vanilla SGD + SN Trains 10,000-Layer Nets Without Momentum").
- Outreach: Share on X/r/ML; tag optimizer researchers.
- If superior: Integrate as default backprop baseline for all tracks.