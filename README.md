# Bioplausible

## 🧬 Introduction

Modern deep learning is built on backpropagation — an algorithm that is mathematically elegant but physically impossible. It demands three things no physical or biological system can provide: symmetric feedback weights (weight transport), a global clock that freezes forward activity to propagate errors backward, and memory proportional to network depth. These constraints anchor deep learning to digital hardware, blocking its realization in analog circuits, neuromorphic chips, optical processors, and — most importantly — the brain.

Bioplausible is a research framework for the alternative: **learning algorithms whose synaptic updates depend only on signals locally available at each connection**. Instead of a global gradient, training emerges from local, energy-based dynamics — networks relax toward equilibrium and contrasts between free and nudged states drive weight changes. The implications are substantial: memory complexity becomes independent of depth, allowing arbitrarily deep networks on fixed hardware. Learning becomes asynchronous and event-driven, naturally matching the physics of analog substrates. Contractive dynamics confer fault tolerance: networks self-heal from perturbation, making them candidates for noisy, low-power, imprecise physical computation. The same locality that makes these algorithms biologically plausible also makes them physically realizable.

The framework demonstrates that capabilities previously reserved for backpropagation can be matched — and in regimes backpropagation cannot reach, exceeded — by algorithms compatible with the actual physics of computation. It provides a **generative physico-computational engine** built on a 5-dimensional ontology that decomposes every learning system into orthogonal, composable primitives, plus the infrastructure to evaluate them rigorously and discover better ones autonomously.

---

## 🔮 The 5-Dimensional Ontology

Every learning system in Bioplausible maps uniquely to a coordinate in a tensor product of five fundamental axes:

```
System = Substrate ⊗ Geometry ⊗ StateDynamics ⊗ CreditAssignment ⊗ ParameterUpdate
```

This decomposition transforms the framework from a "library of models" into a **generative engine** — any valid combination of primitives yields a coherent learning system, and the space of all combinations is the search space for the AutoScientist.

| Axis | Role | Primitives |
|------|------|------------|
| **🔩 Substrate (S)** | Physical state space: precision, noise, sparsity constraints | `Digital`, `Memristive` (conductance, IR-drop), `Neuromorphic` (async spikes), `Photonic` (phase/amplitude), `Quantum` (unitary gates), `Noisy` |
| **🔷 Geometry (G)** | Topology & routing of computational units | `FeedforwardDAG` (MLP/CNN), `RecurrentAttractor` (Hopfield/EqProp), `TileMesh` (TileNet), `FabricPC` (arbitrary node-edge), `SpatialLattice3D` (neural_cube) |
| **🌀 StateDynamics (D)** | Forward evolution & settling (the "forward pass") | `EnergyMinimization` (EqProp), `PredictiveSettling` (Predictive Coding), `SpikeIntegration` (LIF/Izhikevich), `InstantaneousPass` (FF/Backprop), `LazyStateDynamics` (on-demand activation) |
| **💡 CreditAssignment (C)** | Error routing & pseudo-gradient computation | `ThermodynamicContrast` (EqProp free/nudged), `RandomProjectionsCredit` (FA/DFA), `LocalGoodnessCredit` (Forward-Forward/PEPITA), `TemporalTraceCredit` (STDP), `TargetInversionCredit` (Target Prop), `HomeostaticCredit` (autonomous Lipschitz scaling) |
| **🔧 ParameterUpdate (U)** | Physical weight change rule ΔW | `EuclideanUpdate` (SGD/Adam), `RiemannianOrthogonalUpdate` (Muon), `SpectralConstrainedUpdate`, `NaturalGradientUpdate` (Fisher), `ElasticConsolidationUpdate` (EWC) |

### Algebraic Composition (API)

```python
from bioplausible.core.ontology import (
    System, DigitalSubstrate, FeedforwardGeometry,
    InstantaneousDynamics, BackpropCredit, EuclideanUpdate,
    GeometryConfig, RecurrentGeometry, EnergyMinimizationDynamics,
    ThermodynamicContrastCredit, MemristiveSubstrate, TileGeometry,
    TileAlgorithmConfig, LazyStateDynamics, HomeostaticCredit
)

# A standard backprop MLP — no equilibrium dynamics
system = System(
    substrate=DigitalSubstrate(),
    geometry=FeedforwardGeometry(GeometryConfig(input_dim=784, output_dim=10, hidden_dims=(256, 128))),
    dynamics=InstantaneousDynamics(),
    credit=BackpropCredit(),
    update=EuclideanUpdate(step_size=0.01)
)

# Equilibrium Propagation on recurrent geometry
eqprop = System(
    substrate=DigitalSubstrate(),
    geometry=RecurrentGeometry(GeometryConfig(input_dim=784, output_dim=10, hidden_dims=(256, 128)), symmetric=True),
    dynamics=EnergyMinimizationDynamics(StateDynamicsConfig(n_iters=20, beta=0.5)),
    credit=ThermodynamicContrastCredit(),
    update=EuclideanUpdate(step_size=0.01)
)

# TileNet: async tile mesh with configurable credit assignment
tile = System(
    substrate=DigitalSubstrate(),
    geometry=TileGeometry(TileAlgorithmConfig(algorithm="ep", n_tiles=4, tile_size=64)),
    dynamics=InstantaneousDynamics(),
    credit=BackpropCredit(),  # or ThermodynamicContrastCredit, RandomProjectionsCredit, ...
    update=EuclideanUpdate(step_size=0.01)
)

# Memristive EqProp: same algorithm, physical substrate
memristive_eqprop = System(
    substrate=MemristiveSubstrate(MemristiveConfig(conductance_range=(1e-6, 1e-3), ir_drop=0.02)),
    geometry=RecurrentGeometry(...),
    dynamics=EnergyMinimizationDynamics(...),
    credit=ThermodynamicContrastCredit(),
    update=EuclideanUpdate(step_size=0.01)
)
```

Formerly many hardcoded models (e.g., `optical_looped_mlp`, `quantized_looped_mlp`, `crossbar_looped_mlp`, `eqprop_transformer`, `neural_cube`, `sparse_equilibrium`, `momentum_equilibrium`, TileNet variants) are now **emergent coordinates** in this space.

---

## ⚡ Thermodynamic Invariant: Energy as First-Class Object

Energy binds Geometry and StateDynamics. The framework elevates the energy function `E(x)` to a first-class object, enabling mathematical stability proofs *before* implementation:

- **Symmetric topology + EnergyMinimization** → guaranteed fixed-point convergence (Hopfield/EqProp) via LaSalle's invariance principle
- **Directed topology** → requires Control-Lyapunov formulation for stability (formally verified for PredictiveSettlingDynamics)
- **Free energy tracking** → per-iteration Lyapunov certificates (`track_free_energy_per_iter`) for predictive coding and directed FA

This enables the AutoScientist to reason about *physical realizability* as a constraint, not an afterthought.

---

## 🖥️ CLI Commands

All entry points installed with `uv sync --dev`. The CLI has been consolidated under the `biopl` dispatcher:

### Main Dispatcher: `biopl`

```bash
biopl <command> [args]
```

| Subcommand | Purpose | Legacy Alias |
|------------|---------|--------------|
| `biopl run` | Campaign runner (validate/plan/run) | `biopl-run` |
| `biopl report` | Render experiment reports | `biopl-report` |
| `biopl parity` | Backprop parity benchmark | `biopl-parity` |
| `biopl repro` | Reproducibility verification | `biopl-repro-check` |
| `biopl hpo` | Hyperparameter optimization | `biopl-hpo` |
| `biopl audit` | Registry metadata audit | `biopl-registry-audit` |
| `biopl frontier` | Pareto frontier analysis | `biopl-frontier` |
| `biopl rank` | Family ranking from HPO studies | `biopl-compare` |
| `biopl lab` | Interactive experiments & model inspection | — |

### Standalone Commands (for scripting/CI)

| Command | Purpose |
|---------|---------|
| `biopl-scientist` | Autonomous experiment loop (AutoScientist hypercube campaigns) |
| `biopl-failure-manifesto` | Structured negative result documentation |
| `biopl-export-kernel` | Export kernel backend (untrained) |
| `biopl-export-trained-kernel` | Train + export kernel backend with weights |
| `biopl-p2p-worker` | P2P worker for distributed training (renamed from `eqprop-p2p-worker`) |

### Deprecated / Removed

| Old Command | Replacement |
|-------------|-------------|
| `eqprop-verify` | `biopl parity` / `biopl run` with campaign YAML |
| `eqprop-p2p-worker` | `biopl-p2p-worker` (renamed — framework is no longer EqProp-specific) |
| `biopl-run` / `biopl-report` / etc. | Use `biopl <subcommand>` (dispatcher) |

> **Migration**: `biopl run --config campaign.yaml` replaces `biopl-run campaign.yaml`. The dispatcher ensures a single versioned entry point.

---

## 📦 Installation

```bash
uv sync --dev
```

### Quickstart: Interactive Demo

```bash
uv run python demo/main.py
```

Launches a NiceGUI web dashboard at `http://localhost:8080` with:
- Model training across ontology coordinates
- Live loss/accuracy curves
- Hyperparameter controls
- AutoScientist hypothesis proposals

---

## 🏗️ Core Architecture

### 1. Ontology Protocols (`bioplausible/core/ontology.py`)

Five `Protocol` classes with PEP 695 generics, frozen slotted config dataclasses, and reference implementations for every primitive — pure, composable infrastructure.

```python
# Protocol signatures (structural typing — zero-cost abstraction)
class Substrate(Protocol):
    config: SubstrateConfig
    def forward_operator(self, x: Tensor) -> Tensor: ...
    def weight_update_operator(self, delta: Tensor) -> Tensor: ...

class Geometry(Protocol):
    config: GeometryConfig
    def forward(self, x: Tensor, substrate: Substrate) -> Tensor: ...
    def route(self, activations: Tensor) -> Tensor: ...

class StateDynamics(Protocol):
    config: StateDynamicsConfig
    def settle(self, state, geometry, substrate, target) -> SystemState: ...

class CreditAssignment(Protocol):
    config: CreditAssignmentConfig
    def compute_pseudo_gradient(self, free, nudged, loss, geometry) -> list[Tensor]: ...
    def surrogate_objective(self, free, nudged, geometry) -> Tensor: ...  # default provided

class ParameterUpdate(Protocol):
    config: ParameterUpdateConfig
    def step(self, params, pseudo_grads, geometry) -> dict[str, Tensor]: ...
```

### 2. System & Trainers

| Component | Purpose |
|-----------|---------|
| `System[TS, TG, TD, TC, TU]` | Generic 5-layer composition; invalid combos caught at type-check |
| `SystemTrainer` | 5-stage pipeline: Geometry.forward → StateDynamics.settle → CreditAssignment.compute_pseudo_gradient → ParameterUpdate.step → Substrate.weight_update_operator |
| `DistributedSystemTrainer` | In-process P2P coordination; shards along Geometry (TileMesh), federates at ParameterUpdate; CreditAssignment stays local |
| `ModelAdapter` | Strangler Fig adapter: projects legacy Registry models → 5-D System via metadata inference with per-family tolerance calibration |
| `Registry.to_system()` | One-call projection of any registered component |

### 3. Factories (`bioplausible/core/system_trainer.py`)

```python
from bioplausible.core.system_trainer import (
    compose_system, compose_system_from_configs, extract_config,
    create_eqprop_system, create_backprop_system, create_fa_system,
    create_tile_system, create_predictive_coding_system
)

# Config round-trip (L0 schema lock)
configs = extract_config(system)
system2 = compose_system_from_configs(configs)
assert system == system2  # identity verified
```

### 4. Hardware Substrates ✅

| Substrate | Physics Model | Verification |
|-----------|---------------|--------------|
| `MemristiveSubstrate` | Conductance matrices, bounded precision, IR-drop noise | Gradient equivalence vs. digital; positive bounded conductance |
| `NeuromorphicSubstrate` | Async spike routing, strict sparsity, passivity | Property test: deterministic noise cancels in diff (‖na-nb‖ ≤ ‖a-b‖) |
| `OpticalSubstrate` | Phase/amplitude encoding, coherent interference | Phase wrapping to [-π, π]; no NaN/inf outputs |
| `QuantumSubstrate` | Parameterized unitary gates, parameter-shift rule | Parameter-shift matches finite-difference (cosine ≥ 0.999) |

---

## 🔬 Validation Framework: Machine-Certified Hypercube

The framework enforces **correctness by construction** through a layered verification regime. The fast-CI gate certifies the entire hypercube in seconds on CPU.

### ✅ Property Locks (L1–L7 + S/G/D/C/U Axes)

| Lock | Property | Key Assertions |
|------|----------|----------------|
| **L1** | Composed systems train & produce valid metrics | Backprop/FA/Tile systems train; loss≥0, accuracy∈[0,1] |
| **L2** | Pipeline stages pure functions of preceding axes | Geometry.forward deterministic; credit independent of update; substrate noise only effect |
| **L3** | Locality axioms: ThermodynamicContrast invariant to non-local perturb; FA feedback fixed at init & seed-independent | Layer-0 pseudo-gradient invariant; B matrices fixed; different seeds → different B |
| **L4** | Lyapunov/energy: energy non-increasing; Control-Lyapunov for PredictiveSettling | Energy monotonic (EqProp); free energy non-increasing (PredictiveCoding); convergence threshold |
| **L5** | Determinism: same seed + same device = bitwise equal params & metrics (CPU & GPU deterministic) | Parametrized over system factories |
| **L6** | Round-trip & totality: configs round-trip identity; Registry.to_system() projects all registered models | Identity on configs; protocol conformance on projected systems |
| **L7** | Distributed seam: SystemTrainer runs; fault tolerance | gRPC fault injection test captures lost workers, step, partial metrics |
| **S-axis** | Neuromorphic passivity; Quantum parameter-shift equivalence | Deterministic noise cancellation; cosine ≥ 0.999 vs FD |
| **D-axis** | SpikeIntegration Lyapunov (membrane bounded, spike variance non-increasing); LazyStateDynamics | Spike counts tracked; bounded activations |
| **C-axis** | TemporalTrace STDP window (causal +, anti-causal -, antisymmetric, exponential decay); surrogate objectives | Sign matches timing; W(Δt) = -W(-Δt); FD cosine ≥ 0.95 |
| **U-axis** | Muon orthogonalizes gradient (G^T G ≈ I); SpectralConstrained SVD ≤ 1.0; Natural whitens; Elastic moves toward old params | Newton-Schulz converges; diagonal Fisher whitening; δ·(w-old_w) < 0 |

### 🧬 Biology Axiom Property Tests (Hypothesis-based)

| Axiom | Test | Method | Threshold |
|-------|------|--------|-----------|
| **EP Gradient Equivalence** | EqProp gradient aligns with BPTT | Cosine similarity | ≥ 0.5 |
| **Lyapunov Energy Descent** | Free energy monotonically non-increasing along relaxation | Hypothesis | Slack 1e-3; final < initial |
| **Contraction Mapping** | Relaxation operator Lipschitz < 1 | Pairwise distance ratio L < 1; Power iteration σ_max < 1 | Step sizes 0.1–0.5 |
| **Fixed-Point Reliability** | Unique attractor from random initializations | Relative diff < 1e-3; Idempotence \|\|T(h\*)-h\*\|\| < 1e-4 | |
| **Weight-Transport Freeness** | FA backward weights ≠ forward transpose; separate memory | \|\|B - W^T\|\| > 1e-3; data_ptr() distinct | standard_fa, adaptive_fa, DFA |
| **Adaptive-FA Alignment** | Feedback matrices align with forward weights over training | cos(B, W^T) improvement > 0.05 | biologically slow B regime |

### ✅ Integration Verification Gates (All Passing)

| Gate | Result |
|------|--------|
| Gradient equivalence (finite-difference) | CE families cos≥0.9: backprop, FA, DirectFA, StochasticFA, MEP-backprop; MSE families cos≥0.6: EqProp, MEP-EP, CHL |
| Ontology layer equivalence | ThermodynamicContrast=Backprop under InstantaneousDynamics; RiemannianOrthogonal preserves orthogonality; EnergyMinimization converges |
| Energy invariants (formal proofs) | Lyapunov, Control-Lyapunov, Substrate passivity, EqProp energy, Composition |
| Kernel equivalence (Triton vs PyTorch) | max_diff < 1e-5, rel_diff < 1e-4 |
| Kernel accuracy parity | FA, Backprop, PEPITA, DTP: kernel accuracy within 1% of reference on digits/synthetic |
| Registry audit | 0 missing critical fields |
| Reproducibility | Models bitwise reproducible |
| Backprop parity | Runs successfully |
| Static typing | 0 errors in strict mode |
| Formatting | Clean |

### 🧪 Test Commands

```bash
# Property locks (fast CI gate)
uv run pytest tests/property/test_ontology_locks.py -q

# Core ontology unit tests
uv run pytest tests/unit/core/test_ontology.py -q

# Integration: gradient equivalence + energy proofs
uv run pytest tests/integration/test_gradient_equivalence.py tests/integration/test_energy_invariants.py -q

# Kernel equivalence (Triton vs PyTorch)
uv run pytest tests/integration/test_kernel_equivalence.py -q

# Kernel accuracy parity (end-to-end learning)
uv run pytest tests/integration/test_kernel_accuracy_parity.py -q

# gRPC seam test
uv run pytest tests/integration/test_grpc_seam.py -q

# Full suite
uv run pytest tests/ -q

# Type checking (strict)
uv run pyright .

# Formatting & linting
uv run ruff format --check . && uv run ruff check .
```

---

## 🤖 Automated Research: Hypercube Campaigns

The 5-D ontology gives the AutoScientist a **structured search space** instead of a flat model list:

| Campaign Type | Fixed Axes | Varied Axis | Example Hypothesis |
|---------------|------------|-------------|-------------------|
| Substrate Ablation | G, D, C, U | S: Digital → Memristive/Optical/Quantum | At what IR-drop does EqProp parity break? |
| Epistemology Swap | S=Optical, G=TileMesh, D=EnergyMinimization | C: ThermodynamicContrast ↔ RandomProjectionsCredit | Does optical hardware favor FA (lower settling energy)? |
| Kinetics Discovery | S, G, D, C | U: Euclidean ↔ Riemannian ↔ Spectral ↔ Natural | Can Spectral constraints stabilize Memristive settling? |
| Composite | S=Memristive, D=EnergyMinimization | U=SpectralConstrained | "IR-drop (S) + Spectral (U) → stable settling (D)" |

**Key AutoScientist capabilities:**
- 🧠 Chain-of-thought templates operating on ontology axes
- 📚 arXiv retrieval + semantic search for prior art
- 🔀 Counterfactual generator: "What if β schedule changed?"
- 📊 Knowledge Base meta-analysis: scaling laws, algorithm fingerprinting, failure manifold clustering, algorithm phylogeny
- 💾 Campaign persistence/resume (YAML+SQLite, git-like branching)
- 👁️ Human-in-the-loop dashboard (NiceGUI, WebSocket live updates)
- 🖥️ Local LLM support (Ollama auto-pull, llama.cpp quantization, speculative decoding)

---

## 🧪 Flagship Experiments (Implemented)

| Experiment | File | Purpose |
|------------|------|---------|
| TileNet Scaling Sweep | `experiments/tile_scaling.py` | Depth/width scaling on MNIST/CIFAR-10 across tile algorithms + backprop |
| EqProp Vision Parity | `experiments/eqprop_vision_parity.py` | EqProp variants on MNIST/Fashion-MNIST/CIFAR-10/SVHN |
| MEP Preset Tournament | `experiments/mep_tournament.py` | Factorized ablation: gradient×update×constraint×feedback with ANOVA + Sobol |
| FA Depth Scaling | `experiments/fa_depth_scaling.py` | Extreme depth, MNIST + synthetic parity |
| MoT Ablation | `experiments/mot_ablation.py` | Dense vs sparse tile routing (top-k, random, learned) |
| Cross-Domain Transfer | `experiments/cross_domain_transfer.py` | Vision→LM/RL/graph transfer, local vs global learning |
| Tile Algorithm Comparison | `experiments/tile_algorithm_comparison.py` | Fair comparison of PC/EP/FA/TP/Hebbian/SNN/Backprop on same substrate |

---

## 🌐 Evaluation Domains

The framework supports **7 evaluation domains** with 60+ tasks/datasets, unified through a common task interface (`DomainTask` protocol). Each domain has dedicated data loaders, metrics, and task-specific configurations.

### 📊 Domain Overview

| Domain | Tasks | Datasets | Models Tested | Key Metrics |
|--------|-------|----------|---------------|-------------|
| **Vision** | 12 | MNIST, Fashion-MNIST, KMNIST, USPS, CIFAR-10, CIFAR-100, SVHN, Digits, XOR, Spiral, Circles | 25+ | Accuracy, Loss, Energy, FLOPs, Memory |
| **Language (LM)** | 4 | Tiny Shakespeare, Char N-gram, WikiText-2, Penn Treebank | 12+ | Perplexity, BPC, Accuracy, Compression |
| **Reinforcement Learning (RL)** | 6 | CartPole, Pendulum, Acrobot, MountainCar, LunarLander | 8+ | Episode Return, Success Rate, Sample Efficiency |
| **Graph** | 3 | Cora, CiteSeer, PubMed | 6+ | Node Classification Accuracy, F1 |
| **Tabular** | 5 | Breast Cancer, Iris, Wine, Diabetes, California Housing | 10+ | Accuracy, R², AUC |
| **Time Series** | 2 | Synthetic Forecast, (ETT variants planned) | 6+ | MSE, MAE, CRPS |
| **Scientific** | 2 | Synthetic Physics, (PDE variants planned) | 5+ | Relative L2, Conservation Error |

---

### 🖼️ Vision Domain

**Tasks & Datasets**

| Task | Type | Input Dim | Output Dim | Classes | Train/Val/Test Split | Notes |
|------|------|-----------|------------|---------|---------------------|-------|
| `mnist` | Classification | 784 | 10 | 10 | 50k/10k/10k | Standard benchmark |
| `fashion_mnist` | Classification | 784 | 10 | 10 | 50k/10k/10k | Harder than MNIST |
| `kmnist` | Classification | 784 | 10 | 10 | 50k/10k/10k | Kuzushiji-MNIST |
| `usps` | Classification | 784 | 10 | 10 | 7.3k/1.8k/2k | USPS handwritten digits |
| `cifar10` | Classification | 3072 | 10 | 10 | 40k/10k/10k | 32×32 RGB |
| `cifar100` | Classification | 3072 | 100 | 100 | 40k/10k/10k | Fine/coarse labels |
| `svhn` | Classification | 3072 | 10 | 10 | 60k/10k/26k | Street View House Numbers |
| `digits` | Classification | 64 | 10 | 10 | ~1.5k | sklearn digits (8×8) |
| `xor` | Boolean | 2 | 2 | 2 | Synthetic | Non-linear separability |
| `spiral` | Classification | 2 | 3 | 3 | Synthetic | Interlocking spirals |
| `circles` | Classification | 2 | 2 | 2 | Synthetic | Concentric circles |

**Models Registered (Vision-Compatible)**

| Model Family | Variants | Locality Level | Requires Backward |
|--------------|----------|----------------|-------------------|
| `backprop` | mlp, cnn, resnet | global | ✅ |
| `eqprop` | mlp, conv, transformer | equilibrium | ❌ |
| `fa` | standard, dfa, adaptive, stochastic | layerwise | ✅ (random) |
| `predictive_coding` | standard, hierarchical | local | ❌ |
| `hebbian` | 3d, oja, 3-factor | forward-only | ❌ |
| `pepita` | forward-forward, goodness | forward-only | ❌ |
| `target_prop` | standard, difference | layerwise | ❌ |
| `spiking` | lif, izhikevich, stdp | temporal | ❌ |
| `tile` | ep, fa, pc, tp, hebbian | equilibrium/local | varies |

**Quick Commands**

```bash
# Run vision benchmark (all models, all vision tasks)
biopl lab benchmark --domain vision --quick

# Run specific model on MNIST
biopl lab core-train --model eqprop_mlp --task mnist --epochs 10

# Cross-domain transfer: vision → LM
python experiments/cross_domain_transfer.py --source vision --target lm
```

---

### 📝 Language Modeling Domain

**Tasks & Datasets**

| Task | Type | Input Dim | Output Dim | Vocab Size | Sequence Length | Train Tokens |
|------|------|-----------|------------|------------|-----------------|--------------|
| `tiny_shakespeare` | Next-char LM | 65 | 65 | 65 | 100 | ~1M |
| `char_ngram` | N-gram LM | configurable | vocab | 256 | 16–64 | Synthetic |
| `wikitext2` | Word-level LM | 33278 | 33278 | ~33k | 128 | ~36M |
| `penn_treebank` | Word-level LM | 10000 | 10000 | ~10k | 128 | ~1.3M |

**Models Registered (LM-Compatible)**

| Model Family | Variants | Key Feature |
|--------------|----------|-------------|
| `backprop` | lstm, transformer, gpt | Standard autoregressive |
| `eqprop` | causal_transformer, attention_only, recurrent_core | Equilibrium attention |
| `fa` | transformer_fa, lstm_fa | Random feedback in LM |
| `pepita` | ff_lm, goodness_lm | Forward-forward LM |
| `tile` | ep_tile_lm, fa_tile_lm | Tiled language models |

**Key Experiments**

```bash
# EqProp vs Backprop on language modeling (Track 37)
python experiments/language_modeling_comparison.py --epochs 50

# Run LM benchmark
biopl lab benchmark --domain lm --models backprop_transformer,eqprop_causal_transformer

# AutoScientist campaign on LM
biopl scientist --campaign campaigns/lm_hypercube.yaml
```

---

### 🎮 Reinforcement Learning Domain

**Tasks & Environments**

| Task | Type | Observation Space | Action Space | Horizon | Reward Structure |
|------|------|-------------------|--------------|---------|------------------|
| `cartpole` | Classic Control | Box(4) | Discrete(2) | 500 | +1/step |
| `pendulum` | Classic Control | Box(3) | Box(1) | 200 | -θ² - 0.1θ̇² - 0.001u² |
| `acrobot` | Classic Control | Box(6) | Discrete(3) | 500 | -1/step |
| `mountain_car` | Classic Control | Box(2) | Discrete(3) | 200 | -1/step |
| `lunar_lander` | Box2D | Box(8) | Discrete(4) | 1000 | Shaped + sparse |

**Models Registered (RL-Compatible)**

| Model Family | Algorithm | Policy Type | Notes |
|--------------|-----------|-------------|-------|
| `backprop` | PPO, A2C, DQN, SAC | MLP/Gaussian | Standard baselines |
| `eqprop` | EqProp-PPO, EqProp-A2C | Energy-based policy | Equilibrium actor-critic |
| `fa` | FA-PPO, FA-A2C | Random feedback policy | Weight-transport free RL |
| `hebbian` | Hebbian-RL | Local plasticity | Pure Hebbian policy gradient |
| `spiking` | SNN-PPO, STDP-RL | Spiking policy | Neuromorphic RL |
| `tile` | Tile-PPO | Tiled actor-critic | Distributed RL |

**Key Experiments**

```bash
# RL benchmark across algorithms
biopl lab benchmark --domain rl --quick

# EqProp on CartPole (energy-based policy)
biopl lab core-train --model eqprop_ppo --task cartpole --epochs 100

# FA vs Backprop on continuous control
python experiments/fa_rl_comparison.py --env pendulum --seeds 10
```

---

### 🕸️ Graph Domain

**Tasks & Datasets**

| Task | Type | Nodes | Edges | Features | Classes | Split |
|------|------|-------|-------|----------|---------|-------|
| `cora` | Node Classification | 2,708 | 5,429 | 1,433 | 7 | Planetoid |
| `citeseer` | Node Classification | 3,327 | 4,732 | 3,703 | 6 | Planetoid |
| `pubmed` | Node Classification | 19,717 | 44,338 | 500 | 3 | Planetoid |

**Models Registered (Graph-Compatible)**

| Model Family | Variants | Aggregation |
|--------------|----------|-------------|
| `backprop` | GCN, GAT, GraphSAGE | Message passing |
| `eqprop` | EqProp-GCN, EqProp-GAT | Equilibrium message passing |
| `fa` | FA-GCN, FA-GAT | Random feedback GNN |
| `predictive_coding` | PC-GNN | Predictive coding on graphs |
| `tile` | Tile-GCN | Tiled graph learning |

---

### 📋 Tabular Domain

**Tasks & Datasets**

| Task | Type | Samples | Features | Classes/Target | Source |
|------|------|---------|----------|----------------|--------|
| `breast_cancer` | Classification | 569 | 30 | 2 (malignant/benign) | sklearn |
| `iris` | Classification | 150 | 4 | 3 | UCI |
| `wine` | Classification | 178 | 13 | 3 | UCI |
| `diabetes` | Regression | 442 | 10 | Continuous | sklearn |
| `california_housing` | Regression | 20,640 | 8 | Continuous | sklearn |

**Models**: All MLP-based families (backprop, eqprop, fa, pepita, hebbian, tile) support tabular tasks.

---

### 📈 Time Series Domain

**Tasks**

| Task | Type | Sequence Length | Features | Horizon | Source |
|------|------|-----------------|----------|---------|--------|
| `synthetic_forecast` | Forecasting | 100 | 1–5 | 10–50 | Synthetic (sin, AR, chaos) |
| `ett_h1` | Forecasting | 168 | 7 | 24 | ETT (planned) |

**Models**: RNN/LSTM/Transformer families across all credit assignments.

---

### 🔬 Scientific Domain

**Tasks**

| Task | Type | Equation | Dimensions | Resolution | Source |
|------|------|----------|------------|------------|--------|
| `synthetic_physics` | PDE Solving | Heat, Wave, Burgers | 1D/2D | 64×64 | Synthetic |
| `navier_stokes` | PDE Solving | Navier-Stokes | 2D | 64×64 | Synthetic (planned) |

**Models**: Physics-informed variants (PINO, DeepONet, FNO) adapted to bioplausible credit assignments.

---

## 🌐 Distributed Training & P2P

### Multi-GPU Training
PyTorch Lightning with DDP, FSDP, DeepSpeed. `TileShardedBackend` with NCCL `all_reduce_gradients`/`broadcast_params` scales TileNet beyond 1B parameters.

### P2P Coordinator System (gRPC + Kademlia)
Decentralized coordination at `bioplausible/p2p/`:
- 🔑 **Kademlia DHT** (`dht.py`): Peer discovery, KV storage, bootstrap nodes, async background operation. Integration test: 2-node connectivity + best-model propagation with score-based optimistic locking
- 🔗 **gRPC Service** (`proto/tile_mesh.proto`, `grpc_service.py`): `TileMeshService` with `ExecuteStep`, `BroadcastParams`, `AggregateGradients`
- 🏊 **Connection Pool** (`GRPCConnectionPool`): Peer lifecycle, health checks, retry/backoff
- 🔀 **DistributedSystemTrainer**: In-process multi-worker coordination; shards along TileGeometry, federates at ParameterUpdate
- 🛡️ **Fault Tolerance**: `DistributedTrainingError` captures lost workers, step, partial metrics on gRPC failure

CLI: `biopl-p2p-worker` starts a worker node (renamed from `eqprop-p2p-worker` — the P2P layer is algorithm-agnostic).

```bash
# Start a P2P worker
biopl-p2p-worker --bootstrap-ip 192.168.1.100 --task mnist --mode deep

# Run distributed TileNet training
biopl run --config campaigns/distributed_tile.yaml
```

---

## 🚀 Deployment & Inference

### Model Export (`bioplausible/deployment.py`)
- 📦 **ONNX**: dynamic axes, opset 17+, TileNet deployment models export with 0 diff vs PyTorch
- 🔗 **TorchScript**: trace method works for all TileNet models
- 🔢 **INT8 Quantization**: dynamic PTQ, static PTQ, QAT preparation
- ⚖️ **Ternary Quantization**: Post-training conversion to `TernaryLinear` ({-1, 0, +1}), STE-based, bit-operation counting
- 🔬 **HLS/Verilog/NxSDK/SPICE**: FPGA/neuromorphic export via `acceleration/export.py`

### Inference Engine
`InferenceServer` — production-ready async inference:
- 📦 Dynamic batching (configurable max batch size/timeout)
- ⚡ TensorRT optimization (fp16/int8, dynamic shapes)
- 🌐 FastAPI endpoints: `/predict` (async batched), `/predict/sync`, `/health`, `/metrics`
- 🔄 Graceful startup/shutdown via lifespan events

---

## 📊 Analysis & Visualization (`bioplausible/analysis/`)

| Module | Purpose |
|--------|---------|
| `dynamics.py` | Energy trajectories, gradient alignment, tile heatmaps, convergence — interactive Plotly |
| `scaling.py` | Power-law fitting, Chinchilla laws, `ScalingLawFitter`, bootstrap CIs, extrapolation |
| `pareto.py` | Multi-objective Pareto frontier (accuracy, FLOPs, memory, energy, time), knee detection, 3D Plotly |
| `ablation.py` | Leave-one-out, Sobol sensitivity indices, automated HTML/Markdown/JSON/CSV reports |
| `genealogy.py` | Hyperparameter fingerprinting, PCA/t-SNE/UMAP, phylogenetic trees, algorithm maps |
| `interpretability.py` | Weight spectra (SVD, condition number, effective rank), receptive fields, MI, concept alignment, causal mediation |
| `energy_landscape.py` | 2D loss/energy slices (gradient-random, PCA, top-eigen), Hessian spectrum (Lanczos), 3D viz, minima detection |
| `failure_manifesto.py` | Structured negative result docs: what failed, why, search space, partial successes, future hypotheses |
| `tile_dynamics.py` | Tile settling trajectories, utilization, routing patterns |
| `tile_profiler.py` | Per-tile compute/memory profiling |

---

## ⚡ Hardware Acceleration (`bioplausible/acceleration/`)

| Module | Purpose |
|--------|---------|
| `kernels.py` | Pure NumPy/CuPy reference for correctness |
| `triton_kernels.py` | Triton JIT fused ops for EqProp/MEP |
| `fa_kernels.py` | Fused feedback projection, activation derivative, batched outer product |
| `pc_kernels.py` | Fused prediction, error update, contrastive update (Predictive Coding) |
| `hebbian_kernels.py` | Hebbian/Oja's rule, 3-factor, contrastive Hebbian |
| `snn_kernels.py` | LIF step, STDP, contrastive STDP |
| `ff_kernels.py` | Goodness threshold, contrastive FF/PEPITA updates |
| `tp_kernels.py` | Target propagation inverse + target computation |
| `tile_kernels.py` | Complete TileNet suite: 6 algorithms activity/weight update, routing (top-k/random/learned), multi-GPU NCCL sharding |
| `mep_kernels.py` | Muon orthogonalization, Dion SVD, Fisher whitening, EP settle |
| `backprop_kernels.py` | Fused BPTT baseline |
| `contrastive_kernels.py` | O(1) memory contrastive primitives (10 algorithm families) |
| `backends.py` | Auto-dispatch (TRITON > CUDA > CuPy > CPU > NumPy), `AutoDispatcher`, `KernelProfiler` |
| `compile.py` | `torch.compile` integration: custom `EqPropFunction`/`EqPropTritonFunction` autograd, dynamic shapes, compile presets |
| `kernel_backend.py` | `KernelRegistry` with shape-specific auto-tuning cache |

**Key achievements:**
- ⚡ Triton kernels for all tile algorithms + MEP + FA + PC + Hebbian + SNN + FF + TP
- 🔄 Auto-dispatch with profile-guided backend selection
- 🚀 Custom EqProp autograd Function enabling `torch.compile` on settle loops (2–3× speedup)
- 🌐 Multi-GPU tile sharding for >1B parameter models
- ✅ Gradient equivalence CI gate (Triton vs CuPy vs PyTorch on every commit)

---

## 📜 License

MIT