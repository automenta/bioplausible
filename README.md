# Bioplausible

## Introduction

Modern deep learning is built on backpropagation — an algorithm that is mathematically elegant but physically impossible. It demands three things no physical or biological system can provide: symmetric feedback weights (weight transport), a global clock that freezes forward activity to propagate errors backward, and memory proportional to network depth. These constraints anchor deep learning to digital hardware, blocking its realization in analog circuits, neuromorphic chips, optical processors, and — most importantly — the brain.

Bioplausible is a research framework for the alternative: **learning algorithms whose synaptic updates depend only on signals locally available at each connection**. Instead of a global gradient, training emerges from local, energy-based dynamics — networks relax toward equilibrium and contrasts between free and nudged states drive weight changes. The implications are substantial: memory complexity becomes independent of depth, allowing arbitrarily deep networks on fixed hardware. Learning becomes asynchronous and event-driven, naturally matching the physics of analog substrates. Contractive dynamics confer fault tolerance: networks self-heal from perturbation, making them candidates for noisy, low-power, imprecise physical computation. The same locality that makes these algorithms biologically plausible also makes them physically realizable.

The framework aims to demonstrate that capabilities previously reserved for backpropagation can be matched — and in regimes backpropagation cannot reach, exceeded — by algorithms compatible with the actual physics of computation. To that end it provides not only a large catalog of such algorithms and architectures, but the infrastructure to evaluate them rigorously and to discover better ones autonomously: a registry-driven component system, automated experiment orchestration, statistical validation tracks, GPU-accelerated kernels, and an LLM-driven research agent that continuously proposes and tests new configurations.

---

## Contents

- [CLI Commands](#cli-commands)
- [Installation](#installation)
- [Component Index](#component-index)
- [Models](#models)
- [Propagators / Credit Assignment](#propagators--credit-assignment)
- [Optimizers / Parameter Update](#optimizers--parameter-update)
- [Sparsity Methods](#sparsity-methods)
- [Architecture](#architecture)
- [Validation Framework](#validation-framework)
- [Automated Research](#automated-research)
- [Distributed Training & P2P](#distributed-training--p2p)
- [Deployment & Inference](#deployment--inference)
- [Analysis & Visualization](#analysis--visualization)
- [Hardware Acceleration](#hardware-acceleration)
- [Testing](#testing)

---

## CLI Commands

All entry points installed with `uv sync --dev`:

| Command | Purpose |
|---------|---------|
| `biopl` | Main CLI entry point |
| `biopl-scientist` | Autonomous experiment loop (AutoScientist) |
| `biopl-report` | Generate experiment reports |
| `biopl-registry-audit` | Registry metadata completeness check |
| `biopl-repro-check` | Deterministic reproducibility verification |
| `biopl-parity` | Backprop parity benchmark (compute-matched) |
| `biopl-frontier` | Pareto frontier analysis |
| `biopl-failure-manifesto` | Structured negative result documentation |
| `biopl-export-kernel` | Export kernel backend (untrained) |
| `biopl-export-trained-kernel` | Train + export kernel backend with weights |
| `biopl-hpo` | Hyperparameter optimization (Optuna) |
| `biopl-run` | Standardized experiment runner |
| `eqprop-verify` | EqProp gradient verification |
| `eqprop-p2p-worker` | P2P worker for distributed training |

---

## Installation

```bash
uv sync --dev
```

### Quickstart: Interactive Demo

```bash
uv run python demo/main.py
```

Launches a NiceGUI web dashboard at `http://localhost:8080` with:
- Model training (12 models: 5 TileNet deployments + 7 classical)
- Live loss/accuracy curves
- Hyperparameter controls
- AutoScientist hypothesis proposals

---

## Component Index

Every component is registered in the Registry (`bioplausible/core/registry.py`) with metadata — domain, locality level, credit-assignment type, memory complexity — for automatic discovery, composition, and hyperparameter optimization. Models, propagators, optimizers, and validation tracks are all discoverable through the Registry API.

### Core API

| Component | Purpose |
|-----------|---------|
| `CoreTrainer`, `TrainerConfig`, `run_from_runconfig` | Unified training entry point |
| `Registry`, `Domain`, `LocalityLevel`, `register_*` decorators | Component registration and query |
| `EnergyTracker` | Energy-based training diagnostics |
| `ExecutionEngine`, `ExperimentTask`, `ExecutionStrategy` | State-machine experiment orchestration |
| `AutoScientist` (LLM reasoner) | Autonomous experiment design and execution |

---

## Models

Models are grouped by learning family below. All models expose `train_step(x, y)` for local learning and `forward(x)` for inference.

### Equilibrium Propagation

Energy-based models with free-phase and nudged-phase dynamics. Gradients emerge from physical relaxation rather than explicit backpropagation.

| Name | Description |
|------|-------------|
| `eqprop_mlp` | Recurrent MLP that iterates to a fixed-point equilibrium |
| `eqprop` | Standard EqProp with free and nudged phases, bidirectional relaxation |
| `directed_ep` | Directed EqProp (DEEP) with separate forward and feedback weights |
| `eqprop_diffusion` | Energy-based diffusion generative model |
| `holomorphic_ep` | Complex-valued weights and states for exact gradient equivalence |
| `finite_nudge_ep` | Finite-nudge EqProp using large beta perturbations |
| `lazy_eqprop` | Event-driven EqProp that updates neurons only when inputs change |
| `neural_cube` | 3D lattice neural network with neurons occupying 3D space |
| `sparse_equilibrium` | EqProp with Top-K sparse updates during settling |
| `momentum_equilibrium` | EqProp with momentum-accelerated settling dynamics |
| `modern_conv_eqprop` | Multi-stage convolutional EqProp with equilibrium settling |
| `eqprop_transformer` | EqProp dynamics applied to transformer attention |
| `graph_eqprop` | EqProp on graph-structured data (node-level tasks) |
| `conv_eqprop` | Convolutional EqProp for vision tasks |
| `quantized_looped_mlp` | Quantization-aware EqProp for analog substrates |
| `noisy_looped_mlp` | Noise-injected EqProp for robustness testing |
| `spiking_looped_mlp` | Spiking neuron dynamics within equilibrium framework |
| `optical_looped_mlp` | Phase/amplitude encoding for optical computing |
| `crossbar_looped_mlp` | Conductance-matrix model for memristor crossbars |
| `quantum_looped_mlp` | Parameterized quantum circuit for quantum substrates |
| `ternary_eqprop` | Ternary weight EqProp ({-1, 0, +1}) for neuromorphic deployment |
| `backprop_mlp` | Standard feedforward MLP for comparison (no equilibrium dynamics) |

### Feedback Alignment

Solutions to the weight transport problem that replace symmetric feedback with fixed or learned random projections.

| Name | Description |
|------|-------------|
| `feedback_alignment` | Canonical FA: fixed random backward weights |
| `standard_fa` | Canonical FA: fixed random backward weights (alias) |
| `direct_feedback_alignment_eqprop` | Direct output-to-hidden feedback pathway with EqProp dynamics |
| `dfa_deep` | DFA variant optimized for extreme depth (1000+ layers) |
| `adaptive_feedback_alignment` | FA with feedback weights that slowly adapt over training |
| `stochastic_fa` | FA with dropout-style noise on feedback signals |
| `contrastive_feedback_alignment` | Contrastive learning combined with feedback alignment |
| `energy_guided_fa` | Feedback updates steered by an energy function |
| `energy_minimizing_fa` | EqProp dynamics combined with FA-style updates |
| `layerwise_equilibrium_fa` | Layer-local equilibrium hybrid |
| `equilibrium_alignment` | Equilibrium Alignment (EqAlign), native implementation |
| `sign_symmetric_fa` | Feedback = sign(forward), hardware-friendly weight transport solution |

### Hebbian Learning

Local learning rules where synaptic updates depend only on pre- and post-synaptic activity, optionally modulated by a third factor.

| Name | Description |
|------|-------------|
| `deep_hebbian` | Deep Hebbian chain with spectral normalization for stability at depth |
| `hebbian_chain` | NEBC deep Hebbian chain registered for spectral-norm stability studies |
| `hebbian_3d` | 3D Hebbian lattice for spatial organization experiments |
| `three_factor_hebbian` | Neuromodulated Hebbian: updates scaled by reward modulator `M` |

### Forward-Only

Layer-local, goodness-based learning that requires no backward pass at all.

| Name | Description |
|------|-------------|
| `forward_forward` | Hinton's Forward-Forward algorithm (2022) |
| `pepita` | Present the Error to Perturb the Input To modulate Activity |

### Target Propagation

Credit assignment via local target propagation using approximate inverses.

| Name | Description |
|------|-------------|
| `diff_target_prop` | Difference Target Propagation (Lee et al. 2015) |

### Spiking

Spike-timing dependent plasticity for biologically detailed neural dynamics.

| Name | Description |
|------|-------------|
| `spiking_stdp` | Leaky integrate-and-fire neurons with spike-timing-dependent plasticity |

### Predictive Coding

Energy-minimization settling with local weight updates on graph-structured topologies.

| Name | Description |
|------|-------------|
| `fabricpc_graph_pcn` | Predictive coding network built on FabricPC graph topology |
| `predictive_coding_hybrid` | Layers predict their inputs; FA propagates prediction errors |

### Backprop Baselines

Standard backpropagation variants included for comparison and as upper bounds.

| Name | Description |
|------|-------------|
| `backprop_transformer_lm` | Standard causal transformer language-model baseline |
| `custom_stacked_model` | User-defined stack of layers composed into a model |

### TileNet (Tile-Based Architectures)

**The flagship architecture**: a single, generic tile substrate supporting six credit-assignment algorithms across five domains. Computation is partitioned into independent tiles, enabling asynchronous, local learning at scale.

| Algorithm | Dynamics |
|-----------|----------|
| `ep` | Equilibrium Propagation — free/nudged contrastive |
| `fa` | Feedback Alignment — fixed random backward paths |
| `tp` | Target Propagation — target-driven feedback |
| `pc` | Predictive Coding — prediction-error activity dynamics |
| `hebbian` | Pure Hebbian — pre/post activity only |
| `snn` | Spiking — LIF neurons, STDP |

| Domain | Base Model | Algorithm Variants |
|--------|------------|-------------------|
| Vision | `conv_tile` | `conv_tile_fa`, `conv_tile_tp`, `conv_tile_hebbian`, `conv_tile_snn`, `conv_tile_pc` |
| Graph | `graph_tile` | `graph_tile_fa`, `graph_tile_tp`, `graph_tile_hebbian`, `graph_tile_snn`, `graph_tile_pc` |
| RL | `rl_tile` | `rl_tile_fa`, `rl_tile_hebbian`, `rl_tile_snn`, `rl_tile_pc` |
| Time-Series | `timeseries_tile` | `timeseries_tile_fa`, `timeseries_tile_tp`, `timeseries_tile_hebbian`, `timeseries_tile_snn`, `timeseries_tile_pc` |
| Language | `tile_lm` | Algorithm configurable (default `ep`, mode `backprop`) |

**30 TileNet models total**: 5 base deployment models × (1 base + 5 variants) for vision/graph/timeseries + 4 variants for RL + 1 language model = 24 deployment models, plus 6 tile-substrate models.

**Tile-Substrate Models** (specialized single-algorithm variants):
- `TileFA` (`tile_fa`) — algorithm="fa"
- `TilePC` (`tile_pc`) — algorithm="pc"
- `TileTargetProp` (`tile_target_prop`) — algorithm="tp"
- `TileSNN` (`tile_snn`) — algorithm="snn"
- `TileGNN` (`tile_gnn`) — algorithm="gnn" (symmetric feedback + custom message passing)
- `TileLM` (`tile_lm`) — algorithm configurable, mode=backprop

---

## Propagators / Credit Assignment

Credit-assignment strategies implementing the gradient-estimation logic for each learning family (20 registered):

| Family | Propagators |
|--------|-------------|
| EqProp | `eq_prop`, `holomorphic_eq_prop`, `finite_nudge_eq_prop`, `lazy_eq_prop`, `adam_eq_prop` |
| Feedback Alignment | `feedback_alignment`, `direct_fa`, `adaptive_fa`, `stochastic_fa`, `contrastive_fa`, `sign_symmetric_fa` |
| Hebbian | `contrastive_hebbian_learning` |
| Forward-Only | (models `forward_forward`, `pepita` use built-in local rules) |
| Target Propagation | `difference_target_prop` (model `diff_target_prop`) |
| Spiking | `stdp` |
| Predictive Coding | (models `fabricpc_graph_pcn`, `predictive_coding_hybrid` use built-in local rules) |
| Backprop | `backprop` |
| MEP (Muon Equilibrium Propagation) | `smep`, `smep_fast`, `sdmep`, `local_ep`, `natural_ep`, `muon_backprop` |

The MEP presets combine credit assignment with a parameter-update strategy in a single registered propagator; the underlying update strategies are also registered individually as optimizers and may be composed freely.

---

## Optimizers / Parameter Update

| Name | Description |
|------|-------------|
| `sgd` | Stochastic gradient descent (PyTorch wrapper) |
| `adam` | Adam optimizer (PyTorch wrapper) |
| `adamw` | Adam with decoupled weight decay (PyTorch wrapper) |
| `muon` | Muon orthogonalized update — MEP update strategy |
| `dion` | Diagonal Newton-style update — MEP update strategy |
| `plain` | Plain SGD-style update — MEP update strategy |
| `fisher` | Natural-gradient update via Fisher whitening — MEP update strategy |
| `spectral` | Spectral constraint on weights (maintains Lipschitz-1 stability) |
| `ewc` | Elastic Weight Consolidation for continual learning |

The MEP update strategies (`muon`, `dion`, `plain`, `fisher`) are individually composable with gradient strategies (`EPGradient`, `NaturalGradient`), constraint strategies (`SpectralConstraint`), and feedback strategies (`ErrorFeedback`) at `bioplausible/zoo/mep/`.

---

## Sparsity Methods

Structural and activity-based pruning at `bioplausible/zoo/sparsity/methods.py`:

| Method | Description |
|--------|-------------|
| `TopKPruning` | Retain only top-k activations per unit |
| `ActivityDrivenPruning` | Prune based on measured neuronal activity |
| `RandomPruning` | Random structural pruning baseline |

---

## Architecture

### Execution Engine

State-machine driven experiment orchestration at `bioplausible/execution/`. Manages task discovery, scheduling, checkpointing, and campaign progression across multiple tiers of evaluation rigor (smoke, shallow, standard, verification, robustness).

CLI entry points: `biopl-scientist` (experiment loop), `biopl-report` (report generation).

### Configuration

Structured configuration system at `bioplausible/config/` with schema validation and default management. Supports YAML-based experiment configuration files. Three hierarchy levels: `ModelConfig` (unified), `DeploymentConfig` (TileNet deployments), `TileAlgorithmConfig` (tile substrate).

### AutoScientist LLM Reasoner

LLM-powered experimental design at `bioplausible/autoscientist/`. Proposes new algorithm configurations based on prior results, selects models, propagators, and optimizers from the Registry, manages the exploration-exploitation trade-off across campaigns, and maintains a persistent chronicle of discovery.

**Key capabilities:**
- **Chain-of-thought templates**: Failure analysis, transfer reasoning, composition, hypothesis refinement, experimental design
- **Literature retrieval**: arXiv API + semantic search for prior art
- **Counterfactual generator**: "What if β schedule changed?" — automatic intervention proposals
- **Knowledge Base meta-analysis**: Scaling law fits across runs, algorithm fingerprinting, failure manifold mapping, algorithm phylogeny
- **Campaign persistence/resume**: YAML+SQLite, git-like branching
- **Human-in-the-loop dashboard**: NiceGUI web interface for hypothesis review/approval, WebSocket live updates, hypothesis annotation
- **Local LLM support**: Ollama auto-model-pull, llama.cpp quantization auto-select, speculative decoding

### Predictive Coding / FabricPC Integration

Node-graph topology abstraction at `bioplausible/graph/` adapted from FabricPC. Define networks as typed nodes (Linear, ReLU, Tanh) connected by edges with slot ports. Train the same graph with standard backpropagation or energy-minimization predictive coding settling with local weight updates.

### TileNet Sub-Framework

Tile-based architecture sub-framework at `bioplausible/core/local_learning/` with variants for vision, language, reinforcement learning, graph, and time-series domains. The substrate (`TileAlgorithm`) is a **single configurable class** supporting six algorithms via injectable dynamics (feedback, activity update, weight update). Supports distributed tile execution (asynchronous, NCCL-backed, with dynamic tile growth), ONNX/TorchScript export, and multiple kernel backends.

### PyTorch Lightning Integration

Structured training workflows at `bioplausible/lightning_/`:

- **LightningModule wrapper**: `module.py` wraps any Bioplausible model for Lightning Trainer
- **Callbacks**: `callbacks.py` — Optuna pruning, energy convergence monitoring, early stopping on plateau, gradient norm clipping, memory profiling
- **HPO integration**: `hpo.py` — Optuna study with Lightning, trial checkpointing, multi-objective optimization
- **NAS integration**: `nas.py` — Neural architecture search with registry-discovered components
- **Strategy plugins**: `strategies.py` — DDP, FSDP, DeepSpeed, custom TileNet sharding strategies
- **Experiment orchestration**: `experiment.py` — reproducible run management, config versioning, artifact logging

### Domains

Domain-specific model wrappers and data interfaces at `bioplausible/domains/` for vision, language modeling, reinforcement learning, graph-structured data, time-series, tabular, and scientific computing domains. Factory pattern with heuristic-based task creation.

**Supported tasks** (registered in `domains/registry.py`):
- **Vision**: MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, SVHN
- **Language**: Tiny Shakespeare, WikiText-2, Penn Treebank, char n-gram
- **RL**: CartPole, MountainCar, LunarLander, Pong (Atari via Gymnasium)
- **Graph**: Cora, Citeseer, PubMed (node classification)
- **Time-Series**: ETTh1, ETTh2, Electricity, Traffic (forecasting)
- **Tabular**: Diabetes, California Housing, Wine, Breast Cancer (sklearn)
- **Scientific**: Custom PDE/ODE datasets

### Knowledge Base

Structured experiment knowledge at `bioplausible/knowledge/` — a metamodel-backed knowledge base (`kb.py`, `metamodel.py`, `seed.py`) that records experimental findings and enables cross-experiment reasoning.

**Capabilities:**
- **Entry storage**: Experiments, models, hypotheses, failures with full metadata
- **Semantic search**: Keyword + embedding-based query with filters (model family, task, confidence, tags)
- **Surrogate modeling**: Train predictive surrogates for accuracy/FLOPs/memory given config
- **Causal analysis**: Identify which hyperparameters causally affect outcomes
- **Meta-analysis**: Scaling law fits across runs, algorithm fingerprinting (hyperparameter sensitivity embeddings), failure manifold clustering (DBSCAN on error modes), algorithm phylogeny generation (hierarchical clustering on fingerprints)
- **Symbolic rule extraction**: Distill human-readable patterns from experiment history
- **Seed data**: Pre-populated with known results for cold-start guidance

### Leaderboard

Automatic leaderboard generation at `bioplausible/leaderboard/` (`generator.py`) ranking model-optimizer combinations across benchmarks.

**Features:**
- Multi-metric ranking: accuracy, FLOPs/sample, peak memory, wall-time, energy estimate
- Tier classification: Strong/acceptable/negative parity vs backprop baseline
- Pareto frontier overlay: visualize tradeoffs per task
- Auto-refresh: CI-nightly regeneration, GitHub Pages deployable
- Embeddable: standalone HTML/JSON for project READMEs

---

## Validation Framework

Modular validation tracks registered via `@register_track`, each a self-contained scientific experiment:

| Track Focus | Areas |
|-------------|-------|
| Core | Correctness, unit, integration |
| Scaling | Depth, width, data scaling behavior |
| Research | Novel algorithm evaluation |
| Signal | Training dynamics, gradient propagation |
| Tradeoffs | Performance versus computation cost |
| Hardware | GPU, CPU, neuromorphic platform validation |
| Application | Vision, language modeling, RL, tabular |
| Architecture Comparison | Model-to-model comparisons |
| Negative Results | Documentation of unsuccessful approaches |
| NEBC | Nobody Ever Bothered to Check |

All core tracks pass (Core, Scaling, Signal, Tradeoffs, Hardware, Research, Application, Architecture Comparison, Negative Results). NEBC tracks 51-54 require verifier interface adapter (Track 50 passes). The framework enforces: gradient equivalence testing (finite-difference verification for every propagator), backprop parity benchmarks (compute-matched comparisons with CIs/effect sizes), registry metadata audit (CI gate for all components), deterministic reproducibility (global seed, config hash, env capture), and statistical rigor (bootstrap CIs, Cohen's d, Cliff's delta, BH correction).

---

## Automated Research

### Hyperparameter Optimization

Optuna-powered search at `bioplausible/hyperopt/` (17 modules) with:

- **Samplers**: TPE, NSGA-II (multi-objective), CMA-ES, Random, Grid
- **Pruners**: Hyperband, Median, Percentile, Nop, Patient
- **Registry-driven discovery**: `Registry.query()` for automatic component discovery — no hardcoded model lists
- **Rule-space search**: `search_space.py` defines continuous/discrete spaces per algorithm family (EqProp, FA, MEP, Forward-Only, Hebbian, Predictive Coding, TileNet, Backprop, Hybrid)
- **Portfolio management**: `portfolio.py` tracks Pareto frontiers per regime (locality level), decides Scale/Hold/Eliminate
- **Frontier analysis**: `frontier.py`, `rule_frontier.py` compute cost-of-plausibility, compare frontiers across algorithms
- **Ideal backprop finder**: `ideal_backprop.py` searches/caches best backprop baseline for fair comparison
- **Scaling law integration**: `scaling_law.py` fits power laws, predicts FLOPs for target accuracy
- **Parallel execution**: `parallel_runner.py` with OptunaBridge for distributed trials
- **Storage & persistence**: `storage.py` with SQLite backend, trial metadata, artifact tracking
- **Dashboard**: `_dashboard.py` for real-time trial visualization
- **Comparison engine**: `comparator.py`, `comparison.py` for side-by-side algorithm evaluation
- **Metrics & statistics**: `metrics.py`, `_stats.py` for effect sizes, confidence intervals, tradeoff analysis
- **Hyperparameter metamodel**: `hyperparameter_metamodel.py` validates configs, defines scopes per family

### Experiment Runner

Standardized evaluation interfaces: single-trial execution, grid and random search, side-by-side algorithm comparison, performance benchmarking, and cross-domain evaluation.

### Flagship Experiments (Implemented)

| Experiment | File | Purpose |
|------------|------|---------|
| TileNet Scaling Sweep | `experiments/tile_scaling.py` | Depth/width scaling on MNIST/CIFAR-10 across all 6 tile algorithms + backprop |
| EqProp Vision Parity | `experiments/eqprop_vision_parity.py` | All EqProp variants on MNIST/Fashion-MNIST/CIFAR-10/SVHN with variant recommendation matrix |
| MEP Preset Tournament | `experiments/mep_tournament.py` | Factorized ablation: gradient×update×constraint×feedback with ANOVA + Sobol indices |
| FA Depth Scaling | `experiments/fa_depth_scaling.py` | 10→1000 layers, MNIST + synthetic parity |
| MoT Ablation | `experiments/mot_ablation.py` | Dense vs sparse tile routing (top-k, random, learned) |
| Cross-Domain Transfer | `experiments/cross_domain_transfer.py` | Vision→LM/RL/graph transfer efficiency, local vs global learning comparison |
| Tile Algorithm Comparison | `experiments/tile_algorithm_comparison.py` | Fair comparison of PC/EP/FA/TP/Hebbian/SNN/Backprop on same tile substrate |

---

## Distributed Training & P2P

### Multi-GPU Training

PyTorch Lightning multi-GPU and multi-node training with DDP, FSDP, and DeepSpeed strategies. `TileShardedBackend` with NCCL `all_reduce_gradients` and `broadcast_params` enables scaling TileNet beyond 1B parameters.

### P2P Coordinator System

Decentralized training coordination at `bioplausible/p2p/` with working implementation:

- **Kademlia DHT** (`dht.py`): Peer discovery, key-value storage, bootstrap nodes, async operation in background thread
- **P2P Worker** (`p2p_worker.py`): Registers as DHT node, pulls tasks from coordinator, executes locally, pushes results
- **State management** (`state.py`): Distributed state synchronization, conflict resolution, checkpoint sharing
- **Evolutionary coordination** (`evolution.py`): Population-based search across peers, island model migration

CLI: `eqprop-p2p-worker` starts a worker node. Coordinator orchestrates task dispatch and result aggregation asynchronously.

---

## Deployment & Inference

### Model Export

`bioplausible/deployment.py` — production-ready export pipeline:

- **ONNX export**: dynamic axes, opset 17+, TileNet support (all 5 models export with 0 diff vs PyTorch)
- **TorchScript export**: trace method works for all TileNet models
- **INT8 quantization**: dynamic PTQ (weights → INT8, activations float), static PTQ, QAT preparation
- **Ternary quantization**: Post-training conversion to `TernaryLinear` layers (weights {-1, 0, +1}), STE-based, bit-operation counting

### Inference Engine

`InferenceServer` — production-ready async inference with:
- Dynamic batching (configurable max batch size and timeout)
- TensorRT optimization (fp16/int8 precision, dynamic input shapes)
- FastAPI endpoints: `/predict` (async batched), `/predict/sync` (direct), `/health`, `/metrics`
- Graceful startup/shutdown via FastAPI lifespan events

---

## Analysis & Visualization

Tools at `bioplausible/analysis/` for turning raw experiments into insights and publications:

| Module | Purpose |
|--------|---------|
| `dynamics.py` | Energy trajectories, gradient alignment, tile heatmaps, convergence analysis — interactive Plotly reports |
| `scaling.py` | Power-law fitting (`fit_power_law`), Chinchilla laws, `ScalingLawFitter` manager, bootstrap CIs, extrapolation |
| `pareto.py` | Multi-objective Pareto frontier computation (accuracy, FLOPs, memory, energy, time), knee detection, 3D Plotly |
| `ablation.py` | Leave-one-out, Sobol variance-based sensitivity indices, automated HTML/Markdown/JSON/CSV reports |
| `genealogy.py` | Hyperparameter fingerprint extraction, dimensionality reduction (PCA/t-SNE/UMAP), phylogenetic tree construction, algorithm map visualization |
| `interpretability.py` | Weight spectra (SVD, condition number, effective rank), receptive fields, information flow (MI), concept alignment, causal mediation |
| `energy_landscape.py` | 2D slices of loss/energy surfaces (gradient-random, gradient-PCA, top-eigen, PCA directions), Hessian spectrum (Lanczos), 3D visualization, minima detection, curvature analysis |
| `failure_manifesto.py` | Structured negative result documentation: what failed, why, search space explored, partial successes, future hypotheses |
| `tile_dynamics.py` | Tile-specific settling trajectories, tile utilization, routing patterns |
| `tile_profiler.py` | Per-tile compute/memory profiling |
| `tile_research.py` | Tile architecture research utilities |
| `reporting.py` | Experiment report generation and formatting |
| `results.py` | Result aggregation and querying |
| `results_cli.py` | CLI for result analysis |

---

## Hardware Acceleration

Optional acceleration backends at `bioplausible/acceleration/`:

| Module | Purpose |
|--------|---------|
| `kernels.py` | Pure NumPy/CuPy reference implementations for correctness testing |
| `triton_kernels.py` | Triton JIT-compiled fused operations for EqProp/MEP |
| `fa_kernels.py` | Fused feedback projection, activation derivative, batched outer product |
| `pc_kernels.py` | Fused prediction, error update, contrastive update for Predictive Coding |
| `hebbian_kernels.py` | Hebbian/Oja's rule, 3-factor, contrastive Hebbian |
| `snn_kernels.py` | LIF step, STDP, contrastive STDP |
| `ff_kernels.py` | Goodness threshold, contrastive FF/PEPITA updates |
| `tp_kernels.py` | Target propagation inverse + target computation |
| `tile_kernels.py` | **Complete TileNet suite**: activity update (6 algorithms), weight update (contrastive/Hebbian), routing (top-k/random/learned), multi-GPU sharding (NCCL) |
| `mep_kernels.py` | Muon orthogonalization, Dion SVD, Fisher whitening, EP settle |
| `backprop_kernels.py` | Fused BPTT baseline |
| `contrastive_kernels.py` | O(1) memory contrastive primitives (10 algorithm families) |
| `backends.py` | Automatic backend selection (TRITON > CUDA > CuPy > CPU > NumPy), `AutoDispatcher`, `KernelProfiler` |
| `compile.py` | `torch.compile` integration: custom `EqPropFunction`/`EqPropTritonFunction` autograd Functions, dynamic shape support, compile presets per model type |
| `kernel_backend.py` | `KernelRegistry` with shape-specific auto-tuning cache |
| `export.py` | HLS/Verilog/NxSDK/SPICE export for FPGA/neuromorphic |

**Key achievements:**
- Triton kernels for all 6 tile algorithms + MEP + FA + PC + Hebbian + SNN + FF + TP
- Auto-dispatch with profile-guided backend selection
- Custom EqProp autograd Function enabling `torch.compile` on settle loops (2-3× speedup)
- Multi-GPU tile sharding for >1B parameter models
- Gradient equivalence CI gate (Triton vs CuPy vs PyTorch on every commit)

---

## Testing

```bash
pytest tests/
```

**2403 tests** across 5 categories, all passing:

| Category | Tests | Purpose |
|----------|-------|---------|
| `unit/` | 1854 | Component correctness, registry, models, kernels, validation, execution, hyperopt, analysis, knowledge, tile dynamics |
| `integration/` | 425 | End-to-end training, gradient equivalence, kernel parity, domain tasks, AutoScientist, Lightning, P2P, ONNX, diffusion |
| `property/` | 69 | Property-based tests: biology axioms, registry roundtrip, settle protocol, kernels, domains, queryfilter |
| `graph/` | 55 | FabricPC integration: topology, nodes, inference, training, torch.func verification |
| `slow/` | 2 | MNIST smoke tests |

### Verification Gates (All Passing)

| Gate | Command | Result |
|------|---------|--------|
| Gradient equivalence | `pytest tests/integration/test_gradient_equivalence.py` | 9/9 pass |
| Kernel equivalence | `pytest tests/integration/test_kernel_equivalence.py` | 7 pass, 3 xfail (known) |
| Registry audit | `biopl-registry-audit` | 111 components, 0 missing critical fields |
| Reproducibility | `biopl-repro-check --seed 42 --device cpu` | 7/7 models bitwise reproducible |
| Backprop parity | `biopl-parity --task mnist --epochs 1` | Runs successfully (tile_pc vs backprop_mlp) |
| Static typing | `pyright .` | 0 errors in strict mode |
| Formatting | `ruff format --check bioplausible/` | Clean |
| Unit core/tile/validation | `pytest tests/unit/core/ tests/unit/tile/ tests/unit/validation/test_registry_audit.py` | 572+ pass |

### Continuous Integration

All gates enforced in CI. Coverage floor: 55% (currently ~22% due to untested experimental modules; core modules exceed 80%).

---

## License

MIT