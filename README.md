# Bioplausible

## Introduction

Modern deep learning is built on backpropagation, an algorithm that is mathematically elegant but physically impossible. It demands three things no physical or biological system can provide: symmetric feedback weights (weight transport), a global clock that freezes forward activity to propagate errors backward, and memory proportional to network depth. These constraints anchor deep learning to digital hardware, blocking its realization in analog circuits, neuromorphic chips, optical processors, and — most importantly — the brain.

Bioplausible is a research framework for the alternative: learning algorithms whose synaptic updates depend only on signals locally available at each connection. Instead of a global gradient, training emerges from local, energy-based dynamics — networks relax toward equilibrium and contrasts between free and nudged states drive weight changes. The implications are substantial. Memory complexity becomes independent of depth, allowing arbitrarily deep networks on fixed hardware. Learning becomes asynchronous and event-driven, naturally matching the physics of analog substrates. Contractive dynamics confer fault tolerance: networks self-heal from perturbation, making them candidates for noisy, low-power, imprecise physical computation. The same locality that makes these algorithms biologically plausible also makes them physically realizable.

The framework aims to demonstrate that the capabilities previously reserved for backpropagation can be matched — and in regimes backpropagation cannot reach, exceeded — by algorithms compatible with the actual physics of computation. To that end it provides not only a large catalog of such algorithms and architectures, but the infrastructure to evaluate them rigorously and to discover better ones autonomously: a registry-based component system, automated experiment orchestration, statistical validation tracks, and an LLM-driven research agent that continuously proposes and tests new configurations.

## Contents

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

## Installation

```
uv sync --dev
```

## Component Index

Every component is registered in the Registry (`bioplausible/core/registry.py`) with metadata — domain, locality level, credit-assignment type, memory complexity — for automatic discovery, composition, and hyperparameter optimization. Models, propagators, optimizers, and validation tracks are all discoverable through the Registry API.

### Core API

| Component | Purpose |
|-----------|---------|
| `CoreTrainer`, `TrainerConfig`, `run_from_config` | Unified training entry point |
| `Registry`, `Domain`, `LocalityLevel`, `register_*` decorators | Component registration and query |
| `EnergyTracker` | Energy-based training diagnostics |
| `ExecutionEngine`, `ExperimentTask`, `ExecutionStrategy` | State-machine experiment orchestration |
| `AutoScientist` (LLM reasoner) | Autonomous experiment design and execution |

## Models

Models are grouped by learning family below.

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
| `neural_cube` | 3D lattice neural network with neurons occupying 3D space |
| `sparse_equilibrium` | EqProp with Top-K sparse updates during settling |
| `momentum_equilibrium` | EqProp with momentum-accelerated settling dynamics |
| `modern_conv_eqprop` | Multi-stage convolutional EqProp with equilibrium settling |
| `eqprop_transformer` | EqProp dynamics applied to transformer attention |
| `graph_eqprop` | EqProp on graph-structured data (node-level tasks) |
| `backprop_mlp` | Standard feedforward MLP for comparison (no equilibrium dynamics) |

### Feedback Alignment

Solutions to the weight transport problem that replace symmetric feedback with fixed or learned random projections.

| Name | Description |
|------|-------------|
| `feedback_alignment` | EqProp combined with feedback alignment signals |
| `adaptive_feedback_alignment` | FA with feedback weights that slowly adapt over training |
| `stochastic_fa` | FA with dropout-style noise on feedback signals |
| `contrastive_feedback_alignment` | Contrastive learning combined with feedback alignment |
| `direct_feedback_alignment_eqprop` | Direct output-to-hidden feedback pathway with EqProp dynamics |
| `dfa_deep` | DFA variant optimized for extreme depth (1000+ layers) |
| `standard_fa` | Canonical FA: fixed random backward weights |
| `energy_guided_fa` | Feedback updates steered by an energy function |
| `energy_minimizing_fa` | EqProp dynamics combined with FA-style updates |
| `layerwise_equilibrium_fa` | Layer-local equilibrium hybrid |
| `equilibrium_alignment` | Equilibrium Alignment (EqAlign), native implementation |

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

### EquiTile (Tile-Based Architectures)

Partitioned architectures where computation is distributed across independent tiles, enabling asynchronous and local learning at scale.

| Name | Description |
|------|-------------|
| `equitile` | Core tile architecture supporting predictive-coding and EqProp modes |
| `equitile_ep` | EquiTile with strict equilibrium propagation learning |
| `dynamic_equitile` | Tiles grow, prune, merge, and split during training based on error signals and utilization |
| `enhanced_equitile` | Enhanced EquiTile with optional, configurable improvements for ablation |
| `graph_equitile` | Tile message passing over graph-structured data |
| `lm_equitile` | Tile architecture specialized for language modeling |
| `optimized_lm_equitile` | Optimized LM tile variant (transformer-style, mixture-of-tiles sparsity) |
| `rl_equitile` | Tile actor-critic for reinforcement learning |
| `timeseries_equitile` | Temporal forecasting with tile-based attention |
| `conv_equitile` | Vision processing with convolutional tiles |

## Propagators / Credit Assignment

Credit-assignment strategies that implement the gradient-estimation logic for each learning family:

| Family | Propagators |
|--------|-------------|
| EqProp | `eq_prop`, `holomorphic_eq_prop`, `finite_nudge_eq_prop`, `lazy_eq_prop` |
| Feedback Alignment | `feedback_alignment`, `direct_fa`, `adaptive_fa`, `stochastic_fa`, `contrastive_fa` |
| Hebbian | `contrastive_hebbian_learning` |
| Forward-Only | `ff`, `pepita` |
| Target Propagation | `target_prop`, `difference_target_prop` |
| Spiking | `stdp` |
| Predictive Coding | `pcn` |
| Backprop | `backprop` |
| MEP (Muon Equilibrium Propagation) | `smep`, `smep_fast`, `sdmep`, `local_ep`, `natural_ep`, `muon_backprop` |

The MEP presets combine credit assignment with a parameter-update strategy in a single registered propagator; the underlying update strategies are also registered individually as optimizers (see below) and may be composed freely.

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

## Sparsity Methods

Structural and activity-based pruning at `bioplausible/zoo/sparsity/methods.py`:

| Method | Description |
|--------|-------------|
| `TopKPruning` | Retain only top-k activations per unit |
| `ActivityDrivenPruning` | Prune based on measured neuronal activity |
| `RandomPruning` | Random structural pruning baseline |

## Architecture

### Execution Engine

State-machine driven experiment orchestration at `bioplausible/execution/`. Manages task discovery, scheduling, checkpointing, and campaign progression across multiple tiers of evaluation rigor (smoke, shallow, standard, verification, robustness).

CLI entry points: `biopl-scientist` (experiment loop), `biopl-report` (report generation).

### Configuration

Structured configuration system at `bioplausible/config/` with schema validation and default management. Supports YAML-based experiment configuration files.

### AutoScientist LLM Reasoner

LLM-powered experimental design at `bioplausible/autoscientist/`. Proposes new algorithm configurations based on prior results, selects models, propagators, and optimizers from the Registry, manages the exploration-exploitation trade-off across campaigns, and maintains a persistent chronicle of discovery.

### Predictive Coding / FabricPC Integration

Node-graph topology abstraction at `bioplausible/graph/` adapted from FabricPC. Define networks as typed nodes (Linear, ReLU, Tanh) connected by edges with slot ports. Train the same graph with standard backpropagation or energy-minimization predictive coding settling with local weight updates.

### EquiTile Sub-Framework

Tile-based architecture sub-framework at `bioplausible/equitile/` with variants for vision, language, reinforcement learning, graph, and time-series domains. Supports distributed tile execution (asynchronous, NCCL-backed, with dynamic tile growth), ONNX/TorchScript export, and multiple kernel backends.

### PyTorch Lightning Integration

Structured training workflows at `bioplausible/lightning_/`: Lightning module wrapping Bioplausible models, Optuna pruning callbacks, Ray Tune integration, mixed precision support, energy convergence monitoring, and neural architecture search integration.

### Domains

Domain-specific model wrappers and data interfaces at `bioplausible/domains/` for vision, language modeling, reinforcement learning, graph-structured data, time-series, tabular, and scientific computing domains.

### Knowledge Base

Structured experiment knowledge at `bioplausible/knowledge/` — a metamodel-backed knowledge base that records experimental findings and enables cross-experiment reasoning.

### Leaderboard

Automatic leaderboard generation at `bioplausible/leaderboard/` ranking model-optimizer combinations across benchmarks.

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

## Automated Research

### Hyperparameter Optimization

Optuna-powered search at `bioplausible/hyperopt/` with TPE sampler, NSGA-II multi-objective sampler, Hyperband pruner, and Median pruner. Uses `Registry.query()` for automatic component discovery — no hardcoded model lists.

### Experiment Runner

Standardized evaluation interfaces: single-trial execution, grid and random search, side-by-side algorithm comparison, performance benchmarking, and cross-domain evaluation.

## Distributed Training & P2P

### Multi-GPU Training

PyTorch Lightning multi-GPU and multi-node training with DDP, FSDP, and DeepSpeed strategies.

### P2P Coordinator System

Decentralized training coordination at `bioplausible/p2p/` using Kademlia DHT for peer discovery. A coordinator dispatches tasks to distributed workers with asynchronous result aggregation.

## Deployment & Inference

### Model Export

ONNX and TorchScript serialization for cross-platform production deployment. Quantization support for INT8 and ternary weights.

### Inference Engine

High-throughput prediction server with FastAPI REST endpoints and optimized batch processing.

## Analysis & Visualization

Tools at `bioplausible/analysis/` and `bioplausible/visualization_tools.py`:

| Tool | Purpose |
|------|---------|
| `DynamicsAnalyzer` | Training dynamics and convergence analysis |
| `compute_statistics`, `get_rankings` | Statistical analysis with effect sizes, confidence intervals, rankings |
| `compute_pareto_frontier` | Multi-objective efficiency frontier computation |
| `AblationStudy` | Component contribution and hyperparameter sensitivity studies |
| `FailureManifestoGenerator` | Structured negative result documentation |
| `TrainingVisualizer` | Loss curves, convergence plots, speed-accuracy tradeoffs |
| `fit_power_law`, `plot_scaling_curves` | Scaling behavior characterization |

## Hardware Acceleration

Optional acceleration backends at `bioplausible/acceleration/`:

| Module | Purpose |
|--------|---------|
| `kernels.py` | Pure NumPy/CuPy EqProp kernel |
| `triton_kernels.py` | Triton JIT-compiled EqProp operations |
| `backends.py` | Automatic backend selection and dispatch |
| `compile.py` | torch.compile integration with custom EqProp backward |

## Testing

```
pytest tests/ bioplausible/tests/
```

## License

MIT
