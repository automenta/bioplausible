# Bioplausible

Neural network learning algorithms that do not depend on global backpropagation.

Backpropagation requires symmetric weight transport, a global clock for separate forward/backward passes, and activation storage proportional to depth — constraints incompatible with continuous-time analog hardware, ultra-deep architectures, and biological neural computation. Bioplausible implements alternatives that replace global gradient computation with local, energy-based dynamics, using only locally available signals for synaptic updates.

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

Every component is registered in the Registry (`bioplausible/core/registry.py`) with metadata for automatic discovery, composition, and hyperparameter optimization. Models, propagators, optimizers, and validation tracks are all discoverable through the Registry API.

### Core API

| Component | Purpose |
|-----------|---------|
| `CoreTrainer`, `TrainerConfig`, `run_from_config` | Unified training entry point |
| `Registry`, `Domain`, `LocalityLevel`, `register_*` decorators | Component registration and query |
| `EnergyTracker` | Energy-based training diagnostics |
| `ExecutionEngine`, `ExperimentTask`, `ExecutionStrategy` | State-machine experiment orchestration |
| `AutoScientist` (LLM reasoner) | Autonomous experiment design and execution |

## Models

The model zoo spans 46 registered models across learning families.

### Equilibrium Propagation

Energy-based models with free-phase and nudged-phase dynamics. The gradients emerge from physical relaxation rather than explicit backpropagation.

- `eqprop_mlp`, `eqprop`, `directed_ep`, `eqprop_diffusion`, `holomorphic_ep`, `finite_nudge_ep`, `lazy_eqprop`, `neural_cube`, `sparse_equilibrium`, `momentum_equilibrium`, `modern_conv_eqprop`, `eqprop_transformer`, `graph_eqprop`, `backprop_mlp`

### Feedback Alignment

Solve the weight transport problem by replacing symmetric feedback with fixed or learned random projections.

- `feedback_alignment`, `adaptive_feedback_alignment`, `stochastic_fa`, `contrastive_feedback_alignment`, `direct_feedback_alignment_eqprop`, `dfa_deep`, `standard_fa`, `energy_guided_fa`, `energy_minimizing_fa`, `layerwise_equilibrium_fa`, `equilibrium_alignment`

### Hebbian Learning

Classical and modern local learning rules where synaptic updates depend only on pre- and post-synaptic activity, optionally modulated by a third factor.

- `deep_hebbian`, `hebbian_chain`, `hebbian_3d`, `three_factor_hebbian`

### Forward-Only

Layer-local goodness-based learning that requires no backward pass at all.

- `forward_forward`, `pepita`

### Target Propagation

Credit assignment through local target propagation using approximate inverses.

- `diff_target_prop`

### Spiking

Spike-timing dependent plasticity for biologically detailed neural dynamics.

- `spiking_stdp`

### Predictive Coding

Energy-minimization settling with local weight updates on graph-structured topologies.

- `fabricpc_graph_pcn`, `predictive_coding_hybrid`

### Backprop Baselines

Standard backpropagation variants for comparison.

- `backprop_transformer_lm`, `custom_stacked_model`

### EquiTile (Tile-Based Architectures)

Partitioned architectures where computation is distributed across independent tiles, enabling asynchronous and local learning.

- `equitile`, `equitile_ep`, `dynamic_equitile`, `enhanced_equitile`, `graph_equitile`, `lm_equitile`, `optimized_lm_equitile`, `rl_equitile`, `timeseries_equitile`, `conv_equitile`

## Propagators / Credit Assignment

23 credit assignment strategies implementing the gradient estimation logic for each learning family:

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

## Optimizers / Parameter Update

| Optimizer | Source |
|-----------|--------|
| `sgd`, `adam`, `adamw` | Standard PyTorch optimizers |
| `muon`, `dion`, `plain`, `fisher` | MEP optimizer strategies |
| `spectral` | Spectral constraint optimizer |
| `ewc` | Elastic Weight Consolidation |

MEP presets at `bioplausible/zoo/mep/` compose gradient computation, update rule, constraint, and feedback strategies: `smep` (spectral + Muon + equilibrium), `smep_fast`, `sdmep` (low-rank SVD), `local_ep` (layer-local), `natural_ep` (Fisher whitening), and `muon_backprop`. Strategies are individually composable — gradient (`EPGradient`, `NaturalGradient`), update (`MuonUpdate`, `DionUpdate`, `PlainUpdate`, `FisherUpdate`), constraint (`SpectralConstraint`), and feedback (`ErrorFeedback`).

## Sparsity Methods

`TopKPruning`, `ActivityDrivenPruning`, `RandomPruning` — structural and activity-based pruning strategies in `bioplausible/zoo/sparsity/`.

## Architecture

### Execution Engine

State-machine driven experiment orchestration at `bioplausible/execution/`. Manages task discovery, scheduling, checkpointing, and campaign progression across multiple tiers of evaluation rigor (smoke, shallow, standard, verification, robustness).

CLI entry points: `biopl-scientist` (experiment loop), `biopl-report` (report generation).

### AutoScientist LLM Reasoner

LLM-powered experimental design at `bioplausible/autoscientist/`. Proposes new algorithm configurations based on prior results, selects models, propagators, and optimizers from the Registry, manages the exploration-exploitation trade-off across campaigns, and maintains a persistent chronicle of discovery.

### Predictive Coding / FabricPC Integration

Node-graph topology abstraction at `bioplausible/graph/` adapted from FabricPC. Define networks as typed nodes (Linear, ReLU, Tanh) connected by edges with slot ports. Train the same graph with standard backpropagation or energy-minimization predictive coding settling with local weight updates.

### EquiTile

Tile-based architecture sub-framework at `bioplausible/equitile/` with variants for vision, language, reinforcement learning, graph, and time-series domains. Supports distributed tile execution, dynamic tile growth, ONNX/TorchScript export, and multiple kernel backends.

### Configuration

Structured configuration system at `bioplausible/config/` with schema validation and default management. Supports YAML-based experiment configuration files.

### Domains

Domain-specific model wrappers and data interfaces at `bioplausible/domains/` for vision, language modeling, reinforcement learning, graph-structured data, time-series, tabular, and scientific computing domains.

### Knowledge Base

Structured experiment knowledge at `bioplausible/knowledge/` — a metamodel-backed knowledge base that records experimental findings and enables cross-experiment reasoning.

### Leaderboard

Automatic leaderboard generation at `bioplausible/leaderboard/` ranking model-optimizer combinations across benchmarks.

### PyTorch Lightning Integration

Structured training workflows at `bioplausible/lightning_/`: Lightning module wrapping Bioplausible models, Optuna pruning callbacks, Ray Tune integration, mixed precision support, energy convergence monitoring, and neural architecture search integration.

## Validation Framework

11 modular validation tracks registered via `@register_track`, each a self-contained scientific experiment:

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

### EquiTile Parallelism

Asynchronous tile execution across devices with NCCL backends and dynamic tile growth for runtime architecture adaptation.

## Deployment & Inference

### Model Export

ONNX and TorchScript serialization for cross-platform production deployment. Quantization support for INT8 and ternary weights.

### Inference Engine

High-throughput prediction server with FastAPI REST endpoints and optimized batch processing.

## Analysis & Visualization

Tools at `bioplausible/analysis/` and `bioplausible/visualization_tools.py`.

- **DynamicsAnalyzer**: Training dynamics and convergence analysis
- **compute_statistics** / **get_rankings**: Statistical analysis with effect sizes, confidence intervals, rankings
- **AblationStudy**: Component contribution and hyperparameter sensitivity studies
- **FailureManifestoGenerator**: Structured negative result documentation
- **TrainingVisualizer**: Loss curves, convergence plots, speed-accuracy tradeoffs
- **fit_power_law** / **plot_scaling_curves**: Scaling behavior characterization
- **compute_pareto_frontier**: Multi-objective efficiency frontier computation

## Hardware Acceleration

- `kernels.py`: Pure NumPy/CuPy EqProp kernel
- `triton_kernels.py`: Triton JIT-compiled EqProp operations
- `backends.py`: Automatic backend selection and dispatch
- `compile.py`: torch.compile integration with custom EqProp backward

## Testing

```
pytest tests/ bioplausible/tests/
```

## License

MIT
