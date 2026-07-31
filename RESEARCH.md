# Bioplausible Research Roadmap

**Goal**: Transform Bioplausible from a sophisticated framework into a *discovery engine* that produces breakthrough results demonstrating bio-plausible learning can match or exceed backpropagation in regimes where backprop fails.

**Strategy**: GPU-first, low-hanging-fruit prioritized, user-recruiting milestones. Defer non-GPU (neuromorphic hardware, analog simulation) until GPU validation is solid.

---

## Phase 0: Foundation Hardening (GPU-Complete)

*Prerequisites for all downstream work. Must be solid before claiming results.*

### 0.1 Backprop Parity Benchmark Suite — **P0 CRITICAL**
**File**: `bioplausible/validation/backprop_parity.py`

Automated, compute-matched comparison infrastructure:
- Identical architectures (MLP, CNN, Transformer) across backprop vs every bio-plausible family
- Compute-matched: same FLOPs, wall-time, peak memory, parameter count
- Statistical rigor: n≥10 seeds, confidence intervals, effect sizes (Cohen's d, Cliff's delta)
- Output: Publication-ready tables + Pareto frontier plots (accuracy vs compute vs memory) + JSON for programmatic analysis
- CLI: `biopl-parity --model eqprop --tasks mnist,cifar10 --seeds 10`

**Target models for parity**: `eqprop_mlp`, `directed_ep`, `feedback_alignment`, `standard_fa`, `forward_forward`, `pepita`, `equitile`, `equitile_ep`

**Tests**: MNIST, CIFAR-10, Tiny Shakespeare, Penn Treebank
**Metrics**: Accuracy, FLOPs/sample, peak memory, wall-time/epoch, energy estimate (Joules)

### 0.2 Registry Metadata Completeness Audit
**File**: `bioplausible/validation/registry_audit.py`

Verify every registered component for complete, calibrated metadata:
- `bio_plausibility_score` (0.0-1.0, calibrated against ground truth)
- `locality_level` (GLOBAL / LAYERWISE / LOCAL / EQUILIBRIUM / FORWARD_ONLY)
- `memory_complexity` (O(1), O(N), O(N²))
- `requires_backward` (bool)
- `credit_assignment_type` (gradient, equilibrium, hebbian, target, forward-only, spiking)
- `family` tag (eqprop, fa, hebbian, forward_only, target_prop, spiking, predictive_coding, mep, equitile, backprop)
- `provides` / `requires` capabilities (for compatibility checking)
- Instantiates without error on CUDA, runs forward+backward (or local equivalent) on dummy data, produces deterministic output with fixed seed
- Generates registry health report (CI gate)

### 0.3 Deterministic Seeding & Reproducibility
**Files**: `bioplausible/utils/reproducibility.py`, `configs/repro/`

- Global seed manager: all RNGs (torch, numpy, random, CUDA, cuDNN)
- Experiment config hashing for exact reproducibility
- Environment capture: git commit, torch/cuda versions, dependencies hash
- Artifact versioning: model checkpoints, configs, logs with content-addressable storage
- CI smoke test: `biopl-repro-check` runs 1-epoch parity on all models nightly

---

## Phase 1: GPU-Ready Algorithm Portfolio (GPU-Complete, High Visibility)

*Demonstrate range and maturity across all learning families. Low-hanging fruit proving core thesis.*

### 1.1 Equilibrium Propagation Family — **Core Strength**

| Model | Status | Priority | Notes |
|-------|--------|----------|-------|
| `eqprop_mlp` | ✅ | — | Baseline |
| `eqprop` (bidirectional) | ✅ | — | Standard |
| `directed_ep` (DEEP) | ✅ | — | Separate forward/feedback |
| `finite_nudge_ep` | ✅ | — | Large β perturbations |
| `lazy_eqprop` | 🔄 | P1 | Event-driven — **huge for neuromorphic** |
| `holomorphic_ep` | ✅ | — | Complex-valued, exact gradient equiv. |
| `eqprop_diffusion` | ✅ | — | Generative |
| `eqprop_transformer` | ✅ | — | Attention + equilibrium |
| `modern_conv_eqprop` | ✅ | — | Multi-stage conv |
| `graph_eqprop` | ✅ | — | Node-level tasks |
| `momentum_equilibrium` | 🔄 | P1 | Momentum-accelerated settling |
| `sparse_equilibrium` | 🔄 | P1 | Top-K sparse updates |

**P1 Enhancements**:
- **Analytical settling time predictor** → auto-tune `inference_steps`
- **Adaptive β scheduling** (cosine, exponential, learned)
- **Memory-efficient checkpointing** for deep unrolling

### 1.2 Feedback Alignment Family — **Weight Transport Solution**

| Model | Status | Priority |
|-------|--------|----------|
| `standard_fa` | ✅ | — |
| `direct_feedback_alignment_eqprop` | ✅ | — |
| `dfa_deep` | ✅ | — | 1000+ layers |
| `adaptive_feedback_alignment` | ✅ | — |
| `stochastic_fa` | ✅ | — |
| `contrastive_feedback_alignment` | ✅ | — |
| `energy_guided_fa` | ✅ | — |
| `energy_minimizing_fa` | ✅ | — |
| `layerwise_equilibrium_fa` | ✅ | — |
| `equilibrium_alignment` (EqAlign) | ✅ | — |

**Gap**: **Sign-symmetric FA** (feedback = sign(forward)) — more hardware-friendly.

### 1.3 MEP (Muon Equilibrium Propagation) — **Optimizer Innovation**

| Preset | Description | Status |
|--------|-------------|--------|
| `smep` | Spectral MEP | ✅ |
| `smep_fast` | Approximate spectral | ✅ |
| `sdmep` | Diagonal MEP | ✅ |
| `local_ep` | Local equilibrium | ✅ |
| `natural_ep` | Natural gradient EP | ✅ |
| `muon_backprop` | Muon + backprop | ✅ |

**P1**: **Composable MEP builder** — mix/match gradient strategy, update strategy, constraint strategy, feedback strategy.

### 1.4 Forward-Only & Hebbian — **Zero Backward Pass**

| Model | Status | Priority |
|-------|--------|----------|
| `forward_forward` | Hinton 2022 | ✅ |
| `pepita` | Perturb input to modulate activity | ✅ |
| `deep_hebbian` | Spectral norm stability | ✅ |
| `hebbian_chain` | NEBC chain | ✅ |
| `hebbian_3d` | 3D lattice | ✅ |
| `three_factor_hebbian` | Neuromodulated | ✅ |

**Gap**: **Local goodness with homeostatic plasticity** — prevents dead neurons.

### 1.5 Predictive Coding & Target Prop

| Model | Status |
|-------|--------|
| `fabricpc_graph_pcn` | ✅ |
| `predictive_coding_hybrid` | ✅ |
| `diff_target_prop` | ✅ |

### 1.6 Spiking — **Neuromorphic Native**

| Model | Status | Priority |
|-------|--------|----------|
| `spiking_stdp` | LIF + STDP | ✅ |

**P1**: **Surrogate-gradient spiking EquiTile** — unify rate-based + spike-based tiles.

### 1.7 Flagship Experiments (Runnable Now)

#### 1.7.1 EquiTile Scaling Sweep
**File**: `bioplausible/experiments/equitile_scaling.py`

Systematic depth/width scaling on MNIST/CIFAR-10:
- Configs: layers ∈ {2,4,8,16,32}, tiles_per_layer ∈ {2,4,8}, neurons_per_tile ∈ {32,64,128}
- Modes: PC, EP, backprop (control)
- Metrics: test accuracy, settling steps, active tile %, memory, wall-time
- Hypothesis: EquiTile maintains accuracy at depth where backprop degrades (no gradient vanishing)
- Output: scaling law plots + Pareto frontiers (accuracy vs compute)

#### 1.7.2 EqProp Family Parity on Vision
**File**: `bioplausible/experiments/eqprop_vision_parity.py`

Comprehensive EqProp variant comparison:
- Models: `eqprop_mlp`, `directed_ep`, `finite_nudge_ep`, `lazy_eqprop`, `momentum_equilibrium`, `modern_conv_eqprop`, `sparse_equilibrium`
- Tasks: MNIST, Fashion-MNIST, CIFAR-10, SVHN
- Sweep: β ∈ {0.01, 0.05, 0.1, 0.5}, inference_steps ∈ {10, 20, 50, 100}, lr schedules
- Key question: Which variants close the backprop gap? Under what compute budgets?
- Output: variant recommendation matrix per task/compute budget

#### 1.7.3 MEP Preset Tournament
**File**: `bioplausible/experiments/mep_tournament.py`

Systematic MEP variant evaluation (smep, smep_fast, sdmep, local_ep, natural_ep, muon_backprop):
- Factorized design: gradient_strategy × update_strategy × constraint_strategy × feedback_strategy
- Tasks: MNIST, CIFAR-10, tiny_shakespeare (LM)
- Metrics: final accuracy, convergence speed, memory, stability (loss variance)
- Hypothesis: Specific factor combinations dominate for specific regimes
- Output: factor importance analysis + recommended presets per domain

#### 1.7.4 Feedback Alignment Depth Scaling
**File**: `bioplausible/experiments/fa_depth_scaling.py`

Test the weight-transport-free claim at extreme depth:
- Models: `standard_fa`, `adaptive_feedback_alignment`, `dfa_deep`, `direct_feedback_alignment_eqprop`, `energy_guided_fa`
- Architecture: MLP with layers ∈ {10, 20, 50, 100, 200, 500, 1000}
- Tasks: MNIST (flattened), synthetic parity tasks
- Metrics: train/test accuracy, gradient alignment cosine similarity, activation norms
- Output: depth-scaling curves proving FA family viability where backprop fails

---

## Phase 2: Cross-Domain Transfer (GPU-Complete)

*Demonstrate bio-plausible generality: same algorithms work across vision, language, RL, graphs.*

### 2.1 Language Modeling with Local Learning
**File**: `bioplausible/experiments/lm_local_learning.py`

- Models: `eqprop_transformer`, `lm_equitile`, `optimized_lm_equitile`, `backprop_transformer_lm` (baseline)
- Tasks: tiny_shakespeare, wikitext-2, penn-treebank
- Metrics: perplexity, bits-per-character, training FLOPs/token
- Key: Compare settlement steps vs transformer layers for EqProp variants
- Output: First credible local-learning LM results

### 2.2 Reinforcement Learning with EquiTile
**File**: `bioplausible/experiments/rl_equitile.py`

- Models: `rl_equitile`, `recurrent_rl_equitile`, standard actor-critic (baseline)
- Tasks: CartPole, LunarLander, Atari (Pong, Breakout via vectorized envs)
- Metrics: sample efficiency, asymptotic return, training stability
- Hypothesis: Local Hebbian updates + tile importance = natural exploration/exploitation balance
- Output: RL learning curves + tile utilization heatmaps

### 2.3 Graph & Time-Series Domains
**File**: `bioplausible/experiments/graph_timeseries.py`

- Graph: `graph_eqprop`, `graph_equitile` on Cora, Citeseer, ogbn-arxiv (node classification)
- Time-series: `timeseries_equitile` on ETTh1, electricity, traffic (forecasting)
- Metrics: accuracy/MAE, inference latency, memory
- Output: Domain-specific tuning guides

### 2.4 Cross-Domain Transfer Benchmark
**File**: `bioplausible/experiments/cross_domain_transfer.py`

Automated: train on vision → evaluate/adapt on LM/RL/graph:
- Source tasks: MNIST, CIFAR-10
- Target tasks: tiny_shakespeare, CartPole, Cora
- Methods: weight transfer, feature extraction, continual learning (EWC)
- Metrics: transfer efficiency (target performance / from-scratch baseline)
- Hypothesis: Local learning representations transfer better (less co-adaptation)

---

## Phase 3: EquiTile — The Flagship Architecture (GPU-Complete)

> EquiTile is the most user-recruiting component: **scalable, local, production-ready**.

### 3.1 Core Stabilization — **P0**
- [ ] Fix all flaky tests in `tests/unit/equitile/`
- [ ] Deterministic tile initialization
- [ ] Gradient checkpointing for deep tile stacks
- [ ] Mixed-precision stability (FP16/BF16)

### 3.2 Domain Variants — **P1 (User-Visible Demos)**

| Variant | Domain | Status | Demo Target |
|---------|--------|--------|-------------|
| `ConvEquiTile` | Vision | ✅ | CIFAR-100 ≥ 75% |
| `LMEquiTile` | Language | ✅ | Tiny Shakespeare ≤ 1.2 BPB |
| `OptimizedLMEquiTile` | Language (MoT) | ✅ | WikiText-103 ≤ 1.5 BPB |
| `RLEquiTile` | RL | ✅ | CartPole, Atari Pong |
| `GraphEquiTile` | Graph | ✅ | Cora, PubMed |
| `TimeSeriesEquiTile` | Forecasting | ✅ | ETTh1, Electricity |
| `FastLMEquiTile` | Fast viz | ✅ | — |

### 3.3 Advanced Features — **P1-P2**

| Feature | Priority | Impact |
|---------|----------|--------|
| Dynamic tile growth/pruning | P1 | Continual learning, architecture search |
| Async execution (`AsyncEquiTile`) | P1 | Throughput on multi-GPU |
| Distributed (`DistributedEquiTile` + NCCL) | P1 | Multi-node scaling |
| Spiking EquiTile (STDP + tiles) | P2 | Neuromorphic deployment |
| 3D cortical column topology | P2 | Biological realism |
| ONNX/TorchScript export | P1 | Production deployment |

### 3.4 EquiTile Benchmarks — **P0 for Recruitment**

**Standardized suite** (run nightly, publish to leaderboard):
```
Vision:      MNIST, FMNIST, CIFAR-10, CIFAR-100, ImageNet-1k (subset)
Language:    Tiny Shakespeare, WikiText-2, WikiText-103
RL:          CartPole, LunarLander, Pong (Atari)
Graph:       Cora, Citeseer, PubMed, ogbn-arxiv
TimeSeries:  ETTh1, ETTh2, Electricity, Traffic
Continual:   Split MNIST, Permuted MNIST, Split CIFAR-100
```

**Each benchmark**: 5 seeds, compute-matched vs backprop baseline, Pareto plots.

---

## Phase 4: AutoScientist — Autonomous Discovery (GPU-Complete)

### 4.1 LLM Reasoning Upgrade — **P1**
Current: Basic OpenAI calls. Target: **Structured reasoning with experiment context**.

**Enhancements to `LLMHypothesisGenerator`**:
- [ ] **Chain-of-thought templates** for:
  - Failure analysis ("Why did X fail on task Y?")
  - Transfer reasoning ("X works on vision; what changes for language?")
  - Composition reasoning ("Combine EqProp settling with FA feedback")
  - Scaling prediction ("How will this scale to 10x depth?")
- [ ] **Literature retrieval** (arXiv API + semantic search) for prior art
- [ ] **Counterfactual generator** ("What if we changed β schedule?")
- [ ] **Structured output** (JSON schema matching `Hypothesis` dataclass)
- [ ] **Local LLM support** (llama.cpp, ollama) — no API key required

### 4.2 Knowledge Base Synthesis — **P1**
Current: Stores entries. Target: **Generates insights**.

**Add to `KnowledgeBase`**:
- [ ] `run_meta_analysis()` → Periodic synthesis report:
  - Scaling law fits (power law: accuracy ~ params^α, data^β, compute^γ)
  - Algorithm fingerprinting (hyperparameter sensitivity signatures via PCA)
  - Failure manifold mapping (systematic negative results by model/task)
  - Cross-domain transfer matrix (what transfers where)
- [ ] `extract_scaling_laws(model_family, task)` → Returns fitted α, β, γ with confidence
- [ ] `predict_performance(config, target_metric)` → Surrogate prediction with uncertainty
- [ ] `suggest_next_experiment()` → Bayesian optimization over algorithm space

### 4.3 Campaign Persistence & Resume — **P1**
- [ ] AutoScientist state serialization (YAML + SQLite)
- [ ] Resume from arbitrary checkpoint with full context
- [ ] Campaign versioning (git-like: branches for exploration, merges for validation)

### 4.4 Human-in-the-Loop Interface — **P2**
- [ ] Web dashboard for hypothesis review/approval
- [ ] Slack/Discord notifications for milestone results
- [ ] Interactive hypothesis editing

---

## Phase 5: Validation Framework — Scientific Rigor (GPU-Complete)

### 5.1 Validation Tracks — **Complete Registration**

| Track | Status | Priority |
|-------|--------|----------|
| Core (correctness) | ✅ | — |
| Scaling (depth/width/data) | 🔄 | P1 |
| Research (novel algos) | ✅ | — |
| Signal (dynamics/gradients) | 🔄 | P1 |
| Tradeoffs (perf vs compute) | 🔄 | P1 |
| Hardware (GPU/CPU/neuromorphic) | ❌ | P1 |
| Application (vision/language/RL) | ✅ | — |
| Architecture Comparison | ✅ | — |
| Negative Results (NEBC) | 🔄 | P1 |

**P1**: Implement missing tracks with standardized protocols.

### 5.2 Gradient Equivalence Testing — **P1**
> Critical for EqProp/MEP claims.

**Add**: `tests/integration/test_gradient_equivalence.py`
- Finite-difference gradient check for every propagator
- Relative error thresholds per algorithm family
- Automatic failure detection + reporting

### 5.3 Statistical Validation Utilities — **P1**
**File**: `bioplausible/validation/statistics.py`
- [ ] Bootstrap confidence intervals (percentile, BCa)
- [ ] Effect size reporting (Cohen's d, Cliff's delta)
- [ ] Multiple comparison correction (Benjamini-Hochberg)
- [ ] Power analysis for experiment sizing
- [ ] Bayesian A/B testing (rope, HDI)

### 5.4 Negative Result Documentation — **P1**
**File**: `bioplausible/analysis/failure_manifesto.py` — structured negative results:
- What was tried, why it should work, why it failed
- Search space explored (hyperparameters, seeds, architectures)
- Partial successes (what *did* work)
- Hypotheses for future work

---

## Phase 6: Novel Algorithm Development (GPU-Complete)

*New models/variations addressing specific gaps identified in Phases 1-5.*

### 6.1 Hybrid Local-Global Architectures
**Files**: `bioplausible/zoo/models/hybrid/`

Address the "pure local learning underfits complex patterns" hypothesis:
- `local_global_eqprop`: EqProp body + backprop head (last 1-2 layers)
- `fa_backprop_head`: Feedback alignment body + backprop readout
- `equitile_global_readout`: EquiTile tiles + global attention readout
- `progressive_locality`: Start global (backprop), anneal to local (EqProp) over training
- Registration: full metadata, parity tests, AutoScientist discoverable

### 6.2 Spectral/Normalization Variants for Stability
**Files**: `bioplausible/zoo/models/spectral/`, `bioplausible/zoo/optimizers/spectral.py`

- `spectral_eqprop`: Spectral normalization on recurrent weights (Lipschitz-1 guarantee)
- `orthogonal_equitile`: Tile weights constrained to Stiefel manifold
- `normalized_hebbian`: Oja's rule / Sanger's rule variants for Hebbian stability
- `layerwise_lipschitz`: Per-layer Lipschitz constraints via power iteration
- Target: Enable 1000+ layer local learning without explosion/vanishing

### 6.3 Spiking EquiTile (Surrogate Gradients)
**Files**: `bioplausible/zoo/models/spiking_equitile.py`, `bioplausible/acceleration/surrogate.py`

- LIF neurons in tiles, STDP-style local updates
- Surrogate gradients: fast sigmoid, piecewise linear, ATan for backward pass
- Event-driven settlement: only active tiles compute
- Hybrid: spiking tiles + rate-based readout for GPU efficiency
- Benchmark: NMNIST, SHD, DVS-Gesture (neuromorphic datasets)

### 6.4 Structured Topology Variants
**Files**: `bioplausible/equitile/topology/`

Beyond layered grids:
- `cortical_column_equitile`: Mini-columns with local inhibition, long-range excitation
- `hierarchical_equitile`: Multi-scale tiles (coarse → fine) with top-down modulation
- `small_world_equitile`: Watts-Strogatz rewiring for short path length + high clustering
- `modular_equitile`: Specialized tile modules (vision, language, motor) with routing
- Analysis: graph metrics (clustering, path length, modularity) vs performance

### 6.5 Continual/Lifelong EquiTile
**Files**: `bioplausible/equitile/continual/`

- `dynamic_tile_allocation`: New tiles for new tasks, importance-based protection
- `elastic_tile_consolidation`: EWC on tile importance + edge weights
- `replay_from_tiles`: Generative replay using tile dynamics (no stored data)
- `task_agnostic_growth`: Tile splitting/merging driven by error signals, not task labels
- Benchmark: Split MNIST, Permuted MNIST, CIFAR-100 incremental

---

## Phase 7: Analysis & Visualization Toolkit (GPU-Complete)

*Tools that turn raw experiments into insights and publications.*

### 7.1 Training Dynamics Analyzer — **P1**
**File**: `bioplausible/analysis/dynamics.py`
- [ ] Energy trajectory plotting (free vs nudged phase)
- [ ] Convergence rate analysis (linear/quadratic/contraction)
- [ ] Gradient alignment: local update vs true gradient (cosine similarity)
- [ ] Tile activity heatmaps (EquiTile)
- [ ] Sparsity evolution over training
- [ ] Phase transition detection: critical β, learning rate, depth thresholds
- Output: Interactive plots (Plotly) + summary statistics

### 7.2 Scaling Law Characterization — **P1**
**File**: `bioplausible/analysis/scaling.py`
- [ ] `fit_power_law(x, y)` → returns α, β, R², confidence intervals
- [ ] `plot_scaling_curves()` → multi-panel: params, data, compute, depth
- [ ] Chinchilla-style optimal allocation curves
- [ ] Extrapolation with uncertainty bands

### 7.3 Pareto Frontier Computation — **P1**
**File**: `bioplausible/analysis/pareto.py`
- [ ] Multi-objective: accuracy, FLOPs, memory, energy, wall-time
- [ ] Interactive Plotly frontend for exploration
- [ ] Automatic knee-point detection

### 7.4 Ablation Study Framework — **P1**
**File**: `bioplausible/analysis/ablation.py`
- [ ] Component contribution (leave-one-out)
- [ ] Hyperparameter sensitivity (Sobol indices)
- [ ] Automated report generation

### 7.5 Algorithm Similarity & Genealogy
**File**: `bioplausible/analysis/genealogy.py`
- Hyperparameter sensitivity fingerprints → algorithm embeddings
- t-SNE/UMAP of algorithm space (what clusters together?)
- Phylogenetic tree: which algorithms are "descendants" of others?
- Convergent evolution detection: independent discovery of same mechanisms
- Output: Algorithm map for paper figures

### 7.6 Interpretability Toolkit
**File**: `bioplausible/analysis/interpretability.py`
- Tile/neuron receptive fields (synthetic stimuli optimization)
- Weight matrix spectra, singular value distributions
- Information flow: mutual information between layers/tiles
- Causal mediation: which tiles mediate input→output?
- Concept alignment: do tiles learn human-interpretable features?

---

## Phase 8: Hardware Acceleration — GPU-First (GPU-Complete)

> **Defer non-GPU until GPU functionality complete.**

### 8.1 Triton Kernels — **P1**
**File**: `bioplausible/acceleration/triton_kernels.py`
- [ ] EqProp relaxation step (fused activity + error update)
- [ ] Hebbian outer product (batched)
- [ ] Tile prediction + error computation
- [ ] Benchmark vs PyTorch eager + `torch.compile`

### 8.2 Backend Dispatch — **P1**
**File**: `bioplausible/acceleration/backends.py`
- [ ] Auto-select: CUDA → Triton → CPU → NumPy
- [ ] Profile-guided selection per operation
- [ ] Fallback chain with logging

### 8.3 `torch.compile` Integration — **P1**
**File**: `bioplausible/acceleration/compile.py`
- [ ] Custom EqProp backward for `torch.compile`
- [ ] Dynamic shape support (variable tile counts)
- [ ] Graph break minimization

### 8.4 Pure NumPy/CuPy Kernels — **P0** (Reference)
**File**: `bioplausible/acceleration/kernels.py`
- [ ] Reference implementations for correctness testing
- [ ] CPU fallback for CI

---

## Phase 9: Deployment & Export — Production Path (GPU-Complete)

### 9.1 Model Export — **P1**
**File**: `bioplausible/equitile/deployments/deployment.py`
- [ ] ONNX export (dynamic axes, opset 17+)
- [ ] TorchScript export
- [ ] INT8 quantization (PTQ + QAT)
- [ ] Ternary weight quantization (for neuromorphic)

### 9.2 Inference Engine — **P2**
**File**: `bioplausible/deployment.py`
- [ ] FastAPI server with batching
- [ ] TensorRT optimization path
- [ ] Benchmark: throughput, latency (p50/p95/p99)

---

## Phase 10: Distributed & P2P — Scale (GPU-Complete)

### 10.1 Multi-GPU Training — **P1**
- [ ] DDP wrapper for all models
- [ ] FSDP for large EquiTile (>1B params)
- [ ] Gradient accumulation + mixed precision

### 10.2 P2P Coordinator — **P2**
**File**: `bioplausible/p2p/`
- [ ] Kademlia DHT for peer discovery
- [ ] Task dispatch + result aggregation
- [ ] Fault tolerance (peer dropout, stragglers)
- [ ] Incentive mechanism (token/credit system)

---

## Deferred: Non-GPU Work (Post-GPU-Validation)

*Only pursue after GPU parity benchmarks + EquiTile stability + publications.*

| Area | Integration Path | Trigger |
|------|------------------|---------|
| Loihi 2 (Intel) | `lava` backend, spike conversion | Phase 1-2 results published |
| SpiNNaker | `spynnaker` backend | Hardware partner interest |
| BrainScaleS | `hbp` neuromorphic platform | Collaboration established |
| Memristor crossbar | `neuro-sim`, `crossbar` simulators | Funding secured |
| Analog AI | Custom ADC/DAC modeling | Collaboration established |
| Optical Computing | Diffractive/integrated photonic models | Collaboration established |
| ASIC/FPGA | HLS kernels for tile operations | Funding secured |
| Wet Lab Interface | MEA / calcium imaging data integration | Neuroscience collaborator |
| Edge Deployment | TFLite / ONNX Runtime / TensorRT | Product need identified |

---

## Milestone Schedule (GPU-First)

| Week | Milestone | Deliverable | Recruitment Signal |
|------|-----------|-------------|-------------------|
| 1-2 | Foundation | Parity suite, metadata audit, reproducibility | "Run `pytest tests/validation/test_parity.py` — see exactly how close we are to backprop" |
| 3-4 | EqProp/MEP Portfolio | All EqProp variants working + Triton kernels | "EqProp matches backprop on MNIST/CIFAR with 10x less memory" |
| 4-6 | EquiTile Core | Stable core + ConvEquiTile + LMEquiTile demos | "Train EquiTile on CIFAR-10 in 5 min on single GPU" |
| 5-6 | Validation Tracks | Scaling, Signal, Hardware, NEBC tracks complete | "Automated scaling law extraction from your experiments" |
| 6-8 | Analysis Suite | Dynamics, Pareto, Ablation, Scaling laws | "One command: `biopl-analyze experiment.db --pareto`" |
| 7-9 | AutoScientist v1 | CoT reasoning + KB synthesis + campaign persistence | "Leave it running overnight; wake up to 50 tested hypotheses" |
| 8-10 | EquiTile Domains | RL, Graph, TimeSeries benchmarks published | "EquiTile beats backprop on continual learning benchmarks" |
| 10-12 | Deployment | ONNX export + inference server | "Deploy EquiTile to production in 3 commands" |
| 12-14 | Distributed | Multi-GPU + P2P coordinator | "Scale to 8 GPUs with one config change" |
| 14+ | Neuromorphic | Loihi 2 / SpiNNaker deployment | "Same EquiTile code runs on neuromorphic hardware" |

---

## Low-Hanging Fruit (Do First, High Visibility)

1. **`biopl-scientist --demo`** — One-command 5-min demo (MNIST, EquiTile, AutoScientist proposes 3 variants)
2. **Leaderboard page** — Auto-generated, live-updating, embeddable (GitHub Pages)
3. **Colab notebooks** — "Train EquiTile in browser" for each domain
4. **Parity benchmark CI** — Nightly GitHub Action, publishes markdown table to README
5. **Failure manifesto gallery** — "What we tried that didn't work" — builds trust

---

## New Model/Variation Ideas (Research Opportunities)

| Idea | Family | Effort | Potential |
|------|--------|--------|-----------|
| **Sign-Symmetric FA** | FA | Low | Hardware-friendly, no weight transport |
| **EquiTile + MEP** | EquiTile+MEP | Medium | Spectral-constrained tile updates |
| **Spiking EquiTile** | Spiking+EquiTile | High | Neuromorphic native |
| **3D Cortical Column EquiTile** | EquiTile | Medium | Biological realism, structured sparsity |
| **Continual EquiTile (EWC + tile growth)** | EquiTile | Medium | No catastrophic forgetting |
| **Mixture-of-Tiles (MoT) Language Model** | EquiTile | Medium | Sparse activation, scaling |
| **Equilibrium Alignment (EqAlign) + EquiTile** | FA+EquiTile | Medium | Native local alignment |
| **Forward-Forward + EquiTile Tiles** | FF+EquiTile | Low | Zero backward pass tiles |
| **Predictive Coding on Tile Graph** | PC+EquiTile | Low | Local prediction errors |
| **Meta-Learned β Schedule** | EqProp | Medium | Learned nudge annealing |
| **Energy-Based Regularization** | All | Low | Contractive dynamics → robustness |

---

## Tool/Analysis Gaps

| Tool | Status | Priority |
|------|--------|----------|
| **Experiment comparator** (side-by-side diff) | ❌ | P1 |
| **Hyperparameter importance (SHAP/PDP)** | ❌ | P1 |
| **Architecture visualizer** (tile graph, EqProp unrolled) | ❌ | P2 |
| **Energy landscape plotter** (2D slices) | ❌ | P2 |
| **Gradient flow tracker** (per-layer alignment) | ❌ | P1 |
| **Memory profiler** (per-component breakdown) | ❌ | P1 |
| **Automated paper figure generator** | ❌ | P2 |

---

## Recruitment Strategy

### For Users (Researchers/Engineers)
- **5-minute Colab demo** → "See EquiTile train on your data"
- **Parity benchmark results** → "Here's exactly where we match/beat backprop"
- **Leaderboard** → "Compare your method against 50+ bio-plausible baselines"
- **AutoScientist** → "Automate your architecture search"

### For Contributors (Developers/Researchers)
- **Good first issues** tagged in GitHub (tests, docs, benchmarks)
- **Component registry** → "Add your algorithm in 50 lines, get auto-tuning free"
- **Validation tracks** → "Your method gets rigorous evaluation automatically"
- **Knowledge base** → "Your experiments contribute to collective intelligence"

### For Hardware Partners
- **EquiTile deployment path** → ONNX → Loihi/SpiNNaker/BrainScaleS
- **Energy modeling** → "Predict Joules/sample before tape-out"
- **Tile abstraction** → Maps naturally to crossbar arrays, neuromorphic cores

---

## Success Metrics (Track in Knowledge Base)

| Metric | Target (6 mo) | Target (12 mo) |
|--------|---------------|----------------|
| Backprop parity (CIFAR-10) | ≥ 95% of BP accuracy | ≥ 100% (match) |
| EquiTile CIFAR-100 | ≥ 75% | ≥ 80% |
| EquiTile Tiny Shakespeare | ≤ 1.2 BPB | ≤ 1.0 BPB |
| AutoScientist hypotheses/week | 50 | 200 |
| Registered algorithms | 100 | 200 |
| Active contributors | 10 | 30 |
| Neuromorphic deployments | 1 (Loihi 2) | 3 |
| Citations / papers using framework | 5 | 20 |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Bio-plausible methods fundamentally can't match BP | Parity suite *proves* where they do/don't; negative results published |
| AutoScientist generates noise | Human-in-loop approval; surrogate-guided filtering; strict validation tracks |
| EquiTile too complex for adoption | Builder API + presets + Colab demos; "production config" one-liner |
| GPU kernels introduce bugs | Reference NumPy kernels + gradient equivalence testing on every commit |
| Neuromorphic gap too wide | Defer until GPU maturity; use simulators for early validation |

---

## Development Principles

1. **GPU-first**: Every feature must run on CUDA before considering other hardware
2. **Registry-driven**: New components auto-discoverable by AutoScientist/hyperopt
3. **Reproducible by default**: Seeds, env capture, artifact versioning mandatory
4. **Publishable output**: Every experiment generates paper-ready figures/tables
5. **Negative results included**: NEBC track = first-class citizen
6. **Low-friction contribution**: `register_model` decorator + CI validation = 5 min to add algorithm

---

## File/Module Map for New Work

```
bioplausible/
├── validation/
│   ├── backprop_parity.py          # NEW P0
│   ├── statistics.py               # NEW P1
│   ├── registry_audit.py           # NEW P0
│   └── test_gradient_equivalence.py # NEW P1
├── analysis/
│   ├── dynamics.py                 # ENHANCE P1
│   ├── scaling.py                  # NEW P1
│   ├── pareto.py                   # NEW P1
│   ├── ablation.py                 # NEW P1
│   ├── genealogy.py                # NEW P1
│   ├── interpretability.py         # NEW P1
│   └── failure_manifesto.py        # NEW P1
├── acceleration/
│   ├── triton_kernels.py           # ENHANCE P1
│   ├── backends.py                 # ENHANCE P1
│   ├── compile.py                  # ENHANCE P1
│   ├── kernels.py                  # REFERENCE P0
│   └── surrogate.py                # NEW P1
├── autoscientist/
│   ├── reasoner.py                 # ENHANCE (CoT, literature)
│   ├── proposer.py                 # ENHANCE (counterfactuals)
│   └── campaign_v1.py              # NEW P1
├── knowledge/
│   └── kb.py                       # ENHANCE (meta-analysis, scaling laws)
├── equitile/
│   ├── core/model.py               # STABILIZE P0
│   ├── topology/                   # NEW P1 (structured topologies)
│   ├── continual/                  # NEW P1 (lifelong learning)
│   ├── deployments/                # COMPLETE P1
│   └── training/                   # STABILIZE P1
├── zoo/
│   ├── models/
│   │   ├── hybrid/                 # NEW P1
│   │   ├── spectral/               # NEW P1
│   │   └── spiking_equitile.py     # NEW P1
│   ├── optimizers/spectral.py      # ENHANCE P1
│   └── hub.py                      # NEW P1
├── utils/reproducibility.py        # NEW P0
└── validation/tracks/              # COMPLETE registration
```

---

## Immediate Next Actions (Unordered)

- [ ] Implement `backprop_parity.py` with CLI
- [ ] Run EquiTile scaling sweep (Phase 1.7.1)
- [ ] Run EqProp family parity on CIFAR-10 (Phase 1.7.2)
- [ ] Harden AutoScientist LLM prompts with CoT templates
- [ ] Add symbolic regression to KB meta-analysis
- [ ] Create tutorial gallery (3 notebooks minimum)
- [ ] Set up nightly CI with parity smoke test
- [ ] Register hybrid local-global models (Phase 6.1)
- [ ] Implement gradient equivalence testing (Phase 5.2)
- [ ] Build `biopl-scientist --demo` one-command demo

---

*This roadmap is adaptive. Priorities shift based on experimental results. The Knowledge Base meta-analysis (Phase 4.2) continuously informs what to pursue next.*