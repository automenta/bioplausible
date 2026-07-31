# Bioplausible Research Roadmap

**Goal**: Demonstrate that bio-plausible learning matches/exceeds backprop in regimes backprop cannot reach — and build the autonomous discovery engine that systematically finds such algorithms.

**Strategy**: GPU-first, low-hanging-fruit prioritized, user-recruiting milestones every 2-4 weeks.

---

## Phase 0: Foundation Hardening (Week 1-2) — *Prerequisite for Everything*

### 0.1 Backprop Parity Benchmark Suite — **P0 CRITICAL**
> **Why**: Credible claims require apples-to-apples comparison. Without this, papers get rejected.

**Deliverable**: `bioplausible/validation/backprop_parity.py`
- Identical architectures (MLP, CNN, Transformer) across backprop vs every bio-plausible family
- Compute-matched: same FLOPs, wall-time, memory, parameter count
- Statistical rigor: 10 seeds, confidence intervals, effect sizes
- Output: Publication-ready tables + Pareto frontier plots (accuracy vs compute vs memory)

**Tests**: MNIST, CIFAR-10, Tiny Shakespeare, Penn Treebank
**Metrics**: Accuracy, FLOPs/sample, peak memory, wall-time/epoch, energy estimate (Joules)

### 0.2 Registry Metadata Completeness Audit
> **Why**: AutoScientist queries fail silently on incomplete metadata.

**Action**: Script to audit every registered component for:
- `bio_plausibility_score` (0.0-1.0, calibrated)
- `locality_level` (GLOBAL/LAYERWISE/LOCAL/EQUILIBRIUM/FORWARD_ONLY)
- `memory_complexity` (O(1), O(N), O(N²))
- `requires_backward` (bool)
- `credit_assignment_type` (gradient, equilibrium, hebbian, target, forward-only, spiking)
- `family` tag (eqprop, fa, hebbian, forward_only, target_prop, spiking, predictive_coding, mep, equitile, backprop)
- `provides` / `requires` capabilities (for compatibility checking)

### 0.3 Deterministic Seeding & Reproducibility
- Global seed manager (`bioplausible/utils/reproducibility.py`)
- All RNGs: torch, numpy, random, CUDA, cuDNN
- Experiment config hashing for exact reproducibility

---

## Phase 1: GPU-Ready Algorithm Portfolio (Week 2-6) — *Demonstrate Range*

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

---

## Phase 2: EquiTile — The Flagship Architecture (Week 4-10)

> EquiTile is the most user-recruiting component: **scalable, local, production-ready**.

### 2.1 Core Stabilization — **P0**
- [ ] Fix all flaky tests in `tests/unit/equitile/`
- [ ] Deterministic tile initialization
- [ ] Gradient checkpointing for deep tile stacks
- [ ] Mixed-precision stability (FP16/BF16)

### 2.2 Domain Variants — **P1 (User-Visible Demos)**
| Variant | Domain | Status | Demo Target |
|---------|--------|--------|-------------|
| `ConvEquiTile` | Vision | ✅ | CIFAR-100 ≥ 75% |
| `LMEquiTile` | Language | ✅ | Tiny Shakespeare ≤ 1.2 BPB |
| `OptimizedLMEquiTile` | Language (MoT) | ✅ | WikiText-103 ≤ 1.5 BPB |
| `RLEquiTile` | RL | ✅ | CartPole, Atari Pong |
| `GraphEquiTile` | Graph | ✅ | Cora, PubMed |
| `TimeSeriesEquiTile` | Forecasting | ✅ | ETTh1, Electricity |
| `FastLMEquiTile` | Fast viz | ✅ | — |

### 2.3 Advanced Features — **P1-P2**
| Feature | Priority | Impact |
|---------|----------|--------|
| Dynamic tile growth/pruning | P1 | Continual learning, architecture search |
| Async execution (`AsyncEquiTile`) | P1 | Throughput on multi-GPU |
| Distributed (`DistributedEquiTile` + NCCL) | P1 | Multi-node scaling |
| Spiking EquiTile (STDP + tiles) | P2 | Neuromorphic deployment |
| 3D cortical column topology | P2 | Biological realism |
| ONNX/TorchScript export | P1 | Production deployment |

### 2.4 EquiTile Benchmarks — **P0 for Recruitment**
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

## Phase 3: AutoScientist — Autonomous Discovery (Week 6-14)

### 3.1 LLM Reasoning Upgrade — **P1**
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

### 3.2 Knowledge Base Synthesis — **P1**
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

### 3.3 Campaign Persistence & Resume — **P1**
- [ ] AutoScientist state serialization (YAML + SQLite)
- [ ] Resume from arbitrary checkpoint with full context
- [ ] Campaign versioning (git-like: branches for exploration, merges for validation)

### 3.4 Human-in-the-Loop Interface — **P2**
- [ ] Web dashboard for hypothesis review/approval
- [ ] Slack/Discord notifications for milestone results
- [ ] Interactive hypothesis editing

---

## Phase 4: Validation Framework — Scientific Rigor (Week 4-8)

### 4.1 Validation Tracks — **Complete Registration**
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

### 4.2 Gradient Equivalence Testing — **P1**
> Critical for EqProp/MEP claims.

**Add**: `tests/integration/test_gradient_equivalence.py`
- Finite-difference gradient check for every propagator
- Relative error thresholds per algorithm family
- Automatic failure detection + reporting

### 4.3 Statistical Validation Utilities — **P1**
`bioplausible/validation/statistics.py`:
- [ ] Bootstrap confidence intervals (percentile, BCa)
- [ ] Effect size reporting (Cohen's d, Cliff's delta)
- [ ] Multiple comparison correction (Benjamini-Hochberg)
- [ ] Power analysis for experiment sizing
- [ ] Bayesian A/B testing (rope, HDI)

### 4.4 Negative Result Documentation — **P1**
`FailureManifestoGenerator` — structured negative results:
- What was tried, why it should work, why it failed
- Search space explored (hyperparameters, seeds, architectures)
- Partial successes (what *did* work)
- Hypotheses for future work

---

## Phase 5: Analysis & Visualization — Insight Generation (Week 5-9)

### 5.1 Training Dynamics Analyzer — **P1**
`bioplausible/analysis/dynamics.py`:
- [ ] Energy trajectory plotting (free vs nudged phase)
- [ ] Convergence rate analysis (linear/quadratic/contraction)
- [ ] Gradient alignment: local update vs true gradient (cosine similarity)
- [ ] Tile activity heatmaps (EquiTile)
- [ ] Sparsity evolution over training

### 5.2 Scaling Law Characterization — **P1**
`bioplausible/analysis/scaling.py`:
- [ ] `fit_power_law(x, y)` → returns α, β, R², confidence intervals
- [ ] `plot_scaling_curves()` → multi-panel: params, data, compute, depth
- [ ] Chinchilla-style optimal allocation curves
- [ ] Extrapolation with uncertainty bands

### 5.3 Pareto Frontier Computation — **P1**
`bioplausible/analysis/pareto.py`:
- [ ] Multi-objective: accuracy, FLOPs, memory, energy, wall-time
- [ ] Interactive Plotly frontend for exploration
- [ ] Automatic knee-point detection

### 5.4 Ablation Study Framework — **P1**
`bioplausible/analysis/ablation.py`:
- [ ] Component contribution (leave-one-out)
- [ ] Hyperparameter sensitivity (Sobol indices)
- [ ] Automated report generation

---

## Phase 6: Hardware Acceleration — GPU-First (Week 3-7)

> **Defer non-GPU until GPU functionality complete.**

### 6.1 Triton Kernels — **P1**
`bioplausible/acceleration/triton_kernels.py`:
- [ ] EqProp relaxation step (fused activity + error update)
- [ ] Hebbian outer product (batched)
- [ ] Tile prediction + error computation
- [ ] Benchmark vs PyTorch eager + `torch.compile`

### 6.2 Backend Dispatch — **P1**
`bioplausible/acceleration/backends.py`:
- [ ] Auto-select: CUDA → Triton → CPU → NumPy
- [ ] Profile-guided selection per operation
- [ ] Fallback chain with logging

### 6.3 `torch.compile` Integration — **P1**
`bioplausible/acceleration/compile.py`:
- [ ] Custom EqProp backward for `torch.compile`
- [ ] Dynamic shape support (variable tile counts)
- [ ] Graph break minimization

### 6.4 Pure NumPy/CuPy Kernels — **P0** (Reference)
`bioplausible/acceleration/kernels.py`:
- [ ] Reference implementations for correctness testing
- [ ] CPU fallback for CI

---

## Phase 7: Deployment & Export — Production Path (Week 8-12)

### 7.1 Model Export — **P1**
`bioplausible/equitile/deployments/deployment.py`:
- [ ] ONNX export (dynamic axes, opset 17+)
- [ ] TorchScript export
- [ ] INT8 quantization (PTQ + QAT)
- [ ] Ternary weight quantization (for neuromorphic)

### 7.2 Inference Engine — **P2**
`bioplausible/deployment.py`:
- [ ] FastAPI server with batching
- [ ] TensorRT optimization path
- [ ] Benchmark: throughput, latency (p50/p95/p99)

---

## Phase 8: Distributed & P2P — Scale (Week 10-16)

### 8.1 Multi-GPU Training — **P1**
- [ ] DDP wrapper for all models
- [ ] FSDP for large EquiTile (>1B params)
- [ ] Gradient accumulation + mixed precision

### 8.2 P2P Coordinator — **P2**
`bioplausible/p2p/`:
- [ ] Kademlia DHT for peer discovery
- [ ] Task dispatch + result aggregation
- [ ] Fault tolerance (peer dropout, stragglers)
- [ ] Incentive mechanism (token/credit system)

---

## Phase 9: Neuromorphic Deployment — **DEFERRED** (Post-GPU)

> Only after GPU parity benchmarks + EquiTile stability.

| Target | Integration Path |
|--------|------------------|
| Loihi 2 (Intel) | `lava` backend, spike conversion |
| SpiNNaker | `spynnaker` backend |
| BrainScaleS | `hbp` neuromorphic platform |
| Memristor crossbar | `neuro-sim`, `crossbar` simulators |
| Analog AI | Custom ADC/DAC modeling |

**Prerequisites**: Event-driven simulation, spike conversion, energy modeling.

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

## Success Metrics

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

## Appendix: File/Module Map for New Work

```
bioplausible/
├── validation/
│   ├── backprop_parity.py          # NEW P0
│   ├── statistics.py               # NEW P1
│   └── test_gradient_equivalence.py # NEW P1
├── analysis/
│   ├── dynamics.py                 # ENHANCE P1
│   ├── scaling.py                  # NEW P1
│   ├── pareto.py                   # NEW P1
│   └── ablation.py                 # NEW P1
├── acceleration/
│   ├── triton_kernels.py           # ENHANCE P1
│   ├── backends.py                 # ENHANCE P1
│   └── compile.py                  # ENHANCE P1
├── autoscientist/
│   ├── reasoner.py                 # ENHANCE (CoT, literature)
│   └── proposer.py                 # ENHANCE (counterfactuals)
├── knowledge/
│   └── kb.py                       # ENHANCE (meta-analysis, scaling laws)
├── equitile/
│   ├── core/model.py               # STABILIZE P0
│   ├── deployments/                # COMPLETE P1
│   └── training/                   # STABILIZE P1
└── validation/
    └── tracks/                     # COMPLETE registration
```

---

*This roadmap is living. Update weekly based on experimental results. Prioritize what enables the next user-facing demo.*