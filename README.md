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

All entry points installed with `uv sync --dev`:

| Command | Purpose |
|---------|---------|
| `biopl` | Main CLI entry point |
| `biopl-scientist` | Autonomous experiment loop (AutoScientist hypercube campaigns) |
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

CLI: `eqprop-p2p-worker` starts a worker node.

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
