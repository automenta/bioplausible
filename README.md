## 🌌 Searching for Computronium

> **[Computronium](https://en.wikipedia.org/wiki/Computronium) is the theoretical limit of physical computation—matter organized to compute, learn, and adapt.** 
> This framework is the scientific engine designed to discover it. We provide a rigorous, composable ontology for building learning systems out of local, physical dynamics, bridging the gap between abstract algorithms and the thermodynamic reality of optical, memristive, neuromorphic, biological, quantum, and other substrates.

Modern deep learning has achieved miracles in the realm of mathematical abstraction. But abstraction hides physical cost. True intelligence in the natural world does not require global clocks, infinite memory for backward passes, or perfect numerical precision. It emerges from local interactions, energy minimization, and physical constraints. 

The **Search for Computronium** is the quest to find learning algorithms that are native to the physical universe. It is about building systems that learn asynchronously, self-heal from noise, and operate at the thermodynamic limits of efficiency. To achieve this, we must decouple machine learning from the rigid, non-physical requirements of digital abstraction and embrace local, energy-based dynamics where networks relax toward equilibrium.

### 🎯 Why Computronium? (Choose Your Lens)

Computronium is not just a software library; it is a **parameterized algorithm space** and a physics simulator for AI. Depending on your background, here is what the search for computronium unlocks for you:

| Audience | The Paradigm Shift | What You Get |
| :--- | :--- | :--- |
| 🧠 **Natural Scientists & Physicists** | **Thermodynamic AI.** Computation as a physical process. | A mathematically rigorous playground for local, Hebbian, and contrastive learning that obeys thermodynamic constraints, Lyapunov stability, and physical passivity. |
| 📊 **Data Scientists & ML Researchers** | **The Algorithmic Genome.** | A **6-D composable search space** to discover novel learning rules. Achieve O(1) memory credit assignment and train arbitrarily deep networks without the memory wall. |
| 🛠️ **Hardware & Silicon Architects** | **Algorithm-Hardware Co-Design.** | Native physics modeling for Memristive (IR-drop), Optical (phase), and Quantum (unitary) substrates. Test your chip's physical realities against algorithmic stability *before* tape-out. |
| 💻 **Systems Engineers & Developers** | **Correctness by Construction.** | Type-safe, property-locked (Hypothesis), and Triton-accelerated infrastructure. An `AutoScientist` that automates the scientific method across hardware topologies. |

### 🧬 The Telescope: Coupled Dynamical Systems

Historically, ML frameworks treat models as static computational graphs. To search for computronium, we must treat them as **coupled dynamical systems**. 

By elevating the computational rule itself to a dynamical variable, we introduce a **joint transition operator** $z_{t+1} = F_\theta(z_t; G, S)$ that unifies fast neural activity, slow synaptic consolidation, and substrate physics into a single mathematical center. In this framework, standard digital backpropagation is not the default; it is merely recovered as a restricted, null-plasticity slice of a much larger, physically grounded universe.

Computronium provides the ontology, the infrastructure, and the automated scientific engine to map the **physical limits of learning**.

---

## 🔮 The 6-Axis Decomposition

Every learning system in Computronium maps uniquely to a coordinate in a Cartesian product of six fundamental axes:

```
System = Substrate × Geometry × StateDynamics × Plasticity × CreditAssignment × ParameterUpdate
```

This decomposition transforms the framework from a "library of models" into a **parameterized algorithm space** — any valid combination of primitives yields a coherent learning system, and the space of all combinations is the search space for the **AutoScientist**.

| Axis | Symbol | Role | Primitives |
|------|:------:|------|------------|
| **🔩 Substrate** | $S$ | Physical state space: precision, noise, sparsity constraints | `Digital`, `Memristive` (conductance, IR-drop), `Neuromorphic` (async spikes), `Photonic` (phase/amplitude), `Quantum` (unitary gates), `Noisy`, `Complex`, `Sparse`, `Ternary` |
| **🔷 Geometry** | $G$ | Topology & routing of computational units | `FeedforwardDAG` (MLP/CNN), `RecurrentAttractor` (Hopfield/EqProp), `TileMesh` (TileNet), `FabricPC` (arbitrary node-edge), `SpatialLattice3D` (neural_cube) |
| **🌀 StateDynamics** | $D$ | Forward evolution & settling (the "forward pass") | `EnergyMinimization` (EqProp), `PredictiveSettling` (Predictive Coding), `SpikeIntegration` (LIF/Izhikevich), `InstantaneousPass` (FF/Backprop), `LazyStateDynamics` (on-demand activation), `Diffusion` |
| **🧬 Plasticity (MetaDynamics)** | $M$ | The mechanism by which the computational rule becomes a dynamical variable | `NullPlasticity` (Zero-Extension), `RoutingPlasticity` (gating/rerouting), `FastWeightPlasticity` (episode-local memory), `SubstrateCoupledPlasticity` (physical plasticity), `RuleStatePlasticity` (Z3: rule selection) |
| **💡 CreditAssignment** | $C$ | Error routing & pseudo-gradient computation | `ThermodynamicContrast` (EqProp free/nudged), `RandomProjectionsCredit` (FA/DFA), `LocalGoodnessCredit` (Forward-Forward/PEPITA), `TemporalTraceCredit` (STDP), `TargetInversionCredit` (Target Prop), `HomeostaticCredit` (autonomous Lipschitz scaling) |
| **🔧 ParameterUpdate** | $U$ | Slow, persistent parameter consolidation Δθ | `EuclideanUpdate` (SGD/Adam), `RiemannianOrthogonalUpdate` (Muon), `SpectralConstrainedUpdate`, `NaturalGradientUpdate` (Fisher), `ElasticConsolidationUpdate` (EWC) |

### Architecture Diagram

```mermaid
flowchart LR
    S[Substrate] --> G[Geometry]
    G --> D[StateDynamics]
    D --> M[Plasticity]
    M --> C[CreditAssignment]
    C --> U[ParameterUpdate]
    U --> S
```

### Algebraic Composition (API)

Construct systems by composing primitives across the six axes. The `System` generic enforces valid combinations at type-check time.

```python
from computronium.core.ontology import (
    System,
    DigitalSubstrate,
    FeedforwardGeometry,
    InstantaneousDynamics,
    BackpropCredit,
    EuclideanUpdate,
    GeometryConfig,
    RecurrentGeometry,
    EnergyMinimizationDynamics,
    ThermodynamicContrastCredit,
    MemristiveSubstrate,
    TileGeometry,
    TileAlgorithmConfig,
    LazyStateDynamics,
    HomeostaticCredit,
)
from computronium.core.joint import PlasticityConfig
```

**5-D compatible (M = NullPlasticity)** — standard backprop MLP, Equilibrium Propagation, TileNet:
```python
system = System(
    substrate=DigitalSubstrate(),
    geometry=FeedforwardGeometry(
        GeometryConfig(input_dim=784, output_dim=10, hidden_dims=(256, 128))
    ),
    dynamics=InstantaneousDynamics(),
    credit=BackpropCredit(),
    update=EuclideanUpdate(step_size=0.01),
    plasticity=PlasticityConfig.null(),
)
```

**6-D joint systems** — with non-null plasticity:
```python
# RoutingPlasticity: state-dependent gating, sparse pathways
joint_routing = System(..., plasticity=PlasticityConfig.routing(gate_dim=64))

# FastWeightPlasticity: episode-local associative memory
joint_fast_weight = System(
    ...,
    plasticity=PlasticityConfig.fast_weights(
        fast_weight_dim=512, decay=0.9, learning_rate=0.1
    ),
)

# SubstrateCoupledPlasticity: physical memristive drift
memristive_plastic = System(..., plasticity=PlasticityConfig.substrate_coupled())
```

Formerly many hardcoded models (`optical_looped_mlp`, `quantized_looped_mlp`, `crossbar_looped_mlp`, `eqprop_transformer`, `neural_cube`, `sparse_equilibrium`, `momentum_equilibrium`, TileNet variants) are now **emergent coordinates** in this 6-D space. The 5-D coordinates (Sprints 9.0–9.7) are recovered as the `M = NullPlasticity` slice.

### Research Direction Models (Native Ontology Implementations)

The framework includes native implementations of novel research directions as first-class ontology coordinates:

| Model | Coordinate | Description |
|-------|------------|-------------|
| `holomorphic_ep` | `QuantumSubstrate × RecurrentGeometry × EnergyMinimization × ThermodynamicContrast × EuclideanUpdate` | Complex-valued Equilibrium Propagation using holomorphic (analytic) activation functions and conjugate-transpose feedback pathways. Enables complex-domain credit assignment with potential for phase-based computation. |
| `directed_ep` | `DigitalSubstrate × RecurrentGeometry × EnergyMinimization × RandomProjections × EuclideanUpdate` | Directed/Asymmetric Equilibrium Propagation implementing Feedback Alignment within energy-based framework. Fixed random feedback matrices (no weight transport) with thermodynamic settling dynamics. |
| `finite_nudge_ep` | `DigitalSubstrate × RecurrentGeometry × EnergyMinimization × ThermodynamicContrast(beta≥1) × EuclideanUpdate` | Finite-Nudge Equilibrium Propagation using large β (finite nudge) instead of infinitesimal limit. Stronger supervision signals while maintaining equilibrium dynamics. |
| `ternary_eqprop` | `TernarySubstrate × RecurrentGeometry × EnergyMinimization × ThermodynamicContrast × EuclideanUpdate` | Ternary-weight Equilibrium Propagation with STE-based quantization. Weights constrained to {-α, 0, +α}. |
| `momentum_eqprop` | `DigitalSubstrate × RecurrentGeometry × EnergyMinimization(momentum) × ThermodynamicContrast × EuclideanUpdate` | Heavy-ball settling dynamics for faster equilibrium convergence. |
| `sparse_eqprop` | `SparseSubstrate × RecurrentGeometry × EnergyMinimization × ThermodynamicContrast × EuclideanUpdate` | Dynamic sparsity masks with efficient sparse matmul. |
| `diffusion_eqprop` | `DigitalSubstrate × RecurrentGeometry × DiffusionDynamics × ThermodynamicContrast × EuclideanUpdate` | Continuous-time diffusion settling dynamics. |

These models are available via the native API in `computronium.models.native`:
```python
from computronium.models.native import (
    create_native_holomorphic_ep,
    create_native_directed_ep,
    create_native_finite_nudge_ep,
    create_native_ternary_eqprop,
    create_native_momentum_eqprop,
    create_native_sparse_eqprop,
    create_native_diffusion_eqprop,
)
```

---

## ⚡ Thermodynamic Invariant: Energy as First-Class Object

Energy binds Geometry and StateDynamics. The framework elevates the energy function `E(x)` to a first-class object, enabling mathematical stability proofs *before* implementation:

- **Symmetric topology + EnergyMinimization** → guaranteed fixed-point convergence (Hopfield/EqProp) via LaSalle's invariance principle
- **Directed topology** → requires Control-Lyapunov formulation for stability (formally verified for PredictiveSettlingDynamics)
- **Free energy tracking** → per-iteration Lyapunov certificates (`track_free_energy_per_iter`) for predictive coding and directed FA

**Joint Architecture Extension**: The mathematical center is now the **joint transition operator** $z_{t+1} = F_\theta(z_t; G, S)$ acting on composite state $z_t = (x_t, \psi_t, \sigma_t)$. The `StateRegistry` assigns lifecycle metadata to every variable (persistent θ, fast plastic ψ, substrate-owned σ, consolidatable), resolving ontological overlaps where a single physical variable (e.g., memristive conductance) serves multiple roles. Slow learning operates on persistent θ at episode boundaries: $\theta_{e+1} = U(\theta_e, C(\tau_e))$. **The 5-D energy-based dynamics are the restriction `F_θ^Null = D_θ` where `M=Null`, `ψ=∅`, `σ=σ₀`.**

This enables the **AutoScientist** to reason about *physical realizability* and the **stability-plasticity trade-off** as constraints, not afterthoughts.

---

## 🖥️ CLI Commands

All entry points installed with `uv sync --dev`. The CLI has been consolidated under the `comp` dispatcher:

### Main Dispatcher: `comp`

```bash
comp <command> [args]
```

| Subcommand | Purpose | Legacy Alias |
|------------|---------|--------------|
| `comp run` | Campaign runner (validate/plan/run) | `comp-run` |
| `comp report` | Render experiment reports | `comp-report` |
| `comp parity` | Backprop parity benchmark | `comp-parity` |
| `comp repro` | Reproducibility verification | `comp-repro-check` |
| `comp hpo` | Hyperparameter optimization | `comp-hpo` |
| `comp audit` | Registry metadata audit | `comp-registry-audit` |
| `comp frontier` | Pareto frontier analysis | `comp-frontier` |
| `comp rank` | Family ranking from HPO studies | `comp-compare` |
| `comp lab` | Interactive experiments & model inspection | — |
| `comp joint-validate` | Validate arbitrary 6-D joint coordinates | — |
| `comp campaign` | Run/compare/resume joint campaigns | — |
| `comp stability` | Stability-plasticity frontier reports | — |
| `comp benchmark` | Run joint benchmark suites (adaptation, Z3, etc.) | — |

### Standalone Commands (for scripting/CI)

| Command | Purpose |
|---------|---------|
| `comp-scientist` | Autonomous experiment loop (AutoScientist hypercube campaigns) |
| `comp-failure-manifesto` | Structured negative result documentation |
| `comp-export-kernel` | Export kernel backend (untrained) |
| `comp-export-trained-kernel` | Train + export kernel backend with weights |
| `comp-p2p-worker` | P2P worker for distributed training (renamed from `eqprop-p2p-worker`) |

---

## 📦 Installation

```bash
uv sync --dev
```

### Quickstart: Forward-Forward vs Backprop in <2 Minutes

```bash
uv run scripts/quickstart.py
```

Expected output:
```
Backprop:        95% accuracy (3 epochs)
Forward-Forward: 90%+ accuracy (3 epochs)

Both biologically plausible and standard learning work!
```

**Why Forward-Forward?** FF converges in 3 epochs (like Backprop) vs EqProp's 20+. It uses local layer-wise objectives—no backward pass through the network, no weight transport—making it the most biologically plausible algorithm that's competitive with backprop on MNIST. The `scripts/quickstart.py` is the canonical entry point.

### Config-Driven Training

```bash
# Using preset YAML configs
comp run from-config --config configs/presets/eqprop_mnist.yaml
comp run from-config --config configs/presets/backprop_mnist.yaml
comp run from-config --config configs/presets/eqprop_routing_mnist.yaml
```

---

## 🏭 11 Native Factories — One-Line API

All factories are available via `from computronium import ...` and compose 5-D/6-D ontology systems in one call. Each has a matching YAML preset in `configs/presets/`.

| Factory | Axis Coordinate (S × G × D × M × C × U) | Preset YAML | Description |
|---------|-----------------------------------------|-------------|-------------|
| `create_backprop_mlp` | Digital × Feedforward × Instantaneous × Null × Backprop × Euclidean | `backprop_mnist.yaml` | Standard backprop MLP — the gold standard baseline |
| `create_eqprop_mlp` | Digital × Recurrent × EnergyMinimization × Null × ThermodynamicContrast × Euclidean | `eqprop_mnist.yaml` | Equilibrium Propagation: energy-based, local contrastive updates, no weight transport |
| `create_fa_mlp` | Digital × Feedforward × Instantaneous × Null × RandomProjections × Euclidean | `fa_mnist.yaml` | Feedback Alignment: fixed random feedback weights, avoids weight transport |
| `create_ff_mlp` | Digital × Feedforward × Instantaneous × Null × LocalGoodness × Euclidean | `ff_mnist.yaml` | Forward-Forward (Hinton): two forward passes (pos/neg), layer-local goodness objective, 3-epoch MNIST |
| `create_pepita_mlp` | Digital × Feedforward × Instantaneous × Null × LocalGoodness × Euclidean | `pepita_mnist.yaml` | PEPITA: FF variant with per-layer goodness classification, soft thresholding |
| `create_tp_mlp` | Digital × Feedforward × Instantaneous × Null × TargetInversion × Euclidean | `tp_mnist.yaml` | Target Propagation: learns inverse mappings layer-wise, target-based credit assignment |
| `create_pc_mlp` | Digital × Feedforward × PredictiveSettling × Null × ThermodynamicContrast × Euclidean | `pc_mnist.yaml` | Predictive Coding: hierarchical prediction error minimization, convergent dynamics |
| `create_hebbian_mlp` | Digital × Feedforward × Instantaneous × Null × TemporalTrace × Euclidean | `hebbian_mnist.yaml` | Hebbian/STDP: local correlation-based plasticity, biologically grounded |
| `create_snn_mlp` | Digital × Feedforward × SpikeIntegration × Null × TemporalTrace × Euclidean | `snn_mnist.yaml` | Spiking Neural Network: LIF neurons, spike-timing-dependent plasticity |
| `create_tile_mlp` | Digital × TileMesh × Instantaneous × Null × (varies) × Euclidean | `tile_mnist.yaml` | TileNet: modular tiled architecture, supports all credit assignments |
| `create_routing_mlp` | Digital × Recurrent × Instantaneous × RoutingPlasticity × Backprop × Euclidean | `routing_mnist.yaml` | **6-D Joint**: state-dependent gating, sparse pathway routing, dynamic compute |
| `create_fast_weight_mlp` | Digital × Recurrent × Instantaneous × FastWeightPlasticity × Backprop × Euclidean | `fast_weight_mnist.yaml` | **6-D Joint**: episode-local associative memory via fast-weight matrices |

### 5-D Factory Usage Examples

```python
from computronium import (
    create_backprop_mlp, create_eqprop_mlp, create_fa_mlp,
    create_ff_mlp, create_pepita_mlp, create_tp_mlp,
    create_pc_mlp, create_hebbian_mlp, create_snn_mlp,
    create_tile_mlp,
    SystemTrainer, SystemTrainerConfig,
)
from computronium.domains.factory import create_task

device = "cuda" if torch.cuda.is_available() else "cpu"
task = create_task("mnist", device=device, quick_mode=True)
task.setup()
train_loader, val_loader = task.get_dataloader("train"), task.get_dataloader("val")

# All factories share: input_dim, output_dim, hidden_dims (tuple), lr, device
input_dim, output_dim = 784, 10

# Backprop (baseline)
system = create_backprop_mlp(input_dim, (256, 128), output_dim, lr=0.001, device=device)

# Equilibrium Propagation (energy-based)
system = create_eqprop_mlp(input_dim, (512, 512, 512), output_dim,
                           beta=0.1, inference_steps=20, lr=0.001, device=device)

# Feedback Alignment (no weight transport)
system = create_fa_mlp(input_dim, (256, 128), output_dim, lr=0.001, device=device)

# Forward-Forward (local layer-wise, 3 epochs)
system = create_ff_mlp(input_dim, (256, 256), output_dim,
                       layer_lr=0.03, classifier_lr=0.01, threshold=2.0,
                       num_layers=2, device=device)

# PEPITA (FF variant)
system = create_pepita_mlp(input_dim, (256, 128), output_dim, lr=0.01, device=device)

# Target Propagation
system = create_tp_mlp(input_dim, (256,), output_dim, lr=0.001, device=device)

# Predictive Coding
system = create_pc_mlp(input_dim, (256, 256), output_dim, lr=0.001, device=device)

# Hebbian
system = create_hebbian_mlp(input_dim, (256,), output_dim, lr=0.001, device=device)

# Spiking
system = create_snn_mlp(input_dim, (256,), output_dim, lr=0.001, device=device)

# TileNet (modular tiles)
system = create_tile_mlp(input_dim, (256,), output_dim,
                         lr=0.001, neurons_per_tile=16, tiles_per_layer=2, device=device)

# Train with SystemTrainer
trainer = SystemTrainer(
    system=system,
    config=SystemTrainerConfig(max_epochs=3, batch_size=64, device=device, seed=42),
    train_data=train_loader,
    val_data=val_loader,
)
with trainer:
    history = trainer.fit()
```

### 6-D Joint Factories (Non-Null Plasticity)

```python
from computronium import create_routing_mlp, create_fast_weight_mlp

# RoutingPlasticity: state-dependent gating, sparse pathway routing
system = create_routing_mlp(
    input_dim, (256,), output_dim,
    lr=0.001, gate_dim=32, device=device
)

# FastWeightPlasticity: episode-local associative memory
system = create_fast_weight_mlp(
    input_dim, (256,), output_dim,
    lr=0.001, fast_weight_dim=128, device=device
)

# Same training API
trainer = SystemTrainer(
    system=system,
    config=SystemTrainerConfig(max_epochs=3, batch_size=64, device=device, seed=42),
    train_data=train_loader,
    val_data=val_loader,
)
with trainer:
    history = trainer.fit()
```

### YAML Preset Cross-Reference

Run any preset directly:
```bash
# 5-D factories
comp run from-config --config configs/presets/backprop_mnist.yaml
comp run from-config --config configs/presets/eqprop_mnist.yaml
comp run from-config --config configs/presets/fa_mnist.yaml
comp run from-config --config configs/presets/ff_mnist.yaml
comp run from-config --config configs/presets/pepita_mnist.yaml
comp run from-config --config configs/presets/tp_mnist.yaml
comp run from-config --config configs/presets/pc_mnist.yaml
comp run from-config --config configs/presets/hebbian_mnist.yaml
comp run from-config --config configs/presets/snn_mnist.yaml
comp run from-config --config configs/presets/tile_mnist.yaml

# 6-D joint factories
comp run from-config --config configs/presets/routing_mnist.yaml
comp run from-config --config configs/presets/fast_weight_mnist.yaml

# 6-D joint with EqProp dynamics
comp run from-config --config configs/presets/eqprop_routing_mnist.yaml
comp run from-config --config configs/presets/eqprop_fast_weight_mnist.yaml
```

---

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

### 1. Ontology Protocols (`computronium/core/ontology.py`)

Five `Protocol` classes with PEP 695 generics, frozen slotted config dataclasses, and reference implementations for every primitive — pure, composable infrastructure. See `computronium/core/ontology.py` for full Protocol definitions:

- `Substrate` — `forward_operator`, `weight_update_operator`
- `Geometry` — `forward`, `route`
- `StateDynamics` — `settle`
- `CreditAssignment` — `compute_pseudo_gradient`, `surrogate_objective`
- `ParameterUpdate` — `step`

### 2. Joint Architecture Protocols (`computronium/core/joint/`)

The joint dynamical system elevates the computational rule to a dynamical variable via the **CoupledTransition** protocol operating on `CompositeState`. Key types defined in `computronium/core/joint/state.py`, `computronium/core/joint/context.py`, `computronium/core/joint/transition.py`:

- **CompositeState** — joint intra-episode state `z_t = (x_t, ψ_t, σ_t)` with `activity`, `plastic`, `substrate` mappings
- **SystemContext** — immutable context: `theta`, `geometry`, `substrate_physics`, `registry`, `config` (6-axis)
- **StateVariable** — lifecycle metadata: `persistent`, `fast_plastic`, `substrate_owned`, `consolidatable`
- **StateRegistry** — registers variables, validates lifecycle, provides lifecycle groups
- **CoupledTransition** — linchpin protocol: `step(z, context) -> CompositeState` executing `z_{t+1} = F_θ(z_t; G, S)`
- **PlasticityPrimitive** — M-axis protocol: `step(psi, z, context) -> updated psi`
- **StabilityMonitor** — `spectral_radius`, `lyapunov_exponent` estimation

**Key Architectural Rule**: *Plasticity must not become a weight preprocessor.* Plasticity receives the full joint state `z = (x, ψ, σ)`, returns updated plastic state (not modified weights), and the joint transition remains `z_{t+1} = F_θ(z_t; G, S)`. Credit assignment receives the full trajectory `τ = [z_0, ..., z_T]`. Parameter update touches only `persistent`/`consolidatable` variables.

### 3. System & Trainers

| Component | Purpose |
|-----------|---------|
| `System[TS, TG, TD, TM, TC, TU]` | Generic 6-layer composition; invalid combos caught at type-check |
| `JointSystemTrainer` | **Single mathematical center**: executes `CoupledTransition.step` (joint transition `z_{t+1} = F_θ(z_t; G, S)`) → trajectory recording → CreditAssignment → ParameterUpdate.consolidate |
| `SystemTrainer` | Compatibility wrapper: instantiates `JointSystemTrainer` with `plasticity=NullPlasticity`, `ψ=∅`, `σ=σ₀`. The 5-D pipeline `Geometry.forward → StateDynamics.settle → ...` is the **restriction** `F_θ^Null = D_θ` of the joint dynamics. |
| `DistributedSystemTrainer` | In-process P2P coordination; shards along Geometry (TileMesh), federates at ParameterUpdate; CreditAssignment stays local |
| `ModelAdapter` | Strangler Fig adapter: projects legacy Registry models → 5-D System via metadata inference with per-family tolerance calibration |
| `Registry.to_system()` | One-call projection of any registered component |

**Zero-Extension Invariant**: `M=Null, ψ=const, σ=σ₀ ⟹ F_θ(z)|_x = D_θ(x)`. The 5-D system is formally a slice of the 6-D coupled dynamical system, not a parallel architecture. J1 test certifies this equivalence within numerical tolerance.

### 4. Factories (`computronium/core/system_trainer.py`)

Factory functions for composing systems from primitives or configs:

```python
from computronium.core.system_trainer import (
    compose_system,
    compose_system_from_configs,
    extract_config,
    compose_joint_system,
    compose_joint_system_from_configs,
    create_eqprop_system,
    create_backprop_system,
    create_fa_system,
    create_tile_system,
    create_predictive_coding_system,
)

# Config round-trip (L0 schema lock)
configs = extract_config(system)
system2 = compose_system_from_configs(configs)
assert system == system2  # identity verified

# Joint system composition
joint = compose_joint_system(
    substrate=DigitalSubstrate(),
    geometry=RecurrentGeometry(...),
    dynamics=EnergyMinimizationDynamics(...),
    plasticity=RoutingPlasticity(...),
    credit=ThermodynamicContrastCredit(),
    update=EuclideanUpdate(step_size=0.01),
)
```

### 5. Hardware Substrates ✅

| Substrate | Physics Model | Verification |
|-----------|---------------|--------------|
| `DigitalSubstrate` | CPU/GPU | — |
| `MemristiveSubstrate` | Conductance matrices, bounded precision, IR-drop noise | Gradient equivalence vs. digital; positive bounded conductance |
| `NeuromorphicSubstrate` | Async spike routing, strict sparsity, passivity | Property test: deterministic noise cancels in diff (‖na-nb‖ ≤ ‖a-b‖) |
| `OpticalSubstrate` | Phase/amplitude encoding, coherent interference | Phase wrapping to [-π, π]; no NaN/inf outputs |
| `QuantumSubstrate` | Parameterized unitary gates, parameter-shift rule | Parameter-shift matches finite-difference (cosine ≥ 0.999) |

---

## 🔬 Validation Framework: Property-Verified Hypercube

The framework enforces **correctness by construction** through a layered verification regime. The fast-CI gate validates the entire hypercube in seconds on CPU.

### ✅ Property Locks (L1–L7 + S/G/D/C/U/M Axes + J1–J7)

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
| **M-axis** | NullPlasticity Zero-Extension (`F_θ^Null = D_θ`); RoutingPlasticity gate entropy; FastWeightPlasticity decay bounds | Null ≡ 5-D; gate entropy ≥ 0; decay ∈ [0,1] |
| **J1** | NullPlasticity preserves 5-D dynamics (Zero-Extension Invariant) | `F_θ^Null = D_θ` within numerical tolerance |
| **J2** | Persistent θ not mutated during intra-episode steps | θ data_ptr() unchanged during CoupledTransition.step |
| **J3** | fast_plastic variables mutate only through plasticity projection | ψ updates only via PlasticityPrimitive.step |
| **J4** | substrate_owned variables respect substrate physics constraints | σ updates only via Substrate.forward_operator |
| **J5** | consolidatable variables promoted only at episode boundaries | consolidate() only called at episode end |
| **J6** | Cross-adapters preserve joint transition shape & registry semantics | Adapter output is valid CompositeState projection |
| **J7** | Trajectory records contain full z = (x, ψ, σ) | JointTrajectory has activity, plastic, substrate at each step |

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
# Property locks (fast CI gate) — 5-D
uv run pytest tests/property/test_ontology_locks.py -q

# Property locks (fast CI gate) — 6-D Joint Architecture
uv run pytest tests/property/joint/ -q

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

# Joint integration tests
uv run pytest tests/integration/joint/ -q

# Full suite
uv run pytest tests/ -q

# Type checking (strict)
uv run pyright .

# Formatting & linting
uv run ruff format --check . && uv run ruff check .
```

---

## 📐 Stability-Plasticity Trade-off Hypothesis

v1 relied on strict Lyapunov descent and global contraction. In the joint architecture, we recognize that global contraction is a *sufficient* condition for a unique fixed point, but not a *necessary* condition for useful computation. Systems can exhibit local contraction, multiple attractors, limit cycles, or metastable states.

### The Hypothesis

We formulate the research object as:

```
adaptive computation ↔ controlled departure from contraction
```

The hypothesis: **useful rule reconfiguration may require temporarily sacrificing some of the contraction/stability margin that a fixed computational attractor would maximize.**

### Monitoring the Frontier

The framework measures:

| Metric | Purpose |
|--------|---------|
| $\rho(J_F)$ | Spectral radius of joint Jacobian — stability margin |
| Local Lyapunov exponent | Sensitivity/divergence |
| Settling time | Dynamical latency |
| Basin stability | Robustness to perturbation |

**Cheap fast-mode proxies (for CI)**: step-norm ratio, finite-difference perturbation growth, settle iterations, activation variance, gate entropy.

**Deeper estimates (nightly/campaign)**: spectral radius via power iteration, Lyapunov via QR, basin via sampling.

### Resource Vector

The scientific claim is strictly about **resource scaling, locality, energy efficiency, and learnability** under constrained physical resources:

$$\mathcal{C} = (\text{compute}, \text{memory}, \text{energy}, \text{latency}, \text{plastic-state capacity})$$

The campaign asks whether adaptive-rule systems occupy a superior Pareto frontier in $\mathcal{C}$.

### Frontier Record

Defined in `computronium/core/campaign/frontier_record.py`:

```python
@dataclass(frozen=True, slots=True)
class FrontierRecord:
    coordinate: SystemCoordinate
    task_loss: float
    adaptation_time: int
    rho_jacobian: float
    lyapunov_local: float
    settling_time: float
    basin_stability: float
    resources: ResourceUsage
```

### 5-Level Benchmark Hierarchy

| Level | Question | Toy Task | Compare |
|-------|----------|----------|---------|
| **1: Adaptation Efficiency** | Does plasticity adapt faster than Null? | Switching distribution (Phase A: y=f_A(x), Phase B: y=f_B(x)) | Null vs FastWeight vs Routing |
| **2: Compute Efficiency** | Does routing reduce effective ops? | Mixture-of-experts (one route needed per input) | Active units, gate entropy, effective matmul |
| **3: Structural Robustness** | Can system recover after damage? | Zeroed weights, removed nodes, dead channels, noisy memristive | Null vs Routing vs SubstrateCoupled |
| **3.5: Algorithm Migration** | Can ψ switch strategy without θ update? | Task A₀: cumulative sum → Task A₁: last symbol | time(A₀→A₁), energy, ‖θ_after - θ_before‖ == 0 |
| **4: Z3 — Fixed Weights, Changing Algorithm** | Can frozen θ solve multiple tasks via ψ? | **Constraint**: θ frozen. Tasks: parity, last-symbol, threshold. **Operators**: Identity, Threshold, Accumulate, LastSymbol, Parity, SparseTopKRoute, SignFlip, Delay | Adaptation time, energy, operator diversity, parameter invariance |

**Z3 Parameter invariance must be exact**: `||θ_after - θ_before|| == 0`

Commands:
```bash
comp benchmark run --suite adaptation_efficiency
comp benchmark run --suite compute_efficiency
comp benchmark run --suite structural_robustness
comp benchmark run --suite algorithm_migration
comp benchmark run --suite z3_fixed_weights
```

---

## 🤖 Automated Research: Hypercube Campaigns

The 6-axis decomposition gives the **AutoScientist** a **structured search space** instead of a flat model list:

| Campaign Type | Fixed Axes | Varied Axis | Example Hypothesis |
|---------------|------------|-------------|-------------------|
| Substrate Ablation | G, D, M, C, U | S: Digital → Memristive/Optical/Quantum | At what IR-drop does EqProp parity break? |
| Epistemology Swap | S=Optical, G=TileMesh, D=EnergyMinimization | C: ThermodynamicContrast ↔ RandomProjectionsCredit | Does optical hardware favor FA (lower settling energy)? |
| Kinetics Discovery | S, G, D, C | U: Euclidean ↔ Riemannian ↔ Spectral ↔ Natural | Can Spectral constraints stabilize Memristive settling? |
| Plasticity Search | S, G, D, C, U | M: Null ↔ Routing ↔ FastWeight | Does routing reduce compute at stability margin? |
| Composite | S=Memristive, D=EnergyMinimization, M=Routing | U=SpectralConstrained | "IR-drop (S) + Routing (M) + Spectral (U) → stable settling (D)" |
| Stability-Plasticity Trade-off | S, G, D, C, U | M + ρ(J_F) constraint | Maximize adaptation s.t. ρ(J_F) ≈ 0.99 |

**Key AutoScientist capabilities:**
- 🧠 Chain-of-thought templates operating on ontology axes
- 📚 arXiv retrieval + semantic search for prior art
- 🔀 Counterfactual generator: "What if β schedule changed?"
- 📊 Knowledge Base meta-analysis: scaling laws, algorithm fingerprinting, failure manifold clustering, algorithm phylogeny
- 💾 Campaign persistence/resume (YAML+SQLite, git-like branching) — **includes joint state z, θ, ψ, σ**
- 👁️ Human-in-the-loop dashboard (NiceGUI, WebSocket live updates)
- 🖥️ Local LLM support (Ollama auto-pull, llama.cpp quantization, speculative decoding)
- ⚡ **Joint Kernel Cache**: Persisted compiled kernels for `CoupledTransition.step`, plasticity updates, stability estimators
- 🛡️ **Fault Tolerance**: Checkpoint-based recovery for multi-hour campaigns on spot instances

---

## 🧪 Flagship Experiments (Implemented)

### 5-D Experiments (Completed)

| Experiment | File | Purpose |
|------------|------|---------|
| TileNet Scaling Sweep | `experiments/tile_scaling.py` | Depth/width scaling on MNIST/CIFAR-10 across tile algorithms + backprop |
| EqProp Vision Parity | `experiments/eqprop_vision_parity.py` | EqProp variants on MNIST/Fashion-MNIST/CIFAR-10/SVHN |
| MEP Preset Tournament | `experiments/mep_tournament.py` | Factorized ablation: gradient×update×constraint×feedback with ANOVA + Sobol |
| FA Depth Scaling | `experiments/fa_depth_scaling.py` | Extreme depth, MNIST + synthetic parity |
| MoT Ablation | `experiments/mot_ablation.py` | Dense vs sparse tile routing (top-k, random, learned) |
| Cross-Domain Transfer | `experiments/cross_domain_transfer.py` | Vision→LM/RL/graph transfer, local vs global learning |
| Tile Algorithm Comparison | `experiments/tile_algorithm_comparison.py` | Fair comparison of PC/EP/FA/TP/Hebbian/SNN/Backprop on same substrate |

### 6-D Joint Architecture Experiments (In Progress)

| Experiment | File | Purpose |
|------------|------|---------|
| Adaptation Efficiency | `experiments/joint/adaptation_efficiency.py` | Does plasticity adapt faster to non-stationary shifts than Null under matched compute? |
| Compute Efficiency | `experiments/joint/compute_efficiency.py` | Does routing reduce effective operations (dynamic sparsity)? |
| Structural Robustness | `experiments/joint/structural_robustness.py` | Can joint system recover after topology/device damage via autonomous rerouting? |
| Algorithm Migration | `experiments/joint/algorithm_migration.py` | Can ψ switch strategy A₀→A₁ without changing θ? (Experiment 3.5) |
| Z3: Fixed Weights, Changing Algorithm | `experiments/joint/z3.py` | Frozen θ, multiple tasks via ψ-mediated rule selection (Experiment 4) |

---

## 🌐 Evaluation Domains

The framework supports **7 evaluation domains** with 60+ tasks/datasets, unified through a common task interface (`DomainTask` protocol). Each domain has dedicated data loaders, metrics, and task-specific configurations.

### 📊 Domain Overview

| Domain | Tasks | Example Datasets | Models | Key Metrics |
|--------|-------|------------------|--------|-------------|
| **Vision** | 11 | MNIST, CIFAR-10/100, SVHN, Digits, synthetic (XOR, spirals) | 25+ | Accuracy, Loss, Energy, FLOPs |
| **Language (LM)** | 4 | Tiny Shakespeare, WikiText-2, Penn Treebank, char n-gram | 12+ | Perplexity, BPC, Accuracy |
| **Reinforcement Learning (RL)** | 5 | CartPole, Pendulum, Acrobot, MountainCar, LunarLander | 8+ | Episode Return, Success Rate |
| **Graph** | 3 | Cora, CiteSeer, PubMed | 6+ | Node Classification Acc, F1 |
| **Tabular** | 5 | Breast Cancer, Iris, Wine, Diabetes, California Housing | 10+ | Accuracy, R², AUC |
| **Time Series** | 2 | Synthetic Forecast, ETT (planned) | 6+ | MSE, MAE |
| **Scientific** | 2 | Heat/Wave/Burgers PDE, Navier-Stokes (planned) | 5+ | Relative L2, Conservation |

---

### 🖼️ Vision Domain

**Tasks**: `mnist`, `fashion_mnist`, `kmnist`, `usps`, `cifar10`, `cifar100`, `svhn`, `digits`, `xor`, `spiral`, `circles`  
(image classification + synthetic boolean tasks)

**Quick Commands**

```bash
# Run vision benchmark (all models, all vision tasks)
comp lab benchmark --domain vision --quick

# Run specific model on MNIST
comp lab core-train --model eqprop_mlp --task mnist --epochs 10

# Cross-domain transfer: vision → LM
python experiments/cross_domain_transfer.py --source vision --target lm
```

---

### 📝 Language Modeling Domain

**Tasks**: `tiny_shakespeare` (char), `char_ngram`, `wikitext2`, `penn_treebank` (word-level)

**Key Experiments**

```bash
# EqProp vs Backprop on language modeling
python experiments/language_modeling_comparison.py --epochs 50

# Run LM benchmark
comp lab benchmark --domain lm --models backprop_transformer,eqprop_causal_transformer

# AutoScientist campaign on LM
comp scientist --campaign campaigns/lm_hypercube.yaml
```

---

### 🎮 Reinforcement Learning Domain

**Tasks**: `cartpole`, `pendulum`, `acrobot`, `mountain_car`, `lunar_lander` (Gymnasium classic control + Box2D)

**Key Experiments**

```bash
# RL benchmark across algorithms
comp lab benchmark --domain rl --quick

# EqProp on CartPole (energy-based policy)
comp lab core-train --model eqprop_ppo --task cartpole --epochs 100

# FA vs Backprop on continuous control
python experiments/fa_rl_comparison.py --env pendulum --seeds 10
```

---

### 🕸️ Graph Domain

**Tasks**: `cora`, `citeseer`, `pubmed` (Planetoid citation networks, node classification)

---

### 📋 Tabular Domain

**Tasks**: `breast_cancer`, `iris`, `wine` (classification), `diabetes`, `california_housing` (regression) — sklearn/UCI

**Models**: All MLP-based families (backprop, eqprop, fa, pepita, hebbian, tile) support tabular tasks.

---

### 📈 Time Series Domain

**Tasks**: `synthetic_forecast` (sin, AR, chaos), `ett_h1` (planned)

**Models**: RNN/LSTM/Transformer families across all credit assignments.

---

### 🔬 Scientific Domain

**Tasks**: `synthetic_physics` (Heat, Wave, Burgers PDEs), `navier_stokes` (planned)

**Models**: Physics-informed variants (PINO, DeepONet, FNO) adapted to computronium credit assignments.

---

## 🌐 Distributed Training & P2P

### Multi-GPU Training
PyTorch Lightning with DDP, FSDP, DeepSpeed. `TileShardedBackend` with NCCL `all_reduce_gradients`/`broadcast_params` scales TileNet beyond 1B parameters.

### P2P Coordinator System (gRPC + Kademlia)
Decentralized coordination at `computronium/p2p/`:
- 🔑 **Kademlia DHT** (`dht.py`): Peer discovery, KV storage, bootstrap nodes, async background operation. Integration test: 2-node connectivity + best-model propagation with score-based optimistic locking
- 🔗 **gRPC Service** (`proto/tile_mesh.proto`, `grpc_service.py`): `TileMeshService` with `ExecuteStep`, `BroadcastParams`, `AggregateGradients`
- 🏊 **Connection Pool** (`GRPCConnectionPool`): Peer lifecycle, health checks, retry/backoff
- 🔀 **DistributedSystemTrainer**: In-process multi-worker coordination; shards along TileGeometry, federates at ParameterUpdate
- 🛡️ **Fault Tolerance**: `DistributedTrainingError` captures lost workers, step, partial metrics on gRPC failure

CLI: `comp-p2p-worker` starts a worker node (renamed from `eqprop-p2p-worker` — the P2P layer is algorithm-agnostic).

```bash
# Start a P2P worker
comp-p2p-worker --bootstrap-ip 192.168.1.100 --task mnist --mode deep

# Run distributed TileNet training
comp run --config campaigns/distributed_tile.yaml
```

---

## 🚀 Deployment & Inference

### Model Export (`computronium/deployment.py`)
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

## 📊 Analysis & Visualization (`computronium/analysis/`)

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

## ⚡ Hardware Acceleration (`computronium/acceleration/`)

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
