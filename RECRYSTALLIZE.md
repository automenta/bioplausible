# RECRYSTALLIZATION — 5-D Ontology for bioplausible

## Vision: The 5-Layer Physico-Computational Stack

The `autonull/bioplausible` repository represents a monumental leap in biologically and physically plausible machine learning. However, its current ontology—while functionally exhaustive—suffers from **structural entanglement**. Hardware instantiations (`optical_looped_mlp`), mathematical algorithms (`eqprop_transformer`), and architectural patterns (`TileNet`) are flattened into a single, crowded `Model` registry. This conflation obscures the underlying physical laws and creates a combinatorial explosion of redundant code.

To achieve absolute **elegance** and **control**, we recrystallize the ontology into a mathematically pure, decoupled tensor product of fundamental primitives. By separating the physics, topology, and mathematics of learning into orthogonal axes, we transform the framework from a "library of models" into a **generative physico-computational engine**.

Every model in `bioplausible` maps uniquely to a coordinate in this 5-dimensional space:
```
System = Substrate ⊗ Geometry ⊗ StateDynamics ⊗ CreditAssignment ⊗ ParameterUpdate
```

### 1. Substrate (S) — The Physical State Space
Defines constraints on weights/activations: precision, noise, sparsity.
- **Digital:** Infinite precision, continuous time (mathematical ideal)
- **Memristive:** Conductance matrices, bounded precision, IR-drop noise
- **Neuromorphic:** Asynchronous spike routing, strict sparsity (Loihi/TrueNorth)
- **Photonic:** Phase/amplitude encoding, coherent interference
- **Quantum:** Parameterized unitary gates
- *Control:* Exposes `forward_operator` and `weight_update_operator` injecting physically accurate noise

### 2. Geometry (G) — Topology & Routing
Spatial arrangement of computational units and message-passing protocol.
- **Feedforward DAG:** MLPs, CNNs
- **Recurrent Attractor:** Hopfield, EqProp MLPs (symmetric/asymmetric lattices)
- **Asynchronous Tile Mesh:** TileNet — modular independent tiles with local boundaries
- **Neuromorphic Fabric:** Arbitrary node-edge topologies (FabricPC)
- **3D Spatial Lattice:** Voxels embedded in physical space (neural_cube)

### 3. StateDynamics (D) — Forward Evolution & Settling
How activations evolve to process information (the forward pass).
- **Energy Minimization:** Relax toward local minimum of E(x) (Equilibrium Propagation)
- **Predictive Settling:** Hierarchical prediction-error minimization (Predictive Coding)
- **Spike Integration:** Membrane potential accumulation and thresholding (LIF, Izhikevich)
- **Instantaneous Pass:** Pure feedforward mapping (Forward-Forward, standard Backprop)

### 4. CreditAssignment (C) — Error Routing & Pseudo-Gradients
How the network computes the direction of learning using locally available signals.
- **Thermodynamic Contrast:** Difference between nudged and free phase states (EqProp)
- **Random Projections:** Fixed/adaptive matrices projecting errors backward (FA, DFA)
- **Local Goodness:** Layer-local contrastive objectives (Forward-Forward, PEPITA)
- **Temporal Trace:** Spike-timing-dependent correlation (STDP)
- **Target Inversion:** Propagating local targets instead of gradients (Target Propagation)

### 5. ParameterUpdate (U) — Optimization Rule
How computed pseudo-gradients translate into physical weight changes ΔW.
- **Riemannian Orthogonal (Muon):** Enforces orthogonality
- **Spectral Constrained:** Limits Lipschitz constant for stability
- **Natural Gradient (Fisher):** Updates in information geometry space (MEP)
- **Elastic Consolidation (EWC):** Protects past knowledge for continual learning
- **Euclidean (SGD/Adam):** Standard flat-space updates

---

## Algebraic Composition (API Elegance)

```python
from bioplausible.core.ontology import (
    System, DigitalSubstrate, FeedforwardGeometry,
    InstantaneousDynamics, BackpropCredit, EuclideanUpdate,
    GeometryConfig
)

# Old Way: Hardcoded, entangled, brittle
model = Registry.get("optical_looped_mlp")

# New Way: Pure, composable, mathematically rigorous
system = System(
    substrate=DigitalSubstrate(),
    geometry=FeedforwardGeometry(GeometryConfig(input_dim=784, output_dim=10, hidden_dims=(256, 128))),
    dynamics=InstantaneousDynamics(),
    credit=BackpropCredit(),
    update=EuclideanUpdate(step_size=0.01)
)
```

`optical_looped_mlp` is not a unique entity — it's merely the coordinate:
`(Photonic ⊗ RecurrentAttractor ⊗ EnergyMinimization ⊗ ThermodynamicContrast ⊗ Euclidean)`.

---

## Thermodynamic Invariant (Energy)

Energy is the **fundamental invariant** binding the ontology. The Energy Function E bridges Geometry and StateDynamics:
- Symmetric Topology + EnergyMinimization → guaranteed fixed-point convergence (Hopfield/EqProp)
- Directed Topology → requires Control-Lyapunov formulation for stability

Elevating Energy to a first-class object enables mathematical stability proofs for novel combinations before implementation.

---

## AutoScientist: Hypercube Search Space

The 5-D ontology gives the AutoScientist a **Hypercube Search Space** instead of a flat model list:

1. **Substrate Ablation:** Fix G, D, C. Swap Digital → Memristive. At what noise level does parity with backprop break?
2. **Epistemology Swaps:** Fix S=Optical, G=TileNet, D=EnergyMinimization. Swap ThermodynamicContrast (EqProp) ↔ RandomProjections (FA). Does optical hardware favor FA due to settling energy cost?
3. **Kinetics Discovery:** Mix Orthogonal, Natural, Spectral updates without touching credit assignment.

Chain-of-thought templates now operate on *ontology axes*, enabling hypotheses like: *"Because Memristive Crossbars suffer IR-drop (S), applying Spectral Constraints (U) will stabilize EnergyMinimization (D) settling."*

---

## Pragmatic Execution: Strangler Fig, Not Demolition

| Phase | Action | Risk | Status |
|-------|--------|------|--------|
| **1** | Define boundary: 5 Protocols + `System` in `ontology.py` | Zero (additive) | ✅ Done |
| **2** | Wrap, don't rewrite: `ModelAdapter` projects legacy models to 5-D | Zero (Registry intact) | ✅ Done |
| **3** | Migrate incrementally: new code native, old on contact | Low | 🔄 In progress |
| **4** | P2P distribution along seams (not carving monoliths) | Medium | 📋 Planned |

**Anti-Goals (Explicitly NOT Doing):**
- ❌ Big-bang rewrite of 111 components
- ❌ P2P types in Protocols (keep `Tensor` in/out transport-agnostic)
- ❌ Registry API changes (`Registry.get()`, `biopl-*` CLIs stable)
- ❌ Backwards compatibility layer (Protocols *are* the new interface)

---

## Phase Status

| Phase | Scope | Status | Verification |
|-------|-------|--------|--------------|
| **1** | 5 Protocol definitions, configs, `SystemState`, `System` generic | ✅ Complete | Pyright strict, 51 unit tests |
| **2** | Reference implementations (all 5 layers), `SystemTrainer`, factories | ✅ Complete | 18 integration tests (gradient equivalence) |
| **2** | `ModelAdapter`, `Registry.to_system()`, TileGeometry, Hardware Substrates | ✅ Complete | 97+ tests, Ruff clean |
| **2** | PredictiveSettlingDynamics, DistributedSystemTrainer (in-process) | ✅ Complete | 12 energy proofs, 16 property locks |
| **2** | AutoScientist Hypercube Search, Formal Energy Proofs, L1-L7 Locks | ✅ Complete | CORRECTNESS_LOCK.md suite green |
| **3** | P2P RPC layer (gRPC + Kademlia) | 📋 Sprint 2 | — |
| **3** | Control-Lyapunov proofs for directed topologies | 📋 Sprint 3 | — |
| **3** | FA structured init (orthogonal + feedback_scale) | 📋 Sprint 3 | — |
| **4** | ModelAdapter.validate(), legacy migration on contact | 📋 Sprint 4 | — |

**Last verified:** 2026-08-20 — all 97 core/integration/property tests pass, Pyright strict clean, Ruff format clean

---

## Completed Components

| Component | Location | Tests |
|-----------|----------|-------|
| 5 Protocols (`Substrate`, `Geometry`, `StateDynamics`, `CreditAssignment`, `ParameterUpdate`) | `bioplausible/core/ontology.py:216-478` | 51 unit |
| Config dataclasses (frozen, slotted) | `bioplausible/core/ontology.py:78-171` | — |
| `SystemState` mutable container | `bioplausible/core/ontology.py:178-209` | — |
| `System[TS, TG, TD, TC, TU]` with PEP 695 generics | `bioplausible/core/ontology.py:460-546` | — |
| Reference implementations (all 5 layers) | `bioplausible/core/ontology.py:550-1600` | 18 integration |
| Factories (`compose_system`, `create_eqprop_system`, `create_backprop_system`, `create_fa_system`) | `bioplausible/core/system_trainer.py:211-442` | — |
| `SystemTrainer` (5-layer pipeline) | `bioplausible/core/system_trainer.py:75-208` | — |
| `ModelAdapter` (legacy → 5-D projection) | `bioplausible/core/ontology.py:814-1113` | L1 parity |
| `Registry.to_system()` projection | `bioplausible/core/registry.py:566-642` | L6 totality |
| `TileGeometry` (complete TileNet topology) | `bioplausible/core/ontology.py:764-1000` | L7 seam |
| Hardware Substrates (Memristive, Neuromorphic, Optical, Quantum) | `bioplausible/core/ontology.py:1520-1830` | 18 integration |
| `PredictiveSettlingDynamics` (full Predictive Coding) | `bioplausible/core/ontology.py:1914-2110` | L4 Lyapunov |
| `DistributedSystemTrainer` (P2P coordination, in-process) | `bioplausible/core/distributed_trainer.py` | L7 seam |
| `Registry.query_ontology()` + `propose_hypercube_ablation()` | `bioplausible/core/registry.py:646-780`, `bioplausible/autoscientist/proposer.py` | — |
| Formal Energy Proofs | `tests/integration/test_energy_invariants.py` | 12 proofs |
| Ontology Property Locks (L1-L7) | `tests/property/test_ontology_locks.py`, `tests/property/_support.py` | 16 property |

---

## Verified Compositions (End-to-End)

| System | Substrate | Geometry | Dynamics | Credit | Update |
|--------|-----------|----------|----------|--------|--------|
| EqProp | Digital | Recurrent | EnergyMinimization | ThermodynamicContrast | Euclidean |
| Backprop | Digital | Feedforward | Instantaneous | BackpropCredit | Euclidean |
| Feedback Alignment | Digital | Feedforward | Instantaneous | RandomProjectionsCredit | Euclidean |
| Predictive Coding | Digital | Feedforward | PredictiveSettling | ThermodynamicContrast | Euclidean |
| TileNet | Digital | TileMesh | Instantaneous | BackpropCredit | Euclidean |
| Memristive EqProp | Memristive | Recurrent | EnergyMinimization | ThermodynamicContrast | Euclidean |
| Optical FA | Optical | Feedforward | Instantaneous | RandomProjectionsCredit | Euclidean |

Verified via `tests/integration/test_gradient_equivalence.py::TestOntologyLayerEquivalence`.

---

## Key Technical Decisions (Validated)

1. **Protocol-based structural typing** over ABCs — zero-cost abstraction, duck-typing
2. **PEP 695 generics** (`System[TS, TG, TD, TC, TU]`) — invalid compositions caught at type-check
3. **Frozen slotted dataclasses** for all configs — immutability by default, memory efficient
4. **Parameter name consistency** across all 5 layer protocols — matches `SystemTrainer` pipeline exactly
5. **Strangler Fig adapter pattern** — `ModelAdapter` infers 5 layers from metadata (compute_profile, family, gradient_method, locality_level, tags)

---

## Next Sprint Priorities (Ranked by Impact)

| Sprint | Priority | Task | Effort | Blocking | Solution |
|--------|----------|------|--------|----------|----------|
| **2** | **P1** | **P2P RPC Layer (gRPC + Kademlia)** | 3-5 days | Real distributed training | Replace `_fetch_remote_activation`/`_sync_boundary_tiles` with async gRPC; add `kademlia` bootstrap for `DHTRouter` |
| **3** | **P2** | **RandomProjectionsCredit structured init** | 1 day | FA production use | Add `orthogonal_init: bool`, `feedback_scale: float` to `CreditAssignmentConfig`; use `torch.nn.init.orthogonal_` |
| **3** | **P2** | **Control-Lyapunov formal proof** | 2-3 days | Theoretical completeness | Add Lyapunov candidate `V = Σ ‖eₗ‖²` to `test_energy_invariants.py`; prove `dV/dt ≤ 0` for directed topologies; require `dynamics.track_free_energy_per_iter = True` |
| **4** | **P3** | **ModelAdapter.validate()** | 1-2 days | AutoScientist accuracy | Run forward/backward pass comparing metrics with legacy; improve inference priority: metadata → `model.config` → `model.family` → heuristics → defaults |
| **1** | **P3** | **Test file lint cleanup** | 0.5 day | CI hygiene | `ruff check --fix tests/property/test_ontology_locks.py`; replace raw `assert` with pytest assertions; add `# noqa: S101` |

### Sprint Execution Order

```
Sprint 1 (P0 + P1 quick wins):  ✅ COMPLETED
  ├─ P0: ModelAdapter None fallback          (1 day) ✅
  ├─ P1: TileGeometry shape validation       (0.5 day) ✅
  └─ P3: Test lint cleanup                   (0.5 day) ← remaining

Sprint 2 (P1 infrastructure):
  └─ P1: P2P RPC layer (gRPC + kademlia)    (3-5 days)

Sprint 3 (P2 theoretical):
  ├─ P2: FA orthogonal init + feedback_scale (1 day)
  └─ P2: Control-Lyapunov proof + tracking   (2-3 days)

Sprint 4 (P3 polish):
  └─ P3: ModelAdapter.validate()             (1-2 days)
```

---

## Migration Path for Existing Models (Phase 3)

Migrate natively to Protocols when touching:

| Model Family | Target Layers | Effort |
|--------------|---------------|--------|
| `eqprop_*` | S=Digital, G=Recurrent, D=EnergyMinimization, C=ThermodynamicContrast, U=Euclidean | Low |
| `*_fa` / `*_dfa` | C=RandomProjections, D=Instantaneous | Low |
| `*_ff` / `pepita` | C=LocalGoodness, D=Instantaneous | Low |
| `spiking_*` / `*_stdp` | C=TemporalTrace, D=SpikeIntegration | Medium |
| `*_tp` / `*_target_prop` | C=TargetInversion, D=Instantaneous | Medium |
| `*_tile_*` | G=TileMesh, others vary | High |
| `optical_*`, `crossbar_*`, `quantum_*` | S=Optical/Memristive/Quantum, others as base | Medium |

---

## Implementation Notes for Maintainers

### Protocol Signatures (must remain consistent)

```python
# All protocols require a `config` attribute
Substrate.config: SubstrateConfig
Geometry.config: GeometryConfig
StateDynamics.config: StateDynamicsConfig
CreditAssignment.config: CreditAssignmentConfig
ParameterUpdate.config: ParameterUpdateConfig

# Pipeline method signatures (must match for System.train_step)
Geometry.forward(x: Tensor, substrate: Substrate) -> Tensor
Geometry.route(activations: Tensor) -> Tensor
StateDynamics.settle(state, geometry, substrate, target) -> SystemState
CreditAssignment.compute_pseudo_gradient(free, nudged, loss, geometry) -> list[Tensor]
ParameterUpdate.step(params, pseudo_grads, geometry) -> dict[str, Tensor]
```

### Adding New Components
- **Substrate**: Implement all 5 abstract methods. See `MemristiveSubstrate` (IR-drop), `OpticalSubstrate` (phase noise), `QuantumSubstrate` (parameter shift)
- **CreditAssignment**: Implement `compute_pseudo_gradient` returning list matching `Geometry.params` order. `ThermodynamicContrast` uses contrastive Hebbian; `RandomProjectionsCredit` uses fixed feedback matrices
- **Distributed Training**: `DistributedSystemTrainer` shards along Geometry (tile mesh) and federates at ParameterUpdate. CreditAssignment stays local. For new topologies, implement `_distributed_forward` and `_distributed_settle`

---

## Extracted Specifications

- **CORRECTNESS_LOCK.md** → `docs/CORRECTNESS_LOCK.md` (fast-CI property suite L1-L7)
- **Architecture Decisions** → `docs/architecture/decisions.md` (technical rationale)
- **API Reference** → `docs/api/ontology.md` (protocol signatures, factories)

---

## Acceptance Checklist (run in order)

```bash
# 1. Property locks (fast CI gate)
uv run pytest tests/property/test_ontology_locks.py -q

# 2. Core + integration suites
uv run pytest tests/unit/core/test_ontology.py tests/integration/test_gradient_equivalence.py tests/integration/test_energy_invariants.py -q

# 3. Type checking
uv run pyright

# 4. Linting & formatting
uv run ruff format --check . && uv run ruff check .

# 5. Full suite
uv run pytest tests/ -q

# 6. Wall-clock budget check (record in PR)
# GPU ≤ 5 min, CPU ≤ 10 min for lock suite
```

---

*This document is the single source of truth for recrystallization status. Deep specs linked out. Update on every phase transition.*