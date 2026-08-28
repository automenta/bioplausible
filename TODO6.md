# TODO6.md — Computronium Stability Integration & Modularization

> **Scope:** Integrate `computronium-stability` as a first-class library (mirroring `computronium/nn` pattern), clean up fractured `libraries/computronium_stability/`, modularize core types for accessibility, and prepare infrastructure for RESEARCH3 ambitions. **Excludes substrate-specific work** (photonic, memristive, etc.) — CPU/GPU/universal only.

---

## Critical Path Analysis

```
CRITICAL PATH (longest dependency chain):
0.0 CL Re-verify ──────────────────────► (independent, parallel)
0.1 Cleanup ──► 0.2 Move core/stability ──► 1.1-1.6 Stability Lib ──► 2.2 State Types ──► 2.3 Configs ──► 3.1 Fairness ──► 3.2 Campaign ──► 4.1-4.3 Execution
                    │                          │
                    └────► 1.1 (ResourceUsage) ┘
                                                                 
PARALLEL WORKSTREAMS (can overlap):
├─ 0.0 CL Re-verification (blocks design decisions only)
├─ 1.5 Tests (starts after 1.3)
├─ 2.4 Joint Facade (after 2.2, 2.3)
├─ 3.3 L2/𝒞 wiring (after 1.1 ResourceUsage)
├─ 3.4 Algorithm Migration (after 1.1)
├─ 3.5 PR-8 Export (independent)
└─ Documentation / Migration Guide (continuous)
```

**Key Insight:** Phase 0.0 (CL re-verification) is **not** on the critical path for library delivery — it only blocks *design decisions that depend on CL results*. Phase 1 (stability library) is the true critical path.

---

## Status Snapshot

| Track | State |
|---|---|
| `computronium/nn` (CP-C wrapper) | ✅ **COMPLETE** — 26 tests, ruff/pyright clean |
| `libraries/computronium_stability` | 🟡 **FRACTURED** — `.venv/`, `build/`, duplicate source in `core/stability/` |
| Phase 3.6 Audits | ✅ **COMPLETE** — 7 audits pass, 34 regression tests |
| Phase 4 (Regime Discovery) | 🟢 **UNBLOCKED** — Awaits execution |
| Phase 5 (Family-Coverage) | 🟢 **UNBLOCKED** — Awaits coordinate lock |
| Phase 6 (Frontier Cert) | 🟢 **UNBLOCKED** — Awaits flagship coordinate |
| **Phase 2 CL Re-verification** | 🔴 **REQUIRED** — Post-3.6 fixes may have changed behavior; must re-run before design deps |

---

## Phase 0 — Prerequisites & Cleanup (Week 1)

### 0.0 Phase 2 CL Re-verification (Parallel, Non-Blocking for Lib)
**Run immediately in parallel with Phase 1.** Fresh E-1 registration → `scripts/verify_capacity_limited_cl.py` (6 arms, hidden=32, 2 epochs, 5 seeds) → compare vs Session 28 baseline. **Exit:** Null holds → `DECISIONS.md`; signal → new E-1 for full run. **No design deps** on CL outcome until this passes.

### 0.1 Remove Fractured `libraries/computronium_stability/`
- `rm -rf libraries/computronium_stability/.venv/ build/ .pytest_cache/ .ruff_cache/`
- **Retain temporarily:** `pyproject.toml`, `README.md`, `computronium_stability/` (source), `tests/`
- **After Phase 1:** `rm -rf libraries/` — publishing via root `pyproject.toml`

### 0.2 Move `computronium/core/stability/` → `computronium/stability/`
- **Single canonical location** — `core/stability/` becomes deprecated shim (one release) or removed
- Update all internal imports (8+ files, see import audit below)

### 0.3 Standalone Test Suite for Published API
- New: `tests/unit/core/test_stability_standalone.py`
- Imports: `from computronium_stability import attach, StabilityGuard, ...`
- Mirrors `tests/unit/nn/test_computronium_linear.py` pattern
- Validates wheel works identically to internal usage

### 0.4 Publishing Config (Root `pyproject.toml`)
```toml
[tool.setuptools.package-dir]
computronium_stability = "computronium/stability"

[project.optional-dependencies]
stability = []  # zero deps
```
- Verify: `uv build` → `pip install dist/*.whl` → `import computronium_stability` works

---

## Phase 1 — Stability Library at `computronium/stability/` (Weeks 1–2)

**Unification Principle:** `computronium/stability/` = **single canonical implementation** (not wrapper). Public API = internal implementation.

### 1.1 Consolidated Module Structure
```
computronium/
  stability/
    __init__.py           # Public exports (single source of truth)
    guard.py              # StabilityGuard, attach, GuardHandle, calibrate_threshold, quantify_proxy_disagreement, measure_guard_overhead
    spectral_radius.py    # SpectralRadiusEstimator, estimate_spectral_radius, estimate_spectral_radius_full_jacobian
    lyapunov.py           # LyapunovEstimator, estimate_lyapunov_exponent, estimate_lyapunov_spectrum
    settling.py           # SettlingMonitor, measure_settling_time, measure_settling_time_full_state
    basin.py              # BasinStabilityEstimator, estimate_basin_stability, estimate_basin_stability_multistart
    frontier.py           # FrontierRecord, FrontierAggregator (from core.stability.frontier)
    config.py             # Config dataclasses + factories (PEP 695, from_spec/to_spec)
    resources.py          # ResourceUsage (moved from core.profiling)
```

### 1.2 Public API Exports (`__init__.py`)
```python
# Guard API (primary)
from .guard import attach, StabilityGuard, StabilityVerdict, GuardDecision, GuardHandle, DEFAULT_TAU, calibrate_threshold, quantify_proxy_disagreement, measure_guard_overhead
# Estimators
from .spectral_radius import SpectralRadiusEstimator, estimate_spectral_radius, estimate_spectral_radius_full_jacobian
from .lyapunov import LyapunovEstimator, estimate_lyapunov_exponent, estimate_lyapunov_spectrum
from .settling import SettlingMonitor, measure_settling_time, measure_settling_time_full_state
from .basin import BasinStabilityEstimator, estimate_basin_stability, estimate_basin_stability_multistart
# Frontier
from .frontier import FrontierRecord, FrontierAggregator
# Resources
from .resources import ResourceUsage
# Config + Factories
from .config import (SpectralRadiusConfig, LyapunovConfig, SettlingConfig, BasinConfig, GuardConfig,
                     create_spectral_radius_estimator, create_lyapunov_estimator, create_settling_monitor, create_basin_estimator, create_guard)
# Type aliases
from .guard import StepState, TransitionFn, StatisticKind
```

### 1.3 Guard API (`guard.py`) — Consolidate Two Implementations
Merge `core.stability.guard.StabilityGuard` + `libraries/computronium_stability/guard.py`:
- Single `StabilityGuard` class supporting both `CompositeState` (internal) and `dict[str, Tensor]` (external) via `_extract_activity`
- `attach(model, threshold=1.029, statistic="windowed_growth", window=10, transition_fn=None)` → `GuardHandle` with `.check(state, step)` / `.detach()`
- `DEFAULT_TAU = 1.029` calibrated on 16 settling coordinates (FKR 0%, windowed_growth=1.000)

### 1.4 Config Dataclasses (`config.py`) — PEP 695, Frozen, Slotted
```python
@dataclass(frozen=True, slots=True)
class SpectralRadiusConfig:
    num_iterations: int = 20
    perturbation_scale: float = 1e-4
    activity_key: str = "x"
    fast_mode: bool = False

@dataclass(frozen=True, slots=True)
class LyapunovConfig:
    num_steps: int = 50
    perturbation_scale: float = 1e-6
    activity_key: str = "x"
    renormalize_interval: int = 1
    fast_mode: bool = False

@dataclass(frozen=True, slots=True)
class SettlingConfig:
    tolerance: float = 1e-4
    max_steps: int = 1000
    activity_key: str = "x"
    norm_type: str = "relative"
    record_trajectory: bool = False

@dataclass(frozen=True, slots=True)
class BasinConfig:
    num_samples: int = 100
    perturbation_radius: float = 1.0
    max_steps: int = 200
    tolerance: float = 1e-3
    activity_key: str = "x"
    distance_metric: str = "euclidean"
    fast_mode: bool = False

@dataclass(frozen=True, slots=True)
class GuardConfig:
    threshold: float = 1.029
    statistic: Literal["fast_proxy", "windowed_growth"] = "windowed_growth"
    window: int = 10
    estimator_config: SpectralRadiusConfig = field(default_factory=SpectralRadiusConfig)
```
- **All configs:** `to_spec()` / `from_spec(cls, spec)` for YAML/JSON round-trip
- **Factories:** `create_spectral_radius_estimator(config)`, etc. — single entry points

### 1.5 Resources (`resources.py`) — Universal Currency
Move `ResourceUsage` from `core/profiling.py` → here. Keep profiling utilities (`count_flops`, `get_gpu_memory_mb`, `measure_suite_resources`, `EnergyTracker`, `analyze_joint_system`) in `core/profiling.py` — they are *measurement tools*, not the resource vector.

### 1.6 Tests: `tests/unit/stability/test_stability_api.py` (~25 tests)
Mirror `tests/unit/nn/test_computronium_linear.py`:
- `TestStabilityGuardAPI` — attach, check, kill/pass on contractive/expansive
- `TestSpectralRadiusEstimator` — fast/full modes, identity/contractive/expansive
- `TestLyapunovEstimator` — neg/pos exponents, spectrum
- `TestSettlingMonitor` — convergence, max_steps, trajectory, fast_proxy
- `TestBasinStabilityEstimator` — stable/unstable attractors, multistart
- `TestIntegration` — guard kills divergent, passes 16 healthy coordinates
- `TestConfigRoundtrip` — to_spec/from_spec all configs
- `TestDeviceManagement` — CPU/CUDA consistency

### 1.7 CI & Quality Gates
- `ruff format --check . && ruff check . && pyright . && pytest --cov` green
- Coverage ≥85% for `computronium/stability/`
- Deprecation shim in `core/stability/__init__.py` (optional, one release):
  ```python
  import warnings
  warnings.warn("Use computronium.stability", DeprecationWarning)
  from computronium.stability import *
  ```

### Import Updates Required (Phase 1.1 moves)
| From | To | Files |
|---|---|---|
| `computronium.core.stability` | `computronium.stability` | 8+ files (campaign, continual, profiling, stability modules) |
| `computronium.core.profiling import ResourceUsage` | `computronium.stability import ResourceUsage` | 6 files (memory_wall, campaign, profiling, stability) |

---

## Phase 2 — Modularization for Accessibility (Weeks 2–3)

### 2.1 State Types → `computronium/state/` (After Phase 1.1)
```
computronium/
  state/
    __init__.py          # CompositeState, SystemContext, StateRegistry, StateVariable, TransitionFn protocol
    composite.py         # CompositeState (activity/plastic/substrate)
    context.py           # SystemContext (6-D config bundle)
    registry.py          # StateRegistry, StateVariable
    transitions.py       # NullPlasticity, PlasticityConfig, TransitionFn protocol
```
- **Benefit:** External users import state types without joint system internals
- **Protocol:** `TransitionFn` in `transitions.py` for duck-typing
- **Update imports** in: stability modules, campaign, pipeline, plasticity, joint, profiling

### 2.2 Ontology Configs → `computronium/config/` (After Phase 1.1)
Extract from `core/ontology.py` (single 1500-line file) to per-axis modules:
```
computronium/
  config/
    __init__.py          # Re-exports all config classes + factory methods
    substrate.py         # SubstrateConfig, SubstrateType, DigitalSubstrateConfig, etc.
    geometry.py          # GeometryConfig (feedforward, recurrent, tile_mesh)
    dynamics.py          # StateDynamicsConfig (energy_minimization, predictive_settling, etc.)
    plasticity.py        # PlasticityConfig (null, routing, fast_weights, rule_state)
    credit.py            # CreditAssignmentConfig (thermodynamic_contrast, random_projections, etc.)
    update.py            # ParameterUpdateConfig (euclidean, riemannian_orthogonal, etc.)
```
- **Single import:** `from computronium.config import *` gets all
- **Factory pattern preserved:** `GeometryConfig.feedforward(...)`, `PlasticityConfig.fast_weights(...)`
- **Cross-axis validation** (`SystemConfig.validate()`) moves to `config/__init__.py` or stays in `core/ontology.py` as internal

### 2.3 Joint System Facade (`computronium/core/joint/__init__.py`)
Export only the composition API:
```python
from .context import SystemContext
from .state import CompositeState, StateRegistry, StateVariable
from .transition import compose_joint_system, compose_joint_system_from_configs
from .pipeline import run_train_step, run_forward, JointSystem
```
- **Hide internals:** `trajectory.py`, `consolidation.py` behind facade
- **Deprecation path** for direct imports from submodules

---

## Phase 3 — RESEARCH3 Infrastructure (Weeks 3–4)

### 3.1 PR-6 Fairness Contract → Code (`computronium/eval/fairness.py`)
```python
@dataclass(frozen=True, slots=True)
class FairnessContract:
    gpu_hours_per_rule: float
    seeds: int = 5
    early_stopping: str = "best_val"  # or "last"
    data_splits: dict[str, float] = field(default_factory=lambda: {"train": 0.8, "val": 0.1, "test": 0.1})
    
def validate_fairness(contract: FairnessContract, results: list[dict]) -> bool
class BenchmarkRunner:  # base class enforcing contract
    ...
```
- All benchmark runners (Phases 3, 4, 5, 6) inherit `BenchmarkRunner`

### 3.2 PR-9 Campaign Stack Hardening
- **DB schema freeze** + migrations (alembic or custom)
- **ProposalObjective** enum: `ACCURACY`, `STABILITY`, `ENERGY`, `LATENCY`, `PLASTICITY_CAPACITY`
- **Replication gate:** auto-verify ≥5 seeds + ≥2 task families before promoting discovery
- **Counterfactual attribution:** integrate `analysis/counterfactual.py` into evaluation
- **Deliverable:** `CampaignStack.run_campaign(coordinate, objective, max_wall_hours)`

### 3.3 L2 Effective-FLOPs → 𝒞 Vector (After Phase 1.1 ResourceUsage)
- Add `effective_flops: float` field to `ResourceUsage`
- Wire `compute_efficiency.py` gate-entropy-aware route counting → `ResourceUsage.effective_flops`
- Deprecate raw FLOP counting in frontier records
- **Sanctioned feed** per RESEARCH3 L2

### 3.4 Algorithm Migration (L3.5) → First-Class Benchmark
- Promote `experiments/joint/algorithm_migration.py` → `computronium/benchmarks/algorithm_migration.py`
- Cheapest ψ-switching validation (Δθ=0 audit, two-strategy swap)
- **CI smoke test:** <30s, runs on every commit

### 3.5 PR-8 Export Pipeline Reuse
- Single `export_model(model, formats=["onnx", "ternary", "int8"])` in `computronium/deployment.py`
- Wire into Phase 3 memory-wall artifacts + Phase 5 benchmark export

---

## Phase 4 — Phase 4/5/6 Execution Prep (Week 4+)

### 4.1 Phase 4 — Regime Discovery
- **4.1 Prior-Art Gate:** Literature check (mixed credit, hypernetworks, MoE) → `DECISIONS.md`
- **4.2 Bandit Router:** Generalize `RoutingPlasticity` to route **learning rules** per layer; reward = energy descent rate + windowed growth + validation improvement
- **4.3 Memristive IR-Drop** (simulation): Sweep IR-drop on `MemristiveSubstrate`; test `SpectralConstrainedUpdate` + `EnergyMinimization` + `SubstrateCoupledPlasticity`
- **4.4 Photonic Epistemology** (simulation): `OpticalSubstrate` × credit families; test settling-energy preference
- **4.5 Campaign Hygiene:** Enforce `simulated/estimated/measured` labeling; `ProposalObjective` non-accuracy ranking

### 4.2 Phase 5 — Family-Coverage Benchmark
- **5.1 Coordinate Lock:** Lock by rule-family coverage (every credit×update + substrate variants); ≥30 coords; freeze in `DECISIONS.md`
- **5.2 Resource-Vector Runner:** Full `ResourceUsage` per coord/seed; equal GPU-hours (PR-6); ≥5 seeds paired
- **5.3 Dynamical Phylogeny:** Cluster by settling time, windowed growth, gate entropy, ρ via `analysis/genealogy.py`
- **5.4 Full Run:** Capability matrix, Pareto overlays, per-rule stability audits, failure modes

### 4.3 Phase 6 — Frontier Certification
- **6.1 M-Axis Frontier:** Pin S/G/D/C/U at flagship; sweep M ∈ {Null, Routing, FastWeight, RuleState}; `AutoScientistCampaign` with guard live, checkpoint/resume
- **6.2 Goldilocks Map:** ρ(J_F) × 𝒞 scatter; guard boundary (τ=1.029) overlay; annotate M primitive per Pareto knee; "controlled departure from contraction" zones
- **6.3 Manifesto Dataset:** Structured records from every guard kill + E-7 null → standalone dataset

---

## Execution Order & Parallelization

| Session | Focus | Dependencies | Parallel With |
|---|---|---|---|
| **S1** | 0.1 Cleanup + 0.2 Move stability | — | 0.0 CL Re-verify |
| **S2** | 1.1-1.4 Stability impl (guard, estimators, config, resources) | S1 | 0.3 Standalone tests |
| **S3** | 1.5 Tests + 1.7 CI | S2 | 2.1 State types (can start) |
| **S4** | 2.1 State types + 2.2 Configs | S2 (stability moved) | 3.3 L2/𝒞 wiring |
| **S5** | 2.3 Joint facade + 2.4 import sweep | S4 | 3.4 Algorithm Migration |
| **S6** | 3.1 Fairness + 3.2 Campaign | S4 (configs, state) | 3.5 PR-8 Export |
| **S7** | 3.3 L2/𝒞 + 3.4/3.5 | S4, S6 | — |
| **S8+** | Phase 4 Execution | S6, S7, 0.0 | — |

**Import Sweep (S5):** Systematic update of ~100 import sites across codebase (ontology, stability, joint, campaign, pipeline, plasticity, profiling, CLI, zoo, models).

---

## Definition of Done (Library-Level)

- [ ] `computronium[stability]` installs via `pip install -e .[stability]`; `import computronium_stability` works
- [ ] `computronium/stability/` public API complete; tests match `computronium/nn` quality (25+ tests, coverage ≥85%)
- [ ] `libraries/` directory **deleted**
- [ ] `computronium/state/`, `computronium/config/` extracted; all imports updated (~100 sites)
- [ ] `computronium/core/joint/` facade exports only composition API
- [ ] All existing tests pass; no import regressions
- [ ] `ruff format --check . && ruff check . && pyright . && pytest --cov` green
- [ ] `DECISIONS.md` updated: coordinate lock, fairness contract, prior-art gate, CL re-verification outcome
- [ ] Migration guide documented (`docs/migration_stability_v1.md`)

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Import sweep breaks tests | High | High | Incremental: move → update imports → test → commit per module; use `ruff --fix` for trivial renames |
| External API diverges from internal | Low | High | Shared implementation; `test_stability_standalone.py` validates published API |
| Campaign DB schema churn | Med | Med | Freeze schema before Phase 4; migrations for future |
| Effective-FLOPs definition ambiguity | Low | Med | Lock to `compute_efficiency.py` gate-entropy method; document in `ResourceUsage` |
| CL re-verification reveals signal | Low | High | Pre-registered kill criterion honored; design deps wait for 0.0 outcome |

---

## Post-System: Papers (Unchanged)

Writing begins only after system complete and tested. Dependency order:
1. Continual learning without replay (Phase 2) — flagship
2. Resource-axed family-coverage benchmark + phylogeny (Phase 5)
3. Edge memory-wall benchmark (Phase 3)
4. `computronium-stability` + calibration (Phase 1) — software/JOSS
5. Substrate counterfactual campaigns (Phase 4)
6. Z3 boundary memo + operator library (Phase 1) — negative-results venue
7. Goldilocks map + manifesto dataset (Phase 6)
8. Drop-in `ComputroniumLinear` wrapper (post-flagship, CP-C)
9. Theory: ψ-coverage + contraction (if CP-B completes in E-8)
10. Physics-informed conservation (if CP-E reopens post-system)

---

## Explicitly Out of Scope (TODO6)

| Item | Disposition |
|---|---|
| Photonic/Optical substrate work | Phase 4.4 simulation-tier only; no hardware |
| Memristive IR-drop hardware | Phase 4.3 simulation-tier only; no hardware |
| Biological twin | CP-E — deferred indefinitely |
| Hardware co-design pilot | CP-D — gated on PR-3b board arrival |
| ICL bridge | Deferred indefinitely (DECISIONS #2) |
| Physics-informed conservation proof | CP-E — zero coupling to M-axis storyline |

---

## Appendix: Import Migration Checklist (for S5)

**Stability → `computronium.stability`:**
- `computronium/core/campaign/evaluation.py`
- `computronium/core/campaign/__init__.py`
- `computronium/core/campaign/frontier_record.py`
- `computronium/core/continual/stability.py`
- `computronium/core/profiling.py` (ResourceUsage)
- `computronium/core/stability/__init__.py` (ResourceUsage, becomes shim)
- `computronium/core/stability/frontier.py` (ResourceUsage)
- `computronium/experiments/joint/memory_wall.py` (ResourceUsage)

**Ontology Configs → `computronium.config`:**
- 50+ files in `computronium/core/`, `computronium/models/`, `computronium/zoo/`, `computronium/cli/`, `computronium/p2p/`, `computronium/config/experiment.py`

**Joint State/Context → `computronium.state`:**
- `computronium/core/joint/__init__.py`
- `computronium/core/joint/context.py`
- `computronium/core/joint/transition.py`
- `computronium/core/joint/consolidation.py`
- `computronium/core/joint/trajectory.py`
- `computronium/core/stability/*.py` (5 files)
- `computronium/core/plasticity/*.py` (4 files)
- `computronium/core/dynamics/adapters.py`
- `computronium/core/pipeline.py`
- `computronium/core/continual/*.py`
- `computronium/core/campaign/*.py` (5 files)
- `computronium/core/profiling.py`
- `computronium/core/system_trainer.py`
- `computronium/core/distributed_trainer.py`
- `computronium/cli/kernel_profile.py`
- `computronium/p2p/grpc_worker.py`

**Strategy:** Use `ruff` auto-fix for simple renames; manual review for `core.ontology` → `config` splits (multi-import lines).

---

## Appendix B: Automation & Acceleration Strategy

### Import Migration Automation (S5)

**~90% automatable via simple tooling:**

| Migration | Pattern | Tool | Files | Time |
|---|---|---|---|---|
| `core.stability` → `stability` | `from computronium.core.stability import` → `from computronium.stability import` | `sed` / `ruff --fix` | 8 | 1 min |
| `core.profiling.ResourceUsage` → `stability.ResourceUsage` | `from computronium.core.profiling import ResourceUsage` → `from computronium.stability import ResourceUsage` | `sed` | 8 | 1 min |
| `core.joint.state` → `state` | `from computronium.core.joint.state import X` → `from computronium.state import X` | `sed` | ~30 | 1 min |
| `core.joint.context` → `state` | `from computronium.core.joint.context import X` → `from computronium.state import X` | `sed` | ~25 | 1 min |
| `core.ontology` → `config/*` | Multi-symbol splits per symbol map | `libcst` script | 50+ | 30 min to write, then instant |

**Commands for simple renames:**
```bash
# Stability module
find . -name "*.py" -exec sed -i 's/from computronium\.core\.stability import/from computronium.stability import/g' {} +

# ResourceUsage
find . -name "*.py" -exec sed -i 's/from computronium\.core\.profiling import ResourceUsage/from computronium.stability import ResourceUsage/g' {} +

# Joint state/context
find . -name "*.py" -exec sed -i 's/from computronium\.core\.joint\.state import/from computronium.state import/g' {} +
find . -name "*.py" -exec sed -i 's/from computronium\.core\.joint\.context import/from computronium.state import/g' {} +

# Post-cleanup
ruff check --select F401 --fix .  # Remove unused imports
ruff check --select I001 --fix .   # Sort imports
```

**`libcst` script for ontology → config splits:**
```python
# tools/migrate_ontology.py
import libcst as cst
from pathlib import Path

MODULE_MAP = {
    "SubstrateConfig": "computronium.config.substrate",
    "SubstrateType": "computronium.config.substrate",
    "DigitalSubstrate": "computronium.config.substrate",
    "AnalogSubstrate": "computronium.config.substrate",
    "MemristiveSubstrate": "computronium.config.substrate",
    "NeuromorphicSubstrate": "computronium.config.substrate",
    "SparseSubstrate": "computronium.config.substrate",
    "TernarySubstrate": "computronium.config.substrate",
    "OpticalSubstrate": "computronium.config.substrate",
    "QuantumSubstrate": "computronium.config.substrate",
    "ComplexSubstrate": "computronium.config.substrate",
    "GeometryConfig": "computronium.config.geometry",
    "FeedforwardGeometry": "computronium.config.geometry",
    "RecurrentGeometry": "computronium.config.geometry",
    "TileGeometry": "computronium.config.geometry",
    "StateDynamicsConfig": "computronium.config.dynamics",
    "EnergyMinimizationDynamics": "computronium.config.dynamics",
    "PredictiveSettlingDynamics": "computronium.config.dynamics",
    "InstantaneousDynamics": "computronium.config.dynamics",
    "SpikeIntegrationDynamics": "computronium.config.dynamics",
    "DiffusionDynamics": "computronium.config.dynamics",
    "PlasticityConfig": "computronium.config.plasticity",
    "NullPlasticity": "computronium.config.plasticity",
    "RoutingPlasticity": "computronium.config.plasticity",
    "FastWeightPlasticity": "computronium.config.plasticity",
    "RuleStatePlasticity": "computronium.config.plasticity",
    "CreditAssignmentConfig": "computronium.config.credit",
    "ThermodynamicContrast": "computronium.config.credit",
    "RandomProjectionsCredit": "computronium.config.credit",
    "LocalGoodnessCredit": "computronium.config.credit",
    "TemporalTraceCredit": "computronium.config.credit",
    "TargetInversionCredit": "computronium.config.credit",
    "HomeostaticCredit": "computronium.config.credit",
    "BackpropCredit": "computronium.config.credit",
    "ParameterUpdateConfig": "computronium.config.update",
    "EuclideanUpdate": "computronium.config.update",
    "RiemannianOrthogonalUpdate": "computronium.config.update",
    "SpectralConstrainedUpdate": "computronium.config.update",
    "NaturalGradientUpdate": "computronium.config.update",
    "ElasticConsolidationUpdate": "computronium.config.update",
    "SystemConfig": "computronium.config",
    "System": "computronium.config",
    "SystemState": "computronium.config",
    "Phase": "computronium.config",
    "FAMILY_TOLERANCES": "computronium.config",
    "substrate_from_config": "computronium.config.substrate",
}

class ImportRewriter(cst.CSTTransformer):
    def leave_ImportFrom(self, original, updated):
        if original.module and original.module.value == "computronium.core.ontology":
            new_imports = []
            for alias in original.names:
                name = alias.name.value
                if name in MODULE_MAP:
                    new_imports.append(cst.ImportFrom(
                        module=cst.Name(MODULE_MAP[name]),
                        names=[alias],
                    ))
            if new_imports:
                return cst.FlattenSentinel(new_imports)
        return updated

for py_file in Path(".").rglob("*.py"):
    if "test_ontology.py" in str(py_file):  # Skip - tests old imports intentionally
        continue
    source = py_file.read_text()
    tree = cst.parse_module(source)
    new_tree = tree.visit(ImportRewriter())
    new_source = new_tree.code
    if new_source != source:
        py_file.write_text(new_source)
        print(f"Updated: {py_file}")
```

### Other Acceleration Opportunities

| Area | Technique | Savings |
|---|---|---|
| **Test scaffolding** | Cookiecutter template for `Test*Estimator` classes (shared fixtures, parametrized contractive/expansive/identity) | 50% test writing time |
| **Config boilerplate** | Single `make_config_dataclass()` generator for `to_spec`/`from_spec` + factories | Eliminates copy-paste across 5 config classes |
| **Deprecation shims** | `ruff` rule to auto-add `warnings.warn` on old imports | Catches stragglers in CI |
| **Documentation** | `pdoc`/`mkdocstrings` auto-generated from docstrings | Zero manual API docs |
| **Migration verification** | `grep -r "old.import" --include="*.py" \|\| echo "CLEAN"` in CI gate | Prevents regressions |
| **Parallel test runs** | `pytest -n auto --dist=loadfile` (already in `pytest.ini` via `pytest-xdist`) | 3-4x faster test suite |

### S5 Import Sweep: Safe Execution Order

```bash
# 1. Backup
git stash push -m "pre-import-sweep"  # or git commit

# 2. Automated renames (safe, reversible)
./tools/auto_rename_imports.sh

# 3. AST-based ontology splits
python tools/migrate_ontology.py

# 4. Verify
grep -r "computronium.core.stability\|computronium.core.profiling import ResourceUsage\|computronium.core.ontology import\|computronium.core.joint.state import\|computronium.core.joint.context import" --include="*.py" | grep -v test_stability_standalone | grep -v test_ontology.py || echo "CLEAN"

# 5. Fix any stragglers manually
# 6. Run full test suite
pytest --cov -x

# 7. Commit
git add -A && git commit -m "refactor: import migration to stability/state/config modules"
```

### Phase 1 Test Acceleration: Shared Test Infrastructure

```python
# tests/unit/stability/conftest.py
import pytest
import torch

@pytest.fixture
def contractive_transition():
    def _trans(state):
        return {"x": state["x"] * 0.5}
    return _trans

@pytest.fixture
def expansive_transition():
    def _trans(state):
        return {"x": state["x"] * 2.0}
    return _trans

@pytest.fixture
def identity_transition():
    def _trans(state):
        return state
    return _trans

@pytest.fixture
def state_dict():
    return {"x": torch.randn(4, 10)}

# Parametrize estimator tests across modes
@pytest.fixture(params=["fast", "full"])
def estimator_mode(request):
    return request.param
```

Then each estimator test class uses `@pytest.mark.parametrize("transition,expected_sign", [("contractive", -1), ("expansive", 1), ("identity", 0)])` — **one test function covers all modes.**

---

## Updated Execution Order with Automation

| Session | Focus | Automation Used |
|---|---|---|
| **S1** | Cleanup + Move stability | `rm -rf`, `mv` |
| **S2** | Stability impl (1.1-1.4) | Config generator for boilerplate |
| **S3** | Tests + CI | Shared fixtures + parametrize; `pytest -n auto` |
| **S4** | State + Config extraction | `libcst` script for ontology splits |
| **S5** | Import sweep + Joint facade | `sed` + `ruff --fix` + `libcst`; verification script |
| **S6** | Fairness + Campaign | — |
| **S7** | L2/𝒞 + Algorithm Migration + Export | — |
| **S8+** | Phase 4 Execution | — |

**Total S5 import migration time:** ~10 minutes automated + 10 minutes verification vs. 2-3 hours manual.