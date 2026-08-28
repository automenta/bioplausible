# TODO6.md — Computronium Stability Integration & Modularization

> **Scope:** Integrate `computronium-stability` as a first-class library (mirroring `computronium/nn` pattern), clean up fractured `libraries/computronium_stability/`, and plan modularization for RESEARCH3 ambitions. Excludes substrate-specific work (photonic, memristive, etc.) — CPU/GPU/universal only.

---

## Status Snapshot

| Track | State |
|---|---|
| `computronium/nn` (CP-C wrapper) | ✅ **COMPLETE** — 26 tests passing, ruff/pyright clean |
| `libraries/computronium_stability` | 🟡 **FRACTURED** — Separate `.venv/`, build/, duplicate code in `computronium/core/stability/` |
| Phase 3.6 Audits | ✅ **COMPLETE** — All 7 audits pass, 34 regression tests added |
| Phase 4 (Regime Discovery) | 🟢 **UNBLOCKED** — Awaits execution |
| Phase 5 (Family-Coverage Benchmark) | 🟢 **UNBLOCKED** — Awaits coordinate lock |
| Phase 6 (Frontier Certification) | 🟢 **UNBLOCKED** — Awaits flagship coordinate |
| **Phase 2 CL Re-verification** | 🔴 **REQUIRED** — Post-3.6 audit fixes may have changed behavior; must re-run on verified arms before any design dependency |

---

## Phase 0 — Cleanup & Consolidation (Prerequisite)

### 0.0 Phase 2 CL Re-verification (Blocking for Design Dependencies)
**Before any design work that depends on CL results:** re-run the Phase 2 discriminating probe on current codebase.
- **Why:** Phase 3.6 audits fixed in-place ops, device consistency, dynamics settling, plasticity decay/projection, CL pipeline (task masking, replay training, LwF/SI/EWC), memory accounting — all touch CL arms.
- **Protocol:** Fresh E-1 pre-registration → run `scripts/verify_capacity_limited_cl.py` (6 arms, hidden=32, 2 epochs, 5 seeds) → compare forgetting/BWT vs. Session 28 baseline.
- **Exit:** If null still holds → document in `DECISIONS.md`; if signal emerges → new E-1 registration for full run.
- **No design decisions** (benchmark coordinate lock, regime discovery rewards, etc.) should assume CL null/non-null until this passes.

### 0.1 Remove Fractured `libraries/computronium_stability/`
- **Delete** `libraries/computronium_stability/.venv/` (unnecessary venv copy)
- **Delete** `libraries/computronium_stability/build/` (build artifacts)
- **Delete** `libraries/computronium_stability/.pytest_cache/`, `.ruff_cache/`
- **Retain** only: `pyproject.toml`, `README.md`, `computronium_stability/` (source), `tests/` — **temporarily** for migration
- **After Phase 1 complete:** **Delete entire `libraries/` directory** — no longer needed; all publishing handled via root `pyproject.toml` with `package-dir` mapping

### 0.2 Consolidate Stability Code Under `computronium/stability/`
- **Verify** `computronium/core/stability/` is the current canonical implementation (used by all internal code, tests, PR-5 guard)
- **Move** `computronium/core/stability/` → `computronium/stability/` (see Phase 1.1)
- **Migrate** `libraries/computronium_stability/tests/test_stability.py` → `tests/unit/core/test_stability_standalone.py` (see 0.3)
- **Remove** `libraries/computronium_stability/computronium_stability/` duplicate source
- **Retain** `libraries/computronium_stability/pyproject.toml` temporarily as publishing spec reference; **delete after Phase 1**

### 0.3 Create Standalone Test Suite for Published API
- **New file:** `tests/unit/core/test_stability_standalone.py`
- **Purpose:** Test the exact public API that external users get via `pip install computronium[stability]`
- **Imports:** `from computronium_stability import attach, StabilityGuard, StabilityVerdict, SpectralRadiusEstimator, ...`
- **Mirrors** `tests/unit/nn/test_computronium_linear.py` pattern
- **Ensures** the published package works identically to internal usage

### 0.4 Publishing Configuration (PyPI-Ready)
- **Update** `pyproject.toml` (root) — ensure `stability = []` extra has zero extra deps (already true)
- **Add** `tool.setuptools.package-dir` mapping for `computronium_stability = "computronium/stability"` in root `pyproject.toml` for editable installs
- **Verify** `uv build` produces clean wheel with `computronium_stability` as top-level import
- **Document** install: `pip install computronium[stability]` → `import computronium_stability`

---

## Phase 1 — Stability Library Integration (Unified at `computronium/stability/`)

**Unification principle:** `computronium/stability/` is the **single canonical implementation** (not a wrapper). `computronium/core/stability/` becomes a deprecated re-export alias (one-release transition) or is removed.

### 1.1 Consolidate Implementation to `computronium/stability/`
- **Move** all source from `computronium/core/stability/` → `computronium/stability/`:
  ```
  computronium/stability/
    __init__.py           # Public exports (single source of truth)
    guard.py              # StabilityGuard, attach, GuardHandle, calibrate_threshold, etc.
    spectral_radius.py    # SpectralRadiusEstimator, estimate_spectral_radius, estimate_spectral_radius_full_jacobian
    lyapunov.py           # LyapunovEstimator, estimate_lyapunov_exponent, estimate_lyapunov_spectrum
    settling.py           # SettlingMonitor, measure_settling_time, measure_settling_time_full_state
    basin.py              # BasinStabilityEstimator, estimate_basin_stability, estimate_basin_stability_multistart
    frontier.py           # FrontierRecord, FrontierAggregator (from core.stability.frontier)
    config.py             # Config dataclasses (SpectralRadiusConfig, LyapunovConfig, etc.)
    resources.py          # ResourceUsage (moved from core.profiling - see Phase 2.1)
  ```
- **Update** all internal imports: `from computronium.stability import ...` (not `core.stability`)
- **Deprecation shim** (optional, one release): `computronium/core/stability/__init__.py` re-exports from `computronium.stability` with `warnings.warn("Use computronium.stability", DeprecationWarning)`
- **Rationale:** Public API = internal implementation; no duplication, no drift, single test surface.

### 1.2 Public API Exports (`computronium/stability/__init__.py`)
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
# Resources (unified)
from .resources import ResourceUsage
# Config
from .config import SpectralRadiusConfig, LyapunovConfig, SettlingConfig, BasinConfig, GuardConfig
# Factories
from .config import create_spectral_radius_estimator, create_lyapunov_estimator, create_settling_monitor, create_basin_estimator, create_guard
# Type aliases
from .guard import StepState, TransitionFn, StatisticKind
```
- **Zero deps** beyond `torch`, `numpy` — matches `nn = []` extra
- **PEP 695** generics in config dataclasses
- **`from_spec`/`to_spec`** on all config classes for YAML/JSON round-trip

### 1.3 High-Level Guard API (`computronium/stability/guard.py`)
- **Consolidate** `core.stability.guard.StabilityGuard` + `libraries/computronium_stability/computronium_stability/guard.py` into single implementation
- **Support both** `CompositeState` (internal) and plain `dict[str, Tensor]` (external) via `_extract_activity` adapter (already exists)
- **`attach(model, ...)`** returns `GuardHandle` with `.check(state, step)` and `.detach()`

### 1.4 Config Dataclasses (`computronium/stability/config.py`)
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
- **Factory functions** in same module: `create_spectral_radius_estimator(config)`, etc.

### 1.5 Tests: `tests/unit/stability/test_stability_api.py`
- **Mirror** `tests/unit/nn/test_computronium_linear.py` structure (~25 tests)
- **Test classes:** `TestStabilityGuardAPI`, `TestSpectralRadiusEstimator`, `TestLyapunovEstimator`, `TestSettlingMonitor`, `TestBasinStabilityEstimator`, `TestIntegration`, `TestConfigRoundtrip`, `TestDeviceManagement`
- **Standalone test** for published package: `tests/unit/core/test_stability_standalone.py` imports `from computronium_stability import ...` (validates wheel)

### 1.6 Publishing Configuration
- **Root `pyproject.toml`:** `tool.setuptools.package-dir = { "computronium_stability" = "computronium/stability" }` for editable install
- **`stability = []` extra** (zero deps) installs `computronium_stability` top-level package
- **Verify** `uv build` → `pip install dist/*.whl` → `import computronium_stability` works
- **CI:** `ruff format --check . && ruff check . && pyright . && pytest --cov` green; coverage ≥85% for `computronium/stability/`

---

## Phase 2 — Modularization Refactoring (Accessibility)

### 2.1 `ResourceUsage` Unified in `computronium/stability/resources.py`
- **Done in Phase 1.1:** `ResourceUsage` moved from `computronium/core/profiling.py` → `computronium/stability/resources.py`
- **Imports updated:** `from computronium.stability import ResourceUsage` (single source)
- **Profiling utilities** (`count_flops`, `get_gpu_memory_mb`, `measure_suite_resources`, `EnergyTracker`, `analyze_joint_system`) remain in `computronium/core/profiling.py` — they are measurement tools, not the resource vector itself
- **Rationale:** `ResourceUsage` is the **universal currency** (PR-3a, PR-6, frontier, campaigns, stability guard) — belongs in the stability module alongside the metrics that consume it

### 2.2 Extract `State` Types to `computronium/state/`
- **Current:** `computronium/core/joint/state.py` + `context.py` + `transition.py`
- **New structure:**
```
computronium/
  state/
    __init__.py          # CompositeState, SystemContext, StateRegistry, StateVariable
    composite.py         # CompositeState (activity/plastic/substrate)
    context.py           # SystemContext (6-D config bundle)
    registry.py          # StateRegistry, StateVariable
    transitions.py       # NullPlasticity, PlasticityConfig, transition protocols
```
- **Benefit:** External users can import state types without pulling joint system internals
- **Protocol-based:** Define `TransitionFn` protocol in `transitions.py` for duck-typing

### 2.3 Extract `Ontology` Configs to `computronium/config/`
- **Current:** `computronium/core/ontology/` (scattered config dataclasses)
- **New:** `computronium/config/` with per-axis config modules:
  - `substrate.py` — `SubstrateConfig`, `DigitalSubstrateConfig`, etc.
  - `geometry.py` — `GeometryConfig`, `FeedforwardConfig`, `RecurrentConfig`
  - `dynamics.py` — `StateDynamicsConfig`, `EnergyMinimizationConfig`, etc.
  - `plasticity.py` — `PlasticityConfig`, `RoutingConfig`, `FastWeightConfig`, `RuleStateConfig`
  - `credit.py` — `CreditAssignmentConfig`, `BackpropConfig`, `ThermoConfig`, `FAConfig`
  - `update.py` — `ParameterUpdateConfig`, `EuclideanConfig`
- **Single import:** `from computronium.config import *` gets all config types
- **Factory pattern:** `GeometryConfig.feedforward(...)`, `PlasticityConfig.fast_weights(...)` preserved

### 2.4 Unified `computronium.core.joint` Facade
- **Keep** `computronium/core/joint/` as **internal composition layer**
- **New:** `computronium/core/joint/__init__.py` exports only:
  - `compose_joint_system` / `compose_joint_system_from_configs`
  - `JointSystem` (trainer interface)
  - `run_train_step`, `run_forward` (canonical loop)
- **Hide** internal mechanics (`trajectory.py`, `consolidation.py`) behind facade

---

## Phase 3 — RESEARCH3 Ambitions: Infrastructure Work

### 3.1 PR-6 Fairness Contract (Formalize & Enforce)
- **File:** `docs/evaluation_fairness_contract.md` → **promote to code** as `computronium/eval/fairness.py`
- **Contents:**
  - `FairnessContract` dataclass: `gpu_hours_per_rule`, `seeds`, `early_stopping_policy`, `data_splits`
  - `validate_fairness(contract, results)` — programmatic compliance check
  - `BenchmarkRunner` base class enforcing contract
- **Integration:** All benchmark runners (Phase 3, 4, 5, 6) inherit from `BenchmarkRunner`

### 3.2 PR-9 Campaign Stack Hardening
- **Current:** `autoscientist_campaigns/` commissioned (6 episodes, checkpoint/resume verified)
- **Gaps to close:**
  - **Campaign DB schema freeze** — add migrations support (`alembic` or custom)
  - **Proposer objective swap** — `ProposalObjective` enum: `ACCURACY`, `STABILITY`, `ENERGY`, `LATENCY`, `PLASTICITY_CAPACITY`
  - **Replication gate** — auto-verify ≥5 seeds + ≥2 task families before promoting discovery
  - **Counterfactual attribution** — integrate `analysis/counterfactual.py` into campaign evaluation
- **Deliverable:** `CampaignStack` class with `run_campaign(coordinate, objective, max_wall_hours)`

### 3.3 L2 Effective-FLOPs → 𝒞 Vector Wiring
- **Current:** `computronium/experiments/joint/compute_efficiency.py` computes `effective_flops`
- **Wire into:** `ResourceUsage` (add `effective_flops` field) → `FrontierRecord` → Phase 5 runner
- **Sanctioned feed:** Per RESEARCH3 L2, effective-FLOPs (gate-entropy-aware route counting) is the official compute metric for 𝒞 vector
- **Deprecate** raw FLOP counting in favor of effective-FLOPs for all frontier records

### 3.4 Algorithm Migration (L3.5) as ψ-Switching Validation
- **Current:** `computronium/experiments/joint/algorithm_migration.py`
- **Promote to:** `computronium/benchmarks/algorithm_migration.py` (first-class benchmark)
- **Use case:** Cheapest end-to-end validation of ψ-switching machinery (Δθ=0 audit, two-strategy swap)
- **Run as:** CI smoke test for ψ-switching correctness (fast, <30s)

### 3.5 Edge/Green Export Path (PR-8) Reuse
- **Current:** `deployment.py` + `acceleration/export.py` verified for ONNX/ternary
- **Wire into:** Phase 3 memory-wall artifact suite + Phase 5 benchmark export
- **Single pipeline:** `export_model(model, formats=["onnx", "ternary", "int8"])`

---

## Phase 4 — Phase 4/5/6 Execution Prep (Unblocked)

### 4.1 Phase 4 — Regime Discovery (Bandit Router + Substrate Counterfactuals)
- **4.1 Prior-Art Gate** — Literature check (per-layer mixed credit, hypernetwork rule selection, MoE routing) → log in `DECISIONS.md`
- **4.2 Bandit Router** — Generalize `RoutingPlasticity` to route **learning rules** (credit families per layer)
  - Reward: local proxy (energy descent rate, windowed growth, validation improvement)
  - Scope: schedules/regimes/policies only — no novel math
- **4.3 Memristive IR-Drop** (simulation tier) — Sweep IR-drop on `MemristiveSubstrate`; test `SpectralConstrainedUpdate` + `EnergyMinimization` + `SubstrateCoupledPlasticity`
- **4.4 Photonic Epistemology** (simulation tier) — `OpticalSubstrate` × credit families; test settling-energy profile preference
- **4.5 Campaign Hygiene** — Enforce `simulated/estimated/measured` labeling; `ProposalObjective` non-accuracy ranking

### 4.2 Phase 5 — Family-Coverage Benchmark (Resource-Vector Headline)
- **5.1 Coordinate Lock** — Lock by **rule-family coverage** (every credit×update family + substrate variants); target ≥30 coordinates; freeze set in `DECISIONS.md`
- **5.2 Resource-Vector Runner** — Extend runner to emit full `ResourceUsage` per coordinate/seed; equal GPU-hour budgets (PR-6); ≥5 seeds paired
- **5.3 Dynamical Phylogeny** — Cluster by measured dynamics (settling time, windowed growth, gate entropy, ρ) using `analysis/genealogy.py`
- **5.4 Full Run** — Capability matrix, accuracy-per-resource Pareto overlays, per-rule stability audits, failure modes

### 4.3 Phase 6 — Frontier Certification & Goldilocks Map
- **6.1 M-Axis Frontier** — Pin S/G/D/C/U at flagship; sweep M ∈ {Null, Routing, FastWeight, RuleState}; `AutoScientistCampaign` with guard live, checkpoint/resume
- **6.2 Goldilocks Map** — ρ(J_F) × 𝒞 scatter; guard boundary (τ=1.029) overlay; annotate M primitive per Pareto knee; identify "controlled departure from contraction" zones
- **6.3 Manifesto Dataset** — Package failure manifesto as standalone dataset: structured records from every guard kill + E-7 null

---

## Execution Order (Next Sessions)

| Session | Focus | Exit Criteria |
|---|---|---|
| **S1** | Cleanup (0.1–0.4) | `libraries/computronium_stability/` clean; `uv build` works; standalone tests pass |
| **S2** | Stability API (1.1–1.3) | `computronium/stability/` module complete; exports match `computronium/nn` pattern |
| **S3** | Stability Tests (1.5) | 25+ tests passing; ruff/pyright/pytest green; coverage ≥85% |
| **S4** | Modularization (2.2) | `computronium/state/` extracted; imports updated |
| **S5** | Modularization (2.3–2.4) | `computronium/config/` extracted; joint facade cleaned |
| **S6** | PR-6/9 Hardening (3.1–3.2) | `FairnessContract` enforced; `CampaignStack` runnable with non-accuracy objectives |
| **S7** | L2/L3.5/PR-8 Wiring (3.3–3.5) | Effective-FLOPs in 𝒞; algorithm_migration benchmark; export pipeline reused |
| **S8+** | Phase 4 Execution | Bandit router working; substrate campaigns run at simulation tier |

---

## Definition of Done (Library-Level)

- [ ] `computronium[stability]` installs via `pip install -e .[stability]`; `import computronium_stability` works
- [ ] `computronium/stability/` public API complete with tests matching `computronium/nn` quality
- [ ] `libraries/` directory **deleted** (no `.venv`, no build, no duplicate source)
- [ ] `computronium/state/`, `computronium/config/` extracted and imported
- [ ] All existing tests pass; no import regressions
- [ ] `ruff format --check . && ruff check . && pyright . && pytest --cov` green
- [ ] `DECISIONS.md` updated with coordinate lock, fairness contract, prior-art gate

---

## Risk Register (TODO6 Specific)

| Risk | Mitigation |
|---|---|
| Duplicate stability code causes drift | Single source in `computronium/stability/`; library is thin re-export |
| External API diverges from internal | Shared implementation; standalone tests validate published API |
| Modularization breaks internal imports | Incremental: extract → update imports → test → commit per module |
| Campaign stack schema churn | Freeze schema before Phase 4 launch; migrations for future |
| Effective-FLOPs definition ambiguity | Lock to `compute_efficiency.py` gate-entropy-aware method; document in `ResourceUsage` |

---

## Post-System: Papers (Unchanged from TODO5)
Writing begins only after system is complete and tested. Candidate artifacts in dependency order:
1. Continual learning without replay (Phase 2) — flagship
2. Resource-axed family-coverage benchmark + phylogeny (Phase 5)
3. Edge memory-wall benchmark (Phase 3)
4. `computronium-stability` + calibration (Phase 1) — software/JOSS track
5. Substrate counterfactual campaigns (Phase 4)
6. Z3 boundary memo + operator library (Phase 1) — negative-results venue
7. Goldilocks map + manifesto dataset (Phase 6)
8. Drop-in `ComputroniumLinear` wrapper release (post-flagship, per CP-C)
9. Theory: ψ-coverage + contraction (only if CP-B completes in E-8 time)
10. Physics-informed conservation (only if CP-E reopens post-system)

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