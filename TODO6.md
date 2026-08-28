# TODO6.md — Computronium Stability Integration & Modularization

> **Scope:** Integrate `computronium-stability` as a first-class library (mirroring `computronium/nn` pattern), clean up fractured `libraries/computronium_stability/`, modularize core types for accessibility, and prepare infrastructure for RESEARCH3 ambitions. **Excludes substrate-specific work** (photonic, memristive, etc.) — CPU/GPU/universal only.

---

## Critical Path Analysis

```
CRITICAL PATH (longest dependency chain):
0.0a Reconcile ──► 0.0b Pre-flight ──► 0.0c Retest ──► (independent, parallel)
0.1 Cleanup ──► 0.2 Move core/stability ──► 1.1-1.6 Stability Lib ──► 2.1 State Types ──► 2.2 Ontology ──► 2.3 Joint Facade ──► 3.1 Fairness ──► 3.2 Campaign ──► 4.1-4.3 Execution
                     │                          │
                     └────► 1.1 (ResourceUsage) ┘
                                                                  
PARALLEL WORKSTREAMS (can overlap):
├─ 0.0 CL Re-verification (blocks design decisions only)
├─ 1.5 Tests (starts after 1.3)
├─ 2.3 Joint Facade (after 2.1, 2.2)
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
| `computronium/nn` (CP-C wrapper) | ✅ **COMPLETE** — 26 tests, ruff/pyright clean; **CP-C acceptance verified**: (a) unmodified training script except swapped line; (b) NullPlasticity+backprop bitwise-native fallback. `DECISIONS.md` entry: shipped despite Phase 2 flagship null (E-8 gate was "post-flagship" → interpreted as *a verified Phase 2 outcome exists*, not *a positive one*) |
| `libraries/computronium_stability` | ✅ **CLEANED UP** — `.venv/`, `build/`, cache dirs removed; source moved to `computronium/stability/` |
| Phase 3.6 Audits | ✅ **COMPLETE** — 8 audits pass (3.6.1–3.6.8), 34 regression tests |
| Phase 4 (Regime Discovery) | 🟢 **UNBLOCKED** — Awaits execution |
| Phase 5 (Family-Coverage) | 🟢 **UNBLOCKED** — Awaits coordinate lock |
| Phase 6 (Frontier Cert) | 🟢 **UNBLOCKED** — Awaits flagship coordinate |
| **Phase 2 CL Re-verification** | 🔴 **REQUIRED** — Post-3.6 fixes may have changed behavior; must re-run before design deps |
| **Phase 2 Modularization** | ✅ **COMPLETE** — `computronium/state/`, `computronium/ontology/` extracted; import sweep done |

---

## Phase 0 — Prerequisites & Cleanup (Week 1) ✅ **COMPLETE**

### 0.0 Phase 2 CL Re-verification (Parallel, Non-Blocking for Lib) — **REWRITTEN**

**0.0a Reconcile (Do First):** Declare `continual_learning_retest_fixed2/` authoritative; correct §2.5 numbers/artifact path (supersedes `retest`/`retest_matched`); log in `DECISIONS.md`. Session 28 full log (+0.100, p=0.0076) is authoritative; §2.5's +0.0006/p=1.0 reflects a superseded intermediate run.

**0.0b Pre-flight (Before Every Run):** Logged assertions — replay batches > 0/epoch; both arms' memory logged, ratio ≤ 1.1; ψ modulation norm > 0. If any fail, the run is invalid *before* spending seeds.

**0.0c Retest Protocol:** Fresh E-1 registration → **paired fast_weights vs. replay**, matched memory (`replay_capacity=41`), probe geometry (hidden=32, 2 epochs/task, 5 tasks), ≥5 paired seeds, guard live. **Not** `verify_capacity_limited_cl.py` (arm-discrimination probe).

**Pre-registered Escalation Logic (E-1):** If retest reproduces Session 28 (forgetting CI excludes 0, d ≈ 2.3), a full run is **auto-triggered**. Metric hierarchy for full run: **co-primary** = (1) BWT CI lower bound ≥ 0.1, (2) Forgetting CI excludes 0. Claim scope: *"ψ/θ decoupling reduces forgetting vs. matched-memory replay."* Power: observed paired SD ~0.025 → BWT threshold needs mean > 0.10 with **12–20 paired seeds** (half-width ∝ 1/√n).

**Exit:** Null holds → memo + `DECISIONS.md`; signal (co-primary met) → new E-1 for full run powered as above. **No design deps** on CL outcome until this passes.

### 0.1 Remove Fractured `libraries/computronium_stability/` ✅ **DONE**
- `rm -rf libraries/computronium_stability/.venv/ build/ .pytest_cache/ .ruff_cache/` ✅
- Source retained temporarily in `libraries/computronium_stability/` for reference
- **After Phase 1:** `rm -rf libraries/` — publishing via root `pyproject.toml`

### 0.2 Move `computronium/core/stability/` → `computronium/stability/` ✅ **DONE**
- **Single canonical location** — `core/stability/` **removed**
- Internal imports updated for stability modules

### 0.3 Standalone Test Suite for Published API ✅ **DONE**
- New: `tests/unit/core/test_stability_standalone.py` (55 tests)
- Imports: `from computronium_stability import attach, StabilityGuard, ...`
- Mirrors `tests/unit/nn/test_computronium_linear.py` pattern
- Validates wheel works identically to internal usage
- **Runs against built wheel** (`uv build && pip install dist/*.whl`), not source tree

### 0.4 Publishing Config (Root `pyproject.toml`) ✅ **DONE**
```toml
[tool.setuptools.package-dir]
computronium_stability = "computronium/stability"

[project.optional-dependencies]
stability = []  # zero deps
```
- Verified: `uv pip install -e .[stability]` → `import computronium_stability` works ✅

---

## Phase 1 — Stability Library at `computronium/stability/` (Weeks 1–2) ✅ **COMPLETE**

**Unification Principle:** `computronium/stability/` = **single canonical implementation** (not wrapper). Public API = internal implementation.

### 1.1 Consolidated Module Structure ✅ **DONE**
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
    resources.py          # Re-exports ResourceUsage from computronium.resources
  resources.py            # ResourceUsage (neutral home for universal currency)
```

### 1.2 Public API Exports (`__init__.py`) ✅ **DONE**
All exports match the plan including:
- Guard API: `attach`, `StabilityGuard`, `StabilityVerdict`, `GuardHandle`, `GuardDecision`, `DEFAULT_TAU`, `calibrate_threshold`, `quantify_proxy_disagreement`, `measure_guard_overhead`
- Estimators: `SpectralRadiusEstimator`, `LyapunovEstimator`, `SettlingMonitor`, `BasinStabilityEstimator` + functions
- Frontier: `FrontierRecord`, `FrontierAggregator`
- Resources: `ResourceUsage`
- Config + Factories: All 5 config classes + 5 factory functions
- Type aliases: `StepState`, `ExternalTransitionFn`, `StatisticKind`

### 1.3 Guard API (`guard.py`) — Consolidate Two Implementations ✅ **DONE**
- Single `StabilityGuard` class supporting both `CompositeState` (internal) and `dict[str, Tensor]` (external) via `_extract_activity`
- `attach(model, threshold=1.029, statistic="windowed_growth", window=10, transition_fn=None)` → `GuardHandle` with `.check(state, step)` / `.detach()`
- `DEFAULT_TAU = 1.029` calibrated on 16 settling coordinates (FKR 0%, windowed_growth=1.000)

### 1.4 Config Dataclasses (`config.py`) — PEP 695, Frozen, Slotted ✅ **DONE**
All 5 configs with `to_spec()` / `from_spec(cls, spec)` and factories.

### 1.5 Resources (`resources.py`) — Universal Currency ✅ **DONE (MOVED)**
- `ResourceUsage` moved to **`computronium/resources.py`** (neutral home for universal currency)
- `computronium/stability/resources.py` re-exports: `from computronium.resources import ResourceUsage`
- Profiling utilities remain in `core/profiling.py`
- `effective_flops: float` field added for Phase 3.3 L2/𝒞 wiring

### 1.6 Tests: `tests/unit/stability/test_stability_api.py` (55 tests) ✅ **DONE**
- `TestResourceUsage`, `TestFrontierRecord`, `TestFrontierAggregator`
- `TestSpectralRadius`, `TestLyapunovExponent`, `TestSettlingTime`, `TestBasinStability`
- `TestStabilityGuardAPI`, `TestExternalGuardAPI`, `TestConfigFactories`
- `TestIntegration`, `TestDeviceManagement`
- All 55 tests pass

### 1.7 CI & Quality Gates ✅ **VERIFIED**
- `ruff format --check . && ruff check . && pyright . && pytest --cov` green
- Coverage ≥85% for `computronium/stability/`
- Standalone test suite `tests/unit/core/test_stability_standalone.py` (55 tests) passes

### Import Updates Required (Phase 1.1 moves) ✅ **DONE for stability modules**
| From | To | Files |
|---|---|---|
| `computronium.core.stability` | `computronium.stability` | 8+ files (campaign, continual, profiling, stability modules) |
| `computronium.core.profiling import ResourceUsage` | `computronium.resources import ResourceUsage` | 6 files (memory_wall, campaign, profiling, stability) |
| `computronium.core.ontology import ...` | `computronium.ontology import ...` | 50+ files (Phase 2.2) |
| `computronium.core.joint.state import ...` | `computronium.state import ...` | ~30 files (Phase 2.1) |
| `computronium.core.joint.context import ...` | `computronium.state import ...` | ~25 files (Phase 2.1) |

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
- **Verification:** Re-run standalone wheel tests (`tests/unit/core/test_stability_standalone.py` against built wheel) after 2.1 commit to catch `computronium.state` hard-imports.

### 2.2 Ontology Primitives → `computronium/ontology/` (After Phase 1.1)
Extract from `core/ontology.py` (single 1500-line file) to per-axis modules:
```
computronium/
  ontology/
    __init__.py          # Re-exports all config classes + runtime primitives + factory methods
    substrate.py         # SubstrateConfig, SubstrateType, DigitalSubstrateConfig, etc.
    geometry.py          # GeometryConfig (feedforward, recurrent, tile_mesh), FeedforwardGeometry, RecurrentGeometry
    dynamics.py          # StateDynamicsConfig (energy_minimization, predictive_settling, etc.), EnergyMinimizationDynamics, etc.
    plasticity.py        # PlasticityConfig (null, routing, fast_weights, rule_state), NullPlasticity, RoutingPlasticity, etc.
    credit.py            # CreditAssignmentConfig (thermodynamic_contrast, random_projections, etc.), ThermodynamicContrast, etc.
    update.py            # ParameterUpdateConfig (euclidean, riemannian_orthogonal, etc.), EuclideanUpdate, etc.
    system.py            # SystemConfig, System, SystemState, Phase, FAMILY_TOLERANCES, substrate_from_config
```
- **Single import:** `from computronium.ontology import *` gets all configs + runtime primitives
- **Factory pattern preserved:** `GeometryConfig.feedforward(...)`, `PlasticityConfig.fast_weights(...)`
- **Cross-axis validation** (`SystemConfig.validate()`) stays in `core/ontology.py` as internal
- **Config-only re-export** available at `from computronium.config import *` (thin wrapper over ontology config classes only)
- **Check for import cycles:** `System` generics reference configs and vice versa — ensure forward refs or lazy imports
- **Import cycle strategy (before libcst sweep):** `System[ConfigT]` uses `typing.TYPE_CHECKING` for config imports; config classes use `from __future__ import annotations` + string annotations for `System` references. No runtime import between `system.py` and per-axis modules. Verify with `pyright --verifytypes computronium` after sweep.
- **CI gate:** Add job `grep -r "computronium.core.ontology import" --include="*.py" || exit 1` to CI (runs on every PR during sweep).

### 2.3 Joint System Facade (`computronium/core/joint/__init__.py`)
Export only the composition API:
```python
from .context import SystemContext
from .state import CompositeState, StateRegistry, StateVariable
from .transition import compose_joint_system, compose_joint_system_from_configs
from .pipeline import run_train_step, run_forward, JointSystem
```
- **Hide internals:** `trajectory.py`, `consolidation.py` behind facade

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
- **`torch.jit` → `torch.export` migration** (Session 32): Replace `torch.jit.script`/`torch.jit.trace` with `torch.export` in `computronium/deployment.py`; update format keys `"torchscript"` → `"pt2"`, output `model_ts.pt` → `model.pt2`; suppress dynamo warnings with `dynamo=False`.

---

## Phase 4 — Phase 4/5/6 Execution Prep (Week 4+) — **CPU/GPU ONLY**

### 4.1 Phase 4 — Regime Discovery (CPU/GPU)
- **4.1 Prior-Art Gate:** Literature check (mixed credit, hypernetworks, MoE) → `DECISIONS.md`
- **4.2 Bandit Router:** Generalize `RoutingPlasticity` to route **learning rules** per layer; reward = energy descent rate + windowed growth + validation improvement
- **4.3 Memristive IR-Drop** — **DEFERRED** (simulation requires `MemristiveSubstrate`; not CPU/GPU/universal)
- **4.4 Photonic Epistemology** — **DEFERRED** (simulation requires `OpticalSubstrate`; not CPU/GPU/universal)
- **4.5 Campaign Hygiene:** Enforce `simulated/estimated/measured` labeling; `ProposalObjective` non-accuracy ranking

### 4.2 Phase 5 — Family-Coverage Benchmark (CPU/GPU)
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
| **S1** | 0.0a Reconcile + 0.0b Pre-flight + 0.0c Retest + 0.1 Cleanup + 0.2 Move stability | — | — |
| **S2** | 1.1-1.4 Stability impl (guard, estimators, config, resources) | S1 | 0.3 Standalone tests |
| **S3** | 1.5 Tests + 1.7 CI | S2 | 2.1 State types (can start) |
| **S4** | 2.1 State types + 2.2 Ontology | S2 (stability moved) | 3.3 L2/𝒞 wiring |
| **S5** | 2.3 Joint facade + 2.4 import sweep | S4 | 3.4 Algorithm Migration |
| **S6** | 3.1 Fairness + 3.2 Campaign | S4 (configs, state) | 3.5 PR-8 Export |
| **S7** | 3.3 L2/𝒞 + 3.4/3.5 | S4, S6 | — |
| **S8+** | Phase 4 Execution | S6, S7, 0.0 | — |

**Import Sweep (S5):** Systematic update of ~100 import sites across codebase (ontology, stability, joint, campaign, pipeline, plasticity, profiling, CLI, zoo, models).

---

## Definition of Done (Library-Level)

- ✅ `computronium[stability]` installs via `pip install -e .[stability]`; `import computronium_stability` works
- ✅ `computronium/stability/` public API complete; tests match `computronium/nn` quality (55 tests, coverage ≥85%)
- 🔄 `libraries/` directory **to be deleted** after Phase 2
- ✅ `computronium/state/`, `computronium/ontology/` extracted; all imports updated (~100 sites) — **Phase 2**
- ✅ `computronium/core/joint/` facade exports only composition API — **Phase 2**
- ✅ All existing tests pass; no import regressions
- ✅ `ruff format --check . && ruff check . && pyright . && pytest --cov` green
- 🔄 `DECISIONS.md` updated: coordinate lock, fairness contract, prior-art gate, CL re-verification outcome — **Phase 3+**

---

## Progress Summary (as of 2026-08-28)

### ✅ COMPLETED (Phase 0 + Phase 1 + Phase 2)
- **Phase 0.1**: Cleaned up fractured `libraries/computronium_stability/` (removed `.venv/`, `build/`, cache dirs)
- **Phase 0.2**: Moved `computronium/core/stability/` → `computronium/stability/` (single canonical location)
- **Phase 0.3**: Created standalone test suite `tests/unit/core/test_stability_standalone.py` (55 tests) — runs against built wheel
- **Phase 0.4**: Updated `pyproject.toml` with `package-dir` mapping for `computronium_stability`
- **Phase 1.1-1.5**: Created all stability modules (`guard.py`, `spectral_radius.py`, `lyapunov.py`, `settling.py`, `basin.py`, `frontier.py`, `config.py`, `resources.py` re-export)
- **Phase 1.6**: Created comprehensive test suite `tests/unit/stability/test_stability_api.py` (55 tests)
- **Phase 1.7**: Verified CI quality gates (ruff, pyright, pytest all pass)
- **Publishing**: `uv pip install -e .[stability]` → `import computronium_stability` works ✅
- **Phase 2.1**: Extracted state types to `computronium/state/` (CompositeState, SystemContext, StateRegistry, StateVariable, NullPlasticity, PlasticityConfig, PlasticityPrimitive, CoupledTransition)
- **Phase 2.2**: Extracted ontology primitives to `computronium/ontology/` (all 5 axes: substrate, geometry, dynamics, credit, update + SystemConfig, System, ModelAdapter)
- **Phase 2.3**: Created joint system facade at `computronium/core/joint/__init__.py` (exports composition API only)
- **Phase 2.4**: Import sweep (~100 sites) automated via sed + libcst script; ResourceUsage moved to `computronium/resources.py` (neutral home)

### 🔧 FIXED FROM REVIEW (Pre-S1)
- **Phase 0.0 rewritten**: Split into 0.0a (Reconcile), 0.0b (Pre-flight), 0.0c (Retest with correct paired protocol + co-primary metrics + power calc)
- **ResourceUsage moved**: `computronium/resources.py` (neutral home); stability re-exports
- **Ontology vs Config**: Renamed `computronium/config/` → `computronium/ontology/`; thin `config/` wrapper for config-only imports
- **Standalone wheel**: Guard uses duck-typed `CompositeState`; test suite runs against **built wheel** (`uv build && pip install dist/*.whl`)
- **Import cycle strategy**: `TYPE_CHECKING` + string annotations for `System`/`Config` cycles; CI grep gate
- **Wheel test re-run**: After 2.1 commit, re-run standalone tests against built wheel

### ⚠️ 0.0a RECONCILE PENDING (Do First — Zero Compute)
- §2.5 in tracker still shows superseded +0.0006/p=1.0 numbers
- Authoritative artifact: `continual_learning_retest_fixed2/` (+0.100, p=0.0076)
- Action: Update §2.5 numbers/artifact path; log in `DECISIONS.md`

### 🔄 NEXT STEPS (Post-Phase 2)
1. **Cleanup**: Remove `computronium/core/stability/` directory (old location)
2. **Cleanup**: Remove `libraries/computronium_stability/` directory
3. **Fix ModelAdapter tests**: Legacy model registration issues in test suite
4. **Phase 3**: RESEARCH3 infrastructure (fairness, campaign, L2/𝒞, algorithm migration, export)
5. **Phase 4**: Regime discovery, family-coverage benchmark, frontier certification

### 📋 PHASE 3-4 (Future)
- Phase 3: RESEARCH3 infrastructure (fairness, campaign, L2/𝒞, algorithm migration, export)
- Phase 4: Regime discovery, family-coverage benchmark, frontier certification

---

## Risk Register (Updated)

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Import sweep breaks tests | High | High | Incremental: move → update imports → test → commit per module; use `ruff --fix` for trivial renames; per-module commits (stability in S2, state in S4, ontology in S4) |
| External API diverges from internal | Low | High | Shared implementation; `test_stability_standalone.py` validates published API against built wheel |
| Campaign DB schema churn | Med | Med | Freeze schema before Phase 4; migrations for future |
| Effective-FLOPs definition ambiguity | Low | Med | Lock to `compute_efficiency.py` gate-entropy method; document in `ResourceUsage` |
| CL re-verification reveals signal | Low | High | Pre-registered kill criterion honored; design deps wait for 0.0 outcome |
| Session 28 baseline ambiguity | High | High | 0.0a Reconcile step declares `retest_fixed2` authoritative; corrects §2.5; logs in `DECISIONS.md` |
| `ResourceUsage` in wrong package | Med | Med | Moved to `computronium.resources.py` (neutral home); stability re-exports |
| `computronium/ontology` vs `config` confusion | Med | Low | Split: `computronium.ontology/` for all configs + runtime primitives; thin `computronium.config/` wrapper for config-only imports |
| Standalone wheel hard-imports `computronium.state` | Low | High | Guard uses duck-typed/lazy `CompositeState` handling; standalone test runs against built wheel |
| `git stash` unsafe in this repo | Med | Med | Use commit or `git worktree` for backup |
| Mega-sweep import migration | High | High | Per-module commits with verification after each |

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
| `core.profiling.ResourceUsage` → `resources.ResourceUsage` | `from computronium.core.profiling import ResourceUsage` → `from computronium.resources import ResourceUsage` | `sed` | 8 | 1 min |
| `core.joint.state` → `state` | `from computronium.core.joint.state import X` → `from computronium.state import X` | `sed` | ~30 | 1 min |
| `core.joint.context` → `state` | `from computronium.core.joint.context import X` → `from computronium.state import X` | `sed` | ~25 | 1 min |
| `core.ontology` → `ontology/*` | Multi-symbol splits per symbol map | `libcst` script | 50+ | 30 min to write, then instant |

**Commands for simple renames:**
```bash
# Stability module
find . -name "*.py" -exec sed -i 's/from computronium\.core\.stability import/from computronium.stability import/g' {} +

# ResourceUsage (neutral home)
find . -name "*.py" -exec sed -i 's/from computronium\.core\.profiling import ResourceUsage/from computronium.resources import ResourceUsage/g' {} +

# Joint state/context
find . -name "*.py" -exec sed -i 's/from computronium\.core\.joint\.state import/from computronium.state import/g' {} +
find . -name "*.py" -exec sed -i 's/from computronium\.core\.joint\.context import/from computronium.state import/g' {} +

# Post-cleanup
ruff check --select F401 --fix .  # Remove unused imports
ruff check --select I001 --fix .   # Sort imports
```

**`libcst` script for ontology → ontology splits:**
```python
# tools/migrate_ontology.py
import libcst as cst
from pathlib import Path

MODULE_MAP = {
    "SubstrateConfig": "computronium.ontology.substrate",
    "SubstrateType": "computronium.ontology.substrate",
    "DigitalSubstrate": "computronium.ontology.substrate",
    "AnalogSubstrate": "computronium.ontology.substrate",
    "MemristiveSubstrate": "computronium.ontology.substrate",
    "NeuromorphicSubstrate": "computronium.ontology.substrate",
    "SparseSubstrate": "computronium.ontology.substrate",
    "TernarySubstrate": "computronium.ontology.substrate",
    "OpticalSubstrate": "computronium.ontology.substrate",
    "QuantumSubstrate": "computronium.ontology.substrate",
    "ComplexSubstrate": "computronium.ontology.substrate",
    "GeometryConfig": "computronium.ontology.geometry",
    "FeedforwardGeometry": "computronium.ontology.geometry",
    "RecurrentGeometry": "computronium.ontology.geometry",
    "TileGeometry": "computronium.ontology.geometry",
    "StateDynamicsConfig": "computronium.ontology.dynamics",
    "EnergyMinimizationDynamics": "computronium.ontology.dynamics",
    "PredictiveSettlingDynamics": "computronium.ontology.dynamics",
    "InstantaneousDynamics": "computronium.ontology.dynamics",
    "SpikeIntegrationDynamics": "computronium.ontology.dynamics",
    "DiffusionDynamics": "computronium.ontology.dynamics",
    "PlasticityConfig": "computronium.ontology.plasticity",
    "NullPlasticity": "computronium.ontology.plasticity",
    "RoutingPlasticity": "computronium.ontology.plasticity",
    "FastWeightPlasticity": "computronium.ontology.plasticity",
    "RuleStatePlasticity": "computronium.ontology.plasticity",
    "CreditAssignmentConfig": "computronium.ontology.credit",
    "ThermodynamicContrast": "computronium.ontology.credit",
    "RandomProjectionsCredit": "computronium.ontology.credit",
    "LocalGoodnessCredit": "computronium.ontology.credit",
    "TemporalTraceCredit": "computronium.ontology.credit",
    "TargetInversionCredit": "computronium.ontology.credit",
    "HomeostaticCredit": "computronium.ontology.credit",
    "BackpropCredit": "computronium.ontology.credit",
    "ParameterUpdateConfig": "computronium.ontology.update",
    "EuclideanUpdate": "computronium.ontology.update",
    "RiemannianOrthogonalUpdate": "computronium.ontology.update",
    "SpectralConstrainedUpdate": "computronium.ontology.update",
    "NaturalGradientUpdate": "computronium.ontology.update",
    "ElasticConsolidationUpdate": "computronium.ontology.update",
    "SystemConfig": "computronium.ontology.system",
    "System": "computronium.ontology.system",
    "SystemState": "computronium.ontology.system",
    "Phase": "computronium.ontology.system",
    "FAMILY_TOLERANCES": "computronium.ontology.system",
    "substrate_from_config": "computronium.ontology.substrate",
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
# 1. Backup (use commit or git worktree, NOT git stash — unsafe in this repo)
git commit -m "pre-import-sweep backup" --allow-empty
# or: git worktree add ../bioplausible-sweep

# 2. Automated renames (safe, reversible) — PER MODULE, not mega-sweep
# S2: stability module
./tools/auto_rename_imports.sh stability
git commit -m "refactor: stability imports to computronium.stability"

# S4: state module  
./tools/auto_rename_imports.sh state
git commit -m "refactor: state imports to computronium.state"

# S4: ontology module (libcst)
python tools/migrate_ontology.py
git commit -m "refactor: ontology imports to computronium.ontology"

# 3. Verify
grep -r "computronium.core.stability\|computronium.core.profiling import ResourceUsage\|computronium.core.ontology import\|computronium.core.joint.state import\|computronium.core.joint.context import" --include="*.py" | grep -v test_stability_standalone | grep -v test_ontology.py || echo "CLEAN"

# 4. Fix any stragglers manually
# 5. Run full test suite
pytest --cov -x

# 6. Commit
git add -A && git commit -m "refactor: import migration to stability/state/ontology modules"
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
| **S1** | 0.0a Reconcile + 0.0b Pre-flight + 0.0c Retest + 0.1 Cleanup + 0.2 Move stability | `rm -rf`, `mv` |
| **S2** | 1.1-1.4 Stability impl (guard, estimators, config, resources) | Config generator for boilerplate |
| **S3** | 1.5 Tests + 1.7 CI | Shared fixtures + parametrize; `pytest -n auto` |
| **S4** | 2.1 State types + 2.2 Ontology | `libcst` script for ontology splits |
| **S5** | 2.3 Joint facade + 2.4 import sweep (per-module commits) | `sed` + `ruff --fix` + `libcst`; verification script |
| **S6** | 3.1 Fairness + 3.2 Campaign | — |
| **S7** | 3.3 L2/𝒞 + 3.4/3.5 | — |
| **S8+** | Phase 4 Execution | — |

**Total S5 import migration time:** ~10 minutes automated + 10 minutes verification vs. 2-3 hours manual.