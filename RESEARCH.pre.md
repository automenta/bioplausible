# Bioplausible Pre-Development Refactoring Plan

**Goal**: Retire architectural debt before adding new features from RESEARCH.md. Prioritize refactors that unlock maintainability, testability, and downstream velocity — defer mechanical churn to opportunistic "while you're here" passes.

**Scope**: 267 Python files, ~79K LOC. Surgical, not greenfield.

**Guiding principle**: High-impact architecture first. Mechanical cleanups (comments, docstrings, cosmetic) happen opportunistically while touching a file for a higher-priority reason — never as standalone work.

---

## Severity Legend

- **C** — critical: correctness bug, build-breaking, or fails CI gate
- **H** — high: AGENTS.md rule violation that ruff/pyright will flag
- **A** — architectural: high-leverage structural change that reduces future debt
- **M** — medium: idiomatic improvement, localized
- **L** — low: cosmetic, no functional or lint impact

---

## Tier 1: High-Impact Architecture Work

*These refactors change how the codebase is structured, not just how it looks. They unblock future feature work and make the framework easier to reason about.*

### 1.1 Domain Exception Hierarchy (Foundation for ALL Error Handling)
**File**: `bioplausible/core/exceptions.py` (NEW)
**Rule**: AGENTS.md "Define a small custom hierarchy per domain. Always chain."
**Severity**: **A**

There is exactly **one** `raise X from Y` in the entire codebase, and **127** `except Exception` clauses. This is the single highest-leverage refactor: it enables every safe-narrowing and exception-chaining fix downstream.

```python
# bioplausible/core/exceptions.py
class BioplausibleError(Exception):
    """Base for all bioplausible domain errors."""

class ConfigError(BioplausibleError): ...
class RegistryError(BioplausibleError): ...
class IncompatibilityError(RegistryError): ...   # already exists — re-parent
class CheckpointError(BioplausibleError): ...
class LoadStateError(CheckpointError): ...
class KnowledgeBaseError(BioplausibleError): ...
class TrialExecutionError(BioplausibleError): ...
class PropagatorError(BioplausibleError): ...
class TileGraphError(BioplausibleError): ...
```

**Then**: Adopt throughout. Each `except Exception` site either:
1. Narrows to a specific exception set and re-raises a domain error chained (`raise LoadStateError("msg") from e`), OR
2. Stays broad at a true top-level recovery boundary (e.g. `engine.py` trial retry) but logs via `logger.exception(...)` (preserves the traceback) rather than `logger.warning(...)`.

**Downstream impact**: Every subsequent refactor and feature file can rely on a stable error protocol. AutoScientist can catch `TrialExecutionError` instead of bare `Exception`.

### 1.2 Registry Query Architecture — `_QueryFilter` Strategic Refactor
**File**: `bioplausible/core/registry.py:120-165`
**Rule**: AGENTS.md "Composition over Inheritance", `match`/`case`, frozen value objects.
**Severity**: **A**

**Why architectural**: The `Registry` is the heart of the framework — every model/propagator/optimizer/hyperopt is discovered through it. `_QueryFilter.matches` has cc=19 and is the bottleneck for AutoScientist composition. This refactor unlocks future capability matching (Phase 4 AutoScientist discovery in RESEARCH.md).

**Current**: One boolean mega-expression with 9 conjunction clauses.

**Proposed**: Convert to a **predicate dispatch table** — each axis becomes a small `_Predicate` callable, the filter holds a `tuple[_Predicate, ...]`, and `matches` does a short-circuit `all(p(meta) for p in self._predicates)`. Predicates are frozen dataclasses implementing `__call__`.

```python
class _Predicate(Protocol):
    def __call__(self, meta: ComponentMetadata) -> bool: ...

@dataclass(frozen=True, slots=True)
class _DomainIn:
    domains: frozenset[Domain]
    def __call__(self, meta: ComponentMetadata) -> bool:
        return any(d in meta.domains for d in self.domains)
# ...one per axis
```

`_QueryFilter.build()` composes the predicate tuple; `matches()` becomes a one-liner. This also makes the filter trivially testable per-axis with `hypothesis`.

### 1.3 Cyclomatic Complexity Extraction — Registry & Engine Hot Paths
**Rule**: AGENTS.md "Let Ruff rules (`C901`, `PLR09xx`) enforce function size. Extract `_`-prefixed helpers."
**Severity**: **A** (for hot paths) / **M** (for low-traffic code)

Prioritize the **architecturally significant** over-complex functions — those that gate feature work or are on the critical path. Low-traffic helpers can wait.

| Priority | Function | File:Line | cc | Why Architectural |
|----------|----------|-----------|----|---------------------|
| **P1** | `_run_discovery_loop` | `execution/engine.py:203` | 17 | AutoScientist main loop — blocks Campaign v1 (RESEARCH.md Phase 4.1) |
| **P1** | `_process_with_retry` | `execution/engine.py:410` | 12 | Same loop — retry/backoff logic entangled with classification |
| **P1** | `_relax` | `equitile/core/model.py:428` | 16 | EquiTile training critical path — blocks scaling sweep (RESEARCH.md 1.7.1) |
| **P1** | `_apply_hebbian_updates` | `equitile/core/model.py:770` | 13 | Same critical path — error propagation + weight update entangled |
| **P2** | `analyze_knowledge_base` | `autoscientist/reasoner.py:218` | 12 | Blocks LLM reasoning upgrade (RESEARCH.md 4.1) |
| **P2** | `load_state` | `equitile/core/model.py:967` | 11 | Checkpoint round-trip — blocks reproducibility work (RESEARCH.md 0.3) |
| **P3** | `search` / `_matches_filters` | `knowledge/kb.py:432, 479` | 12, 11 | KB synthesis — blocks meta-analysis (RESEARCH.md 4.2) |

**Pattern**: Extract `_`-prefixed helpers with single responsibility. Each helper should be independently testable. Use guard-clause flattening throughout.

### 1.4 Control Flow Modernization — `match`/`case` Conversion
**Rule**: AGENTS.md "Use `match`/`case` for complex state/data routing, favoring it over chained `if/elif`."
**Severity**: **A** (where state is closed enum) / **M** (where pattern)

Targeted, not blanket. Only convert where:
- The dispatched value is a closed `StrEnum` / `Literal` (exhaustive `match` catches new variants at review time)
- The chain is ≥3 branches deep
- `BioModel._get_activation` already uses `match` (line 86) — `EquiTile._get_activation` (model.py:292) is inconsistent **in the same project**

| Location | Chain | Closed? | Action |
|----------|-------|---------|--------|
| `equitile/core/model.py:292` `_get_activation` | 5-way on `activation: Literal[...]` | ✅ | Convert — consistency with `BioModel` |
| `equitile/core/model.py:489` `train_step` | 3-way on `mode: Literal["pc","ep","backprop"]` | ✅ | Convert |
| `execution/engine.py:384` `_log_task_start` | 7-way on `task.__dict__` fields | ⚠️ open | Convert with dataclass extraction first |
| `execution/engine.py:551` `_prepare_fixed_config` | 6-way on task flags | ⚠️ open | Convert with dataclass extraction first |

### 1.5 Module Boundary Hardening — Public API Surface Audit
**Rule**: AGENTS.md "internal modules are `_`-prefixed", "`__init__.py` exposes only the public API via `__all__`"
**Severity**: **A**

The `bioplausible/__init__.py` facade imports nearly everything for registration side-effects. This is intentional but undocumented. Audit:

1. **`bioplausible/equitile/utils/`** — internal helpers (`init_utils.py`, `reproducibility.py`). Should be `_utils/` or moved under `_internal/` unless something external imports them.
2. **`bioplausible/__init__.py`** — split heavy registration from the importable public API. Consider a `bioplausible/_register_all.py` that `__init__` calls explicitly, so `import bioplausible.types` doesn't trigger model registration.
3. **Verify no external/test code imports `_internal/` internals** — add a CI check (`ruff` `TID252` for relative-private-import) if needed.

This unblocks the "low-friction contribution" goal in RESEARCH.md — a clear public/boundary split lets new contributors know what's stable.

### 1.6 SQLite Resource Audit — Standardize on `with`
**Rule**: AGENTS.md "Use context managers for all resource lifecycles."
**Severity**: **A**

`knowledge/kb.py` uses `with sqlite3.connect(...) as conn:` consistently (the model). `execution/_state.py` mixes `with` and manual `try/finally: conn.close()` in 12+ methods.

**Refactor**: Standardize on a small helper:
```python
@contextmanager
def _connect(db_path: str) -> Iterator[sqlite3.Connection]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        yield conn
```
This unblocks RESEARCH.md Phase 4.2 (KB meta-analysis) which will add many new query paths — a single safe helper prevents future resource leaks.

---

## Tier 2: Critical Correctness Fixes (Pre-CI-Gate)

*Fixes that prevent `ruff check` / `pyright` from passing cleanly. Necessary regardless of architecture work.*

### 2.1 `print()` → `logging` in Library Code
**Rule**: AGENTS.md "Never `print()`."
**Severity**: **H**

| File | Print count |
|------|------------|
| `bioplausible/equitile/lm/ablation_study.py` | 52 |
| `bioplausible/equitile/benchmarks/compare_nanoGPT.py` | 38 |
| `bioplausible/equitile/benchmarks/rigorous.py` | 26 |
| `bioplausible/equitile/benchmarks/mot_benchmark.py` | 4 |

**Pattern**: Module-level `logger = logging.getLogger(__name__)`, replace `print(f"…")` → `logger.info("…", arg)` (lazy interpolation).

**Acceptable `print()`**: CLI scripts (`cli/__main__.py`), `if __name__ == "__main__":` demo blocks.

### 2.2 Overly Broad `except Exception` Swallowing Tracebacks
**Rule**: AGENTS.md "Always chain", AGENTS.md Logging.
**Severity**: **H**

After Tier 1.1 (exception hierarchy), narrow and chain these specific swallowers:

| Location | Issue | Fix |
|----------|-------|-----|
| `equitile/core/model.py:998-1001` | `except Exception: logger.warning(...)` for scheduler state | `except (KeyError, RuntimeError) as e: logger.warning("…", exc_info=e)` |
| `equitile/core/model.py:1222-1225` | `except Exception: torch.load(..., weights_only=False)` silent fallback | Catch `(RuntimeError, pickle.UnpicklingError)`; log reason; re-raise `LoadStateError` if both attempts fail |
| `knowledge/kb.py:742, 763, 831, 858, 921` | 5× `except Exception as e: return None` | `except (sqlite3.Error, KeyError, ValueError) as e: logger.exception("…"); raise KnowledgeBaseError("…") from e` |

**Keep broad**: Top-level retry boundary in `engine.py:_process_with_retry` (intentional recovery). But switch `logger.warning` → `logger.exception` so stack traces aren't lost.

### 2.3 Bare-Exception Tuple Syntax (Missing Parens)
**Rule**: ruff `E722` adjacency / consistency.
**Severity**: **H** (lint) / not **C** (verified: parses as tuple form, not Python 2 binding)

In modern Python, `except X, Y:` parses as `except (X, Y):` (tuple form), NOT Python 2 binding. So this is a **consistency/legibility** issue, not a correctness bug. Still worth fixing because the bare form reads as Python 2 to humans.

**17 sites across 12 files** — fix opportunistically while touching a file for Tier 1 work, OR do as one mechanical pass at the end. Full list:

```
bioplausible/core/registry.py:247
bioplausible/execution/_state.py:291
bioplausible/execution/_guards.py:219
bioplausible/execution/synthesizer.py:230, 245
bioplausible/execution/training_dynamics.py:150
bioplausible/analysis/ablation.py:39
bioplausible/analysis/results.py:135
bioplausible/acceleration/backends.py:149
bioplausible/acceleration/kernels.py:57
bioplausible/hyperopt/comparison.py:208, 227
bioplausible/zoo/mep/optimizers/strategies/update.py:204
bioplausible/zoo/mep/optimizers/energy.py:138
bioplausible/equitile/language/canonical.py:856
bioplausible/equitile/lm/components.py:369, 468
```

**Pattern**: `except ValueError, TypeError:` → `except (ValueError, TypeError):`

---

## Tier 3: Type System & Domain Modeling

*Improves pyright strict-mode signal and domain clarity. Do file-by-file alongside Tier 1 work to reduce churn.*

### 3.1 Eliminate `Any` at Internal Boundaries
**Rule**: AGENTS.md "No `Any`: Replace with `object`, generics, or `Protocol`."
**Severity**: **M**

| Location | Current | Refactor |
|----------|---------|----------|
| `bioplausible/core/trainer.py:17` | `**kwargs: Any` | `**kwargs: object` |
| `bioplausible/config/__init__.py:8` | `dict[str, Any]` | `dict[str, object]` at boundary; validate via Pydantic |
| `bioplausible/config/schema.py:11, 221` | `Any` I/O | `Mapping[str, object] \| DictConfig` |
| `bioplausible/equitile/core/config.py:9` | `from typing import Any` | Audit each usage; replace with `object` or narrow |
| `bioplausible/equitile/core/config.py:445, 464, 484, 501, 515` | 5× factory `**kwargs: Any` | `**kwargs: object` (factories pass to `dataclasses.fields()` which takes `object`) |
| `bioplausible/equitile/core/config.py:599` | `dict[str, Any]` field default | TypedDict or `dict[str, object]` |

### 3.2 `Literal` for Open String Fields
**Rule**: AGENTS.md "Use `Literal` / `StrEnum` instead of bare strings."
**Severity**: **M**

- `core/registry.py:93` `credit_assignment_type: str` → `Literal["gradient", "equilibrium", "hebbian", "target", "forward-only", "spiking"]`

**Downstream**: AutoScientist can rely on closed membership when composing capability queries (RESEARCH.md Tier 4.1).

### 3.3 PEP 695 Generics (Where Natural)
**Rule**: AGENTS.md "Use `class Cache[T]: ...` and `type UserId = int`."
**Severity**: **L** (only worth it where the generic actually helps)

- `Registry[T]` parameterized by `ComponentCategory` — only if `_QueryFilter` refactor (1.2) reveals a real type-axis
- `_QueryFilter[T]` — defer until 1.2 lands
- Do **not** retrofit PEP 695 speculatively across the codebase

### 3.4 Dataclass `frozen=True, slots=True` Audit
**Rule**: AGENTS.md internal value objects.
**Severity**: **M** (3 sites)

| Class | File | Missing |
|-------|------|---------|
| `TrainingMetrics` | `core/trainer.py:168` | `slots=True` |
| `LazyStats` | `zoo/models/eqprop/lazy_eqprop.py:19` | `frozen=True` |
| `TileTask` | `equitile/training/async_execution.py:64` | both |

33 other dataclasses are already correct — these are the stragglers.

### 3.5 `builtins.list` → `list` in Annotations
**Severity**: **L**

`core/registry.py:135, 317, 339, 360, 380, 385`. Mechanical.

---

## Tier 4: Opportunistic Hygiene (Do Not Schedule Standalone)

*These are real AGENTS.md violations but have zero architectural impact. Handle them while touching a file for a Tier 1-3 reason — never as dedicated tasks. Scheduling them as standalone work wastes reviewer attention.*

### While-here Cleanups

- **`# WHAT`-comments**: Delete or rewrite as `# WHY` when touching the function. Examples: `# Get …`, `# Initialize …`, `# Compute …`, `# Save to file`, `# Metadata`, `# Determine Job ID`.
- **Missing Google-style docstrings**: Add to public APIs you're already editing. Pending: `EquiTile.to`, `get_stats`, `summarize`, `KnowledgeEntry.to_dict`, `from_dict`.
- **Lazy `%s` logging**: Already dominant (verified: 0 f-strings in logger calls). Maintain for new code.
- **Duplicate `distributed` docstring entry** in `bioplausible/equitile/__init__.py:36-37` — delete on next edit.
- **Adopt PEP 750 t-strings** for new logger code only — do not churn existing.

### What NOT to Refactor

- **OmegaConf config classes** (`config/schema.py`): mutable-by-design.
- **`frozen=False` dataclasses at I/O boundaries**: Pydantic schemas, Optuna trials — intentional.
- **Module-level registration side-effects** (`bioplausible/__init__.py` facade): intentional; document via 1.5 instead of restructuring.
- **`print()` in `__main__` blocks**: legitimate CLI output.
- **Broad `except Exception` at top-level retry boundaries**: intentional recovery — switch to `logger.exception` only.

---

## Execution Order (Dependency-Aware)

```
TIER 1 — Architecture (do first, in this order)
  1.1 Exception hierarchy        ← enables every narrowing/chain fix below
   ├─ 2.2 Narrow except Exception (after 1.1 — uses new types)
  1.2 _QueryFilter refactor      ← independent; enables hypothesis tests
  1.3 Complexity extraction       ← depends on 1.1 for clean re-raises
      ▸ hot paths first (engine loop, EquiTile relax/hebbian)
      ▸ cool paths (KB, reasoner, load_state) in second pass
  1.4 match/case conversion       ← fold into 1.3 extractions
  1.6 SQLite helper               ← independent; prereq for KB work in RESEARCH.md
  1.5 Module boundary audit       ← independent; do alongside 1.3

TIER 2 — CI Gate (parallel with Tier 1, blocks merges)
  2.1 print → logging             ← independent, do early to unlock CI clarity
  2.3 bare-except parens          ← opportunity-cost: do as one final pass

TIER 3 — Types (file-by-file with Tier 1)
  3.1–3.5                          ← ride along with whichever Tier 1 file you touch

TIER 4 — Hygiene (never standalone)
  fold into any Tier 1-3 PR touching the same lines
```

---

## Validation Strategy (Per-Tier Gate)

After **each tier or sub-task**:

1. `uv run ruff format --check .` then `uv run ruff check .` — zero new violations, no regressions
2. `uv run pyright .` — zero new errors (warnings should *decrease*, not stay flat)
3. `uv run pytest --cov` — coverage floor maintained (≥50% per `pyproject.toml`; raise toward ≥85% per AGENTS.md as refactors land)
4. **Behavior parity**: 1-epoch smoke on all models in `tests/slow/` unchanged before/after each architectural refactor. Lock these as snapshot tests if not already.

**Before declaring refactoring complete** (gate for starting RESEARCH.md Phase 0):
- Nightly CI smoke green
- Tier 1 fully landed (all 6 sub-items)
- Tier 2 fully landed (CI clean)
- No new pyright warnings vs. baseline
- Coverage ≥ baseline + delta from any tests added during refactors

---

## Snapshot Tests Guarding Behavior

For each Tier 1.3 refactor (hot-path complexity extraction), **first** add a snapshot/property test that captures current behavior:
- `_relax`: input tensor in → fixed output tensor out (deterministic seed)
- `_apply_hebbian_updates`: weight state before/after
- `_run_discovery_loop`: a mock strategy plan → expected trial dispatch sequence

Then refactor against the snapshot. This makes refactors provably behavior-preserving and leaves a regression net for future feature work (directly serves RESEARCH.md Phase 0.3 "Reproducibility").

---

## Success Criteria

| Metric | Target |
|--------|--------|
| `ruff check` new violations | 0 |
| `pyright` new errors | 0 (new warnings ≤ -baseline) |
| `pytest --cov` floor | maintained, trending up |
| Functions with cc > 10 | 0 |
| `except Exception` sites that swallow (Tier 2.2 list) | 0 |
| `print()` in library code | 0 |
| Bare `except X, Y:` syntax | 0 |
| Domain exception usage | `<module>Error` raised (not `Exception`) at every internal boundary |
| Behavior parity snapshots | All green |

---

## Relationship to RESEARCH.md

This plan is a **precondition**, not a substitute. Refactoring completion is the gate for RESEARCH.md Phase 0 (Foundation Hardening):

- **RESEARCH.md 0.1** (Parity suite) depends on Tier 1.1 (exceptions) + 1.3 (clean hot paths)
- **RESEARCH.md 0.2** (Registry audit) depends on Tier 1.2 (`_QueryFilter` testability)
- **RESEARCH.md 0.3** (Reproducibility) depends on Tier 1.6 (clean SQLite) + snapshot tests from this plan
- **RESEARCH.md Phase 4** (AutoScientist) depends on Tier 1.2, 1.3 (engine loop), 1.4 (match/case)
- **RESEARCH.md Phase 4.2** (KB synthesis) depends on Tier 1.6

*Update this plan as refactors reveal additional debt. The snapshot tests added here become part of RESEARCH.md Phase 0's reproducibility infrastructure.*

---

## Appendix: Additional Architectural Findings (from Deep Codebase Study)

These items emerged from a systematic architectural review of the 267-file codebase. They represent structural opportunities that don't fit neatly into the Tier 1-4 refactors but should inform future work. **Do not schedule as standalone refactors** — fold into relevant feature work from RESEARCH.md.

### A1. Configuration Unification (Three-Layer Config Problem)
**Files**: `bioplausible/core/config.py` (ModelConfig), `bioplausible/config/schema.py` (OmegaConf configs), `bioplausible/equitile/core/config.py` (EquiTileConfig + 7 factory functions)
**Issue**: Three independent config hierarchies with manual conversions (`to_internal()`, `_build_model_config()`). No shared base class or conversion protocol. EquiTileConfig has 70+ flat fields; core ModelConfig has 25; OmegaConf configs have their own structure.
**Impact**: Configuration errors are hard to trace; RESEARCH.md Phase 0.1 parity suite will need to construct configs in all three dialects.
**Approach**: Define a `ConfigProtocol` with `to_internal() -> InternalModelConfig` and `from_internal(config: InternalModelConfig) -> Self`. Make all three config types implement it. Move factory functions to a `ConfigBuilder` that composes the three layers.

### A2. Domain/Execution Task Duplication
**Files**: `bioplausible/domains/base.py` (DomainTask ABC), `bioplausible/execution/task.py` (ExperimentTask dataclass), `bioplausible/validation/tracks/` (tracks consume tasks)
**Issue**: Two task abstractions with different interfaces. `DomainTask` is ABC with `setup()`, `get_dataloader()`, `evaluate()`. `ExperimentTask` is a dataclass with tier, model_name, config dict. Validation tracks expect yet another format. No unified `TaskProtocol` used consistently.
**Impact**: AutoScientist (execution layer) can't cleanly hand off to domain evaluation; validation tracks need adapters.
**Approach**: Extract a `TaskProtocol` (PEP 544) that both satisfy. Make `ExperimentTask` carry a `DomainTask` instance. Validation tracks consume `TaskProtocol`.

### A3. State Persistence Fragmentation (Multiple SQLite Databases)
**Files**: `bioplausible/execution/_state.py` (bioplausible.db for Optuna), `bioplausible/knowledge/kb.py` (bioplausible_kb.db for KB), `bioplausible/validation/tracks/` (various artifact dirs), `checkpoints/` (model checkpoints)
**Issue**: Four+ independent persistence layers with no cross-referencing. Optuna study doesn't link to KB entries; checkpoints don't link to trial IDs.
**Impact**: RESEARCH.md Phase 4.2 (KB meta-analysis) cannot correlate hyperopt results with knowledge entries.
**Approach**: Add a `persistence_id: str` (UUID) to every trial/experiment/checkpoint/KB entry. Create a lightweight `PersistenceIndex` (single table) that maps `persistence_id` → `{trial_id, kb_entry_id, checkpoint_path, track_id}`. Use existing `experiment_id` from KB as the key.

### A4. Optuna Hard Dependency
**Files**: `bioplausible/hyperopt/optuna_bridge.py`, `bioplausible/execution/_state.py`, `bioplausible/execution/engine.py` all import `optuna` at module level.
**Issue**: Optuna is a required dependency, not optional. No abstraction layer — `create_study()`, `TPESampler`, `HyperbandPruner` are used directly. If a different backend is needed (Ray Tune, nevergrad), it's invasive.
**Impact**: Blocks RESEARCH.md Phase 4.3 (surrogate-guided optimization with custom acquisition) which may want a different backend.
**Approach**: Extract a `TrialBackend` protocol with `create_study()`, `suggest()`, `complete_trial()`, `get_best_trials()`. Implement `OptunaBackend` as the default. Make `hyperopt` import Optuna lazily inside the backend. Configuration chooses backend.

### A5. ExecutionEngine / AutoScientistCampaign Split
**Files**: `bioplausible/execution/engine.py` (ExecutionEngine — 900+ lines), `bioplausible/autoscientist/campaign.py` (AutoScientistCampaign — 300+ lines), `bioplausible/autoscientist/proposer.py`, `reasoner.py`, `bridge.py`
**Issue**: Two overlapping autonomous agents. `ExecutionEngine` runs continuous trials with Optuna. `AutoScientistCampaign` uses LLM for hypothesis generation. They share no common interface; `campaign.py` imports `engine.py` but not vice versa. `bridge.py` attempts to connect them but is underused.
**Impact**: RESEARCH.md Phase 4 (AutoScientist) will need to pick one or unify. Duplicate state management, duplicate reporting.
**Approach**: Define a `DiscoveryEngine` protocol (`run_cycle() -> list[TrialResult]`, `get_state() -> EngineState`). Make `ExecutionEngine` and `AutoScientistCampaign` implement it. Create a `CompositeEngine` that alternates or delegates based on tier.

### A6. Parallel Execution Fragmentation
**Files**: `bioplausible/hyperopt/parallel_runner.py` (ParallelTrialRunner), `bioplausible/execution/engine.py` (uses ParallelTrialRunner), `bioplausible/equitile/training/async_execution.py` (AsyncEquiTile), `bioplausible/equitile/training/distributed.py` (DistributedEquiTile), `bioplausible/equitile/training/_nccl.py`
**Issue**: Four parallel execution subsystems with no shared primitives. `ParallelTrialRunner` uses `concurrent.futures.ProcessPoolExecutor`. `AsyncEquiTile` uses `asyncio.TaskGroup`. `DistributedEquiTile` uses NCCL. No common `WorkerPool` abstraction.
**Impact**: Resource contention (GPU memory, CPU) when multiple run simultaneously. No unified backpressure.
**Approach**: Extract a `ComputePool` protocol (`submit(fn)`, `map(fn, iterable)`, `shutdown()`). Implement `ProcessPool`, `AsyncTaskGroup`, `NCCLCommunicator` as backends. `ExecutionEngine` and `EquiTile` both take a `ComputePool` dependency.

### A7. Missing Protocol Interfaces (ABCs over Protocols)
**Files**: `bioplausible/domains/base.py` (DomainTask ABC), `bioplausible/core/model.py` (BioModel ABC), `bioplausible/zoo/propagators/base.py` (BioOptimizer/LearningRuleOptimizer ABCs), `bioplausible/evaluation/base.py` (EvaluatorBase ABC)
**Issue**: AGENTS.md says "Prefer `Protocol` over ABCs". The codebase uses ABCs for key interfaces. Protocols enable structural subtyping (no inheritance required) which is critical for cross-package composition (e.g., EquiTile + Zoo models).
**Impact**: New algorithm implementations must inherit from specific base classes rather than just implementing the structural contract.
**Approach**: For each ABC, define a matching `Protocol` (e.g., `DomainTaskProtocol`, `BioModelProtocol`). Keep ABCs as convenience base classes but type-hint against protocols in consumers. This is a gradual migration — do when touching the ABC file for Tier 1-3 work.

### A8. Global Mutable State Audit
**Files**: `bioplausible/core/registry.py` (`Registry._components` class-level dict), `bioplausible/validation/tracks/track_registry.py` (`ALL_TRACKS` module-level dict), `bioplausible/knowledge/kb.py` (`DEFAULT_KB` lazy via `__getattr__`), `bioplausible/execution/engine.py` (module-level `logging.basicConfig`)
**Issue**: Multiple global singletons with mutable state. Not thread-safe (PEP 703). `Registry._components` is modified at import time via decorators. `ALL_TRACKS` is populated at import. `logging.basicConfig` in `engine.py` affects root logger for entire process.
**Impact**: Test isolation issues; parallel execution may corrupt state; `import bioplausible` has side effects (registration).
**Approach**: 
- `Registry`: Make it a true singleton class with `instance()` method; clear between tests via fixture.
- `ALL_TRACKS`: Move to a `TrackRegistry` class with `register()` and `get_all()`.
- `logging.basicConfig`: Remove from `engine.py`; configure in CLI entry points only.

### A9. Registry Metadata Inference Complexity
**File**: `bioplausible/core/registry.py:254-277` (`_infer_metadata` uses `object.__setattr__` on frozen dataclass)
**Issue**: The registry bypasses frozen dataclass immutability to infer metadata from component classes. This is a workaround for the registration API design where metadata is declared both on the component class and in the decorator.
**Impact**: Obscures mutation; makes `ComponentMetadata` not truly frozen; harder to reason about.
**Approach**: Redesign registration to be explicit: decorator provides ALL metadata; component class is just implementation. Remove `_infer_metadata` and `object.__setattr__` hacks.

### A10. Validation Track Magic Registration
**File**: `bioplausible/validation/tracks/track_registry.py:33-64` (`register_tracks_from_module` inspects for `track_` functions or TRACKS dict)
**Issue**: Track discovery relies on naming conventions and `inspect.getmembers`. Fragile; hard to know which tracks exist without running; no static type checking.
**Impact**: Adding a track is error-prone; IDE can't autocomplete track IDs; CI can't validate track signatures.
**Approach**: Replace with explicit registration: each track module exports a `TRACKS: dict[int, TrackFn]` dict. `track_registry.py` imports and merges them. Add a `TrackProtocol` for the function signature.

### A11. EquiTile Configuration Factory Sprawl
**File**: `bioplausible/equitile/core/config.py` (7 factory functions: `create_production_config`, `create_research_config`, `create_fast_config`, `create_enhanced_config`, `create_dynamic_config`, etc.)
**Issue**: 7 factory functions with overlapping parameter spaces. No clear taxonomy — "production" vs "research" vs "enhanced" are not mutually exclusive. Each has 50+ kwargs. Hard to compose.
**Impact**: RESEARCH.md Phase 1.7.1 (EquiTile scaling sweep) needs clean config composition, not preset selection.
**Approach**: Replace factories with a `EquiTileConfigBuilder` (fluent API) that composes base → domain → mode → hardware overrides. Presets become named builder configurations, not functions.

### A12. Knowledge Base — Feature Bloat in Single Class
**File**: `bioplausible/knowledge/kb.py` (950 lines, 40+ methods: CRUD, vector search, surrogate training, symbolic regression, causal discovery, meta-analysis, export)
**Issue**: `KnowledgeBase` is a god class mixing storage, search, ML (surrogates), statistics, and analysis. Violates SRP. Hard to test; surrogate training pulls in sklearn/pandas at import time.
**Impact**: RESEARCH.md Phase 4.2 (KB meta-analysis) will add more methods. Circular imports risk (KB imports metamodel which imports KB).
**Approach**: Split into: `KnowledgeStore` (CRUD + vector search), `SurrogateModelRegistry` (training/prediction), `MetaAnalyzer` (scaling laws, symbolic regression, causal discovery), `KnowledgeBase` (facade composing the three). Use dependency injection.

### A13. Leaderboard / Validation Track Disconnect
**Files**: `bioplausible/leaderboard/generator.py`, `bioplausible/validation/tracks/track_registry.py`
**Issue**: Leaderboard generates markdown from experiment results. Validation tracks produce structured results. They don't share a common result schema. Leaderboard re-queries databases; tracks don't publish to leaderboard automatically.
**Impact**: Manual step to update leaderboard; results may diverge.
**Approach**: Define a `BenchmarkResult` protocol (already exists in `evaluation/__init__.py`). Validation tracks emit `BenchmarkResult` events. Leaderboard subscribes and updates incrementally.

### A14. Graph Domain Isolation
**Files**: `bioplausible/graph/` (inference.py, initialization.py, nodes.py, topology.py, training.py), `bioplausible/domains/graph.py`
**Issue**: Graph has its own training/inference modules separate from the domains abstraction. `domains/graph.py` wraps some graph functionality but not all. Two independent graph implementations.
**Impact**: RESEARCH.md Phase 2.3 (Graph domain) may need to unify or pick one.
**Approach**: Move `graph/training.py` logic into `domains/graph.py` as a `GraphTask` implementation. Keep `graph/topology.py` and `graph/nodes.py` as utilities. Deprecate duplicate code.

### A15. P2P / Distributed Separation
**Files**: `bioplausible/p2p/` (DHT, evolution, state), `bioplausible/equitile/training/distributed.py` (DistributedEquiTile with NCCL)
**Issue**: Two distributed training approaches: P2P (Kademlia DHT, decentralized) and EquiTile NCCL (centralized, GPU). No shared abstraction; they serve different use cases but could share `Communicator` protocol.
**Impact**: RESEARCH.md Phase 10 (Distributed & P2P) will need to integrate both.
**Approach**: Define a `Communicator` protocol (`all_reduce`, `broadcast`, `barrier`). Implement `NCCLCommunicator` and `DHTCommunicator`. `DistributedEquiTile` and `P2PEvolution` both take `Communicator`.

### A16. Test Architecture — Integration Test Bloat
**Files**: `tests/integration/` (35 files, some >15KB: `test_smoke_training.py` 16KB, `test_lm_demo.py` 19KB, `test_equitile_domains.py` 16KB)
**Issue**: Many "integration" tests are actually end-to-end training runs (download MNIST, train for epochs, assert accuracy). They're slow, flaky, and hard to debug. They duplicate unit test coverage.
**Impact**: CI slow; hard to isolate failures; coverage floor maintained by slow tests, not unit tests.
**Approach**: 
- Move true integration tests (cross-module) to `tests/integration/` (keep ~10).
- Move training smoke tests to `tests/slow/` (already exists, mark with `@pytest.mark.slow`).
- Add property-based tests in `tests/property/` for pure functions (registry, config, kernels).
- Use synthetic data fixtures for fast unit tests (already in `conftest.py`).

### A17. Property-Based Testing — Underutilized
**Files**: `tests/property/` (exists but minimal)
**Issue**: `hypothesis` is in dev dependencies but few property tests exist. Pure functions in `core/registry.py` (`_QueryFilter.matches`), `core/config.py` (`resolve_hidden_dims`), `acceleration/kernels.py` are excellent candidates.
**Impact**: Missing opportunity for high-confidence refactoring guards (especially for Tier 1.2, 1.3).
**Approach**: Add `@given` tests for:
- `_QueryFilter.matches` with strategies for `ComponentMetadata`
- Config resolution with various `hidden_dim`/`num_layers` combos
- Kernel numerical properties (transpose, matmul equivalences)
- Registry query round-trips

### A18. CLI Entry Point Consolidation
**Files**: `pyproject.toml` (4 scripts), `bioplausible/cli/__main__.py`, `bioplausible/cli/lab.py`, `bioplausible/cli/run.py`, `bioplausible/cli/rank.py`, `bioplausible/execution/cli.py`
**Issue**: Multiple CLI modules with overlapping commands. `bioplausible/cli` is the main user-facing CLI; `execution/cli.py` has `main_scientist`, `main_reporter`. No unified command hierarchy.
**Impact**: User confusion; hard to discover features; `biopl-scientist` vs `bioplausible run` do similar things.
**Approach**: Single `bioplausible` CLI with subcommands (`train`, `search`, `scientist`, `report`, `leaderboard`, `validate`). Migrate all entry points to `cli/` using `typer` or `click`.

### A19. Experiment Presets — Untyped Configuration
**Files**: `bioplausible/experiments/presets.py`
**Issue**: Many preset configurations as raw dicts. No validation, no type checking, no documentation of what each preset tests. Used by validation tracks and CI but not integrated with config schema.
**Impact**: Hard to add new presets; drift between preset and actual config schema.
**Approach**: Convert presets to `ExperimentConfig` (OmegaConf) objects. Validate at import time. Generate preset documentation from type annotations.

### A20. Two-Tier Architecture (Model vs Propagator) — Cross-Reference Maintenance
**Files**: `bioplausible/__init__.py` (re-exports model-side classes from propagators), `bioplausible/core/registry.py` (`_PROPAGATOR_TO_MODEL` mapping), `bioplausible/zoo/propagators/__init__.py`
**Issue**: Some algorithms exist in both tiers (FF, PEPITA, TargetProp, PCN are models but re-exported as propagators). The `_PROPAGATOR_TO_MODEL` mapping is manually maintained. When a new model-side algorithm is added, it's easy to forget the cross-reference.
**Impact**: AutoScientist may query wrong tier; users get confusing errors.
**Approach**: Make the two-tier distinction a first-class concept in the registry:
- `ComponentCategory.MODEL` and `ComponentCategory.PROPAGATOR` are separate
- Algorithms that are MODEL-ONLY register only as models; registry returns a typed error with the model name when queried as propagator
- Remove manual `_PROPAGATOR_TO_MODEL` mapping; use a `ModelOnly` marker in metadata
