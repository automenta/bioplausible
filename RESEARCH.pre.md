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
