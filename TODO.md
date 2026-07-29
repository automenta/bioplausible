# Refactoring TODO — Bioplausible Codebase

> Goal: improve elegance, clarity, DRY, maintainability, and alignment with `@AGENTS.md`.
> **Strategy**: Archive noise first (dead code, syntax fixes), then fix foundations (registries, configs, core types), then deduplicate algorithms, then harden type system.

---

## 0. Survey Summary (Updated)

### Completed Dead Code Removal (Package — ~1800 lines removed)

| Item | Lines | Status |
|------|-------|--------|
| `analysis/legacy_report/` | 4 files, ~1777 | **ARCHIVED** → `docs/archive/20260729/dead_package_code/analysis/legacy_report/` |
| `zoo/mep/optimizers/o1_memory.py` | 435 | **ARCHIVED** → `docs/archive/20260729/dead_package_code/zoo/mep/optimizers/` |
| `zoo/mep/optimizers/inspector.py` | 167 | **ARCHIVED** — `ModelInspector` removed from exports |
| `zoo/mep/optimizers/monitor.py` | 262 | **ARCHIVED** — `EPMonitor`/`monitor_ep_training` removed from exports |
| `equitile/lm_demo/profiling.py` | 508 | **ARCHIVED** — duplicates `equitile/profiler.py` |
| `EqPropLMWrapper` | 31 | **REMOVED** from `eqprop_lm_variants.py` + `__init__.py` |
| `_apply_feedback_alignment()` + call | 2 | **REMOVED** empty `pass` in `fa.py:71` and call site |
| `_apply_direct_feedback()` + call | 2 | **REMOVED** empty `pass` in `fa.py:140` and call site |
| `_settle()` wrapper | 12 | **REMOVED** trivial delegate in `eqprop.py:102` → direct calls updated |
| `train_step` no-op | 8 | **REMOVED** just calls `super()` in `finite_nudge_ep.py:33` |
| Duplicate `return` | 1 | **REMOVED** unreachable in `wrappers.py:102` |
| Commented code | 1 | **REMOVED** `# logits_nudged = ...` in `base.py:428` |
| Duplicate import | 1 | **REMOVED** already imported in `hebbian.py:19` |

**Net: ~1800 lines removed from package** (archived for history)

### Completed Dead Code Removal (Outside Package — ~4000 lines)

| Item | Lines | Status |
|------|-------|--------|
| `examples/legacy/` | 17 files, ~2800 | **ARCHIVED** → `docs/archive/20260729/legacy_examples/` |
| `scripts/legacy/` | 16 files, ~1076 | **ARCHIVED** → `docs/archive/20260729/legacy_scripts/` |
| `tests/test_report_generation.py` | 191 | **ARCHIVED** → `docs/archive/20260729/legacy_tests/` (tested dead code) |
| `tests/test_report_analysis_robustness.py` | 77 | **ARCHIVED** (tested dead code) |

### Completed Dead Demo/Config Class Removal

| Item | Status |
|------|--------|
| `ArchitectureConfig` (config.py) | **REMOVED** — never used outside config.py |
| `OptimizationConfig` (config.py) | **REMOVED** — never used outside config.py |
| `DynamicsConfig` (config.py) | **REMOVED** — duplicated builder.py version; collided with `DynamicEquiTileConfig as DynamicsConfig` |
| `to_architecture_config/to_optimization_config/to_dynamics_config` (config.py) | **REMOVED** — unused methods referencing removed classes |
| `CurriculumScheduler` (enhanced.py) | **REMOVED** — `get_sample_weights()` always returns `torch.ones()` (dead logic); name collides with `data/curricula.py` |
| `enable_curriculum()` (builder.py) | **REMOVED** — used dead `CurriculumScheduler` |
| `build_enhanced_model` `enable_curriculum` param | **REMOVED** |

### Completed Syntax Fixes

| Issue | Files Fixed |
|-------|-------------|
| `steps: int = None` → `int \| None = None` | `ternary.py:106`, `causal_transformer_eqprop.py:128`, `neural_cube.py:152`, `eqprop_lm_variants.py:175,278,350,436,499` |
| Legacy `except X, Y:` → `except (X, Y):` | `core/trainer.py:733,882`, `core/registry.py:245` |

### Completed `print()` → `logging` Migration

| File | Replacements |
|------|-------------|
| `equitile/lm_demo/training.py` | 7 |
| `equitile/lm_demo/train_tinystories.py` | 22 |
| `equitile/lm_demo/demo.py` | 43 |
| `equitile/lm_demo/data_advanced.py` | 6 |
| `cli/rank.py` | 3 |
| `experiments/__init__.py` | 0 (print() only in docstring examples) |

**Total: 81 `print()` → `logger.info()` / `logger.warning()` replacements across 5 files.**

### Remaining Duplications (Not Yet Tackled)

| Pattern | Count | Files |
|---------|-------|-------|
| `train_step` implementations | **26** | `zoo/models/*` |
| `forward_step` implementations | **11** | `zoo/models/eqprop/*` |
| `_build_layers` implementations | **12** | `zoo/models/*` |
| `build` classmethods | **18+** | `zoo/models/*` |
| `hidden_dims = [...]` computation | **17** | `zoo/models/*` |
| `DynamicsConfig` name collision (builder.py) | **2** | builder.py `DynamicsConfig` vs `DynamicEquiTileConfig as DynamicsConfig` |
| `MemoryProfiler` class | **2** | `equitile/profiler.py` (the `lm_demo/profiling.py` was archived) |
| `ProfileResult` class | **2** | `equitile/profiler.py` (the `lm_demo/profiling.py` was archived) |
| Tile communicator classes | **2** | `TileCommunicator` vs `NCCLCommunicator` |
| `LearningConfig` (builder.py) | **1** | Deeply integrated into builder — deferred to Phase 1 |

---

## Phase 0: Quick Wins — **COMPLETED** ✅

*All items in Phase 0 are complete. Net result:*

| Metric | Value |
|--------|-------|
| Files changed | **121** |
| Lines inserted | **+623** |
| Lines deleted | **-8532** |
| Net reduction | **-7909 lines** |
| Archives created | `docs/archive/20260729/` (dead code preserved for history) |
| Archives line count | **~5800 lines** in 4 subdirs |
| Tests passing | **1117 passed**, 15 skipped (55% coverage) |
| pyright errors | **5 pre-existing** (no new errors introduced) |

---

## Phase 1: Foundational Architecture (2–3 days)

### 1.1 Unify 4 Registries → 1

| Registry | File | Action |
|----------|------|--------|
| `NEBCRegistry` | `zoo/nebc_base.py:73-104` | **Replace** with `register_model` + `Registry.get` |
| `TaskRegistry` | `hyperopt/task_registry.py` | **Replace** — add `ComponentCategory.TASK`, register there |
| `track_registry` | `validation/tracks/track_registry.py` | **Refactor** — add `ComponentCategory.TRACK` |
| `register_nebc` decorator | `zoo/nebc_base.py:104` | **Remove** |

**Archive location**: `docs/archive/20260729/registries/`

### 1.2 Unify Config Dataclasses

After Phase 0 removals, only one config per concern remains. Add `frozen=True, slots=True`.

### 1.3 Add `frozen=True, slots=True` to Core Dataclasses

**Files**: `zoo/base.py`, `core/trainer.py`, `core/registry.py`, `equitile/config.py`, `equitile/builder.py`, `data/curricula.py`.

**Note**: `LearningConfig` in `builder.py:80-101` is still present — it's deeply wired into the builder's internal state. Defer removal to Phase 1 (or skip — it's a simple internal dataclass with no duplication cost).

---

## Phase 2: Core Type Safety (3–4 days)

### 2.1 Eliminate `Any` from Core Files

| File | Lines | Strategy |
|------|-------|----------|
| `core/registry.py` | 13, 113, 253, 263, 269, 275, 419 | `dict[str, object]`, `Component: TypeVar`, `Any` → `object` |
| `core/trainer.py` | 17, 88, 92, 96, 100, 145, 148 | `dict[str, object]`, `TypedDict` |
| `zoo/base.py` | 14, 46 | `ModelConfig.extra: dict[str, object]` |
| `zoo/models/base.py` | 3, 30, 68, 454 | `ctx: object`, `dict[str, object]` |
| `zoo/propagators/base.py` | 31 | `params: Iterable[nn.Parameter]` |
| `acceleration/_array_ops.py` | 7, 14, 23, 38, 47, 59, 86, 96 | `xp: object`, `type[Protocol]` |
| `equitile/state_types.py` | 3 | `Any` in `TypedDict` → `object` |

### 2.2 Add Type Annotations to Propagator `params`

### 2.3 Replace f-string Logging with t-strings (Core First)

---

## Phase 3: Algorithmic Deduplication (5–7 days)

*Now easier because Phase 0 removed ~6000 lines of noise.*

### 3.1 Extract Settling Loop Helper

**New module**: `bioplausible/zoo/_settling.py`

**Refactor targets (13+ classes)**: `EqPropModel`, `StandardEqProp`, `DirectedEP`, `HolomorphicEP`, `GraphEqProp`, `LoopedMLP`, `MemoryEfficientLoopedMLP`, EqProp LM variants, `EqPropDiffusion`, `NeuralCube`, `TemporalResonanceEqProp`, `HomeostaticEqProp`, `LazyEqProp`.

### 3.2 Extract Long Functions (>50 lines)

13 functions: `EqPropModel.forward()` (122), `.contrastive_update()` (89), `.train_step()` (93), `EquilibriumFunction.backward()` (115), `CoreTrainer._train_step()` (70), `._validate()` (74), `._train_epoch()` (73), `StandardEqProp.train_step()` (67), `DirectedEP.train_step()` (59), `HolomorphicEP.train_step()` (58), `GraphEqProp.train_step()` (67), `AdaptiveFeedbackAlignment.train_step()` (65), `StandardFA.train_step()` (70).

### 3.3 Unify Feedback Alignment Backward Passes

**File**: `zoo/models/fa.py` — 9 classes with similar `train_step`. Extract `_fa_backward(activation_derivative_fn)` helper.

### 3.4 Consolidate Language Model Files

**Files**: `equitile/language.py` (1192), `language_optimized.py` (687), `fast_lm.py` (613) = **2492 lines total**

**Plan**:
1. Create `equitile/_components.py` with shared: `TileAttention`, `TileFeedForward`, `PositionalEncoding`, `CausalMask`
2. `language_optimized.py` already imports from `language.py` — extend
3. Move `FastLMConfig` → `equitile/config.py`
4. `fast_lm.py` imports from `_components.py`

**Estimated savings: ~800–1000 lines**

### 3.5 Unify Hidden Dims Computation (17 occurrences)

**Extract**: `_compute_hidden_dims(hidden_dim, num_layers, max_layers=5) -> list[int]`

### 3.6 Consolidate Duplicate `build` Classmethods (18+)

**New helper**: `_build_from_spec(cls, spec, ...)` in `zoo/models/base.py`.

### 3.7 Consolidate Duplicate `_build_layers` (12)

### 3.8 Unify Profiling Code

**Files**: `equitile/profiler.py` (1076) — `lm_demo/profiling.py` archived in Phase 0.

### 3.9 Unify Distributed/Multi-GPU Code

**Files**: `equitile/distributed.py` (994) and `equitile/multigpu.py` (950) = **1944 lines**

Investigate if `TileCommunicator` / `NCCLCommunicator` and `DistributedEquiTile` / `MultiGPUEquiTile` can be unified.

---

## Phase 4: Type System Hardening (3–5 days)

### 4.1 Eliminate Remaining `Any` (All Files)

### 4.2 Replace `dict[str, Any]` with `TypedDict` or `dict[str, object]`

### 4.3 Add `__all__` to All Modules

---

## Verification Gates

After each phase:
```bash
ruff format . && ruff check --fix .
pyright .
pytest --cov
```

**Phase-specific**:
- [x] Phase 0: `git diff --stat` → **-8532 lines** in working tree (archived ~5800 to docs/archive/)
- [x] Phase 0: All syntax errors fixed; print() migrated to logging (81 replacements)
- [ ] Phase 1: `grep -r "NEBCRegistry\|TaskRegistry\|O1MemoryEP\b\|ModelInspector\|EPMonitor" --include="*.py" | grep -v test` → empty
- [ ] Phase 2: `pyright` zero errors on core files
- [ ] Phase 3: Settling loops use shared helper
- [ ] Phase 4: `grep -r "from typing import Any" --include="*.py" bioplausible/` → only tests

**Known pre-existing issues** (not caused by refactoring):
- `deployment.py:717` — `InferenceRequest` undefined (missing import)
- `hyperopt/graph_task.py:28-32` — `os` undefined (missing import)
- `equitile/async_execution.py:325`, `distributed.py:684`, `multigpu.py:674` — `lambda_error` attribute access on `ModelConfig`

---

## Archive Structure

All archived code lives at `docs/archive/20260729/`:

```
docs/archive/20260729/
├── dead_package_code/          # From bioplausible/ package
│   ├── analysis/legacy_report/
│   ├── zoo/mep/optimizers/o1_memory.py
│   ├── zoo/mep/optimizers/inspector.py
│   ├── zoo/mep/optimizers/monitor.py
│   └── equitile/lm_demo/profiling.py
├── legacy_examples/            # From examples/legacy/
├── legacy_scripts/             # From scripts/legacy/
├── legacy_tests/               # Tests for removed dead code
└── registries/                 # (Future: NEBCRegistry, TaskRegistry)
```

**Note**: Archive contains ~53 files. Some `.cover` and `__pycache__` artifacts snuck in — can clean with:
```bash
find docs/archive/20260729/ -name "*,cover" -delete
find docs/archive/20260729/ -name "__pycache__" -exec rm -rf {} + 2>/dev/null
```

This preserves history while cleaning the working tree.

---

## Effort Summary (Updated)

| Phase | Focus | Est. Days | Working Tree Delta | Status |
|-------|-------|-----------|--------------------|--------|
| **0** | Archive dead code & fix syntax | **2–3** | **−7909 lines** (archived ~5800) | **DONE** ✅ |
| 1 | Registries, configs, frozen dataclasses | 2–3 | −200 lines (dedup) | ⏳ NEXT |
| 2 | Core type safety | 3–4 | +200 lines (annotations) | |
| 3 | Algorithmic dedup | 5–7 | **−3000+ lines** (shared helpers) | |
| 4 | Full type hardening | 3–5 | +500 lines (TypedDict, exports) | |

**Total**: ~15–22 days. **Phase 0 complete**: ~7909 lines removed from working tree (archived for history).

**Next recommended step**: Phase 1.1 — Unify 4 registries into 1. The `NEBCRegistry` and `TaskRegistry` are small, self-contained modules ideal for starting Phase 1.

---

## Session Wrap-Up (2026-07-29)

### What Was Done This Session

All Phase 0 work is **complete in the working tree** (121 files changed, +623/-8532) but **NOT committed**. The 3 latest git commits (90984b6, 436c99a, d434cb9) are plan-documentation-only. To commit:

```bash
git add -A && git commit -m "refactor: Phase 0 — archive dead code, fix syntax, migrate print()->logging"
```

### Verification Results (Current Working Tree)

| Gate | Result |
|------|--------|
| `ruff format --check .` | 659 files already formatted — **PASS** |
| `ruff check --select I .` | All checks passed — **PASS** |
| `ruff check --fix .` | 5316 `@typing.override` suggestions (style-only, not actionable) |
| `pyright` | 5 errors (all pre-existing), 1388 warnings — **no new errors** |
| `pytest -q` | 1117 passed, 15 skipped, **55% coverage** (floor=40%) |

### New Discoveries / Issues

1. **pyright error count fixed**: TODO previously claimed "8 pre-existing errors". Actual count is **5** — the earlier count may have included warnings. Corrected.

2. **Archive has `.cover` + `__pycache__` cruft**: The `docs/archive/20260729/` directory preserved `.py,cover` files and `.pyc` caches from the original file tree. Cleanup commands added above.

3. **`ruff check --fix` is noisy**: It flags 5316 missing `@typing.override` decorators. These are style-only suggestions from a new ruff rule. Ignore unless the team explicitly adopts this convention.

4. **Phase 0 work is in working tree, not HEAD**: All refactoring is uncommitted. A future session should `git add -A && git commit` first, then proceed to Phase 1.

### Pointers for Future Sessions

- **Starting Phase 1**: The small registries to tackle first are `TaskRegistry` (hyperopt/task_registry.py — 1 file, ~80 lines) and `NEBCRegistry` (zoo/nebc_base.py:73-104). Archive originals to `docs/archive/20260729/registries/`.
- **Phase 1.2 (frozen dataclasses)**: After registry unification, add `frozen=True, slots=True` to all core dataclasses. This may break some internal mutation code — test aggressively.
- **`LearningConfig` in builder.py**: The 2026-07-29 session confirmed this is deeply wired (40+ `self._learning.*` refs). Consider keeping it as an internal detail; the cost of extraction may exceed the benefit.
- **`EquiTileBuilder` mutation patterns**: The builder mutates `self._learning.*` fields during construction. If you make `LearningConfig` frozen, you'll need to switch to a builder pattern or a mutable proxy.
- **Test coverage is 55%**: The minimum is 40%, so there's headroom. But Phase 3 algorithmic dedup has high refactoring risk — consider writing additional unit tests for the settling loop and FA backward helpers *before* extracting them.
- **Pyright `reportOptionalMemberAccess` warnings**: Propagators (`eqprop.py`, `fa.py`, `hebbian.py`) have ~40 warnings about `.train()` called on `None` (the `_solver` attribute). These are not errors but indicate `None`-guarding could be improved across the board — worth a Phase 2 or 4 sweep.

---

## Discoveries During Phase 0 Work

1. **`builder.py:LearningConfig` kept**: The TODO says to remove `LearningConfig` because it duplicates `EquiTileConfig` fields, but removing it requires significant refactoring of `EquiTileBuilder` internals (40+ references to `self._learning.*`). Deferred to Phase 1 or may be kept as-is since it's an internal implementation detail with no external cost.

2. **`eqprop_diffusion.py` line 28**: The TODO listed `steps: int = None` at `eqprop_diffusion.py:28` but this file's `__init__` signature doesn't have that pattern. The file was verified and no fix was needed.

3. **`.cover` files**: The repo has stale `.cover` copies (e.g., `__init__.py,cover`, `deployment.py,cover`) from a previous coverage run. These can be cleaned up with `find . -name "*,cover" -delete`.

4. **`_settle()` was called from production code**: The `AdamEqProp` class in `eqprop.py:335-336` called `self._settle()`. These calls had to be updated to `self._settle_phase_direct()` alongside the test updates.

5. **Test file deletion**: Two test files (`test_report_generation.py`, `test_report_analysis_robustness.py`) imported the removed `legacy_report` module and had to be archived. The `test_refactor2_bugfixes.py` module-import test also referenced `legacy_report` and was updated.