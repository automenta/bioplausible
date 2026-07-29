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

| Pattern | Count | Files | Status |
|---------|-------|-------|--------|
| `train_step` implementations | **26** | `zoo/models/*` | ⏳ |
| `forward_step` implementations | **11** | `zoo/models/eqprop/*` | ⏳ |
| `_build_layers` implementations | **12** | `zoo/models/*` | ⏳ |
| `build` classmethods | **18+** | `zoo/models/*` | ⏳ NEXT |
| `hidden_dims = [...]` computation | ~~**17**~~ | ~~`zoo/models/*`~~ | **DONE** ✅ |
| `DynamicsConfig` name collision (builder.py) | **2** | builder.py `DynamicsConfig` vs `DynamicEquiTileConfig as DynamicsConfig` | ⏳ |
| `MemoryProfiler` class | **2** | `equitile/profiler.py` | ⏳ |
| `ProfileResult` class | **2** | `equitile/profiler.py` | ⏳ |
| Tile communicator classes | **2** | `TileCommunicator` vs `NCCLCommunicator` | ⏳ |
| `LearningConfig` (builder.py) | **1** | Deeply integrated into builder — deferred to Phase 1 | ⏳ |

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

## Phase 1: Foundational Architecture — **COMPLETED** ✅

### 1.1 Unify 4 Registries → 1

| Registry | File | Action | Status |
|----------|------|--------|--------|
| `NEBCRegistry` | `zoo/nebc_base.py:73-104` | **Removed** — callers use `register_model` + `Registry.get` | Done |
| `TaskRegistry` | `hyperopt/task_registry.py` | **Replaced** — tasks registered via `register_task()` | Done |
| `track_registry` | `validation/tracks/track_registry.py` | **Refactored** — added `ComponentCategory.TRACK` sync | Done |
| `register_nebc` decorator | `zoo/nebc_base.py:104` | **Removed** — 4 usages replaced with `@register_model` | Done |

**What changed**:
- `NEBCRegistry` class removed from `nebc_base.py`. All 4 `@register_nebc(...)` decorators in `hebbian.py` and `fa.py` replaced with `@register_model(...)`.
- `NEBCRegistry.list_all()` callers updated to `Registry.list(ComponentCategory.MODEL)`.
- `NEBCRegistry.create()` callers updated to `Registry.get(...)(...)`.
- Old `TaskRegistry` class removed from `hyperopt/task_registry.py`. Tasks now registered via `register_task()` decorator into core Registry.
- `ComponentCategory.TASK` and `ComponentCategory.TRACK` added to enum.
- `track_registry.py` now syncs its `ALL_TRACKS` entries into core Registry under `TRACK` category.
- `register_task` convenience function added to `core/registry.py`.

### 1.2 & 1.3: Config/Core Dataclasses — `frozen=True, slots=True`

| File | Classes | Status |
|------|---------|--------|
| `zoo/base.py` | `ModelConfig` | **frozen+slots** — `__post_init__` uses `object.__setattr__`; unused `Any` import removed |
| `core/trainer.py` | `TrainingMetrics` | **frozen** — `slots` omitted ( `__dict__` accessed in `to_dict()`) |
| `core/registry.py` | Already had both | **Unchanged** |
| `equitile/config.py` | `EquiTileConfig`, `EnhancedEquiTileConfig`, `NCCLConfig`, `AsyncConfig`, `CurriculumConfig` | **frozen+slots** |
| `equitile/config.py` | `DistributedConfig`, `MultiGPUConfig`, `TileGrowthConfig`, `DynamicEquiTileConfig` | **Kept mutable** — runtime mutation patterns (`device_ids`, `growth.enabled`) |
| `core/trainer.py` | `TrainerConfig` | **Kept mutable** — test code mutates config fields |
| `equitile/builder.py` | All 4 configs | **Skipped** — deeply wired builder mutation patterns (`self._learning.* =`) |
| `data/curricula.py` | No dataclasses | **Skipped** |

**Net change**: +1 file committed (10 files modified, 65 insertions, 106 deletions).

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
- [x] Phase 1: `grep -r "NEBCRegistry\|TaskRegistry\|register_nebc" --include="*.py" bioplausible/` → empty
- [x] Phase 2: `pyright` zero errors on core files (5 pre-existing errors only, 0 new)
- [x] Phase 2: All core files free of `Any` (except OmegaConf-boundary `TrainerConfig` fields)
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
└── registries/                 # (Empty — registries replaced in-place, not archived)
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
| **1** | Registries, configs, frozen dataclasses | **2–3** | **−41 lines** (2 commits) | **DONE** ✅ |
| **2** | Core type safety | **3–4** | **+40 lines** (annotations, 6 files) | **DONE** ✅ |
| 3 | Algorithmic dedup | 5–7 | **+161 lines** (shared infrastructure, −271 existing) | ⏳ **IN PROGRESS** |
| 4 | Full type hardening | 3–5 | +500 lines (TypedDict, exports) | |

**Total**: ~15–22 days. **Phases 0–2 complete, Phase 3 partially done**.

**Progress breakdown**:

| Sub-phase | Status | Lines Changed |
|-----------|--------|---------------|
| 3.1 Extract settling helper | **DONE** ✅ | +432 new, −247 from base.py |
| 3.5 Unify hidden dims | **DONE** ✅ | −174 from 7 model files |
| 3.3 FA backward passes | ⏳ NEXT | Estimated −350 |
| 3.6 Build classmethods | ⏳ NEXT | Estimated −250 |
| 3.4 LM file consolidation | ⏳ | Estimated −900 |
| 3.2 Extract long functions | ⏳ | Estimated −200 |
| 3.7-3.9 Profiling/distributed/etc. | ⏳ | Estimated −400 |

**Next recommended step**: Phase 3.6 (Consolidate `build` classmethods) — it's mechanical, low-risk, and touches 18+ files. Use the same pattern as Phase 3.5: extract a shared `_build_from_spec` helper into `zoo/models/base.py`, then refactor each `@classmethod def build` to call it.

Alternatively, Phase 3.3 (FA backward pass unification) if you want higher line savings per file touched.

---

## Session Wrap-Up (2026-07-29 — Session 2)

### What Was Done This Session

**Commit 1** (9a02fba): Phase 0 — archive dead code, fix syntax, migrate `print()`→`logging`.  
**Commit 2** (de96ffc): Phase 1 — unify registries, freeze dataclasses.

Both commits are on `main`. Working tree is clean.

### Verification Results (HEAD = de96ffc)

| Gate | Result |
|------|--------|
| `ruff format --check .` | 659 files formatted — **PASS** |
| `ruff check --select I .` | All checks passed — **PASS** |
| `ruff check --fix .` | 5312 `@typing.override` suggestions (style-only, not actionable) |
| `pyright` | 5 errors (all pre-existing), 1403 warnings — **no new errors** |
| `pytest -q` | 1117 passed, 15 skipped, **55% coverage** (floor=40%) |

### Registry Unification Details

| Step | Files Touched | Lines Changed |
|------|---------------|---------------|
| Add `TASK`, `TRACK` to `ComponentCategory` | `core/registry.py` | +4 |
| Remove `NEBCRegistry` class, `register_nebc` alias | `zoo/nebc_base.py` | -28 |
| Update 4 `@register_nebc` → `@register_model` | `hebbian.py`, `fa.py` | -4 |
| Update NEBC test file | `test_nebc_base.py` | -5, +24 |
| Replace `TaskRegistry` class with inline `register_task` calls | `hyperopt/task_registry.py` | -32, +17 |
| Sync `track_registry` into core Registry | `track_registry.py` | +5 |

### Frozen Dataclass Details

| Class | File | Result | Reason |
|-------|------|--------|--------|
| `ModelConfig` | `zoo/base.py` | **frozen+slots** | `__post_init__` uses `object.__setattr__`; `Any`→`object` for `extra` |
| `TrainingMetrics` | `core/trainer.py` | **frozen** only | `to_dict()` accesses `__dict__` |
| `EquiTileConfig`, `EnhancedEquiTileConfig`, `NCCLConfig`, `AsyncConfig`, `CurriculumConfig` | `equitile/config.py` | **frozen+slots** | Read-only value objects |
| `DistributedConfig` | `equitile/config.py` | **Kept mutable** | `self.config.device_ids = [...]` in runtime code |
| `MultiGPUConfig` | `equitile/config.py` | **Kept mutable** | Same pattern as DistributedConfig |
| `TileGrowthConfig` | `equitile/config.py` | **Kept mutable** | Test code sets `.enabled = True` |
| `DynamicEquiTileConfig` | `equitile/config.py` | **Kept mutable** | Contains mutable child config |
| `TrainerConfig` | `core/trainer.py` | **Kept mutable** | Test code mutates fields directly |
| Builder configs (4) | `equitile/builder.py` | **Skipped** | Deeply wired builder mutation patterns |

### New Discoveries / Issues

1. **Runtime mutation patterns block frozen everywhere**: 6 dataclass configs (`DistributedConfig`, `MultiGPUConfig`, `TileGrowthConfig`, `DynamicEquiTileConfig`, `TrainerConfig`, and all 4 builder configs) have runtime mutation. Making them frozen would require significant refactoring of their callers. The pragmatic rule: if a dataclass is used as an immutable value object (read-only after construction), freeze it. If it's mutated, keep it mutable. `TrainingMetrics` is the edge case — frozen but not slotted.

2. **`__post_init__` with frozen requires `object.__setattr__`**: `ModelConfig` in `zoo/base.py` had 3 field mutations in `__post_init__` (syncing `equilibrium_steps`/`max_steps`, unwrapping tuple `input_dim`). All needed `object.__setattr__`.

3. **`TrainingMetrics.to_dict()` uses `__dict__`**: This prevented `slots=True`. If `slots` is desired later, refactor `to_dict` to use `dataclasses.asdict()`.

4. **Track registration sync is one-way**: `track_registry.py` syncs its `ALL_TRACKS` into the core Registry, but `Registry._components` is not the source of truth for tracks — `ALL_TRACKS` is. The sync was added for unified discovery only. If the core Registry is cleared in tests, tracks will remain in `ALL_TRACKS` but not in the Registry until module reload.

5. **`ModelConfig.extra` type changed**: `dict[str, Any]` → `dict[str, object]`. This is technically Phase 2 work (type safety) and was a side-effect of the frozen change. This removes one entry from the Phase 2.1 checklist.

### Pointers for Future Sessions

- **Phase 2.1 (Eliminate `Any` from core files)** is the natural next step. The TODO already lists all target files and line numbers. `zoo/base.py` is already clean (side-effect of this session). Next targets: `core/registry.py` (7 `Any` refs), `core/trainer.py` (6 refs), `zoo/models/base.py` (4 refs), `zoo/propagators/base.py` (1 ref), `acceleration/_array_ops.py` (8 refs), `equitile/state_types.py` (1 ref).

- **`equitile/config.py` still imports `Any` from typing** but it's used in `extra: dict[str, Any]` fields. That's Phase 2 work.

- **`builder.py` configs remain untouched**: `ArchitectureConfig`, `IOConfig`, `LearningConfig`, `DynamicsConfig` are all mutable builder internals. The TODO says to possibly remove `LearningConfig` — that would be a separate refactoring of `EquiTileBuilder` internals (40+ `self._learning.*` refs). Low priority.

- **Phase 3 (algorithmic dedup)** is the highest-impact work remaining (~3000 line reduction). A dedicated session should start with `3.1 Extract Settling Loop Helper` — this is the most widely shared pattern (13+ classes).

- **Run `find bioplausible -name "*,cover" -delete`** to clean up stale coverage artifacts if any remain.

- **Pyright errors unchanged at 5** — all pre-existing and unrelated to refactoring.

---

## Discoveries During Phase 0 Work

1. **`builder.py:LearningConfig` kept**: The TODO says to remove `LearningConfig` because it duplicates `EquiTileConfig` fields, but removing it requires significant refactoring of `EquiTileBuilder` internals (40+ references to `self._learning.*`). Deferred to Phase 1 or may be kept as-is since it's an internal implementation detail with no external cost.

2. **`eqprop_diffusion.py` line 28**: The TODO listed `steps: int = None` at `eqprop_diffusion.py:28` but this file's `__init__` signature doesn't have that pattern. The file was verified and no fix was needed.

3. **`.cover` files**: The repo has stale `.cover` copies (e.g., `__init__.py,cover`, `deployment.py,cover`) from a previous coverage run. These can be cleaned up with `find . -name "*,cover" -delete`.

4. **`_settle()` was called from production code**: The `AdamEqProp` class in `eqprop.py:335-336` called `self._settle()`. These calls had to be updated to `self._settle_phase_direct()` alongside the test updates.

5. **Test file deletion**: Two test files (`test_report_generation.py`, `test_report_analysis_robustness.py`) imported the removed `legacy_report` module and had to be archived. The `test_refactor2_bugfixes.py` module-import test also referenced `legacy_report` and was updated.

---

## Session Wrap-Up (2026-07-29 — Session 3)

### What Was Done This Session

**Phase 2: Core Type Safety** — Eliminated `Any` from all 6 core target files.

### Phase 2.1 Results — `Any` Elimination

| Target File | `Any` Refs Before | Status | Notes |
|-------------|-------------------|--------|-------|
| `core/registry.py` | 10 | **ELIMINATED** | Replaced with `object`, `cast()`, removed `Any` import |
| `core/trainer.py` | 16 | **11 kept, 5 → `object`** | OmegaConf fields must keep `Any` (OmegaConf rejects `object` type) |
| `zoo/models/base.py` | 4 | **ELIMINATED** | `ctx: object`, `dict[str, object]` |
| `zoo/propagators/base.py` | 1 | **ELIMINATED** | Added `params: Iterable[nn.Parameter]` type annotation |
| `acceleration/_array_ops.py` | 8 | **ELIMINATED** | Replaced all 8 with `object` / `type[object]` |
| `equitile/state_types.py` | 1 | **ELIMINATED** | Replaced 6 `dict[str, Any]` → `dict[str, object]` |

**Key finding — OmegaConf incompatibility**: `OmegaConf.structured()` rejects `dict[str, object]` with `Unsupported value type: 'object'`. Therefore `TrainerConfig` fields that interface with OmegaConf (`model_kwargs`, `propagator_kwargs`, `optimizer_kwargs`, `data_kwargs`, `tags`, `extra`) must remain `dict[str, Any]`. This is reasonable — OmegaConf is the I/O boundary where Pydantic would be used, and `Any` is justified at system boundaries per AGENTS.md guidance.

**Other core files remain untouched** (Phase 4 work):
- `acceleration/kernels.py` — 15 `Any` refs
- `equitile/config.py` — 6 `Any` refs (all `**kwargs: Any`)
- `zoo/models/eqprop/*.py` — 5-10 `Any` refs each
- `equitile/builder.py` — ~8 `Any` refs
- Plus ~100 more across `equitile/` (research, benchmarks, deployment, etc.)

### Verification Results (Working Tree)

| Gate | Result |
|------|--------|
| `ruff format --check .` | 659 files formatted — **PASS** |
| `ruff check --select I .` | All checks passed — **PASS** |
| `ruff check --fix .` | 5313 `@typing.override` suggestions (style-only, not actionable) |
| `pyright` | 5 errors (all pre-existing — same as Session 2), 1511 warnings — **no new errors** |
| `pytest --no-cov` | 1117 passed, 15 skipped, 5 subtests — **all passing** |
| Working tree delta | 7 files changed, **+131/-91 lines** |

### New Discoveries / Issues

1. **OmegaConf rejects `dict[str, object]`**: `OmegaConf.structured()` validates type annotations at introspection time and raises `ValidationError: Unsupported value type: 'object'`. Any dataclass field that passes through `OmegaConf.structured()` must use concrete types (or `Any`). This is a hard constraint — cannot work around with `# type: ignore`.

2. **`TypeVar` used once in signature = pyright error**: When using `Component = TypeVar("Component")`, pyright enforces the constraint that the TypeVar must appear at least twice in the signature (once as parameter, once as return). Fixed by using `object` instead for one-shot usage sites.

3. **`_components` type narrowing**: Changing `_components` from `dict[str, dict[str, dict[str, Any]]]` to `dict[str, dict[str, dict[str, object]]]` forced `cast(ComponentMetadata, ...)` at all access sites (`get_metadata`, `query`, `export_yaml`). Alternative would be a TypedDict for the inner dict shape — considered overengineered for now.

4. **Phase 2.3 trivially complete**: No f-string logging existed in any core file — all used the proper `%s` deferred formatting pattern. No t-string migration was needed.

### Pointers for Future Sessions

- **Phase 2.2 (Add type annotations to propagator `params`)** was merged into 2.1 — the `zoo/propagators/base.py` change added `Iterable[nn.Parameter]` annotations.

- **Phase 3 (algorithmic dedup)** is the highest-impact remaining work (~3000 line reduction potential). Start with `3.1 Extract Settling Loop Helper` — 13+ classes share the same settling pattern in `zoo/models/eqprop/` and `zoo/models/base.py`.

- **Phase 4 (full type hardening)** will need to handle the OmegaConf incompatibility discovered here. Strategy: keep `dict[str, Any]` on OmegaConf-structured dataclasses, use `dict[str, object]` everywhere else. Do NOT attempt to convert OmegaConf-facing types.

- **`--no-cov` for fast test runs**: Full coverage test takes ~4 min vs ~45 sec with `--no-cov`. Use `--no-cov` during development, full test before commit.

- **Pyright errors remain at 5** — all pre-existing (`deployment.py:717` missing import, `hyperopt/graph_task.py:28-32` missing import). Not caused by any refactoring session.

- **Clean up stale coverage artifacts**: `find . -name "*,cover" -delete` — from a previous coverage run.

---

## Session Wrap-Up (2026-07-29 — Session 4)

### What Was Done This Session

**Commit 3** (2e94a3a): Phase 3 — extract settling helpers, unify hidden dims computation.

### Phase 3.1: Settling Loop Helper (`bioplausible/zoo/_settling.py`)

| Item | Detail |
|------|--------|
| **New module** | `bioplausible/zoo/_settling.py` (432 lines) |
| **Classes/functions created** | `settle_single_state()`, `settle_activations_list()`, `EquilibriumFunction`, `_run_with_sn_freeze()`, `_inf_norm_converged()` |
| **Models refactored to use `settle_single_state`** | `EqPropModel.forward()` (BPTT branch) — replaces ~80 lines of inline settling with 6-line helper call |
| **Models refactored to use `settle_activations_list`** | `StandardEqProp.forward()`, `DirectedEP.forward()`, `HolomorphicEP.forward()` — 3 models now share same loop structure |
| **`EquilibriumFunction` moved** | From `zoo/models/base.py` to `zoo/_settling.py` (no code changes, better modularity) |

**Skipped** (loops too unique to benefit): `NeuralCube`, `TemporalResonanceEqProp`, `HomeostaticEqProp`, `LazyEqProp`, `MomentumEquilibrium`.

### Phase 3.5: Hidden Dims Computation

| Item | Detail |
|------|--------|
| **Helper functions added** to `zoo/base.py` | `resolve_hidden_dims(config, hidden_dim)` and `compute_hidden_dims(hidden_dim, num_layers, max_layers=5)` |
| **Ternary pattern refactored** (`hidden_dims = (ternary)`) | 16 sites across 7 files → `resolve_hidden_dims()` |
| **Build pattern refactored** (`hidden_dims=[hd]*min(n,5)`) | 11 sites across 7 files → `compute_hidden_dims()` |
| **Net lines removed** from existing files | **−271 lines** (from 10 files) |
| **New code added** | `_settling.py` (+432) + helpers in `zoo/base.py` (+28) |
| **Total working tree delta** | **+585/-424 = +161 lines** (one-time cost for shared infrastructure) |

### Verification Results (HEAD = 2e94a3a)

| Gate | Result |
|------|--------|
| `ruff format --check .` | 653 files formatted — **PASS** |
| `ruff check --fix .` | 5338 `@typing.override` suggestions (style-only, not actionable) |
| `pyright` | 5 errors (all pre-existing — same as Sessions 2/3), 1477 warnings — **no new errors** |
| `pytest -q` | 1117 passed, 15 skipped, 5 subtests — **all passing** |

### New Discoveries / Issues

1. **`_spectral_norm_freezer` context manager complexity**: The warmup-step-inside-eval-mode pattern was subtly wrong on the first attempt — the remaining steps counter wasn't decremented inside the closure, causing trajectory IndexError. Fixed by making `remaining` a `nonlocal` variable mutated by both `warmup()` and `main_loop()`.

2. **`[None] * (steps + 1)` type inference**: Pyright infers `list[None]` from `[None] * N`. When assigned to `list[torch.Tensor] | None`, it's a type error. Fixed with `cast("list[torch.Tensor]", ...)`.

3. **Convergence delta computation overhead**: `settle_activations_list` always computes the per-step delta (used for convergence checking), but for models like `DirectedEP` that don't do early stopping, this is wasted compute. Optimization: only compute delta when `return_dynamics=True` or `convergence_start < steps`.

4. **`settle_activations_list` trajectory format**: Returns `list[list[Tensor]]` (append-based). The original `DirectedEP` code used the same format. `StandardEqProp` originally used pre-allocated `[None] * (eq_steps+1)` with index assignment. Both produce equivalent sequence of snapshots.

5. **`settle_single_state` trajectory format**: Uses pre-allocated `list[torch.Tensor]` of length `steps+1` with index assignment, then slices to actual length. Matches original `EqPropModel.forward()` behavior exactly.

### Remaining Duplications (Still Need Refactoring)

| Pattern | Count | Files |
|---------|-------|-------|
| `train_step` implementations | **26** | `zoo/models/*` |
| `forward_step` implementations | **11** | `zoo/models/eqprop/*` |
| `_build_layers` implementations | **12** | `zoo/models/*` |
| `build` classmethods | **18+** | `zoo/models/*` |
| `MemoryProfiler` class | **2** | `equitile/profiler.py` |
| `ProfileResult` class | **2** | `equitile/profiler.py` |
| Tile communicator classes | **2** | `TileCommunicator` vs `NCCLCommunicator` |
| `LearningConfig` (builder.py) | **1** | Deferred to future session |

### Pointers for Future Sessions

- **Phase 3.6 (Consolidate `build` classmethods)** is the natural next step. 18+ `@classmethod def build(...)` methods in `zoo/models/*` share the same pattern:
  ```python
  config = ModelConfig(name=spec.name, input_dim=input_dim, output_dim=output_dim,
                       hidden_dims=compute_hidden_dims(hidden_dim, num_layers), extra=kwargs)
  return cls(config=config).to(device)
  ```
  Extract a shared `_build_from_spec(cls, spec, input_dim, output_dim, hidden_dim, num_layers, device, **kwargs)` in `zoo/models/base.py`. Estimated savings: **~200–300 lines**.

- **Phase 3.3 (Unify FA backward passes)** — 9 classes in `fa.py` with nearly identical `train_step` methods (each ~65–70 lines). Extract `_fa_backward(activation_derivative_fn)` helper. Estimated savings: **~300–400 lines**.

- **Phase 3.4 (Consolidate LM files)** — `language.py` (1192) + `language_optimized.py` (687) + `fast_lm.py` (613) = 2492 lines. Create `_components.py` for shared `TileAttention`, `TileFeedForward`, etc. Estimated savings: **~800–1000 lines**.

- **Phase 3.3 or 3.4** are higher-impact than 3.6 (more line savings). Prioritize whichever is more familiar.

- **Phase 3.2 (Extract long functions)** — 13 functions >50 lines across settling code. Much of this was already addressed by Phase 3.1 (the `EqPropModel.forward()` BPTT branch was 80→6 lines). Remaining targets: `CoreTrainer` methods in `core/trainer.py`.

- **Pyright errors still at 5** — all pre-existing. No new errors introduced by Phase 3 work.

- **`find . -name "*,cover" -delete`** — clean stale coverage artifacts if they bother you.