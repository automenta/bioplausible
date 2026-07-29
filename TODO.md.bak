# Refactoring TODO — Bioplausible Codebase

> Goal: improve elegance, clarity, DRY, maintainability, and alignment with `@AGENTS.md`.
> **Strategy**: Archive noise first (dead code, syntax fixes), then fix foundations (registries, configs, core types), then deduplicate algorithms, then harden type system.

---

## 0. Survey Summary (Updated)

### Confirmed Dead Code (Package — ~1400 lines)

| Item | Lines | Status |
|------|-------|--------|
| `analysis/legacy_report/` | 4 files, ~1777 | Superseded by `analysis/reporting.py` |
| `zoo/mep/optimizers/o1_memory.py` | 435 | Superseded by `o1_memory_v2.py` |
| `zoo/mep/optimizers/inspector.py` | 167 | `ModelInspector` exported but never used |
| `zoo/mep/optimizers/monitor.py` | 262 | `EPMonitor`/`monitor_ep_training` exported but never used |
| `equitile/lm_demo/profiling.py` | 508 | Only imports itself in docstring; duplicates `equitile/profiler.py` |
| `EqPropLMWrapper` | 31 | Proxy class, no behavior, never used |
| `_apply_feedback_alignment()` | 2 | Empty `pass` in `fa.py:71` |
| `_apply_direct_feedback()` | 2 | Empty `pass` in `fa.py:140` |
| `_settle()` wrapper | 12 | Trivial delegate in `eqprop.py:102` |
| `train_step` no-op | 8 | Just calls `super()` in `finite_nudge_ep.py:33` |
| Duplicate `return` | 1 | Unreachable in `wrappers.py:102` |
| Commented code | 1 | `# logits_nudged = ...` in `base.py:428` |
| Duplicate import | 1 | Already imported line 18 in `hebbian.py:19` |

**NOT dead (used in tests):** `HolomorphicEqProp`, `FiniteNudgeEqProp`, `LazyEqProp` — keep.

### Confirmed Dead Code (Outside Package — ~5000 lines)

| Item | Lines | Status |
|------|-------|--------|
| `examples/legacy/` | 17 files, ~2800 | Not imported anywhere |
| `scripts/legacy/` | 16 files, ~1076 | Not imported anywhere |

### Duplications Identified

| Pattern | Count | Files |
|---------|-------|-------|
| `train_step` implementations | **26** | `zoo/models/*` |
| `forward_step` implementations | **11** | `zoo/models/eqprop/*` |
| `_build_layers` implementations | **12** | `zoo/models/*` |
| `build` classmethods | **18+** | `zoo/models/*` |
| `hidden_dims = [...]` computation | **17** | `zoo/models/*` |
| `CurriculumScheduler` name collision | **2** | `equitile/enhanced.py`, `data/curricula.py` |
| `DynamicsConfig` name collision | **2** | `equitile/config.py`, `equitile/builder.py` |
| `MemoryProfiler` class | **2** | `equitile/profiler.py`, `lm_demo/profiling.py` |
| `ProfileResult` class | **2** | `equitile/profiler.py`, `lm_demo/profiling.py` |
| Tile communicator classes | **2** | `TileCommunicator` vs `NCCLCommunicator` |

---

## Phase 0: Quick Wins — Archive Noise & Fix Syntax (2–3 days)

*Immediate code reduction, clarifies what's actually used.*

### 0.1 Archive Dead Code in `bioplausible/` Package

**Action**: Move to `docs/archive/YYYYMMDD/dead_package_code/` preserving directory structure.

| Item | Source | Archive Target |
|------|--------|----------------|
| `analysis/legacy_report/` | `bioplausible/analysis/legacy_report/` | `dead_package_code/analysis/legacy_report/` |
| `o1_memory.py` | `bioplausible/zoo/mep/optimizers/o1_memory.py` | `dead_package_code/zoo/mep/optimizers/o1_memory.py` |
| `inspector.py` | `bioplausible/zoo/mep/optimizers/inspector.py` | `dead_package_code/zoo/mep/optimizers/inspector.py` |
| `monitor.py` | `bioplausible/zoo/mep/optimizers/monitor.py` | `dead_package_code/zoo/mep/optimizers/monitor.py` |
| `profiling.py` (lm_demo) | `bioplausible/equitile/lm_demo/profiling.py` | `dead_package_code/equitile/lm_demo/profiling.py` |
| `EqPropLMWrapper` class | `bioplausible/zoo/models/eqprop/eqprop_lm_variants.py:564-594` | Remove from file; archive original file version |
| `_apply_feedback_alignment()` | `bioplausible/zoo/propagators/fa.py:71-72` | Remove lines |
| `_apply_direct_feedback()` | `bioplausible/zoo/propagators/fa.py:140-141` | Remove lines |
| `_settle()` wrapper | `bioplausible/zoo/propagators/eqprop.py:102-113` | Remove lines |
| `train_step` no-op | `bioplausible/zoo/models/eqprop/finite_nudge_ep.py:33-40` | Remove lines |
| Duplicate `return` | `bioplausible/zoo/models/wrappers.py:102` | Remove line |
| Commented code | `bioplausible/zoo/models/base.py:428` | Remove line |
| Duplicate import | `bioplausible/zoo/models/hebbian.py:19` | Remove line |

**Net savings: ~1400 lines removed from package** (archived for history)

### 0.2 Archive Dead Code Outside Package

**Action**: Move to `docs/archive/YYYYMMDD/legacy_examples/` and `legacy_scripts/`.

| Item | Source | Archive Target |
|------|--------|----------------|
| `examples/legacy/` | `examples/legacy/*` | `docs/archive/YYYYMMDD/legacy_examples/` |
| `scripts/legacy/` | `scripts/legacy/*` | `docs/archive/YYYYMMDD/legacy_scripts/` |

**Net savings: ~4000 lines removed from working tree** (archived for history)

### 0.3 Remove Dead Demo/Config Classes

| Item | Location | Action |
|------|----------|--------|
| `ArchitectureConfig` | `equitile/config.py:16-23` | **Remove** — never used outside config.py |
| `OptimizationConfig` | `equitile/config.py:26-38` | **Remove** — never used outside config.py |
| `DynamicsConfig` | `equitile/config.py:42-58` | **Remove** — duplicates `builder.py:DynamicsConfig`; name collides with `DynamicEquiTileConfig as DynamicsConfig` in `__init__.py:167` |
| `LearningConfig` | `equitile/builder.py:80-101` | **Remove** — duplicates fields in `EquiTileConfig` |
| `CurriculumScheduler` (enhanced.py) | `equitile/enhanced.py:39-100` | **Remove** — `get_sample_weights` always returns `torch.ones(n_samples)` (dead logic); name collides with `data/curricula.py:CurriculumScheduler` |
| `enable_curriculum()` | `equitile/builder.py:672` | **Remove** — uses dead `CurriculumScheduler` |

### 0.4 Fix Type Syntax Errors

| Issue | Files | Fix |
|-------|-------|-----|
| `steps: int = None` → `int \| None = None` | `zoo/models/eqprop/ternary.py:106`, `causal_transformer_eqprop.py:128`, `neural_cube.py:152`, `eqprop_lm_variants.py:61,175,278,350,436,499`, `eqprop_diffusion.py:28` | Change annotation |
| Legacy `except X, Y:` → `except (X, Y):` | `core/trainer.py:733`, `core/registry.py:245` | Fix syntax |

### 0.5 Replace `print()` with `logging`

| Files | Lines |
|-------|-------|
| `equitile/lm_demo/training.py` | 664, 724, 741, 761, 777, 794, 795 |
| `equitile/lm_demo/train_tinystories.py` | 74, 91, 125-129, 267, 286, 335, 344, 419, 425, 439, 442, 469-470, 476-478, 493-495, 527, 531-541, 546, 549 |
| `equitile/lm_demo/demo.py` | 201, 211, 260, 327, 495-623, 635-642, 664, 682-683, 694 |
| `equitile/lm_demo/data_advanced.py` | 147, 356, 459, 473, 480 |
| `cli/rank.py` | 14, 18, 40 |
| `experiments/__init__.py` | 30, 31 |

---

## Phase 1: Foundational Architecture (2–3 days)

### 1.1 Unify 4 Registries → 1

| Registry | File | Action |
|----------|------|--------|
| `NEBCRegistry` | `zoo/nebc_base.py:73-104` | **Archive** — replace with `register_model` + `Registry.get` |
| `TaskRegistry` | `hyperopt/task_registry.py` | **Archive** — add `ComponentCategory.TASK`, register there |
| `track_registry` | `validation/tracks/track_registry.py` | **Refactor** — add `ComponentCategory.TRACK` |
| `register_nebc` decorator | `zoo/nebc_base.py:104` | **Remove** |

**Archive location**: `docs/archive/YYYYMMDD/registries/`

### 1.2 Unify Config Dataclasses

After Phase 0 removals, only one config per concern remains. Add `frozen=True, slots=True`.

### 1.3 Add `frozen=True, slots=True` to Core Dataclasses

**Files**: `zoo/base.py`, `core/trainer.py`, `core/registry.py`, `equitile/config.py`, `equitile/builder.py`, `data/curricula.py`.

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

*Now easier because Phase 0 removed ~5000 lines of noise.*

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
- Phase 0: `git diff --stat` → ~-5500 lines in working tree (archived ~5500 to docs/archive/)
- Phase 1: `grep -r "NEBCRegistry\|TaskRegistry\|O1MemoryEP\b\|ModelInspector\|EPMonitor" --include="*.py" | grep -v test` → empty
- Phase 2: `pyright` zero errors on core files
- Phase 3: Settling loops use shared helper
- Phase 4: `grep -r "from typing import Any" --include="*.py" bioplausible/` → only tests

---

## Archive Structure

All archived code goes to `docs/archive/YYYYMMDD/` following existing pattern:

```
docs/archive/20260729/          # or current date
├── dead_package_code/          # From bioplausible/ package
│   ├── analysis/legacy_report/
│   ├── zoo/mep/optimizers/o1_memory.py
│   ├── zoo/mep/optimizers/inspector.py
│   ├── zoo/mep/optimizers/monitor.py
│   └── equitile/lm_demo/profiling.py
├── legacy_examples/            # From examples/legacy/
├── legacy_scripts/             # From scripts/legacy/
└── registries/                 # NEBCRegistry, TaskRegistry
```

This preserves history while cleaning the working tree.

---

## Effort Summary (Corrected)

| Phase | Focus | Est. Days | Working Tree Delta |
|-------|-------|-----------|--------------------|
| **0** | Archive dead code & fix syntax | **2–3** | **−5500 lines** (archived ~5500) |
| 1 | Registries, configs, frozen dataclasses | 2–3 | −200 lines (dedup) |
| 2 | Core type safety | 3–4 | +200 lines (annotations) |
| 3 | Algorithmic dedup | 5–7 | **−3000+ lines** (shared helpers) |
| 4 | Full type hardening | 3–5 | +500 lines (TypedDict, exports) |

**Total**: ~15–22 days. **Phase 0 alone removes ~5500 lines from working tree** (archived for history), unblocking everything else.

**Total potential reduction: ~8,500+ lines from working tree** (from ~81,500 to ~73,000 — ~10% smaller; ~5500 archived separately).