# Refactoring TODO — Bioplausible Codebase

> Goal: improve elegance, clarity, DRY, maintainability, and alignment with `@AGENTS.md`.
> **Strategy**: Remove noise first (dead code, syntax fixes), then fix foundations (registries, configs, core types), then deduplicate algorithms, then harden type system.

---

## 0. Survey Summary (Updated)

### Newly Discovered Dead Code (Additional ~3000+ lines)

| Item | Lines | Status |
|------|-------|--------|
| `examples/legacy/` (17 files) | ~2800 | Not imported anywhere |
| `scripts/legacy/` (16 files) | ~1076 | Not imported anywhere |
| `docs/archive/20260722/` (~100 files) | massive | Archived, not part of active code |
| `zoo/mep/optimizers/o1_memory.py` | 435 | Superseded by `o1_memory_v2.py` |
| `zoo/mep/optimizers/inspector.py` | 167 | `ModelInspector` exported but never used |
| `zoo/mep/optimizers/monitor.py` | 262 | `EPMonitor`/`monitor_ep_training` exported but never used |
| `equitile/lm_demo/profiling.py` | 508 | Only imports itself in docstring; never actually used |

### Newly Discovered Duplications

| Pattern | Count | Files |
|---------|-------|-------|
| `train_step` implementations | **26** | zoo/models/* |
| `forward_step` implementations | **11** | zoo/models/eqprop/* |
| `_build_layers` implementations | **12** | zoo/models/* |
| `build` classmethods | **18+** | zoo/models/* |
| `hidden_dims = [...]` computation | **17** | zoo/models/* |
| `CurriculumScheduler` class (name collision!) | **2** | `equitile/enhanced.py` and `data/curricula.py` |
| `DynamicsConfig` class (name collision!) | **2** | `equitile/config.py` and `equitile/builder.py` |
| `MemoryProfiler` class | **2** | `equitile/profiler.py` and `equitile/lm_demo/profiling.py` |
| `ProfileResult` class | **2** | `equitile/profiler.py` and `equitile/lm_demo/profiling.py` |
| Tile communicator classes | **2** | `TileCommunicator` (distributed.py) vs `NCCLCommunicator` (multigpu.py) |

---

## Phase 0: Quick Wins — Delete Noise & Fix Syntax (2–3 days)

*Immediate code reduction, clarifies what's actually used.*

### 0.1 Delete Dead Code in `bioplausible/` Package

**Confirmed dead code (~1400 lines in package):**

| Item | Files/Lines | Action |
|------|-------------|--------|
| `analysis/legacy_report/` | 4 files, ~1777 lines | **Delete** — superseded by `analysis/reporting.py` |
| `zoo/mep/optimizers/o1_memory.py` | 435 lines | **Delete** — `O1MemoryEP` superseded by `O1MemoryEPv2`; `energy_from_states`/`manual_energy_compute`/`settle_manual` never used |
| `zoo/mep/optimizers/inspector.py` | 167 lines | **Delete** — `ModelInspector` exported in `__init__.py` but never called |
| `zoo/mep/optimizers/monitor.py` | 262 lines | **Delete** — `EPMonitor`/`monitor_ep_training` exported but never called |
| `equitile/lm_demo/profiling.py` | 508 lines | **Delete** — only imported in its own docstring; duplicates `equitile/profiler.py` |
| `HolomorphicEqProp` | `zoo/propagators/eqprop.py:315-351` | **Delete** — stub, just calls `loss.backward()` |
| `FiniteNudgeEqProp` | `zoo/propagators/eqprop.py:354-389` | **Delete** — stub, just scales gradients |
| `LazyEqProp` | `zoo/propagators/eqprop.py:392-432` | **Delete** — stub, just backprop with threshold |
| `_apply_feedback_alignment()` | `zoo/propagators/fa.py:71-72` | **Delete** — empty `pass` |
| `_apply_direct_feedback()` | `zoo/propagators/fa.py:140-141` | **Delete** — empty `pass` |
| `_settle()` wrapper | `zoo/propagators/eqprop.py:102-113` | **Delete** — trivial delegate |
| `EqPropLMWrapper` | `zoo/models/eqprop/eqprop_lm_variants.py:564-594` | **Delete** — proxy class, no behavior |
| `train_step` no-op | `zoo/models/eqprop/finite_nudge_ep.py:33-40` | **Delete** — just calls `super()` |
| Duplicate `return` | `zoo/models/wrappers.py:102` | **Delete** (unreachable) |
| Commented code | `zoo/models/base.py:428` | **Delete** |
| Duplicate import | `zoo/models/hebbian.py:19` | **Delete** |

**Net savings: ~3500 lines**

### 0.2 Delete Dead Code Outside Package

| Item | Files/Lines | Action |
|------|-------------|--------|
| `examples/legacy/` | 17 files, ~2800 lines | **Delete** — not imported anywhere |
| `scripts/legacy/` | 16 files, ~1076 lines | **Delete** — not imported anywhere |
| `docs/archive/20260722/` | ~100 files | **Move out of repo** or **delete** — archived, not active |

**Net savings: ~5000+ lines**

### 0.3 Delete Dead Demo/Config Classes

| Item | Location | Action |
|------|----------|--------|
| `ArchitectureConfig` (config.py) | `equitile/config.py:16-23` | **Delete** — never used outside config.py |
| `OptimizationConfig` (config.py) | `equitile/config.py:26-38` | **Delete** — never used outside config.py |
| `DynamicsConfig` (config.py) | `equitile/config.py:42-58` | **Delete** — duplicates `builder.py:DynamicsConfig`; name collides with `DynamicEquiTileConfig as DynamicsConfig` in `__init__.py:167` |
| `LearningConfig` (builder.py) | `equitile/builder.py:80-101` | **Delete** — duplicates fields in `EquiTileConfig` |
| `CurriculumScheduler` (enhanced.py) | `equitile/enhanced.py:39` | **Delete** — `get_sample_weights` always returns `torch.ones(n_samples)` (dead logic); name collides with `data/curricula.py:CurriculumScheduler` |
| `enable_curriculum()` | `equitile/builder.py:672` | **Delete** — uses dead `CurriculumScheduler` |

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
| `NEBCRegistry` | `zoo/nebc_base.py:73-104` | **Delete** — replace with `register_model` + `Registry.get` |
| `TaskRegistry` | `hyperopt/task_registry.py` | **Delete** — add `ComponentCategory.TASK`, register there |
| `track_registry` | `validation/tracks/track_registry.py` | **Refactor** — add `ComponentCategory.TRACK` |
| `register_nebc` decorator | `zoo/nebc_base.py:104` | **Delete** |

### 1.2 Unify Config Dataclasses

After Phase 0 deletions, only one config per concern remains. Add `frozen=True, slots=True`.

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

**Files**: `equitile/profiler.py` (1076) and `equitile/lm_demo/profiling.py` (508 = removed in Phase 0)
After Phase 0, focus on deduplication within remaining profiler.

### 3.9 Unify Distributed/Multi-GPU Code

**Files**: `equitile/distributed.py` (994) and `equitile/multigpu.py` (950) = **1944 lines**

Investigate if `TileCommunicator` (distributed.py) and `NCCLCommunicator` (multigpu.py) can be unified. Same for `DistributedEquiTile` vs `MultiGPUEquiTile`.

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
- Phase 0: `git diff --stat` → -8500+ lines
- Phase 1: `grep -r "NEBCRegistry\|TaskRegistry\|O1MemoryEP\b\|ModelInspector\|EPMonitor" --include="*.py" | grep -v test` → empty
- Phase 2: `pyright` zero errors on core files
- Phase 3: Settling loops use shared helper
- Phase 4: `grep -r "from typing import Any" --include="*.py" bioplausible/` → only tests

---

## Effort Summary (Updated)

| Phase | Focus | Est. Days | Code Delta |
|-------|-------|-----------|------------|
| **0** | Delete dead code & fix syntax | **2–3** | **−8500 lines** (Phase 0.1: ~3500 + Phase 0.2: ~5000) |
| 1 | Registries, configs, frozen dataclasses | 2–3 | −200 lines (dedup) |
| 2 | Core type safety | 3–4 | +200 lines (annotations) |
| 3 | Algorithmic dedup | 5–7 | **−3000+ lines** (settling loop, FA, LM, hidden dims, build, build_layers) |
| 4 | Full type hardening | 3–5 | +500 lines (TypedDict, exports) |

**Total**: ~15–22 days. **Phases 0–1 alone remove ~9000 lines** with minimal risk, unblocking everything else.

**Total potential reduction: ~12,000+ lines** (from ~81,500 to ~69,500 — ~15% smaller).