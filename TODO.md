# Refactoring TODO — Bioplausible Codebase

> Goal: improve elegance, clarity, DRY, maintainability, and alignment with `@AGENTS.md`.
> **Strategy**: Remove noise first (dead code, syntax fixes), then fix foundations (registries, configs, core types), then deduplicate algorithms, then harden type system.

---

## Phase 0: Quick Wins — Delete Noise & Fix Syntax (1–2 days)

*Immediate code reduction, clarifies what's actually used.*

### 0.1 Delete Dead Code

| Item | Files/Lines | Action |
|------|-------------|--------|
| `analysis/legacy_report/` | Entire directory (5 files + `__pycache__`) | **Delete** — superseded by `analysis/reporting.py` |
| `HolomorphicEqProp` | `zoo/propagators/eqprop.py:315-351` | **Delete** — stub, just calls `loss.backward()` |
| `FiniteNudgeEqProp` | `zoo/propagators/eqprop.py:354-389` | **Delete** — stub, just scales gradients |
| `LazyEqProp` | `zoo/propagators/eqprop.py:392-432` | **Delete** — stub, just backprop with threshold |
| `_apply_feedback_alignment()` | `zoo/propagators/fa.py:71-72` | **Delete** — empty `pass` |
| `_apply_direct_feedback()` | `zoo/propagators/fa.py:140-141` | **Delete** — empty `pass` |
| `_settle()` wrapper | `zoo/propagators/eqprop.py:102-113` | **Delete** — trivial delegate to `_settle_phase_direct` |
| `EqPropLMWrapper` | `zoo/models/eqprop/eqprop_lm_variants.py:564-594` | **Delete** — proxy class, no behavior |
| `train_step` no-op | `zoo/models/eqprop/finite_nudge_ep.py:33-40` | **Delete** — just calls `super()` |
| Duplicate `return` | `zoo/models/wrappers.py:102` | **Delete** line 102 (unreachable) |
| Commented code | `zoo/models/base.py:428` | **Delete** `# logits_nudged = ...` |
| Duplicate import | `zoo/models/hebbian.py:19` | **Delete** line 19 (already imported line 18) |

### 0.2 Fix Type Syntax Errors

| Issue | Files | Fix |
|-------|-------|-----|
| `steps: int = None` → `int \| None = None` | `zoo/models/eqprop/ternary.py:106`, `causal_transformer_eqprop.py:128`, `neural_cube.py:152`, `eqprop_lm_variants.py:61,175,278,350,436,499`, `eqprop_diffusion.py:28` | Change annotation |
| Legacy `except X, Y:` → `except (X, Y):` | `core/trainer.py:733`, `core/registry.py:245` | Fix syntax |

### 0.3 Replace `print()` with `logging` in Demos

| Files | Lines |
|-------|-------|
| `equitile/lm_demo/training.py` | 664, 724, 741, 761, 777, 794, 795 |
| `equitile/lm_demo/train_tinystories.py` | 74, 91, 125-129, 267, 286, 335, 344, 419, 425, 439, 442, 469-470, 476-478, 493-495, 527, 531-541, 546, 549 |
| `equitile/lm_demo/demo.py` | 201, 211, 260, 327, 495-623, 635-642, 664, 682-683, 694 |
| `equitile/lm_demo/data_advanced.py` | 147, 356, 459, 473, 480 |
| `cli/rank.py` | 14, 18, 40 |
| `experiments/__init__.py` | 30, 31 |

**Action**: Add `logger = logging.getLogger(__name__)`; replace `print()` → `logger.info()`.

---

## Phase 1: Foundational Architecture (2–3 days)

*Fix structural duplication that propagates through the codebase.*

### 1.1 Unify 4 Registries → 1

| Registry | File | Action |
|----------|------|--------|
| `NEBCRegistry` | `zoo/nebc_base.py:73-104` | **Delete** — replace `NEBCRegistry.register` → `register_model`, `NEBCRegistry.get` → `Registry.get(ComponentCategory.MODEL, ...)` |
| `TaskRegistry` | `hyperopt/task_registry.py` | **Delete** — add `ComponentCategory.TASK` to `Registry`, register `LMTask`, `VisionTask`, `RLTask` there |
| `track_registry` | `validation/tracks/track_registry.py` | **Refactor** — keep as-is but add `ComponentCategory.TRACK` to `Registry` for consistency; or migrate to `Registry` |
| `register_nebc` decorator | `zoo/nebc_base.py:104` | **Delete** |

**Files to modify**: `core/registry.py` (add categories), `zoo/nebc_base.py`, `hyperopt/task_registry.py`, `validation/tracks/track_registry.py`, all call sites.

### 1.2 Unify Config Dataclasses

| Duplicate | Canonical Location | Action |
|-----------|-------------------|--------|
| `ArchitectureConfig` | `equitile/config.py:16-23` | Keep in `config.py`; `builder.py` imports from `config.py` |
| `DynamicsConfig` | `equitile/config.py:42-58` | Keep in `config.py`; `builder.py` imports from `config.py` |
| `LearningConfig` (builder) | `equitile/builder.py:80-101` | **Merge** into `OptimizationConfig` (`config.py:26-38`) |
| `IOConfig` (builder) | `equitile/builder.py:64-77` | Keep in `builder.py` (builder-specific) |

**Also**: Add `frozen=True, slots=True` to all config dataclasses in `config.py` and `builder.py` (AGENTS.md).

### 1.3 Add `frozen=True, slots=True` to Core Dataclasses

**Files**: `zoo/base.py` (`ModelConfig`), `core/trainer.py` (`TrainerConfig`, `TrainingMetrics`), `core/registry.py` (`ComponentMetadata`, `_QueryFilter`), `equitile/config.py`, `equitile/builder.py`.

---

## Phase 2: Core Type Safety (3–4 days)

*Type-safety on the most-imported files first (registry, trainer, base models).*

### 2.1 Eliminate `Any` from Core Files

**Priority order** (most imported first):

| File | Lines | Strategy |
|------|-------|----------|
| `core/registry.py` | 13, 113, 253, 263, 269, 275, 419 | `ComponentMetadata.extra: dict[str, object]`; `Component: TypeVar`; `Any` → `object` in `_infer_metadata` |
| `core/trainer.py` | 17, 88, 92, 96, 100, 145, 148, 336, 343, 854, 1067, 1072, 1083, 1086, 1089, 1130, 1134 | `dict[str, object]`, `TypedDict` for config sections |
| `zoo/base.py` | 14, 46 | `ModelConfig.extra: dict[str, object]` |
| `zoo/models/base.py` | 3, 30, 68, 454 | `ctx: object` in autograd; `dict[str, object]` returns |
| `zoo/propagators/base.py` | 31 | `params: Iterable[nn.Parameter] \| list[nn.Parameter]` |
| `acceleration/_array_ops.py` | 7, 14, 23, 38, 47, 59, 86, 96 | `xp: object` for array lib; return `type[Protocol]` |
| `equitile/state_types.py` | 3 | `Any` in `TypedDict` → `object` |

**Strategy per pattern**:
- `dict[str, Any]` → `dict[str, object]` or `TypedDict`
- `type[Any]` → `type[Protocol]` or concrete type
- `Any` in autograd `ctx` → `object`
- `Any` in config `extra` → `dict[str, object]`

### 2.2 Add Type Annotations to Propagator `params`

**Files**: All `zoo/propagators/*.py` `__init__` methods — add `params: Iterable[nn.Parameter] \| list[nn.Parameter]`.

### 2.3 Replace f-string Logging with t-strings (Core First)

**Priority files** (core infrastructure):
- `core/trainer.py` (lines 941-955, 963)
- `core/registry.py` (line 247)
- `execution/engine.py` (11 occurrences)
- `hyperopt/parallel_runner.py` (2)
- `knowledge/kb.py` (12)
- `autoscientist/campaign.py` (6)

**Then**: remaining 80+ occurrences across execution, evaluation, p2p, etc.

---

## Phase 3: Algorithmic Deduplication (5–7 days)

*Heavy lifting — now easier because noise is gone.*

### 3.1 Extract Settling Loop Helper

**New module**: `bioplausible/zoo/_settling.py` (or `models/_settling.py`)

**Shared logic**:
- Spectral norm freeze/warmup pattern
- Convergence checking (delta threshold, early exit)
- Trajectory/dynamics tracking
- Step counting

**Refactor targets** (12+ classes):
| Class | File | Method |
|-------|------|--------|
| `EqPropModel` | `zoo/models/base.py:445-566` | `forward()` |
| `StandardEqProp` | `zoo/models/eqprop/standard_eqprop.py:178-244` | `train_step()` |
| `DirectedEP` | `zoo/models/eqprop/deep_ep.py:141-199` | `train_step()` |
| `HolomorphicEP` | `zoo/models/eqprop/holomorphic_ep.py:123-180` | `train_step()` |
| `GraphEqProp` | `zoo/models/eqprop/graph_eqprop.py:76-142` | `train_step()` |
| `LoopedMLP` | `zoo/models/eqprop/looped_mlp.py:31-273` | `forward()` + settling |
| `MemoryEfficientLoopedMLP` | `zoo/models/eqprop/memory_efficient.py:14-65` | `forward()` |
| `EqPropLM` variants | `zoo/models/eqprop/eqprop_lm_variants.py` | multiple `forward()` |
| `EqPropDiffusion` | `zoo/models/eqprop/eqprop_diffusion.py:17-172` | `forward()` + `sample()` |
| `NeuralCube` | `zoo/models/eqprop/neural_cube.py:38-120` | `forward()` |
| `TemporalResonanceEqProp` | `zoo/models/eqprop/temporal_resonance.py:15-80` | `forward()` |
| `HomeostaticEqProp` | `zoo/models/eqprop/homeostatic.py:22-200` | `forward()` |
| `LazyEqProp` | `zoo/models/eqprop/lazy_eqprop.py:43-120` | `forward()` |

### 3.2 Extract Long Functions (>50 lines)

| Function | Lines | Split Into |
|----------|-------|------------|
| `EqPropModel.forward()` | 122 | `_forward_bptt()`, `_forward_equilibrium()`, `_forward_contrastive()` |
| `EqPropModel.contrastive_update()` | 89 | `_compute_hebbian_gradients()` |
| `EqPropModel.train_step()` | 93 | `_free_phase()`, `_nudged_phase()`, `_apply_contrastive_update()` |
| `EquilibriumFunction.backward()` | 115 | `_compute_adjoint_state()`, `_compute_param_gradients()` |
| `CoreTrainer._train_step()` | 70 | `_compute_loss()`, `_compute_metrics()` |
| `CoreTrainer._validate()` | 74 | `_validate_batch()` |
| `CoreTrainer._train_epoch()` | 73 | `_fetch_batch()`, `_track_energy()` |
| `StandardEqProp.train_step()` | 67 | `_compute_contrastive_gradients()` |
| `DirectedEP.train_step()` | 59 | `_compute_feedback_gradients()` |
| `HolomorphicEP.train_step()` | 58 | `_compute_complex_gradients()` |
| `GraphEqProp.train_step()` | 67 | `_compute_graph_gradients()` |
| `AdaptiveFeedbackAlignment.train_step()` | 65 | `_compute_fa_gradients()` |
| `StandardFA.train_step()` | 70 | `_compute_fa_backward()` |

### 3.3 Unify Feedback Alignment Backward Passes

**File**: `zoo/models/fa.py` — 3 nearly identical `train_step` implementations:
- `AdaptiveFeedbackAlignment` (213-277)
- `StandardFA` (668-737)
- `StochasticFA` (355-393)

**Extract**: `_fa_backward(activation_derivative_fn)` helper taking the derivative function as parameter.

### 3.4 Consolidate Language Model Files

**Files**: `equitile/language.py` (1192), `language_optimized.py` (687), `fast_lm.py` (613)

**Plan**:
1. Create `equitile/_components.py` with shared: `TileAttention`, `TileFeedForward`, `PositionalEncoding`, `CausalMask`
2. `language_optimized.py` already imports from `language.py` — extend this
3. Move `FastLMConfig` → `equitile/config.py`
4. `fast_lm.py` imports from `_components.py` and `config.py`

### 3.5 Unify Hidden Dims Computation

**Pattern** (10+ occurrences): `hidden_dims = [hidden_dim] * num_layers` with variations

**Extract**: `_compute_hidden_dims(hidden_dim: int, num_layers: int, max_layers: int = 5) -> list[int]` in `zoo/base.py` or `zoo/models/base.py`.

**Occurrences**: `fa.py` (6), `standard_eqprop.py`, `holomorphic_ep.py`, `deep_ep.py`, `mom_eq.py`, `sparse_eq.py`.

### 3.6 Consolidate Duplicate `build` Classmethods

**New helper**: `_build_from_spec(cls, spec, input_dim, output_dim, hidden_dim, num_layers, device, task_type, **kwargs)` in `zoo/models/base.py`.

**Refactor**: 15+ `build` classmethods across `fa.py`, `standard_eqprop.py`, `holomorphic_ep.py`, `deep_ep.py`, `graph_eqprop.py`, `modern_conv_eqprop.py`, `eqprop_lm_variants.py`, `causal_transformer_eqprop.py`, `neural_cube.py`, `ternary.py`, `eqprop_diffusion.py`, `forward_only.py`, `target_prop.py`, `predictive_coding.py`, `spiking.py`, `backprop.py`.

**Also**: Add return type annotations to all `build` methods.

---

## Phase 4: Type System Hardening (3–5 days)

*Complete the type-safety sweep.*

### 4.1 Eliminate Remaining `Any` (All Files)

**Remaining packages**: `execution/`, `hyperopt/`, `analysis/`, `evaluation/`, `knowledge/`, `validation/`, `lightning_/`, `p2p/`, `config/`, `autoscientist/`, `tracking.py`, `deployment.py`, `sklearn_interface.py`, `visualization.py`, `training/`, `experiments/`, `leaderboard/`, `zoo/mep/`, `equitile/` (non-core), `acceleration/kernels.py`.

**Pattern**: Mechanical replacement per Phase 2.1 strategy.

### 4.2 Replace `dict[str, Any]` with `TypedDict` or `dict[str, object]`

**Define `TypedDict` for structured configs**:
- `TrainerConfigDict` for `core/trainer.py`
- `OptimizerConfigDict` for hyperopt
- `ModelConfigDict` for zoo models

### 4.3 Add `__all__` to All Modules

**Files**: Every `__init__.py` and public module — export only public API; internal modules stay `_`-prefixed.

---

## Verification Gates

After **each phase**:
```bash
ruff format . && ruff check --fix .
pyright .
pytest --cov
```

**Phase-specific checks**:
- Phase 0: `git diff --stat` should show significant line reduction
- Phase 1: `grep -r "NEBCRegistry\|TaskRegistry" --include="*.py" | grep -v test` → empty
- Phase 2: `pyright` zero errors on core files
- Phase 3: `grep -r "forward_step.*settle" --include="*.py" zoo/` → uses shared helper
- Phase 4: `grep -r "from typing import Any" --include="*.py" bioplausible/` → only in tests/compat

---

## Effort Summary

| Phase | Focus | Est. Days | Code Delta |
|-------|-------|-----------|------------|
| 0 | Delete noise, fix syntax | 1–2 | **-2000+ lines** (dead code removal) |
| 1 | Registries, configs, frozen dataclasses | 2–3 | -500 lines (dedup) |
| 2 | Core type safety | 3–4 | +200 lines (type annotations) |
| 3 | Algorithmic dedup | 5–7 | -3000+ lines (shared helpers) |
| 4 | Full type hardening | 3–5 | +500 lines (TypedDict, exports) |

**Total**: ~14–21 days for full plan. Phases 0–1 are high-impact, low-risk, and unblock everything else.