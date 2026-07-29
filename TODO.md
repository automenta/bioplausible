# Refactoring TODO — Bioplausible Codebase

> Goal: Improve elegance, clarity, DRY, maintainability, and alignment with `@AGENTS.md`.
> **Strategy**: Build on Phases 0–3 completed (see `TODO0.md`). Focus remaining work on:
> 1. **Algorithmic deduplication** (remaining `train_step` variants)
> 2. **Type system hardening** (eliminate `Any`, add `TypedDict`, strict exports)
> 3. **Architecture clarity** (consolidate overlapping modules, clean imports)
> 4. **Test infrastructure** (fixtures, property-based tests, coverage floor)

> **Excluded**: `docs/` and all archives under it.

---

## Current State Summary (from TODO0.md)

| Phase | Focus | Status | Net Lines |
|-------|-------|--------|-----------|
| 0 | Archive dead code, fix syntax, print→logging | ✅ **DONE** | −7,909 |
| 1 | Unify registries, frozen dataclasses | ✅ **DONE** | −41 |
| 2 | Core type safety (eliminate `Any`) | ✅ **DONE** | +40 |
| 3 | Algorithmic deduplication | 🟡 **IN PROGRESS** | +81 |
| 4 | Full type hardening | ⏳ **PENDING** | — |

**Tests**: 1,117 passed, 15 skipped (55% coverage, floor=40%)
**Pyright**: 5 pre-existing errors (2 files), 0 new errors from refactoring
**Ruff**: Clean

---

## Phase 3 (Remaining): Algorithmic Deduplication

### 3.A Remaining `train_step` Implementations (26 total → 23 remaining)

The `_contrastive_step` helper in `zoo/models/eqprop/_contrastive.py` already unified 3 EqProp variants (−168 lines). Remaining groups:

| Group | Models | Pattern | Est. Savings | Priority |
|-------|--------|---------|--------------|----------|
| **E** Manual-gradient FA | `StochasticFA`, `EquilibriumAlignment`, `FeedbackAlignmentEqProp`, `DirectFeedbackAlignmentEqProp` | Compute `wgrads`/`bgrads` manually, then `param.data -= lr * grad` | ~30 lines | **HIGH** |
| **C** Kernel backend | `LoopedMLP`, `MemoryEfficientLoopedMLP` | Identical numpy→torch dispatch logic | ~15 lines | **HIGH** |
| **B** New-optimizer-each-call | `StandardFA`, `EnergyMinimizingFA`, `LayerwiseEquilibriumFA` | `opt = Adam(model.parameters())` inside `train_step` (no momentum!) | ~5 lines + **bug fix** | **HIGH** |
| **D** Autograd FA (unique) | `AdaptiveFeedbackAlignment`, `EnergyGuidedFA` | Custom gradient computations | — | Low |
| **F** EqProp variants | `NeuralCube`, `TemporalResonanceEqProp`, `HomeostaticEqProp`, `LazyEqProp`, `SparseEquilibrium`, `MOMEquilibrium`, `EqPropDiffusion`, `GraphEqProp` | Diverse settling dynamics | — | Low |
| **G** Predictive Coding | `PredictiveCodingHybrid`, `FabricPCGraphPCN` | Different energy functions | — | Low |

**Recommended approach for Group E (Manual FA)**:
```python
# New helper in zoo/models/fa.py
def _apply_manual_fa_gradients(
    model: nn.Module,
    wgrads: list[torch.Tensor],
    bgrads: list[torch.Tensor] | None,
    lr: float,
) -> None:
    """Apply manually computed gradients to model parameters."""
    for param, wg in zip(model.parameters(), wgrads):
        if wg is not None:
            param.data -= lr * wg
    if bgrads:
        for param, bg in zip(model.parameters(), bgrads):
            if bg is not None:
                param.data -= lr * bg
```

**Recommended approach for Group B (New optimizer bug)**:
```python
# New helper in zoo/models/fa.py
def _ensure_optimizer(model: nn.Module, lr: float, cache_attr: str = "_fa_optimizer") -> torch.optim.Optimizer:
    """Get or create cached Adam optimizer to preserve momentum."""
    if not hasattr(model, cache_attr):
        setattr(model, cache_attr, torch.optim.Adam(model.parameters(), lr=lr))
    return getattr(model, cache_attr)
```

### 3.B Remaining `forward_step` Implementations (11)

All in `zoo/models/eqprop/`. Most are unique (complex dynamics). Low ROI for deduplication — **deprioritize**.

### 3.C `_build_layers` (12 implementations)

**DEPRIORITIZED** — 12 divergent architectures (Linear, Conv2d/3d, GCNConv, Hebbian, Embeddings, Attention). Only shared pattern is `if use_spectral_norm: layer = spectral_norm(...)` (~1 line each). Estimated savings <50 lines.

---

## Phase 4: Full Type System Hardening

### 4.1 Eliminate Remaining `Any` (Non-Core Files)

| File | `Any` Refs | Strategy |
|------|------------|----------|
| `equitile/config.py` | 6 | `extra: dict[str, object]` (OmegaConf fields stay `Any`) |
| `equitile/builder.py` | ~8 | `dict[str, object]` for internal dicts |
| `equitile/research.py` | ~12 | `object`, `Protocol` |
| `equitile/deployment.py` | ~15 | Fix `InferenceRequest` import + `TypedDict` |
| `equitile/benchmarks/*.py` | ~20 | `dict[str, object]` |
| `zoo/models/eqprop/*.py` | 5–10 each | `dict[str, object]`, remove `Any` imports |
| `hyperopt/graph_task.py` | 4 | Add missing `os` import, `dict[str, object]` |
| `validation/tracks/*.py` | ~30 | `TypedDict` for metrics dicts |

**Rule**: At I/O boundaries (OmegaConf, YAML, external APIs) → `dict[str, Any]` is acceptable. Everywhere else → `dict[str, object]` or `TypedDict`.

### 4.2 Replace `dict[str, object]` with `TypedDict` at Boundaries

| Boundary | Current | Target |
|----------|---------|--------|
| `CoreTrainer` config | `dict[str, Any]` | `TrainerConfigDict` (TypedDict) |
| `TaskProtocol.get_batch` return | `tuple[Tensor, Tensor]` | Keep (explicit) |
| `Registry.get_metadata` return | `ComponentMetadata` | Keep (dataclass) |
| `run_from_runconfig` input | `RunConfig` (OmegaConf) | Keep |
| Checkpoint state dicts | `dict[str, Any]` | `EquiTileStateDict`, `ModelStateDict` (TypedDict) |

### 4.3 Add `__all__` to All Public Modules

**Rule**: Every `bioplausible/**/*.py` (except tests, scripts, `__pycache__`) must have `__all__` listing public exports. Internal modules prefixed `_` excluded.

```bash
# Find modules missing __all__
grep -rL "__all__" bioplausible/ --include="*.py" | grep -v test | grep -v __pycache__
```

### 4.4 Strict Pyright Config (Future)

Move pyright from per-rule warnings to `strict = true` once `Any` count → 0 in non-boundary files.

---

## Phase 5: Architecture Clarity & Consolidation

### 5.1 Consolidate EquiTile Language Model Files

**Files**: `equitile/language.py` (1,192), `language_optimized.py` (687), `fast_lm.py` (613) = **2,492 lines**

**Status**: Config (`LMEquiTileConfig`) already consolidated in Phase 3.4.

**Remaining**: Three *different* architectures:
- `language.py` — Canonical EquiTile with tiles
- `language_optimized.py` — Fused kernels for throughput
- `fast_lm.py` — Demo with integrated training loop

**Action**: Create `equitile/_components.py` with shared building blocks:
- `TileAttention` (multi-head causal attention as tile)
- `TileFeedForward` (GLU/SwiGLU as tile)
- `PositionalEncoding` (RoPE or learned)
- `CausalMask` (buffer)

**Estimated savings**: ~100–200 lines (shared components only — architectures remain distinct).

### 5.2 Unify Distributed / Multi-GPU Code

**Files**: `equitile/distributed.py` (994), `equitile/multigpu.py` (950) = **1,944 lines**

**Overlap**: `TileCommunicator` vs `NCCLCommunicator`, `DistributedEquiTile` vs `MultiGPUEquiTile`.

**Action**: 
1. Extract common NCCL primitive wrappers → `equitile/_nccl.py`
2. Make `DistributedEquiTile` the single class; `MultiGPUEquiTile` becomes alias or thin wrapper
3. Deprecate `TileCommunicator` in favor of `NCCLCommunicator`

**Estimated savings**: ~300–500 lines.

### 5.3 Consolidate Profiling Code

**Files**: `equitile/profiler.py` (1,076), plus inline profiling in `lm_demo/training.py`

**Status**: `lm_demo/profiling.py` archived in Phase 0.

**Action**: Keep `profiler.py` as single source. Remove inline profiling from demo files, import from `profiler`.

### 5.4 Clean Up `zoo/models/eqprop/` Directory Structure

**Current**: 20 files in `eqprop/` subdirectory.

**Action**: Group by algorithm family:
```
eqprop/
├── __init__.py
├── standard_eqprop.py      # StandardEqProp, DirectedEP, HolomorphicEP
├── contrastive.py          # _contrastive_step helper (exists)
├── variants/
│   ├── neural_cube.py
│   ├── temporal_resonance.py
│   ├── homeostatic.py
│   ├── lazy.py
│   ├── sparse.py
│   ├── mom.py
│   ├── diffusion.py
│   └── graph_eqprop.py
└── lm/
    ├── causal_transformer.py
    └── ternary.py
```

**Benefit**: Clearer imports, easier discovery, `__init__.py` exports organized by family.

---

## Phase 6: Test Infrastructure & Quality Gates

### 6.1 Pytest Fixtures for Common Patterns

**File**: `tests/conftest.py` (extend)

```python
@pytest.fixture
def mnist_task() -> VisionTask:
    """Pre-configured MNIST VisionTask (quick_mode=True)."""
    task = VisionTask("mnist", quick_mode=True)
    task.setup()
    return task

@pytest.fixture
def lm_task() -> LMTask:
    """Pre-configured Shakespeare LMTask (quick_mode=True)."""
    task = LMTask("tiny_shakespeare", quick_mode=True)
    task.setup()
    return task

@pytest.fixture
def synthetic_data():
    """Deterministic synthetic classification data."""
    torch.manual_seed(42)
    X = torch.randn(200, 64)
    y = (X.sum(dim=1) > 0).long() % 10
    return X, y

@pytest.fixture
def equitile_model(synthetic_data) -> EquiTile:
    """Minimal EquiTile for fast tests."""
    X, y = synthetic_data
    return EquiTile(input_dim=64, output_dim=10, num_layers=2)
```

### 6.2 Property-Based Tests (Hypothesis)

**Targets** (pure logic, no GPU):
- `zoo/base.py`: `resolve_hidden_dims`, `compute_hidden_dims`
- `zoo/_settling.py`: `_inf_norm_converged`, `settle_single_state` convergence
- `core/registry.py`: `_QueryFilter.matches` logic
- `hyperopt/search_space.py`: Parameter space validation
- `validation/tracks/core_tracks.py`: Metric computations

### 6.3 Enforce Coverage Floor ≥85%

Current: 55% (floor=40% in pyproject.toml).

**Action**:
1. Raise `--cov-fail-under=85` in `pyproject.toml`
2. Add tests for uncovered core modules:
   - `core/registry.py` (query, compatibility check)
   - `core/trainer.py` (checkpointing, early stopping, callbacks)
   - `equitile/config.py` (validation, frozen dataclass behavior)
   - `zoo/_settling.py` (convergence, trajectory shapes)

### 6.4 Test Organization

```
tests/
├── conftest.py              # Shared fixtures
├── unit/                    # Fast, isolated unit tests
│   ├── test_registry.py
│   ├── test_configs.py
│   ├── test_settling.py
│   └── test_utils.py
├── integration/             # Model + trainer + data
│   ├── test_core_trainer.py
│   ├── test_equitile.py
│   ├── test_eqprop.py
│   └── test_fa.py
├── property/                # Hypothesis property tests
│   ├── test_hidden_dims.py
│   ├── test_convergence.py
│   └── test_registry_query.py
└── slow/                    # Full training runs (marked @pytest.mark.slow)
    ├── test_mnist_full.py
    └── test_lm_full.py
```

---

## Phase 7: Documentation & Developer Experience

### 7.1 API Reference Generation

**Tool**: `pdoc` or `sphinx` (configured in `pyproject.toml`)

**Target**: Auto-generate from Google-style docstrings + type hints.

### 7.2 Architecture Decision Records (ADRs)

**Location**: `docs/adr/` (new, not in archive)

**Template**:
```markdown
# ADR-NNN: Title

## Status
Accepted / Superseded

## Context
What problem are we solving?

## Decision
What did we decide?

## Consequences
Trade-offs, follow-up work.
```

**Priority ADRs**:
1. Registry unification (Phase 1)
2. Frozen dataclass policy (Phase 1.2)
3. `Any` elimination strategy & OmegaConf boundary (Phase 2)
4. Settling loop abstraction (Phase 3.1)
5. Train_step helper pattern (Phase 3.x)

### 7.3 Pre-commit Hooks (Verify)

Current `.pre-commit-config.yaml` should include:
- `ruff format --check`
- `ruff check --fix`
- `pyright` (strict mode once Phase 4 done)
- `pytest --cov --cov-fail-under=85`
- `pip-audit`

---

## Quick-Start: Next Session Priorities

### Session A: Train_Step Deduplication (Highest ROI)
1. **Group E (Manual FA)** — Extract `_apply_manual_fa_gradients` in `fa.py`, refactor 4 classes
2. **Group C (Kernel Backend)** — Extract `_kernel_backend_step` for `LoopedMLP`/`MemoryEfficientLoopedMLP`
3. **Group B (Optimizer Bug)** — Extract `_ensure_optimizer`, fix momentum loss in 3 classes

**Estimated**: −50 lines + 1 bug fix, ~2 hours.

### Session B: Type Hardening (Phase 4)
1. Eliminate `Any` from `equitile/config.py`, `builder.py`, `research.py`, `deployment.py`
2. Add `TypedDict` for checkpoint state dicts
3. Add `__all__` to all public modules

**Estimated**: +200 lines (annotations), ~3 hours.

### Session C: EquiTile LM Consolidation (Phase 5.1)
1. Create `equitile/_components.py` with shared `TileAttention`, `TileFeedForward`, etc.
2. Refactor `language_optimized.py` and `fast_lm.py` to import from `_components.py`

**Estimated**: −150 lines, ~3 hours.

### Session D: Distributed/Profiling Consolidation (Phase 5.2, 5.3)
1. Merge `distributed.py` + `multigpu.py` → single `DistributedEquiTile`
2. Extract NCCL primitives → `_nccl.py`
3. Clean up inline profiling in demo files

**Estimated**: −400 lines, ~4 hours.

---

## Verification Gates (Run After Each Session)

```bash
# Formatting & Linting
ruff format . && ruff check --fix .

# Type Checking
pyright .

# Tests (fast)
pytest -q --no-cov

# Tests (full, before commit)
pytest --cov=bioplausible --cov-fail-under=85
```

---

## Notes

- **Do not touch `docs/archive/`** — historical record only.
- **Do not bypass pre-commit/CI gates** — they enforce the standards in `@AGENTS.md`.
- **Prefer small, focused PRs** — each session above is a logical commit.
- **Update this TODO.md** after each session with actual results (like TODO0.md sessions).

---

*Generated from codebase analysis + TODO0.md history. Phases 0–3 partially complete. Next: Phase 3.A (train_step dedup).*