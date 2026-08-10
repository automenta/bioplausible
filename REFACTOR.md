# REFACTOR.md — Comprehensive Refactoring Plan for bioplausible

**Generated**: 2025-08-09  
**Updated**: 2026-08-10  
**Codebase**: 316 Python files, ~41K lines (91K total with blanks/comments)  
**Goal**: Drastically reduce size via deduplication, DRY, and structural consolidation

---

## Progress Summary (Completed: ~2,010 lines / 4.9% reduction across 68+ files)

| Phase | Task | Lines Saved | Key Files |
|-------|------|-------------|-----------|
| **1. Quick Wins** | `core/utils/activations.py` — unified `_get_activation`, `_approx_spectral_norm`, `softmax`, `cross_entropy`, `spectral_normalize`, `get_backend`, `to_numpy` | ~200 | 15 files (7 model + 8 acceleration) |
| | `core/utils/seeds.py` — unified `set_all_seeds(seed, deterministic)` replacing 7 `_set_seed` variants | ~100 | 7 files (`cli/run.py`, `core/trainer.py`, `equitile/benchmarks/rigorous.py`, `equitile/utils/reproducibility.py`) |
| | `core/utils/device.py` — unified `get_device(device="auto")` replacing 30+ inline patterns | ~150 | 32 files |
| | `core/logging.py` — `get_logger()` helper created (opt-in; 113 legacy call sites remain) | — | 1 new file |
| | Acceleration array ops — `kernels.py` + `_array_ops.py` now re-export from `core.utils.activations` | ~100 | 2 files |
| | MEP Benchmarks — `BenchmarkConfig`, `get_dataloaders`, `get_input_dim`, `get_num_classes`, `cnn_classifier` → `_shared.py` | ~120 | 2 files |
| **2. Model Architecture** | `TrainingMixin` / `SpectralMixin` / `CheckpointMixin` — composition-based `BioModel` | ~380 | 4 new + 3 refactored |
| **3. Deployment Configs** | `equitile/deployments/base.py` — unified config hierarchy + `create_deployment_model` factory | ~400 | 1 new + 4 refactored + 1 `_feature_extractors.py` |
| **10. Metrics** | `core/metrics.py` — canonical `BaseMetrics` + `EpochMetrics` (frozen+slots, `to_dict()`) | ~55 | 1 new |
| | `TrainingMetrics` (trainer.py) + `BenchmarkMetrics` (runner.py) → extend `BaseMetrics` | ~50 | 3 files |
| **Config Unification** | `config/unified.py` — `BaseConfig` (frozen) + `BaseStructuredConfig` (OmegaConf) + load/save helpers; `BenchmarkConfig` migrated | ~245 | 3 files |

**Blockers Resolved (2026-08-10)**:
- **OmegaConf + frozen dataclasses**: OmegaConf 2.3+ handles `@dataclass(frozen=True, slots=True)` correctly for `structured()`, `merge()`, `to_object()`, and save/load round-trips. The dual-pattern in `config/unified.py` (frozen runtime + OmegaConf mirror) is proven.
- **Pareto/ND Sorting**: Investigated — not a true duplication. `hyperopt/metrics.py` operates on `TrialMetrics` (4 objectives); `analysis/results.py` operates on raw `dict` (3 objectives). Different input types/objectives → leave as-is.

**Deferred**:
- **FastLMEquiTile merge**: Two fundamentally different implementations — `lm/fast_lm.py` (canonical, extends `BioModel`) vs `language/fast.py` (demo, extends `OptimizedLMEquiTile`). Blocked on architecture decision.

---

## Remaining Work — Prioritized by Impact

### 🔴 CRITICAL — Config Unification (Est. ~1,500 lines)
**Status**: Pattern established in `config/unified.py`; 1 of ~60 config classes migrated (`BenchmarkConfig`)

**Remaining config pairs to migrate** (search `core/config.py`, `config/schema.py`, `equitile/core/config.py`):
| Config | Current Locations | Target |
|--------|-------------------|--------|
| `ModelConfig` | `core/config.py` + `config/schema.py` (different!) | `unified.py` hierarchy |
| `TrainingConfig` | `equitile/lm/training.py`, `config/schema.py`, `equitile/language/fast.py` | `unified.py` hierarchy |
| `FastLMConfig` | `equitile/language/components.py` + `equitile/lm/fast_lm.py` (dup) | `unified.py` hierarchy |
| `OptimizerConfig` | `zoo/mep/benchmarks/tuned_compare.py`, `config/schema.py` | `unified.py` hierarchy |
| `ExperimentConfig` | `experiments/utils.py`, `equitile/utils/reproducibility.py`, `config/schema.py` | `unified.py` hierarchy |
| `RLConfig` / `VisionConfig` / etc. | Deployment modules | Already on unified base |

**Migration Pattern** (established in `config/unified.py`):
```python
# For configs needing OmegaConf YAML interop:
@dataclass  # non-frozen
class MyStructuredConfig(BaseStructuredConfig):
    field: int = 42
    def to_internal(self) -> MyConfig: ...

# For pure runtime configs (no YAML needed):
@dataclass(frozen=True, slots=True)
class MyConfig(BaseConfig):
    field: int = 42
```

**Steps per config**:
1. Define frozen runtime config in `config/unified.py` (or domain-specific unified file)
2. If YAML interop needed, add non-frozen `StructuredConfig` mirror with `to_internal()`
3. Update all imports: `from bioplausible.config.unified import BaseConfig, load_config, save_config`
4. Delete obsolete config class files
5. Run `ruff check --fix . && pyright . && pytest --cov`

**Verification**: All tests pass; `load_config(MyConfig, path)` round-trips; no `config/schema.py` imports remain in runtime code.

---

### 🟠 HIGH — Logging Migration (Est. ~110 lines)
**Status**: `core/logging.py` exists; 113 call sites use `logging.getLogger(__name__)`

**Action**: Mechanical replacement across codebase
```bash
# Pattern: logger = logging.getLogger(__name__)
# → from bioplausible.core.logging import get_logger; logger = get_logger()
```

**Files**: `cli/`, `zoo/`, `equitile/`, `tests/` — grep for `logging.getLogger`

**Verification**: `grep -r "logging.getLogger" --include="*.py" bioplausible/ | grep -v "__pycache__" | wc -l` → 0

---

### 🟠 HIGH — FastLMEquiTile Decision (Est. ~500 lines)
**Blocked**: Architecture decision needed

**Options**:
1. **Keep canonical only**: Delete `language/fast.py`, move demo hooks to `lm/fast_lm.py` behind config flag
2. **Wrapper pattern**: `language/fast.py` → thin `DemoFastLMEquiTile(FastLMEquiTile)` subclass (~150 lines)
3. **Unify base**: Extract common `FastEquiTileLayer`/`MixtureOfTiles`/`TileLocalAttention` to shared module

**Recommendation**: Option 2 (wrapper) — preserves demo API, minimal risk, ~500 lines saved.
**Decision needed before implementation**.

---

### 🟡 MEDIUM — Acceleration Cleanup (Est. ~30 lines)
**Status**: `_array_ops.py` is now a thin re-exporter after Phase 1.2

**Action**: Delete `acceleration/_array_ops.py` once all importers use `core.utils.activations`
- Check imports: `grep -r "_array_ops" --include="*.py" bioplausible/`
- Update any remaining imports
- Delete file

---

### 🟡 MEDIUM — BenchmarkMetrics Naming Reconciliation (Est. ~40 lines + schema)
**Issue**: `BenchmarkMetrics` uses `train_acc`/`val_acc`; `TrainingMetrics` uses `train_accuracy`/`val_accuracy`
**Impact**: Touches SQL schemas (`hyperopt/storage.py`, `execution/_lifecycle.py`), `TrainingCheckpoint` (`execution/training_dynamics.py`), many call sites
**Verdict**: Low value per line — **defer** unless future ticket unifies trial representation

---

### 🟡 MEDIUM — Optimizer Factory Consolidation (Est. ~200 lines)
**Problem**: 40+ locations create `torch.optim.Adam`/`AdamW`/`SGD` directly with hardcoded or config-driven parameters

**Locations** (sample):
- `equitile/deployments/*.py` — 4 files, each creates optimizer in `__init__`
- `equitile/lm/*.py` — 3 files (`training.py`, `fast_lm.py`, `ablation_study.py`)
- `zoo/models/fa.py` — 6 optimizer creations
- `zoo/mep/benchmarks/runner.py` — `get_optimizer()` factory exists but not used everywhere
- `validation/tracks/*.py` — 10+ files with inline optimizer creation

**Solution**: Create `core/utils/optimizer.py` with factory:
```python
from dataclasses import dataclass
from typing import Literal
import torch
from torch import nn

@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    name: Literal["adam", "adamw", "sgd"] = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9  # for SGD
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

def create_optimizer(model: nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
    match config.name:
        case "adam":
            return torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
                                    betas=config.betas, eps=config.eps)
        case "adamw":
            return torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
                                     betas=config.betas, eps=config.eps)
        case "sgd":
            return torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum,
                                   weight_decay=config.weight_decay)
        case _:
            raise ValueError(f"Unknown optimizer: {config.name}")
```

**Migration**: Replace inline optimizer creation with `create_optimizer(model, config.optimizer)`.
Add `optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)` to relevant configs.

---

### 🟡 MEDIUM — Hyperopt/Execution Storage Unification (Est. ~300 lines)
**Problem**: Two independent checkpoint/trajectory storage systems with overlapping dataclasses:

| Module | Dataclasses | Purpose |
|--------|-------------|---------|
| `hyperopt/storage.py` | TrialMetrics, epoch_metrics table, training_trajectories table, training_checkpoints table | Optuna trial persistence |
| `execution/training_dynamics.py` | `TrainingCheckpoint`, `TrainingTrajectory`, `ContinuousTrainingSchedule` | Training analysis & pruning |

**Overlap**: Both track epoch-level metrics (loss, accuracy, time), checkpoints, and trajectories.
**Difference**: `hyperopt` is trial-centric with SQL; `execution` is trajectory-centric with analytics.

**Solution**: 
1. Extract shared `EpochCheckpoint` dataclass to `core/checkpoint.py` (or new `core/training_state.py`)
2. Make `TrainingCheckpoint` (execution) and `TrialMetrics` (hyperopt) extend/embed it
3. Unify SQL schema in `hyperopt/storage.py` to use canonical fields
4. `execution/training_dynamics.py` imports from shared location

**Verification**: Both modules pass tests; no field duplication; `TrainingTrajectory` can reconstruct from `hyperopt` data.

---

### 🔴 CRITICAL — EquiTile Generification (Est. ~800 lines + enables future reuse)
**Problem**: EquiTile's tile-based local learning infrastructure (topology, kernels, optimizers, task handling) is trapped in `equitile/` and cannot be reused by other algorithms (FA, Target Prop, Hierarchical PC, Spiking, Graph NNs).

**Components to Lift to `core/`**:
| Component | Current Location | Target Location | Reusability |
|-----------|------------------|-----------------|-------------|
| `TileGraph`, `TileState` | `equitile/core/topology.py` | `core/tile/topology.py` | Universal graph substrate |
| Kernels (`compute_activity_update`, `compute_hebbian_update`, `compute_contrastive_hebbian_update`, `compute_tile_prediction`) | `equitile/core/kernels.py` | `core/tile/kernels.py` | All local learning |
| `MultiOptimizerMixin` (weight/importance/full optimizers) | `equitile/training/optimizer_mixin.py` | `core/local_learning/mixins.py` | Any multi-optimizer model |
| `TaskHandler` (task-type loss/grad/metrics) | `equitile/training/task_handler.py` | `core/local_learning/task.py` | All models |
| `LocalLearningConfig` base | `equitile/core/config.py` | `core/local_learning/config.py` | Algorithm configs |
| Feature extractors (Conv/Temporal/RL/Graph) | `equitile/deployments/_feature_extractors.py` | `core/tile/feature_extractors.py` | Deployments, zoo |

**New Algorithms Enabled** (post-generification):
- `TileFA` — Feedback Alignment on tile substrate
- `TileTargetProp` — Target Propagation with tile graph
- `HierarchicalPC` — Multi-scale Predictive Coding
- `TileSNN` — Spiking tile models
- `TileGNN` — Graph NNs with local learning

**Solution**: 4-phase migration (see `EQUITILE_GENERIFICATION.md` for full plan):
1. Create `core/tile/` + `core/local_learning/` infrastructure (Week 1-2)
2. Refactor EquiTile to consume core primitives (Week 2-3)
3. Update deployments & enable zoo models (Week 3-4)
4. Validation & docs (Week 4)

**Verification**: EquiTile PC/EP/Backprop modes unchanged; deployments work; new example algorithms in `zoo/models/`.

---

### 🟢 LOW — Data Loading Consolidation (Est. ~150 lines)
**Problem**: `domains/base.py` has `DomainTask.get_dataloader()` pattern, but many modules create `DataLoader` inline with similar transforms.

**Locations**:
- `domains/vision.py`, `timeseries.py`, `tabular.py`, `scientific.py`, `lm.py` — each has `setup()` with DataLoader creation
- `zoo/mep/benchmarks/_shared.py` — `get_dataloaders()` for MNIST/Fashion/CIFAR
- `zoo/mep/benchmarks/runner.py` — `get_dataloader()` with similar transforms
- `validation/tracks/*.py` — inline DataLoader creation

**Shared transforms** (canonical):
```python
MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
CIFAR10_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
```

**Solution**: Add `data/transforms.py` with canonical transforms and `create_dataloader(dataset, config, split)` factory. Migrate inline creations.

---

### 🟢 LOW — Validation Tracks Infrastructure (Est. ~200 lines)
**Problem**: 11 track files in `validation/tracks/` each define `track_N_*(verifier) -> TrackResult` with boilerplate:
- Dataset creation (`create_synthetic_dataset`)
- Model instantiation (`LoopedMLP`, `BackpropMLP`)
- Training loop (`train_model`)
- Evaluation (`evaluate_accuracy`)
- Evidence markdown generation
- `TrackResult` construction

**Shared helpers** in `validation/utils.py`:
- `create_synthetic_dataset()`
- `train_model()`
- `evaluate_accuracy()`

**Missing**: Common `TrackResult` builder, evidence template, model factory.

**Solution**: Add `validation/tracks/_base.py` with:
```python
def run_track(verifier, model_factory, train_fn, eval_fn, name, description, category):
    """Execute standard track pattern."""
    # ... boilerplate extraction
```

Each track becomes ~20 lines of configuration + assertions.

---

### 🟢 LOW — Energy Profiling Consolidation (Est. ~100 lines)
**Status**: Two related but separate modules:
- `core/energy.py` — `EnergyProfile`, `EnergyTracker` for FLOPs/sparsity/memory profiling
- `core/energy_model.py` — `EnergyModel` protocol + `EBMTrainer` for PC/EP/CHL training

**Overlap**: Both deal with "energy" but different semantics (performance vs. algorithm).
**Action**: Rename for clarity:
- `core/energy.py` → `core/profiling.py` (performance profiling)
- `core/energy_model.py` → `core/ebm.py` (energy-based model protocol)

No code consolidation needed — just naming clarity to prevent confusion.

---

## Implementation Sequence

### Week 1: Config Unification (Core)
| Step | Task | Files | Est. Time |
|------|------|-------|-----------|
| 1.1 | Define `ModelConfig` + `TrainingConfig` hierarchies in `config/unified.py` | 1 new | 2h |
| 1.2 | Migrate `core/config.py` → `unified.py` (delete `core/config.py`) | 2 files | 1h |
| 1.3 | Migrate `config/schema.py` → `unified.py` StructuredConfig mirrors | 1 file | 2h |
| 1.4 | Migrate `equitile/core/config.py` | 1 file | 1h |
| 1.5 | Migrate `equitile/lm/training.py` `TrainingConfig` | 1 file | 1h |
| 1.6 | Migrate `FastLMConfig` (delete dup in `components.py`) | 2 files | 1h |
| 1.7 | Migrate remaining configs incrementally (grep for `@dataclass.*Config`) | ~10 files | 4h |
| 1.8 | Full validation: `ruff format . && ruff check --fix . && pyright . && pytest --cov` | — | 2h |

### Week 2: Logging + Acceleration + Optimizer Factory
| Step | Task | Files | Est. Time |
|------|------|-------|-----------|
| 2.1 | Mechanical `get_logger()` migration (codemod or sed) | 113 call sites | 2h |
| 2.2 | Delete `acceleration/_array_ops.py` after import audit | 1 file | 30m |
| 2.3 | Create `core/utils/optimizer.py` factory | 1 new | 1h |
| 2.4 | Migrate 5-10 optimizer creation sites to factory | ~10 files | 2h |
| 2.5 | Validation suite | — | 1h |

### Week 3: EquiTile Generification (Core Infrastructure)
| Step | Task | Files | Est. Time |
|------|------|-------|-----------|
| 3.1 | Create `core/tile/topology.py`, `core/tile/kernels.py`, `core/tile/state.py` | 3 new | 2h |
| 3.2 | Create `core/local_learning/mixins.py`, `core/local_learning/task.py`, `core/local_learning/config.py` | 3 new | 2h |
| 3.3 | Move `_feature_extractors.py` → `core/tile/feature_extractors.py` | 1 move | 1h |
| 3.4 | Refactor `EquiTileConfig` to extend `LocalLearningConfig` | 1 edit | 1h |
| 3.5 | Refactor `EquiTile` to use core mixins + tile primitives | 1 major edit | 3h |
| 3.6 | Update imports across `equitile/`, `deployments/`, `zoo/` | ~15 files | 2h |
| 3.7 | Verify PC/EP/Backprop modes + deployments | Tests | 2h |

### Week 4: Storage Unification + FastLMEquiTile + EquiTile Extensions
| Step | Task | Files | Est. Time |
|------|------|-------|-----------|
| 4.1 | Extract shared `EpochCheckpoint` to `core/training_state.py` | 1 new | 2h |
| 4.2 | Update `hyperopt/storage.py` and `execution/training_dynamics.py` | 2 files | 2h |
| 4.3 | Architecture decision on FastLMEquiTile (team sync) | — | 30m |
| 4.4 | Implement chosen option | 2-3 files | 2-4h |
| 4.5 | Enable `TileFA`, `TileTargetProp` in `zoo/models/` using core primitives | 2 new | 2h |
| 4.6 | Validation | — | 1h |

### Week 5: Data Loading + Validation Tracks + Energy Rename
| Step | Task | Files | Est. Time |
|------|------|-------|-----------|
| 5.1 | Create `data/transforms.py` with canonical transforms | 1 new | 1h |
| 5.2 | Migrate inline DataLoader creations | ~10 files | 2h |
| 5.3 | Create `validation/tracks/_base.py` infrastructure | 1 new | 2h |
| 5.4 | Refactor 3-5 tracks to use shared pattern | ~5 files | 2h |
| 5.5 | Rename `energy.py` → `profiling.py`, `energy_model.py` → `ebm.py` | 2 files | 1h |
| 5.6 | Full validation suite | — | 2h |

### Week 6: Final Validation
| Step | Task | Est. Time |
|------|------|-----------|
| 6.1 | Full test suite + coverage check | 2h |
| 6.2 | Performance benchmarks (smoke) | 1h |
| 6.3 | Update docs/README | 1h |

---

## Verification Checklist (Per Phase)

- [ ] `ruff format . && ruff check --fix .`
- [ ] `pyright .` — zero errors in strict mode
- [ ] `pytest --cov` — all tests pass, coverage ≥85%
- [ ] `pip-audit` — no new vulnerabilities
- [ ] Smoke test: `uv run python -m bioplausible.cli.run --help`
- [ ] Config round-trip test: `load_config(MyConfig, save_config(cfg, "tmp.yaml")) == cfg`

---

## File-Level Impact Map (Remaining)

```
bioplausible/
├── config/
│   ├── unified.py          ← EXTEND (add ModelConfig, TrainingConfig, FastLMConfig, OptimizerConfig, LocalLearningConfig hierarchies)
│   ├── __init__.py         ← UPDATE (re-export new configs)
│   ├── schema.py           ← MIGRATE to StructuredConfig mirrors, then DEPRECATE
│   └── defaults.py         ← KEEP
├── core/
│   ├── config.py           ← DELETE (replaced by unified.py)
│   ├── trainer.py          ← UPDATE (TrainingMetrics already on BaseMetrics)
│   ├── utils/
│   │   ├── logging.py      ← EXISTS (migrate 113 call sites)
│   │   └── optimizer.py    ← NEW (optimizer factory)
│   ├── tile/               ← NEW (generified tile infrastructure)
│   │   ├── topology.py     ← TileGraph, TileState (from equitile/core/topology.py)
│   │   ├── kernels.py      ← compute_activity_update, hebbian, contrastive, tile_prediction (from equitile/core/kernels.py)
│   │   ├── state.py        ← TileStateDict, checkpoint helpers
│   │   └── feature_extractors.py ← Conv/Temporal/RL/Graph extractors (from equitile/deployments/_feature_extractors.py)
│   ├── local_learning/     ← NEW (generified local learning infrastructure)
│   │   ├── mixins.py       ← MultiOptimizerMixin, LocalLearningMixin (from equitile/training/optimizer_mixin.py)
│   │   ├── task.py         ← TaskHandler (from equitile/training/task_handler.py)
│   │   └── config.py       ← LocalLearningConfig base (from equitile/core/config.py)
│   ├── training_state.py   ← NEW (shared EpochCheckpoint, TrainingTrajectory)
│   ├── profiling.py        ← RENAME from energy.py
│   └── ebm.py              ← RENAME from energy_model.py
├── equitile/
│   ├── core/config.py      ← MIGRATE EquiTileConfig to extend LocalLearningConfig
│   ├── lm/
│   │   ├── fast_lm.py      ← KEEP (canonical)
│   │   ├── components.py   ← DELETE FastLMConfig dup
│   │   └── training.py     ← MIGRATE TrainingConfig, use optimizer factory
│   ├── language/
│   │   └── fast.py         ← REFACTOR (pending decision)
│   ├── deployments/*.py    ← USE optimizer factory, import feature extractors from core.tile
│   └── utils/reproducibility.py  ← UPDATE (use core.utils.seeds)
├── zoo/
│   ├── models/fa.py        ← USE optimizer factory
│   ├── models/tile_fa.py   ← NEW (Feedback Alignment on tile substrate)
│   ├── models/tile_tp.py   ← NEW (Target Prop on tile substrate)
│   └── mep/benchmarks/     ← ALREADY on BaseMetrics/BaseConfig
├── experiments/utils.py    ← MIGRATE ExperimentConfig
├── validation/
│   ├── tracks/_base.py     ← NEW (shared track infrastructure)
│   └── tracks/*.py         ← REFACTOR to use _base.py
├── data/
│   └── transforms.py       ← NEW (canonical transforms + DataLoader factory)
├── hyperopt/storage.py     ← UPDATE (use shared EpochCheckpoint)
└── execution/
    └── training_dynamics.py ← UPDATE (use shared EpochCheckpoint)
```

---

## Technical Notes (for implementers)

### Config Migration Gotchas
- **Frozen inheritance**: Non-frozen dataclass cannot inherit frozen → all `BaseConfig` subclasses must be `frozen=True, slots=True`
- **Mutation**: No config is mutated post-construction (verified). If needed later, use `object.__setattr__(self, 'field', value)` — never flip `frozen=False`
- **OmegaConf**: `BaseStructuredConfig` mirror only for YAML interop. Pure runtime configs use frozen form directly.
- **Field ordering**: Parent fields with defaults must come before child fields with defaults. Use `field(default=...)` or reorder.

### Metrics Hierarchy (Already Wired)
- `BaseMetrics(loss, accuracy?, epoch=0, step=0, extra={})` — canonical base with `to_dict()` (filters `None`)
- `EpochMetrics(BaseMetrics)` — adds `train_loss`, `train_acc`, `val_loss`, `val_acc`, `epoch_time`
- `TrainingMetrics(BaseMetrics)` — adds `lr`, `grad_norm`; `step=global_step` set in `_run_epoch`
- `BenchmarkMetrics(BaseMetrics)` — adds `model_name`, `config`, `param_count`, `iteration_time`, `perplexity`, `status`

### Deployment Configs (Already Done)
- `vision.py` / `rl.py` → inherit `ConvDeploymentConfig` / `RLDeploymentConfig` from `base.py` (use PC/EP fields)
- `timeseries.py` / `graph.py` → standalone frozen configs (NO PC/EP fields; tests assert absence)
- All shared NN layers in `_feature_extractors.py`; public modules re-export historical names

---

## Risk Assessment

| Refactor | Risk | Mitigation |
|----------|------|------------|
| Config hierarchy | HIGH — touches 60+ files | Automated codemod + incremental migration + tests |
| Logging migration | LOW — mechanical | Batch replace, verify zero `getLogger` remain |
| FastLMEquiTile | MEDIUM — API change | Wrapper pattern preserves demo API |
| Acceleration cleanup | LOW — internal only | Import audit before delete |
| Optimizer factory | LOW — additive | Old code deprecated, not removed |
| Storage unification | MEDIUM — two systems | Extract shared base, migrate incrementally |
| Data loading | LOW — internal only | Factory pattern, preserve API |
| EquiTile generification | MEDIUM — core refactor | Phased migration, backward compat |

---

## Quick Reference: Completed vs Remaining

| Category | Completed | Remaining |
|----------|-----------|-----------|
| Activation/Utility Functions | ✅ `core/utils/activations.py` | — |
| Seed Setting | ✅ `core/utils/seeds.py` | — |
| Device Resolution | ✅ `core/utils/device.py` | — |
| Model Base Classes | ✅ Composition mixins | — |
| train_step Boilerplate | ✅ `TrainingMixin` | — |
| Checkpointing | ✅ `CheckpointMixin` | — |
| Deployment Configs | ✅ `deployments/base.py` + modules | — |
| Metrics Classes | ✅ `BaseMetrics` hierarchy wired | — |
| Logging Helper | ✅ `core/logging.py` | 🟠 113 call sites to migrate |
| Config Hierarchy | 🟡 Pattern + 1 config | 🔴 ~60 configs to migrate |
| FastLMEquiTile | — | 🟠 Blocked on decision |
| Acceleration Backend | ✅ Consolidated | 🟡 Delete `_array_ops.py` |
| Pareto/ND Sorting | ✅ Investigated — not a dup | — |
| Optimizer Creation | — | 🟡 Factory needed (~40 sites) |
| Storage/Checkpoints | — | 🟡 Two systems to unify |
| Data Loading | — | 🟢 Canonical transforms + factory |
| Validation Tracks | — | 🟢 Shared infrastructure |
| Energy Profiling | — | 🟢 Rename for clarity |
| **EquiTile Generification** | — | 🔴 Core tile/local_learning infra to lift |

---

## New Improvement Opportunities (Discovered 2026-08-10)

| Opportunity | Where | Est. Lines | Priority |
|-------------|-------|-----------|----------|
| **`config/unified.py` migration** — Migrate remaining config pairs (`core/config.py:ModelConfig` ↔ `config/schema.py:ModelConfig`, `TrainerConfig` variants) onto the unified `BaseConfig` hierarchy. The OmegaConf frozen-dataclass blocker is **resolved** — the dual-pattern (frozen runtime + `BaseStructuredConfig` OmegaConf wrapper) is proven. | `core/config.py`, `config/schema.py`, `equitile/core/config.py` | ~1,500 (est.) | 🔴 CRITICAL |
| **EquiTile Generification** — Lift tile-based local learning infrastructure (topology, kernels, optimizers, task handling) from `equitile/` to `core/tile/` + `core/local_learning/` for reuse by FA, Target Prop, Hierarchical PC, Spiking, Graph NNs. See `EQUITILE_GENERIFICATION.md`. | `equitile/core/`, `equitile/training/`, `equitile/deployments/` | ~800 + future reuse | 🔴 CRITICAL |
| **Optimizer factory** — 40+ inline `torch.optim.Adam/AdamW/SGD` creations. Single factory in `core/utils/optimizer.py` with `OptimizerConfig` dataclass. | `equitile/deployments/`, `equitile/lm/`, `zoo/models/fa.py`, `validation/tracks/` | ~200 | 🟡 MEDIUM |
| **Hyperopt/Execution storage unification** — Two independent checkpoint/trajectory systems (`hyperopt/storage.py` + `execution/training_dynamics.py`). Shared `EpochCheckpoint` in `core/training_state.py`. | `hyperopt/storage.py`, `execution/training_dynamics.py` | ~300 | 🟡 MEDIUM |
| **`core/logging.py` `get_logger()` migration** — 113 call sites still use `logging.getLogger(__name__)`. | `cli/`, `zoo/`, `equitile/`, `tests/` | ~110 | 🟡 MEDIUM |
| **`acceleration/_array_ops.py` deletion** — After Phase 1.2 consolidation it is a thin re-exporter. Safe to delete once all importers switch to `core.utils.activations`. | `acceleration/` | ~30 | 🟡 MEDIUM |
| **Canonical DataLoader transforms** — MNIST/CIFAR transforms duplicated across domains, benchmarks, validation. | `domains/`, `zoo/mep/benchmarks/`, `validation/tracks/` | ~150 | 🟢 LOW |
| **Validation tracks infrastructure** — 11 track files with repeated boilerplate. Shared `run_track()` helper in `validation/tracks/_base.py`. | `validation/tracks/` | ~200 | 🟢 LOW |
| **Energy profiling rename** — `core/energy.py` (performance) vs `core/energy_model.py` (EBM protocol). Rename to `profiling.py` and `ebm.py`. | `core/energy.py`, `core/energy_model.py` | ~100 | 🟢 LOW |
| **`BenchmarkMetrics` naming reconciliation** — `train_acc`/`val_acc` vs `TrainingMetrics`'s `train_accuracy`/`val_accuracy`. Touches SQL schemas, checkpoints, call sites. Low value per line — defer. | `zoo/mep/benchmarks/runner.py`, `execution/training_dynamics.py`, `hyperopt/storage.py` | ~40 + schema | 🟢 LOW |

---

## Verified Behavior Preserved (Regression Checklist)

- `ConvEquiTileConfig(input_channels=1, input_size=28, num_classes=10, equitile_kwargs={"sparsity_threshold":0.5})` → `model.config.sparsity_threshold` propagates to `model.head.get_config()`
- `RLEquiTileConfig(obs_dim=8, action_dim=4, equitile_kwargs={"dropout":0.3})` → `model.feature_extractor.get_config().dropout == 0.3`
- `ConvEquiTileConfig()` defaults: `learning_rate=0.01`, `neurons_per_tile=64`
- `RLEquiTileConfig()` defaults: `mode="backprop"`, `learning_rate=3e-4`, `inference_steps=5`, `neurons_per_tile=32`
- Graph/timeseries `__all__` exports unchanged (re-exports from `_feature_extractors`)

### Test Baselines
- **2026-08-09 full run**: 621 passed, 4 skipped (14 pre-existing failures unrelated to refactor: EP numerical parity CPU vs GPU, ONNX export under strict `torch==2.6`, Triton kernel tolerance on CUDA 12.x). 0 regressions.
- **2026-08-10 metrics + config pass**: New `tests/unit/core/test_config_unified.py` (11 tests). All 58 relevant tests pass. 0 regressions.

### Test Plan for `config/unified.py`
```bash
uv run python -m pytest tests/unit/core/test_config_unified.py \
  tests/unit/core/test_config_schema.py \
  tests/unit/core/test_config_defaults.py -x -q --no-cov
```

---

**Total Estimated Remaining Effort**: ~45 hours over 5-6 weeks  
**Expected Additional Reduction**: ~3,400 lines (8.3%)  
**Projected Total**: ~5,400 lines (13.2%) reduced

**Next Immediate Action**: Begin Config Unification Week 1 (Step 1.1) — highest impact, blocker resolved.  
**Parallel Track**: EquiTile Generification Week 3 (Step 3.1) — enables algorithm reuse across codebase.