# REFACTOR.md — Strategic Refactoring Plan for bioplausible

**Codebase**: 316 Python files, ~41K lines  
**Goal**: Maximize size reduction via deduplication, DRY, structural consolidation  
**Completed**: ~2,010 lines saved (4.9%) across 68+ files

---

## ✅ COMPLETED (No Further Action)

| Area | Result |
|------|--------|
| Activation/Utility Functions | `core/utils/activations.py` — unified 7 functions |
| Seed Setting | `core/utils/seeds.py` — single `set_all_seeds()` |
| Device Resolution | `core/utils/device.py` — single `get_device()` |
| Model Base Classes | Composition mixins (`TrainingMixin`, `SpectralMixin`, `CheckpointMixin`) → `BioModel` |
| Deployment Configs | `equitile/deployments/base.py` — unified hierarchy + factory |
| Metrics Classes | `core/metrics.py` — `BaseMetrics` hierarchy wired |
| Logging Helper | `core/logging.py` — `get_logger()`; **119 call sites migrated**, dup deleted |
| Config Pattern | `config/unified.py` — frozen `BaseConfig` + OmegaConf `BaseStructuredConfig` proven; `ModelConfig` migrated, `core/config.py` deleted |
| Acceleration Backend | `acceleration/_array_ops.py` deleted (thin re-exporter) |
| Energy Profiling | Renamed: `energy.py`→`profiling.py`, `energy_model.py`→`ebm.py` |

---

## 🔴 CRITICAL — Maximum Impact

### 1. Config Unification (~1,300 lines remaining)
**Pattern proven**: `config/unified.py` with frozen runtime configs + optional `StructuredConfig` mirrors for YAML interop.

**Remaining config pairs to migrate** (search `config/schema.py`, `equitile/core/config.py`):
- `TrainingConfig` — `equitile/lm/training.py`, `config/schema.py`, `equitile/language/fast.py`
- `FastLMConfig` — `equitile/language/components.py` + `equitile/lm/fast_lm.py` (dup)
- `OptimizerConfig` — `zoo/mep/benchmarks/tuned_compare.py`, `config/schema.py`
- `ExperimentConfig` — `experiments/utils.py`, `equitile/utils/reproducibility.py`, `config/schema.py`
- `RLConfig`/`VisionConfig`/etc. — Deployment modules (already on unified base)

**Migration steps per config**:
1. Define frozen runtime config in `config/unified.py`
2. If YAML needed, add non-frozen `StructuredConfig` mirror with `to_internal()`
3. Update imports: `from bioplausible.config.unified import BaseConfig, load_config, save_config`
4. Delete obsolete config class files
5. Verify: `ruff check --fix . && pyright . && pytest --cov`

**Blocker resolved**: OmegaConf 2.3+ handles `@dataclass(frozen=True, slots=True)` correctly.

---

### 2. EquiTile Generification (~800 lines + enables algorithm reuse)
**Problem**: Tile-based local learning infrastructure trapped in `equitile/`, unavailable to FA, Target Prop, Hierarchical PC, Spiking, Graph NNs.

**Components to lift to `core/`**:

| Component | From | To | Reusability |
|-----------|------|-----|-------------|
| `TileGraph`, `TileState` | `equitile/core/topology.py` | `core/tile/topology.py` | Universal graph substrate |
| Kernels (activity/hebbian/contrastive/prediction) | `equitile/core/kernels.py` | `core/tile/kernels.py` | All local learning |
| `MultiOptimizerMixin` | `equitile/training/optimizer_mixin.py` | `core/local_learning/mixins.py` | Any multi-optimizer model |
| `TaskHandler` | `equitile/training/task_handler.py` | `core/local_learning/task.py` | All models |
| `LocalLearningConfig` base | `equitile/core/config.py` | `core/local_learning/config.py` | Algorithm configs |
| Feature extractors (Conv/Temporal/RL/Graph) | `equitile/deployments/_feature_extractors.py` | `core/tile/feature_extractors.py` | Deployments, zoo |

**New algorithms enabled post-generification**:
- `TileFA` — Feedback Alignment on tile substrate
- `TileTargetProp` — Target Propagation with tile graph
- `HierarchicalPC` — Multi-scale Predictive Coding
- `TileSNN` — Spiking tile models
- `TileGNN` — Graph NNs with local learning

**4-phase migration** (see `EQUITILE_GENERIFICATION.md`):
1. Create `core/tile/` + `core/local_learning/` infrastructure
2. Refactor EquiTile to consume core primitives
3. Update deployments & enable zoo models
4. Validation & docs

---

## 🟠 HIGH — Blocked on Decision

### FastLMEquiTile (~500 lines)
Two implementations exist:
- `lm/fast_lm.py` — canonical, extends `BioModel`
- `language/fast.py` — demo, extends `OptimizedLMEquiTile`

**Options**:
1. **Keep canonical only**: Delete `language/fast.py`, move demo hooks to `lm/fast_lm.py` behind config flag
2. **Wrapper pattern**: `language/fast.py` → thin `DemoFastLMEquiTile(FastLMEquiTile)` subclass (~150 lines) — **recommended**
3. **Unify base**: Extract common layer primitives to shared module

**Decision needed before implementation**.

---

## 🟡 MEDIUM — High Leverage

### 3. Optimizer Factory Consolidation (~150 lines)
**Factory created**: `core/utils/optimizer.py` — `OptimizerConfig` (frozen+slots) + `create_optimizer(model, config)` supporting `adam`/`adamw`/`sgd`.

**3 of ~60 sites migrated**: `validation/utils.py`, `equitile/lm/ablation_study.py`, `validation/tracks/application_tracks.py`.

**Remaining high-impact sites**:
- `equitile/deployments/*.py` (4 files, optimizer in `__init__`)
- `equitile/lm/training.py`, `equitile/lm/fast_lm.py`
- `zoo/models/fa.py` (6 creations)
- `validation/tracks/*.py` (10+ files)
- `core/trainer.py` (dynamic `opt_cls`)

**Action**: Migrate inline `torch.optim.Adam/AdamW/SGD` → `create_optimizer(model, config.optimizer)`.

---

### 4. Hyperopt/Execution Storage Unification (~300 lines)
Two independent checkpoint/trajectory systems:

| Module | Dataclasses | Purpose |
|--------|-------------|---------|
| `hyperopt/storage.py` | `TrialMetrics`, epoch_metrics table, training_trajectories table | Optuna trial persistence |
| `execution/training_dynamics.py` | `TrainingCheckpoint`, `TrainingTrajectory`, `ContinuousTrainingSchedule` | Training analysis & pruning |

**Overlap**: Both track epoch-level metrics, checkpoints, trajectories.

**Solution**:
1. Extract shared `EpochCheckpoint` to `core/training_state.py`
2. Make `TrainingCheckpoint` and `TrialMetrics` extend/embed it
3. Unify SQL schema in `hyperopt/storage.py`
4. `execution/training_dynamics.py` imports from shared location

---

## 🟢 LOW — Incremental Polish

### 5. Data Loading Consolidation (~150 lines)
Canonical transforms duplicated across `domains/`, `zoo/mep/benchmarks/`, `validation/tracks/`.

**Solution**: Add `data/transforms.py` with:
```python
MNIST_TRANSFORM = transforms.Compose([...])
CIFAR10_TRANSFORM = transforms.Compose([...])
def create_dataloader(dataset, config, split): ...
```
Migrate inline `DataLoader` creations.

---

### 6. Validation Tracks Infrastructure (~200 lines)
11 track files in `validation/tracks/` each repeat boilerplate: dataset creation, model instantiation, training loop, evaluation, evidence generation, `TrackResult` construction.

**Solution**: Add `validation/tracks/_base.py`:
```python
def run_track(verifier, model_factory, train_fn, eval_fn, name, description, category):
    """Execute standard track pattern."""
```
Each track becomes ~20 lines of configuration + assertions.

---

### 7. BenchmarkMetrics Naming Reconciliation (~40 lines + schema)
`BenchmarkMetrics` uses `train_acc`/`val_acc`; `TrainingMetrics` uses `train_accuracy`/`val_accuracy`. Touches SQL schemas, checkpoints, call sites.

**Verdict**: Low value per line — **defer** unless future ticket unifies trial representation.

---

## Next Immediate Actions (Priority Order)

1. **Continue Config Unification** — Migrate remaining config pairs onto `config/unified.py` using proven `ModelConfig` pattern
2. **EquiTile Generification Week 1** — Create `core/tile/` + `core/local_learning/` infrastructure (enables algorithm reuse across codebase)
3. **Optimizer Factory Migration** — Migrate 5-10 high-impact creation sites to `create_optimizer()`
4. **Storage Unification** — Extract `EpochCheckpoint` to `core/training_state.py`
5. **FastLMEquiTile Decision** — Resolve architecture choice (Option 2 recommended)

---

## Verification Gates (Per Phase)
```bash
ruff format . && ruff check --fix .
pyright .                          # zero errors in strict mode
pytest --cov                       # all tests pass, coverage ≥85%
pip-audit                          # no new vulnerabilities
```

---

## Projected Outcome
- **Additional reduction**: ~3,100 lines (7.6%)
- **Total reduction**: ~5,650 lines (13.8%)
- **Key multiplier**: EquiTile generification unlocks 5+ new algorithms with minimal new code