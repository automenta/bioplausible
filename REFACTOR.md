# REFACTOR.md — Strategic Refactoring Plan for bioplausible

**Codebase**: 316 Python files, ~41K lines  
**Goal**: Maximize size reduction via deduplication, DRY, structural consolidation  
**Completed**: ~2,010 lines saved (4.9%) across 68+ files

**2026-08-10 progress**: +~250 lines saved. Optimizer factory migration (8 sites), storage unification (`core/training_state.py`), FastLMEquiTile decision re-opened (wrapper requires re-architecture). See "Session Notes" below.

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
| Optimizer Factory | `core/utils/optimizer.py` now drives 14 sites (deployments, LM variants) |
| Training-State Types | `core/training_state.py` — shared `EpochCheckpoint`/`TrainingTrajectory` |

---

## 📝 Session Notes (2026-08-10)

### Optimizer Factory Migration (§3) — 8 sites migrated this session
Migrated inline `torch.optim.Adam/AdamW` → `create_optimizer(model, OptimizerConfig)`:
- `equitile/deployments/base.py` (2: feature + head)
- `equitile/deployments/vision.py` (2: conv + head)
- `equitile/deployments/rl.py`, `timeseries.py`, `graph.py` (1 each)
- `equitile/lm/fast_lm.py`, `equitile/lm/training.py`, `equitile/language/optimized.py` (AdamW with `betas=(0.9, 0.95)`)

**Deferred (need param-subset support)**: `zoo/models/fa.py` (6 sites use `[p for p in model.parameters() if p.requires_grad]` or split w/b optimizers) and `equitile/training/optimizer_mixin.py` (3 sites, `W_in+W_out` subset). Extend `create_optimizer` with an optional `param_filter`/`param_groups` arg to unlock these.

### Storage Unification (§4) — `core/training_state.py`
Extracted shared `EpochCheckpoint` (frozen+slots) and `TrainingTrajectory` (with `compute_convergence_speed` / `compute_sample_efficiency` / `detect_overfitting`). `execution/training_dynamics.py` now re-imports these (net **−150 lines**); `hyperopt/storage.py` imports from the shared home. `TrialMetrics` was left in `hyperopt/metrics.py` (it models whole-trial objectives, not epoch checkpoints — merging would force artificial fields; see §7-dependent future ticket).

### Config Unification (§1) — assessment this session
**Do NOT add the speculated unified configs** (`OptimizerConfig`, `RLConfig`, `VisionConfig`, `GraphConfig`, `TimeSeriesConfig`) to `config/unified.py`:
- `OptimizerConfig` already exists canonically in `core/utils/optimizer.py` (frozen+slots) and is what `create_optimizer` consumes.
- RL/Vision/Graph/Timeseries deployment configs already canonically exist in `equitile/deployments/base.py`.

Remaining targeted migrations require *field-level reconciliation*, not a merge:
- `config/schema.py:TrainingConfig` (OmegaConf structured: `log_every_n_steps`, `save_every_n_epochs`, `early_stopping_*`) ≠ `equitile/lm/training.py:TrainingConfig` (`save_every`, `log_every`, `generate_every`, `use_amp`). Different trainers, different knobs.
- `experiments/utils.py:ExperimentConfig` (model/optimizer/runner) ≠ `equitile/utils/reproducibility.py` (seed + 4 config dicts) ≠ `equitile/analysis/research.py` (name/description/tags).
- `FastLMConfig` appears 3×: `equitile/lm/components.py` (canonical), `equitile/language/fast.py` (demo, extends `LMEquiTileConfig`), `config/schema` n/a.

**Do this instead**: pick ONE trainer family per session and reconcile its fields onto `unified.py`, deleting the consumer's local class. Cheapest first: `equitile/lm/training.py:TrainingConfig` → subtree of `TrainerConfig` in `core/trainer.py`.

### FastLMEquiTile decision (§2.1) — re-opened
The "Option 2 wrapper" (`language/fast.py → DemoFastLMEquiTile(FastLMEquiTile)`) is NOT a thin wrapper: the demo extends `OptimizedLMEquiTile` (different architecture: pre-norm transformer + tile importance gating), while canonical `lm/fast_lm.py` extends `BioModel` (MoT + SwiGLU). Conflating them would silently change the demo's behavior. **Recommendation**: keep both, delete the demo's `FastLMConfig` (make it `FastLMConfig(**shared)`), and add a shared `get_lm_dataset` sink. ~100-line save without behavioral risk.

### Verification
- `ruff check` on all changed files: neutral or improved vs baseline (training_dynamics 13→10).
- `pyright` on all changed files: 0 errors (warnings are pre-existing `Optional` access patterns).
- `pytest`: 121 unit/equitile + integration tests pass; the 7 pre-existing integration failures (onnx/smoke/triton/model-integration) are identical on clean HEAD — none introduced by this session.

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
2. **Wrapper pattern**: `language/fast.py` → thin `DemoFastLMEquiTile(FastLMEquiTile)` subclass (~150 lines)
3. **Unify base**: Extract common layer primitives to shared module

**Decision (2026-08-10)**: Option 2 is **NOT** a thin wrapper — the demo extends `OptimizedLMEquiTile` (pre-norm transformer + tile-importance sigmoid gating) while canonical `lm/fast_lm.py` extends `BioModel` (Mixture-of-Tiles + SwiGLU). Retargeting the demo onto the canonical architecture changes demo behavior. **Chosen**: keep both models; the only clean dedup is the demo's private `FastLMConfig` (+) `_load_data`/dataset plumbing — merge those onto the shared `FastLMConfig` and a shared LM-data helper (~100 lines) while leaving the models distinct.

Safe next step: make `language/fast.py`'s `FastLMConfig(LMEquiTileConfig)` import shared base fields instead of re-declaring, and route both through one `get_lm_dataset` path.

---

## 🟡 MEDIUM — High Leverage

### 3. Optimizer Factory Consolidation (~150 lines)
**Factory created**: `core/utils/optimizer.py` — `OptimizerConfig` (frozen+slots) + `create_optimizer(model, config)` supporting `adam`/`adamw`/`sgd`.

**11 of ~60 sites migrated**: `validation/utils.py`, `equitile/lm/ablation_study.py`, `validation/tracks/application_tracks.py`, `equitile/deployments/base.py`, `equitile/deployments/vision.py`, `equitile/deployments/rl.py`, `equitile/deployments/timeseries.py`, `equitile/deployments/graph.py`, `equitile/lm/fast_lm.py`, `equitile/lm/training.py`, `equitile/language/optimized.py` (2026-08-10).

**Remaining high-impact sites**:
- `zoo/models/fa.py` (6 creations — need param-subset support in factory)
- `validation/tracks/*.py` (10+ files)
- `core/trainer.py` (dynamic `opt_cls`)
- `equitile/training/optimizer_mixin.py` (3, uses `W_in+W_out` param subset)

**Action**: Extend `create_optimizer` with an optional `params=`/`param_groups=` override so subset-sites (`fa.py`, `optimizer_mixin.py`) can migrate; then migrate the rest.

---

### 4. Hyperopt/Execution Storage Unification (~300 lines)
Two independent checkpoint/trajectory systems:

| Module | Dataclasses | Purpose |
|--------|-------------|---------|
| `hyperopt/storage.py` | `TrialMetrics`, epoch_metrics table, training_trajectories table | Optuna trial persistence |
| `execution/training_dynamics.py` | `TrainingCheckpoint`, `TrainingTrajectory`, `ContinuousTrainingSchedule` | Training analysis & pruning |

**Progress (2026-08-10)**: Shared `EpochCheckpoint` + `TrainingTrajectory` extracted to `core/training_state.py`. `execution/training_dynamics.py` (−150 lines) and `hyperopt/storage.py` both import from it. `TrialMetrics` intentionally left in `hyperopt/metrics.py` (whole-trial objectives, not epoch-level); revisit alongside §7 trial-representation unification.

**Remaining**:
1. Unify SQL schema in `hyperopt/storage.py` (epoch_metrics table → shared column set) — ~40 lines
2. Consider `EpochCheckpoint` aliasing over the `epoch_metrics` insert path

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

1. **Continue Config Unification** — Migrate ONE trainer family at a time (cheapest: `equitile/lm/training.py:TrainingConfig` → reconcile onto unified). Do NOT bulk-add speculative configs (see Session Notes).
2. **EquiTile Generification Week 1** — Create `core/tile/` + `core/local_learning/` infrastructure (enables algorithm reuse across codebase). Audit: only `equitile/core/model.py`, `_internal/enhanced.py`, `training/async_execution.py`, `training/distributed.py`, `training/optimizer_mixin.py` + 2 tests import the tile primitives — low blast radius.
3. **Optimizer Factory Migration** — Add `params=`/`param_groups=` override to `create_optimizer`, then migrate `zoo/models/fa.py` (6) + `optimizer_mixin.py` (3).
4. **Storage Unification** — Unify `epoch_metrics` SQL schema insert path over `EpochCheckpoint` (~40 lines).
5. **FastLMEquiTile** — Merge the demo's `FastLMConfig` onto shared base + shared LM-data helper; keep the two models distinct.

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
- **Additional reduction**: ~2,850 lines (7.0%) [revised down after storage/config scope correction: `TrialMetrics` left separate, deployment configs already canonical, FastLM wrapper rejected]
- **Total reduction**: ~5,400 lines (13.2%)
- **Key multiplier**: EquiTile generification unlocks 5+ new algorithms with minimal new code