# REFACTOR.md — Strategic Refactoring Plan for bioplausible

**Codebase**: 316 Python files, ~41K lines  
**Goal**: Maximize size reduction via deduplication, DRY, structural consolidation  
**Completed**: ~2,010 lines saved (4.9%) across 68+ files

**2026-08-10 progress**: +~250 lines saved. Optimizer factory migration (8 sites), storage unification (`core/training_state.py`), FastLMEquiTile decision re-opened (wrapper requires re-architecture). See "Session Notes" below.

**2026-08-10 (Session 2) progress**: **Optimizer factory** extended with param-subset/param-group support → migrated `zoo/models/fa.py` (5), `equitile/training/optimizer_mixin.py` (3), `equitile/language/fast.py` (1 param-group) — 26 sites factory-driven. **LM `TrainingConfig`** reconciled onto unified `core/trainer.py:LMTrainingConfig(TrainerConfig)` (local ~87-line dataclass deleted; shared epoch knobs inherited). **`epoch_metrics` SQL schema** unified with the `EpochCheckpoint` column set in `hyperopt/storage.py`. **Shared `get_lm_dataset` friendly-name normalization** sink in `data/lm.py`. See "Session Notes 2" below.

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
| Optimizer Factory | `core/utils/optimizer.py` now drives 26 sites (deployments, LM variants, zoo/fa, equitile mixins, language/fast) |
| Training-State Types | `core/training_state.py` — shared `EpochCheckpoint`/`TrainingTrajectory` |
| LM Trainer Config | `equitile/lm/training.py:TrainingConfig` → unified `core/trainer.py:LMTrainingConfig(TrainerConfig)`, local class deleted |

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

## 📝 Session Notes 2 (2026-08-10, second session)

### Optimizer factory: param-subset/param-group support (§3) — 9 more sites
`create_optimizer` signature changed from `(model, config)` to `(model_or_params, config)` — the first arg is now a `nn.Module` **or** an explicit iterable of parameters / param-group dicts. This unlocked every former "needs subset support" blocker with no API break (module path unchanged):

- `zoo/models/fa.py` — 5 sites: `_ensure_optimizer`, `AdaptiveFeedbackAlignment.w_optimizer`/`b_optimizer`, `ContrastiveFeedbackAlignment`, `StandardFA` (incl. `[p for p in model.parameters() if p.requires_grad]` filters).
- `equitile/training/optimizer_mixin.py` — 3 sites, `W_in+W_out`, `[tile_importance, edge_importance]`, full-model.
- `equitile/language/fast.py` — 1 param-groups site (`AdamW` with `weight_params`/`importance_params` LR groups).

Factory now drives **26 sites** total. Remaining high-value: `core/trainer.py` dynamic `opt_cls` (line 710, `torch.optim.Adam`), `zoo/models/target_prop.py` (3), `equitile/_internal/enhanced.py` (3, duplicate of `optimizer_mixin`), `zoo/mep/benchmarks/runner.py`, `validation/tracks/*.py`.

### LM TrainingConfig → unified `LMTrainingConfig` (§1, cheapest config pair done)
Reconciled `equitile/lm/training.py:TrainingConfig` onto `core/trainer.py:LMTrainingConfig(TrainerConfig)`. The local ~87-line dataclass (incl. 50-line docstring) is **deleted**; `equitile/lm/training.py` now imports `LMTrainingConfig as TrainingConfig` and re-exports unchanged (`equitile.lm.TrainingConfig` API stable). Key design points:

- Shared epoch-wise knobs (`epochs`, `device`, `num_workers`, `checkpoint_dir`, `grad_clip`) are **inherited** from `TrainerConfig` rather than redeclared.
- `model: str | None = None` override lets LM construction omit the registry-name (LMTrainer binds an explicit model instance).
- `__post_init__` preserved the `device="auto"` → concrete-backend resolution so `equitile.lm.demo` and `tests/integration/test_lm_demo.py` (`TestTrainingConfig`) pass unmodified.

**Do NOT do**: `config/schema.py:TrainingConfig` (OmegaConf structured, `log_every_n_steps`/`save_every_n_epochs`/early-stopping knobs) and the LM one are still different trainers with different knobs — leave schema's as-is. `FastLMConfig` dup itself remains (two distinct architectures, see below).

### Storage: `epoch_metrics` schema unified with `EpochCheckpoint` (§4)
`hyperopt/storage.py:epoch_metrics` table upgraded from 6 minimal columns (`loss`/`accuracy`/`perplexity`/`time`) to the full `EpochCheckpoint` column set — now a byte-for-byte schema twin of `training_checkpoints`. `log_epoch` now constructs an `EpochCheckpoint` internally (with legacy-arg fallbacks so `hyperopt/experiment.py` call site needs no change) and inserts the shared column set. Favoring the RUF013 cleanup, moved `.metrics` import to the top (E402 fixed), net lint −1.

**Deliberately not done**: merging `epoch_metrics` INTO `training_checkpoints` — different FK (trial vs trajectory), both are write-only sinks consumed by different analysis paths; keeping both avoids a migration job. Existing DBs keep the old schema (backwards-compat is NONE per AGENTS.md; the CREATE TABLE IF NOT exists path just leaves old files untouched).

### FastLM data plumbing (§5) — shared `get_lm_dataset` sink
`data/lm.py` gained `_normalize_lm_name()`: friendly display names ("Tiny Shakespeare", "WikiText-2", "PTB") normalize to canonical IDs, centralizing the friendly-name → ID mapping that `equitile/language/fast.py:_load_data` previously inlined. Demo's `_load_data` dropped ~6 lines of mapping logic. Also migrated the demo's `torch.optim.AdamW` param-groups optimiser to the factory.

**Config merge deliberately NOT attempted**: `language/fast.py:FastLMConfig(LMEquiTileConfig)` and `lm/components.py:FastLMConfig` describe different architectures (OptimizedLMEquiTile pre-norm + sigmoid gating vs BioModel MoT+SwiGLU) with different field defaults — force-merging would silently change demo behavior. The `FastLMConfig` dup line in §1.1 estimate is retained as a known, intentional duplicate.

### Verification (Session 2)
- `ruff check` changed files: neutral or better (storage 17→16, language/fast 20→19; trainer + equitile/lm/training unchanged at 50 pre-existing).
- `pyright` changed files: **0 errors**, 212 warnings (all pre-existing `Optional`/`Path`/mixin-override patterns).
- `pytest`: 242 unit (core/data/refactor2/scheduler) + 111 unit (zoo/equitile) + 101 (hyperopt/execution) + 62 (trainer-coverage/lm-demo) + 6 (hyperopt-integration/continuous) + 12 (phase2/scientist) all pass. One rerun-flaky failure (`test_lm_equitile`, `torch.multinomial` NaN) is a local-numerics flake, not a regression — passes back-to-back on rerun and on clean HEAD when it did fail.

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

**Decision (2026-08-10)**: Option 2 is **NOT** a thin wrapper — the demo extends `OptimizedLMEquiTile` (pre-norm transformer + tile-importance sigmoid gating) while canonical `lm/fast_lm.py` extends `BioModel` (Mixture-of-Tiles + SwiGLU). Retargeting the demo onto the canonical architecture changes demo behavior. **Chosen**: keep both models; the only clean dedup is the demo's `_load_data`/dataset plumbing — route both through one `get_lm_dataset` path while leaving the models and their configs distinct. **Done (2026-08-10 session 2)**: shared friendly-name normalization in `data/lm.py::_normalize_lm_name`; demo `_load_data` simplified. The demo optimizer (param-groups AdamW) migrated to the factory. Config merge of the two `FastLMConfig` dataclasses is **final-rejected**: different bases, different defaults, different architectures.

---

## 🟡 MEDIUM — High Leverage

### 3. Optimizer Factory Consolidation (~150 lines)
**Factory created**: `core/utils/optimizer.py` — `OptimizerConfig` (frozen+slots) + `create_optimizer(model_or_params, config)` supporting `adam`/`adamw`/`sgd`. First arg is a `nn.Module` **or** explicit parameter iterable / param-group dicts (param-subset support added 2026-08-10 session 2).

**26 of ~60 sites migrated**: `validation/utils.py`, `equitile/lm/ablation_study.py`, `validation/tracks/application_tracks.py`, `equitile/deployments/{base,vision,rl,timeseries,graph}.py`, `equitile/lm/fast_lm.py`, `equitile/lm/training.py`, `equitile/language/optimized.py` (session 1); `zoo/models/fa.py` (5), `equitile/training/optimizer_mixin.py` (3), `equitile/language/fast.py` (1, param-groups) (session 2).

**Remaining high-impact sites** (param-subset support no longer a blocker):
- `core/trainer.py` (dynamic `opt_cls`, `torch.optim.Adam` fallback)
- `zoo/models/target_prop.py` (3: forward/inverse/output optimizers)
- `zoo/models/forward_only.py` (2), `zoo/models/eqprop/_unified.py`
- `equitile/_internal/enhanced.py` (3 — clone of `optimizer_mixin._setup_optimizers`; consolidate once mixin is lifted to `core/local_learning`)
- `zoo/mep/benchmarks/runner.py`, `validation/tracks/*.py` (10+ files)

---

### 4. Hyperopt/Execution Storage Unification (~300 lines)
Two independent checkpoint/trajectory systems:

| Module | Dataclasses | Purpose |
|--------|-------------|---------|
| `hyperopt/storage.py` | `TrialMetrics`, epoch_metrics table, training_trajectories table | Optuna trial persistence |
| `execution/training_dynamics.py` | `TrainingCheckpoint`, `TrainingTrajectory`, `ContinuousTrainingSchedule` | Training analysis & pruning |

**Progress (2026-08-10)**: Shared `EpochCheckpoint` + `TrainingTrajectory` extracted to `core/training_state.py`. `execution/training_dynamics.py` (−150 lines) and `hyperopt/storage.py` both import from it. `TrialMetrics` intentionally left in `hyperopt/metrics.py` (whole-trial objectives, not epoch-level); revisit alongside §7 trial-representation unification.

**Progress (2026-08-10 session 2)**: `epoch_metrics` SQL schema unified to the shared `EpochCheckpoint` column set; `log_epoch` builds an `EpochCheckpoint` and inserts the shared columns (legacy `loss`/`accuracy`/`perplexity`/`time` positional args kept, with keyword fallbacks for the richer fields).

**Remaining**:
1. ~~Unify SQL schema in `hyperopt/storage.py`~~ DONE — `epoch_metrics` matches `training_checkpoints` column-for-column.
2. Consider `EpochCheckpoint` aliasing over the `epoch_metrics` insert path — deferred: only write-only today; merge tables only if a shared reader appears.

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

Done this session: **(3) optimizer factory** param-subset support + fa.py/mixin/language-fast migration, **(4) storage** `epoch_metrics` schema unification, **(1) LM `TrainingConfig`** → `LMTrainingConfig(TrainerConfig)`, **(5) FastLM** shared `get_lm_dataset` sink. Remaining, in priority order:

1. **Continue Config Unification** — Next cheapest trainer family: `equitile/lm/components.py:FastLMConfig` is already the single canonical for `lm/fast_lm.py` (the `language/fast.py` variant is a genuinely different architecture — leave both, see Session Notes 2 §5). Next candidate: `zoo/mep/benchmarks/tuned_compare.py:OptimizerConfig` + `config/schema.py:OptimizerConfig` → these are *per-algorithm* configs with different fields (`gamma`, per-family values) and are **not** the factory's `OptimizerConfig` — reconcile only if a shared subset is extracted. Do NOT bulk-add speculative configs.
2. **EquiTile Generification Week 1** — Create `core/tile/` + `core/local_learning/` infrastructure (enables algorithm reuse across codebase). Audit: only `equitile/core/model.py`, `_internal/enhanced.py`, `training/async_execution.py`, `training/distributed.py`, `training/optimizer_mixin.py` + 2 tests import the tile primitives — low blast radius.
3. **Optimizer Factory Migration (cont.)** — Remaining sites now that param-subset/groups are supported: `core/trainer.py` (dynamic `opt_cls`, line ~710), `zoo/models/target_prop.py` (3), `zoo/models/forward_only.py` (2), `zoo/models/eqprop/_unified.py`, `equitile/_internal/enhanced.py` (3 — mirror of `optimizer_mixin`; could import from the mixin once mixin is lifted to core), `zoo/mep/benchmarks/runner.py`, `validation/tracks/*.py`.
4. **Storage Unification (cont.)** — Consider aliasing `EpochCheckpoint` directly over the `epoch_metrics` read path if a reader is added; merging the two tables is deliberately deferred (different FK, no consumer needs both).
5. **FastLMEquiTile** — Models stay distinct (decision upheld). Remaining soft dedup: share the demo's `update_activity_ema`/gate instrumentation primitives if a future algorithm needs binary-gate tile selection.

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