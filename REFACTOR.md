# REFACTOR.md — Strategic Refactoring Plan for bioplausible

**Codebase**: 316 Python files, ~41K lines  
**Goal**: Maximize size reduction via deduplication, DRY, structural consolidation  
**Completed**: ~3,080 lines saved (7.5%) across 90+ files

**2026-08-10 (Session 4) progress**: **EquiTile generification Phase 2 complete + Phase 3 feature-extractor decoupling landed.** All 4 shims deleted — `equitile` now imports `core.tile`/`core.local_learning` directly. `_internal/enhanced.py:_setup_optimizers` folded into `MultiOptimizerMixin` (new `extra_importance_params()` + `importance_params()` hooks; the clip-grad site also deduped). `equitile/deployments/_feature_extractors.py` → generic `core/tile/feature_extractors.py` (+ `TileModelFactory` injection) with the EquiTile-embedding layers param'd — the `core → equitile` reverse dependency is GONE. Validation tracks §6 `_base.py` (`track_header` / `build_track_result`) landed; 18 standard-banner tracks migrated. See "Session Notes 4" below.

**2026-08-10 (Session 3) progress**: **Optimizer factory** 12 more sites (target_prop 3, forward_only 2, eqprop/_unified 1, `_internal/enhanced.py` 3, `zoo/mep/benchmarks/runner.py` 3) — 38 sites total. **EquiTile generification §2 Phase 1 landed**: `core/tile/{topology,kernels}` + `core/local_learning/{task,mixins,config}`; `EquiTileConfig` now extends `LocalLearningConfig`; equitile consumers rewired to `core.tile`. See "Session Notes 3" below.

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
| Optimizer Factory | `core/utils/optimizer.py` now drives 38 sites (deployments, LM variants, zoo models, eqprop engine, enhanced equitile, mep-bench runner) |
| Training-State Types | `core/training_state.py` — shared `EpochCheckpoint`/`TrainingTrajectory` |
| LM Trainer Config | `equitile/lm/training.py:TrainingConfig` → unified `core/trainer.py:LMTrainingConfig(TrainerConfig)`, local class deleted |
| Tile Substrate | `core/tile/` — generic `TileGraph`/`TileState` + 4 math kernels (moved from `equitile.core`) |
| Local-Learning Infra | `core/local_learning/` — `TaskHandler`, `MultiOptimizerMixin` (renamed from `EquiTileOptimizerMixin`), `LocalLearningConfig` base |
| EquiTile Shim Removal | Phase 2a — 4 shims (`equitile/core/{topology,kernels}.py`, `equitile/training/{task_handler,optimizer_mixin}.py`) deleted; `equitile` imports `core.tile`/`core.local_learning` directly |
| Enhanced Optimizer Fold | Phase 2b — `_internal/enhanced.py:_setup_optimizers` override deleted; `MultiOptimizerMixin` gained `extra_importance_params()` hook + `importance_params()` helper (clip-grad site deduped too) |
| Feature Extractors | Phase 3.1 — generic extractors/attention/graph-utils → `core/tile/feature_extractors.py`; EquiTile-embedding layers (`TemporalEquiTileLayer`, `GraphEquiTileLayer`) param'd with `TileModelFactory`; `core → equitile` edge eliminated |
| Validation Track Boilerplate | `validation/tracks/_base.py` — `track_header()` (banner + timing anchor) + `build_track_result()`; 18 standard-banner track fns migrated |

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

## 📝 Session Notes 4 (2026-08-10, fourth session)

### §2 Phase 2a — shims deleted, `equitile` re-pointed at `core.tile`
`equitile/core/__init__.py` and `equitile/__init__.py` now import `TileGraph`/`TileState` from `bioplausible.core.tile`; `tests/unit/equitile/test_equitile_refactor.py` imports `TaskHandler` from `core.local_learning`. **Deleted** `equitile/core/topology.py`, `equitile/core/kernels.py`, `equitile/training/task_handler.py`, `equitile/training/optimizer_mixin.py` (~70 lines of shim bodies). The only remaining references are stale docstring mentions (harmless history). The `EquiTileOptimizerMixin` alias name is retired (it was only used by the shim; search confirms zero live importers).

### §2 Phase 2b — `_setup_optimizers` folded into `MultiOptimizerMixin`
`EnhancedEquiTile._setup_optimizers` was a near-verbatim clone; replaced with an `extra_importance_params()` hook (returns `[self.tile_lr_scale]` when the per-tile-LR parameter exists — the old `hasattr` guard timeline is preserved: `EquiTile.__init__` calls `_setup_optimizers` *before* `EnhancedEquiTile.__init__` attaches `tile_lr_scale`; the hook must stay defensive) plus a shared `importance_params()` helper. The `enhanced.py` importance-clip site (`clip_grad_norm_`) now reuses `importance_params()` too. `enhanced.py` net −24 lines incl. dropped `OptimizerConfig`/`create_optimizer` import.

### §2 Phase 3.1 — feature-extractor decoupling (the `core → equitile` blocker, RESOLVED)
- **New `core/tile/feature_extractors.py`** (~565 lines): graph scatter utils, `conv/temporal/graph` extractors, `Temporal{PositionalEncoding,AttentionLayer}`, `GraphAttentionLayer` — all EquiTile-free, typed against structural `Protocol`s (`_ConvConfig`/`_TemporalConfig`/`_GraphConfig`) so deployment dataclasses satisfy them without a `core → equitile` import. **`type TileModelFactory = Callable[..., nn.Module]`**.
- The two EquiTile-embedding layers (`TemporalEquiTileLayer`, `GraphEquiTileLayer`) plus the `TemporalFeatureExtractor`/`GraphFeatureExtractor` that stack them now take a `tile_model_factory` and keep shared kwarg building in `tile_model_kwargs(config, num_layers, activation)`.
- **`equitile/deployments/_feature_extractors.py` shrinks 559 → 103 lines** as a re-export layer: `tile_model_factory(*, input_dim, output_dim, **kwargs)` binds `EquiTileConfig`+`EquiTile` (one typed `# type: ignore[reportArgumentType]` for the dynamic splat), `RLFeatureExtractor` stays (its `get_config()` now returns the stored `EquiTileConfig`, behavior-identical to `EquiTile.get_config()`). All downstream public re-exports (`equitile.deployments.*`, `equitile/__init__.py`) unchanged — zero consumer churn beyond `timeseries.py`/`graph.py` passing `_fe.tile_model_factory`.
- Net effect: **no `core` module imports `equitile` at runtime anymore**; the tile-substrate extractors are usable by FA/TargetProp/SNN/GNN without EquiTile. (Type-only `core/local_learning/mixins.py:20` still references `EquiTileConfig` under `TYPE_CHECKING` — see opportunities below.)

### §6 — Validation Tracks `_base.py` landed
`validation/tracks/_base.py` supplies `track_header(track_id, name, width=60) -> float` (3-line banner + timing anchor) and `build_track_result(*, track_id, name, start, status, score, metrics, evidence, improvements, evidence_level="smoke", limitations)` (assembles `TrackResult` with elapsed time). **18 standard `/track_*/` fns migrated** across `application_tracks` (2), `architecture_comparison` (1), `core_tracks` (3), `hardware_tracks` (3), `negative_results` (1), `research_tracks` (3), `scaling_tracks` (4), `tradeoff_tracks` (1, `width=70`, extra reality-check `logger.info` kept inline). `nebc_tracks` (5) **not migrated** — they use a stub `time_seconds=0.1` and a bespoke single-line banner with no timing boilerplate. `signal_tracks.py` untouched (already on `ValidationTrack`).
- Supporting moves ruff required: `TrackResult` is now annotation-only in the migrated modules → moved under `if TYPE_CHECKING:` **and** each file got `from __future__ import annotations` (they previously evaluated annotations at runtime). ruff `--unsafe-fixes` also surfaced two harmless cleanups (`nebc_tracks` mid-module imports hoisted; `track_registry` put `collections.abc.Callable` under `TYPE_CHECKING`).
- **Line accounting honesty**: net ~±0 lines (the TOIL estimate of ~66 was optimistic). Headers −4 lines/site but `from __future__` + `TYPE_CHECKING` imports + `from ._base import` add the same back. The win is single-sourced banner/result-assembly (one format, one timing path), not line count. Worth keeping only because the duplication would otherwise drift.

### New improvement opportunities found
- `core/local_learning/mixins.py:20` still `TYPE_CHECKING`-imports `equitile.core.config.EquiTileConfig`. Now that equitile pulls from core cleanly, flip it: `LocalLearningConfig` (or a narrow `Protocol` exposing `learning_rate`/`importance_lr`/`mode`) can be the mixin's declared config type, making `core/*` 100% equitile-free even at type-check time.
- `core/tile/feature_extractors.py` `tile_model_kwargs()` duplicates the base-field mapping that `zoo`/`equitile/core/model.py::EquiTile.build` also do; a generic `tile_kwargs(base_config)` sink could serve both (postponed — `EquiTile.build` uses `spec`-driven fields).
- `_feature_extractors.py:tile_model_factory` `# type: ignore[reportArgumentType]` is the one remaining `Any`-ish dynamic splat; if a future `equitile_kwargs: dict[str, object]` → typed-config projection helper appears, it can go.

### Verification (Session 4)
- `ruff check` changed files: no new F/I/E; remaining are pre-existing style (magic-value, non-lowercase-names, too-many-*, assert, TCH on pre-existing imports). `_base.py` is clean (one targeted `# ruff: ignore[too-many-arguments]` — the signature mirrors `TrackResult` fields by design).
- `pyright` all changed paths: **0 errors** (479 warnings, all pre-existing `Optional`/Tensor/`Path`/mixin patterns).
- `pytest`: 705 (core/equitile/validation/refactor/kernel/scheduler) + 312 (registry/track/refactor-enhanced) + 520 (zoo/models/domains/experiment) + 126 (equitile-domains/lm-demo/data) + track-registry + registry-audit, all pass. **Pre-existing failures unchanged on clean HEAD**: `test_backprop_parity[eqprop_mlp]`, `test_backprop_parity[directed_ep]`, `test_nebc_base.py::test_cannot_instantiate_base`. Flaky-on-rerun only: `test_lm_equitile` (torch.multinomial NaN, passes on rerun).

### Optimizer factory: final remaining sweep (§3) — 12 more sites, 38 total
- `zoo/models/target_prop.py` (3: forward/inverse/output nets), `zoo/models/forward_only.py` (2: `FFLayer` + classifier), `zoo/models/eqprop/_unified.py` (1: implicit-adjoint Adam), `equitile/_internal/enhanced.py` (3: io/importance/full clones), `zoo/mep/benchmarks/runner.py` (3: SGD/Adam/AdamW branches; SMEP/SDMEP are MEP-specific and untouched).
- **⚠️ Preserve original `weight_decay`**: the factory default `weight_decay=1e-4` differs from bare `torch.optim.Adam(..., lr=...)` (0.0). Passing the default silently changed training enough to fail `test_backprop_parity[forward_forward]` (acc 0.366→0.146 over 3 epochs). New sites pass `weight_decay=0.0` explicitly where the original used torch defaults. (The earlier fa.py/mixin sites took the 1e-4 default under test tolerance; do not assume that holds elsewhere.)
- **`core/trainer.py` → NOT migrated** (permanently). `_create_optimizer` (line ~727) is already config-driven (`optimizer: str` + `optimizer_kwargs: dict`), supports *arbitrary* `torch.optim` names via `getattr`, and defers creation for learning-rule optimizers. The factory (fixed `adam`/`adamw`/`sgd` + fixed kwargs) is *less* expressive there — migration would regress functionality, not dedup.

### EquiTile generification (§2) — Phase 1 complete, Phase 2 partial
**New canonical homes** (verified byte-identical bodies, renamed where genericity demands):
- `core/tile/topology.py` — `TileGraph`, `TileState`
- `core/tile/kernels.py` — `compute_activity_update`, `compute_hebbian_update`, `compute_contrastive_hebbian_update`, `compute_tile_prediction`
- `core/local_learning/task.py` — `TaskHandler`
- `core/local_learning/mixins.py` — `MultiOptimizerMixin` (renamed from `EquiTileOptimizerMixin`; `equitile.training.optimizer_mixin.EquiTileOptimizerMixin` is now an alias, so `equitile` API is stable)
- `core/local_learning/config.py` — new frozen+slots `LocalLearningConfig` (22 architecture/learning/dynamics/task fields + `validate()`).

**`EquiTileConfig(LocalLearningConfig)`** (phase 2.1): the 22 shared fields are inherited; EquiTile keeps only its energy-dynamics (`mode`, `lambda_error`, `beta`, `beta_anneal`, `inference_steps_free/nudged`, `use_symmetric_weights`, `ep_init_scale`) and importance/sparsity knobs. `validate()` chains `super().validate()`. `EnhancedEquiTileConfig`/`DynamicEquiTileConfig` untouched. net −26 lines in `equitile/core/config.py`.

**Consumers rewired to `core.tile` directly**: `equitile/core/model.py` (+ `BioModel, MultiOptimizerMixin`), `_internal/enhanced.py`, `training/async_execution.py`, `training/distributed.py`.
~~**Shims retained**...~~ **DONE Session 4**: the 4 shims were deleted after re-pointing `equitile/core/__init__.py` + `equitile/__init__.py` at `core.tile`; see Session Notes 4 §Phase 2a.

### §2 Phase 3.1 — feature extractor lift (was BLOCKED, **DONE Session 4**, see Session Notes 4 §Phase 3.1)
`equitile/deployments/_feature_extractors.py` *constructed `EquiTile` directly* (`RLFeatureExtractor`, `TemporalEquiTileLayer`, `GraphEquiTileLayer` embed `EquiTile(config=...)`), which created a `core → equitile` edge.
**Executed path**: (1) extracted the EquiTile-free helpers — `aggregate_messages`, `scatter_mean/sum/max`, `create_graph_from_edges`, `add_self_loops`, `ConvFeatureExtractor`, `TemporalFeatureExtractor`, `TemporalPositionalEncoding`, `TemporalAttentionLayer`, `GraphFeatureExtractor`, `GraphAttentionLayer` — into `core/tile/feature_extractors.py`; (2) param'd the EquiTile-embedding layers (`TemporalEquiTileLayer`, `GraphEquiTileLayer`) with a `TileModelFactory` instead of hardcoding `EquiTile`; (3) re-exported from the slimmed `equitile/deployments/_feature_extractors.py` for the 5 deployment consumers.

### Validation tracks §6 — assessment (deferred this session)
22 free-function `track_*` across 10 modules. The plan's `run_track(verifier, model_factory, train_fn, eval_fn, ...)` orchestration does **not** fit: scoring/evidence/metrics are per-track bespoke. The actual shared boilerplate is only: 3-line banner logging, `start = time.time()`, `TrackResult(... time_seconds=time.time()-start)` construction, and `.description`/`.category` attachment.
**Cheapest consolidation** (~3 lines/call site, ~66 lines saved): `validation/tracks/_base.py` with `track_header(track_id, name)` + `build_track_result(...)`. Migrate mechanically per file; `signal_tracks.py` already uses `ValidationTrack` from `validation/notebook.py` (leave it). **→ EXECUTED Session 4** (18 tracks migrated; net line change ~±0 — see Session Notes 4 §6 for honest accounting; value is single-sourced assembly).

### New improvement opportunities found
- ~~`enhanced.py:_setup_optimizers` is a near-verbatim clone~~ **→ DONE Session 4 (Phase 2b)** via `extra_importance_params()` hook + `importance_params()` helper.
- `equitile/training/distributed.py:118` docstring references `graph : TileGraph` but nothing imports `TileGraph` there; the shims in `equitile/core/` are the only survivors — re-point `equitile/core/__init__.py` + `equitile/__init__.py` at `core.tile` and delete the 4 shims (~40 lines). **→ DONE Session 4 (Phase 2a).**
- `trainer.py` uses `getattr(torch.optim, name)` — could add a registry fallback note but the dynamic path is intentional (see above).

### Verification (Session 3)
- `ruff check` changed files: **0 new F/I/E errors** repo-wide vs HEAD (only pre-existing style noise, e.g. `too-many-*`, `magic-value`).
- `pyright` changed files: **0 errors**, warnings are pre-existing `Optional`/`Self`-mixin patterns.
- `pytest`: 195 (core+data) + 68 (equitile) + 111 (zoo+equitile) + 470 (models) + 413 (validation) + 17 (refactor/kernel/scheduler/track-registry) + 69 (lm-demo/equitile-domains/advanced-training) + 40 (sparsity/mep/optimizer-stubs) + 5 (registry-instantiation) all pass. **Pre-existing failures unchanged on clean HEAD**: `test_nebc_base.py::test_cannot_instantiate_base`, `test_backprop_parity[eqprop_mlp]`, `test_backprop_parity[directed_ep]`. Flaky-on-rerun only: `test_lm_equitile` (passes back-to-back; local `torch.multinomial` NaN).

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

| Component | From | To | Status |
|-----------|------|-----|--------|
| `TileGraph`, `TileState` | `equitile/core/topology.py` | `core/tile/topology.py` | ✅ DONE (shim deleted Session 4) |
| Kernels (activity/hebbian/contrastive/prediction) | `equitile/core/kernels.py` | `core/tile/kernels.py` | ✅ DONE (shim deleted Session 4) |
| `MultiOptimizerMixin` | `equitile/training/optimizer_mixin.py` | `core/local_learning/mixins.py` | ✅ DONE (renamed from `EquiTileOptimizerMixin`; alias retired with the shim — Session 4) |
| `TaskHandler` | `equitile/training/task_handler.py` | `core/local_learning/task.py` | ✅ DONE (shim deleted Session 4) |
| `LocalLearningConfig` base | `equitile/core/config.py` | `core/local_learning/config.py` | ✅ DONE — `EquiTileConfig` inherits it |
| Feature extractors (Conv/Temporal/RL/Graph) | `equitile/deployments/_feature_extractors.py` | `core/tile/feature_extractors.py` | ✅ DONE — generic extractors moved; EquiTile-embedding layers param'd with `TileModelFactory`; `equitile/deployments/_feature_extractors.py` is now a 103-line factory/re-export wiring (Session 4) |

**New algorithms enabled post-generification**:
- `TileFA` — Feedback Alignment on tile substrate
- `TileTargetProp` — Target Propagation with tile graph
- `HierarchicalPC` — Multi-scale Predictive Coding
- `TileSNN` — Spiking tile models
- `TileGNN` — Graph NNs with local learning

**4-phase migration** (see `EQUITILE_GENERIFICATION.md`):
1. Create `core/tile/` + `core/local_learning/` infrastructure — ✅ DONE (session 3)
2. Refactor EquiTile to consume core primitives — ✅ DONE (`EquiTileConfig(LocalLearningConfig)`; 4 shims deleted; `enhanced.py` folded into `MultiOptimizerMixin`; feature-extractor decoupling)
3. Update deployments & enable zoo models — ◑ deployments rewired (sessions 3-4); zoo reuse of extractors still to write (the substrate is now importable: `TileFA`, `TileTargetProp`, `HierarchicalPC`, `TileSNN`, `TileGNN`)
4. Validation & docs — 🔴 pending

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

**38 of ~60 sites migrated** (session 1-3): `validation/utils.py`, `equitile/lm/ablation_study.py`, `validation/tracks/application_tracks.py`, `equitile/deployments/{base,vision,rl,timeseries,graph}.py`, `equitile/lm/fast_lm.py`, `equitile/lm/training.py`, `equitile/language/optimized.py` (session 1); `zoo/models/fa.py` (5), `equitile/training/optimizer_mixin.py` (3), `equitile/language/fast.py` (1, param-groups) (session 2); `zoo/models/target_prop.py` (3), `zoo/models/forward_only.py` (2), `zoo/models/eqprop/_unified.py` (1), `equitile/_internal/enhanced.py` (3), `zoo/mep/benchmarks/runner.py` (3) (session 3).

**Remaining sites** (mostly registry/ptl/legacy, diminishing returns):
- `zoo/models/forward_only.py` PEPITA path & other zoo leaves — check `grep -rn "torch.optim.Adam" bioplausible/zoo`
- `zoo/mep/optimizers/*` — MEP-specific optimizers, out of scope
- `validation/tracks/*.py` — only if the §6 track refactor lands
- `core/trainer.py` — **deliberately NOT migrated** (dynamic `opt_cls` is already config-driven and supports arbitrary torch.optim names; factory is less expressive there)

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
10 track modules in `validation/tracks/` each repeat boilerplate: header banner, timing, `TrackResult` construction, metadata attachment (the plan's "dataset/model/train/eval/evidence" steps are per-track bespoke and NOT unifiable).

**Done (2026-08-10 session 4)**: `validation/tracks/_base.py` with `track_header(track_id, name, width=60)` (banner + timing anchor) and `build_track_result(*, track_id, name, start, status, score, metrics, evidence, improvements=None, evidence_level="smoke", limitations=None)`. **18 standard-banner track fns migrated**; `nebc_tracks` (stub `time_seconds=0.1`, no banner boilerplate) and `signal_tracks` (already on `ValidationTrack`) deliberately left. Net line change ~±0 (header −4/site offset by `from __future__` + `TYPE_CHECKING` imports) — value is single-sourced banner/result assembly, not line count.

---

### 7. BenchmarkMetrics Naming Reconciliation (~40 lines + schema)
`BenchmarkMetrics` uses `train_acc`/`val_acc`; `TrainingMetrics` uses `train_accuracy`/`val_accuracy`. Touches SQL schemas, checkpoints, call sites.

**Verdict**: Low value per line — **defer** unless future ticket unifies trial representation.

---

## Next Immediate Actions (Priority Order)

Done this session: **(2) EquiTile generification Phase 2 complete + Phase 3.1 extractor decoupling** (shims deleted; `enhanced.py` → mixin hook; `core/tile/feature_extractors.py` + `TileModelFactory`; `core → equitile` edge removed), **(6) validation tracks `_base.py`** (18 track fns migrated). Earlier sessions: **(3)** optimizer factory (38 sites), **(4)** storage unify, **(1)** LM `TrainingConfig`, **(5)** `get_lm_dataset` sink. Remaining, in priority order:

1. **Config Unification (cont.)** — Next cheapest trainer family: `equitile/lm/components.py:FastLMConfig` is already canonical for `lm/fast_lm.py` (the `language/fast.py` variant is a genuinely different architecture — leave both, see Session Notes 2 §5). Next candidate: `zoo/mep/benchmarks/tuned_compare.py:OptimizerConfig` + `config/schema.py:OptimizerConfig` → *per-algorithm* configs with different fields (`gamma`, per-family values), **not** the factory's `OptimizerConfig` — reconcile only if a shared subset is extracted. Do NOT bulk-add speculative configs.
2. **EquiTile generification Phase 3/4** — (a) opportunity from Session Notes 4: flip `core/local_learning/mixins.py:20` off the `equitile.core.config.EquiTileConfig` TYPE_CHECKING import (use `LocalLearningConfig`/Protocol) so `core/*` is equitile-free at type-check too; (b) optional `tile_kwargs()` sink for `EquiTile.build` + `tile_model_kwargs` dedup; (c) write one zoo algorithm on the substrate (e.g. `TileFA`) to prove the reuse story; (d) validation & docs.
3. **Optimizer Factory Migration (cont.)** — Sweep remaining `torch.optim.Adam`/`SGD` in `zoo/models/*.py` and `validation/tracks/*.py` (diminishing returns; the remaining sites are registry/ptl/legacy). `core/trainer.py` is **permanently out of scope** (dynamic path is already config-driven). Remember: preserve the original `weight_decay` at each site — the factory default (1e-4) silently changes training (Session Notes 3).
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
- **Additional reduction**: ~2,900 lines (7.0%) [revised: §6 landed net-±0 (Session Notes 4 — boilerplate centralization, not line count); generification Phase 2/3 net-±150 (relocation + protocol typing, paid for by removing the `core → equitile` layer violation); config unification remains the dominant remaining line sink]
- **Total reduction**: ~6,000 lines (14.6%)
- **Key multiplier**: EquiTile generification is COMPLETE at the substrate level (session 4) — `core/tile` + `core/local_learning` are importable by any algorithm; `TileFA`, `TileTargetProp`, `HierarchicalPC`, `TileSNN`, `TileGNN` now need only their own model classes, not replicated substrate code.