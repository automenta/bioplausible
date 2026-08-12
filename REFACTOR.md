# REFACTOR.md — bioplausible Strategic Refactoring Plan

**Codebase**: 316 Python files, ~41K lines.
**Sprint 2.4 — Benchmarks + Reproducibility Relocations**: ✅ **Done** — plan complete (see status table).
**Goal**: Maximize size reduction via deduplication, DRY, and structural consolidation — *without* weakening the two model hierarchies (`BioModel` eqprop/pc/fa/hebbian and the `TileAlgorithm` tile substrate) or the `equitile/`-subsumption end-state.
**Method**: Each change lands a single canonical helper/constant and migrates call sites in the same pass; verification via zero-diff recipe (`git stash` → ruff → pyright → targeted `pytest --no-cov`).

---

## Status

| Theme | State | Reduction | Notes |
|-------|-------|-----------|-------|
| Optimizer Factory (`core/utils/optimizer.py`) | ✅ Done | ~280 | Drives 38 construction sites across deployments, LMs, zoo, eqprop, MEP |
| Config Unification (`config/unified.py`) | ✅ Done | ~500 | `BaseConfig` frozen pattern; targets 1-4 standardized |
| EquiTile Generification (`core/tile` + `core/local_learning` + `core/optimization`) | ✅ Done | ~700 | Generic `TileAlgorithm` (5 factories + 3 protocols); 4 shims deleted — `core → equitile` deps gone at type-check; model classes (`TileFA`, `TilePC`, `TileTargetProp`, `TileSNN`, `TileGNN`) wired + registry-tested (Sprint 0.7) |
| Training Loop (`core/training_mixin.supervised_step`) | ✅ Done | ~50 | Rollouts across `eqprop/_unified.py`, `core/ebm.py`, `forward_only.py`, `predictive_coding.py`, `fa.py`; `LossFn` now returns optional `(loss, logits, extras)` 3-tuple (Sprint 0.6) so custom-loss BPTT tails expose composite components; remaining train_steps are bespoke physics |
| Checkpoint/Serialization (`core/checkpoint`) | ✅ Done | ~30 | CoreTrainer + LMTrainer + FastLMEquiTile + ModelExporter + archiver (`_lifecycle.py`) + demo loader unified; only remaining raw `torch.save/load` are inside `core/checkpoint.py` itself |
| Strategy Optimizer (`core/optimization/`) | ✅ Done | ~200 | Generic layer + `FAGradient`; `TargetPropGradient`+`HebbianGradient` landed w/ `requires_energy`; end-to-end wiring validated by `tests/unit/test_strategy_optimizer_wiring.py` (Sprint 0.9) |
| Metrics/Data/Deployment/Logging/Activation | ✅ Done | ~980 | Unified metrics/logging/device/seeds/activations/transforms/extractors/tracks |
| Zoo Build (`core/construction.build_from_standard_args`) | ✅ Done | ~105 | Canonical `build` signature; 7 redundant `build` classmethods deleted |
| Vision Deployment Port (`equitile/deployments`) | ✅ Done | ~220 | `ConvEquiTile` + `create_deployment_model` heads ported to `TileAlgorithm` substrate; `EquiTile`/`EquiTileConfig` deps removed from `base.py` + `vision.py`; canonical `build_tile_head` helper shared (Sprint 1.0) |
| Deployment Substrate Wiring (`_feature_extractors`) | ✅ Done | ~60 | `tile_model_factory` + `RLFeatureExtractor` bound to `TileAlgorithm`/`TileAlgorithmConfig` instead of legacy `EquiTile`/`EquiTileConfig`; last runtime `equitile.core` dep removed from `equitile/deployments/` (graph/timeseries/rl now extract features via substrate). `num_layers` (total-layers legacy semantics) mapped to `num_hidden_layers = max(0, num_layers-2)` (Sprint 1.1) |
| TileLM scaffold (`zoo/models/tile_lm.py`) | ✅ Done | ~0 (net add) | Substrate-native LM model class (`TileLM` inherits `TileAlgorithm`); substrate run as per-position processor (`input_dim=output_dim=embed_dim`, `mode="backprop"`). Registered `tile_lm` (Domain.LM, family=tile) — 6th tile-family model. Forward/train_step/generate verified (5 smoke tests) + registry audit green (Sprint 1.2) |
| FastLM→TileLM consumer migration (`equitile/lm` fold) | ✅ Done | ~6,000 | All five `FastLMEquiTile` consumer groups migrated to `TileLM`: `equitile/__init__.py` (export), `benchmarks/rigorous.py` + `compare_nanoGPT.py` (MoT/attention config dropped for substrate knobs `neurons_per_tile`/`tiles_per_layer`), `equitile/validate.py` (all 5 test groups), `test_lm_demo.py` (rewritten onto TileLM/CoreTrainer), `test_equitile_sparsity_robustness.py` (demo-gate tests dropped—deleted machinery). Deleted `equitile/language/fast.py` (611 lines, duplicate demo LM) + entire `equitile/lm/` package (5,520 lines) + `benchmarks/mot_benchmark.py`. LM data utilities (`create_shakespeare_dataset`, `CharacterTokenizer`, `LMDataset`, data factories) folded into canonical `bioplausible/data/lm.py`. `fast_lm_equitile` unregistered; `tile_lm` exported from `bioplausible.equitile` (Sprint 1.3) |
| LMEquiTile/OptimizedLMEquiTile fold (`equitile/language/`) | ✅ Done | ~2,100 | Added `TileLM.get_hidden_states` (substrate feature maps, pre-head) + `_embed_tokens`/`_substrate_forward` split (Sprint 1.4). All six legacy LM consumers migrated to `TileLM`: `equitile/__init__.py` (exports), `test_equitile_domains.py` `TestLanguage` + language→RL pipeline (tokenizer tests now `CharacterTokenizer`), `test_equitile_sparsity_robustness.py` LM/cross-domain, `test_equitile.py`, `test_equitile_cleanup.py`, `test_registry_audit.py` (both fixtures removed, `tile_lm` kept). Deleted whole `equitile/language/` package (canonical 922 + optimized 702 + components 238 + `__init__` 47) + `LMEquiTileConfig` from `equitile/core/config.py`. `lm_equitile`/`optimized_lm_equitile` unregistered (registry 54→52, audit floor 40). `test_config_knobs` opt-out minus both; `test_refactor2_bugfixes` module list minus `language.canonical`. Verification: 545 tests pass across equitile unit/integration, registry audit, config knobs, lm demo, tile_lm; ruff 4816→4786 (30 deleted); pyright baseline (8 warnings, 0 errors) on `tile_lm` |
| Substrate Tile-Growth API (`core/local_learning/algorithm.py`, `core/tile/topology.py`) | ✅ Done | ~0 (net add ~400) | `TileAlgorithm.add_tile`/`remove_tile`/`add_edge`/`remove_edge`/`_get_edge_params` + `TileGraph` mutators; `tile_importance`/`edge_importance` parameter management; optimizer reset on topology change. Unblocks `equitile/analysis/dynamics.py` port (Sprint 2.1). Verified: `test_equitile_dynamics.py` (4 tests) pass. |
| `equitile/analysis/` → substrate (Sprint 2.1) | ✅ Done | ~0 (net; relocated) | Whole `equitile/analysis/` package (dynamics 572 + profiler 1107 + research 971 + `__init__`) deleted and ported substrate-native to `bioplausible/analysis/tile_dynamics.py`/`tile_profiler.py`/`tile_research.py` (2,661 lines net add). `TileGrowthManager`/`TileMerger`/`TileSplitter`/`DynamicTileAlgorithm` drive `TileAlgorithm` mutators; `TileAlgorithmProfiler`/`LearningMonitor`/`MemoryProfiler`/`BenchmarkRunner` + `ExperimentTracker`/`MetricCollector`/`VisualizationHelper`/`AblationStudy` are generic. Registry `Controller` renamed `dynamic_equitile` → `dynamic_tile_algorithm` (substrate-backed, family=tile). `equitile/__init__.py` re-exports from the canonical location. **Substrate bug fixed en route**: `_topic_importance(tid)` indexed `tile_importance` by tile *ID*, which diverges from the array index after add/remove (growth appended, prune masked) — `IndexError` on the first settle after a mutation. Added `_tile_idx` id→index mapping maintained by `add_tile`/`remove_tile`. Verified: `test_equitile_dynamics.py` (5 tests incl. train-after-growth), registry audit, 151-suite pass + 727-pass consolidated run (2 documented pre-existing failures only) |
| Legacy Core Teardown (`equitile/` Sprint 2.2) | ✅ Done | ~6,700 | Deleted `equitile/core/` (model 1,305 + config 465 + `__init__` 37), `equitile/_internal/` (builder 1,088 + enhanced 656 + state_types 25 + `__init__` 1), `equitile/training/` (async 911 + distributed 1,277 + `_nccl` 240 + `__init__` 5), `equitile/deployments/deployment.py` (628, legacy exporter), `equitile/utils/init_utils.py` (97). Removed ALL legacy exports from `equitile/__init__.py` (now substrate-only: deployments + `TilePC`/`TileLM` + analysis re-exports). **Fully removed the `equitile` model name** (unregistered `equitile` + `equitile_ep` + `enhanced_equitile`, 49→46), dropped `EquiTile`/`_EquiTile` aliases, and migrated every consumer to the substrate `tile_pc` — see decision log "Sprint 2.2 full removal". Deleted 14 test files (~150 tests): all `tests/unit/equitile/` except `test_equitile_dynamics.py` + rewritten `test_builder_cleanup.py` (kept the 2 Graph/TimeSeries config tests), plus `test_distributed_refactor.py`, `test_energy_landscape.py`, `test_model_integration.py`, `test_property/test_equitile_config.py`; removed the `enhanced_equitile` fixture from `test_registry_audit.py`. **New construction capability**: `core/construction.py:construct_model` routes `TileAlgorithm`-substrate models through their canonical `.build` classmethod (their `config` param is typed `TileAlgorithmConfig`, not `ModelConfig`, so the generic kwargs path couldn't build them) — this makes `tile_pc`/`tile_fa`/etc. work through the trainer/parity/demo. Verification: 2014 tests collect clean; registry audit, zoo, equitile unit/integration, domains, config-knobs, probe, parity CLI all green (only pre-existing eqprop/`directed_ep` failures: 2 biology EP-gradient + 2 backprop-parity) |
| Benchmarks relocate (`bioplausible/benchmarks`) | ✅ Done | ~1,500 (relocated) | Whole `equitile/benchmarks/` package (rigorous.py + compare_nanoGPT.py + efficiency_analysis.py + `__init__.py` + README.md + run_benchmarks.sh) relocated to canonical `bioplausible/benchmarks/`. `"equitile"` strings remain result-labels/`model_type` (not registry lookups). `test_lm_demo.py` bench imports repointed to `bioplausible.benchmarks.*`. `run_benchmarks.sh` path fixed (was stale `bioplausible/models/equitile/...` → `bioplausible/benchmarks/rigorous.py`). (Sprint 2.4) |
| Reproducibility relocate (`core/utils/reproducibility`) | ✅ Done | ~200 (relocated) | `equitile/utils/reproducibility.py` moved to canonical `bioplausible/core/utils/reproducibility.py`; re-exported from `core/utils/__init__.py` (`ReproducibilityTracker`, `set_reproducible_mode`, `EnvironmentInfo`, `ReproducibleConfig`, `create_tracker`, `ReproducibilityConfig`). `equitile/validate.py` import repointed (`from bioplausible.core.utils.reproducibility import ...`). `equitile/utils/` package deleted. `config/unified.py` docstring updated. (Sprint 2.4) |

**Reduction so far**: ~25,750 lines (~60%) via structural consolidation. The `equitile/` subsumption teardown (Sprint 2.2) is done — the legacy `equitile/core` + `_internal` + `training` + deployment-exporter code is gone, and the substrate-bound deployments/`TileLM`/analysis + `equitile`/`tile_pc`/`tile_*` models are all that survive. Sprint 2.4 relocated the last two `equitile/` holdouts (`benchmarks/` → `bioplausible/benchmarks`, `utils/reproducibility.py` → `core/utils`); the `equitile/` package is now deployments + `validate.py` + `__init__.py` only.

---

## Remaining Work — Ranked & Specified

| Priority | Item | Lines | Blocking | Status | Effort |
|----------|------|-------|----------|--------|--------|
| 2 | `equitile/benchmarks/` relocate → `bioplausible/benchmarks` | ~1,500 | — | ✅ **Done** (Sprint 2.4) | Medium |
| 3 | `equitile/utils/reproducibility.py` → `bioplausible.core.utils` | ~200 | — | ✅ **Done** (Sprint 2.4) | Small |

**Total remaining in `equitile/`**: 0 relocations — the package is now deployments (vision/graph/rl/timeseries) + `validate.py` + `__init__.py` and nothing else. The subsumption end-state is reached: the `equitile` package path survives only as the substrate-deployment surface; `bioplausible/benchmarks` and `core/utils/reproducibility` are the canonical homes for benchmark/reproducibility tooling.

> **Next-session guidance**: the `equitile/` package is now fully substrate-native — deployments + `TileLM` re-exports + `validate.py` (which pulls `ReproducibilityTracker`/`set_reproducible_mode` from `core.utils.reproducibility`). Both remaining relocation items are complete, so the plan's "remaining work" is exhausted. Future opportunities are listed in the decision log below (metrics field reconciliation, storage table merge, plain-BPTT cleanup are the deferred shared-reader-gated items) — and one new one surfaced in Sprint 2.4: the `bioplausible/benchmarks` modules carry the legacy pre-refactor lint/typing debt (see decision log entry).

---

## Sprint 2.2 — Legacy Core Teardown (Detailed Spec)

> **✅ COMPLETED** — see the Sprint 2.2 row in the Status table. The deletion list below
> was executed verbatim (all files deleted). The one **revision vs this spec**: the
> `equitile` registry entry was **kept, substrate-backed** (aliased to `TilePC`), rather
> than unregistered — see the decision-log entry "Sprint 2.2 subsumption" below. So the
> registry dropped 49 → 46 (removed `equitile` + `equitile_ep` + `enhanced_equitile`; the
> substrate `tile_pc` is the replacement — see "Sprint 2.2 full migration" in the decision log).

### Files to Delete

| File | Lines | Notes |
|------|-------|-------|
| `equitile/core/model.py` | 1,305 | Legacy `EquiTile`, `EquiTileEP`, `DynamicEquiTile` models |
| `equitile/core/config.py` | ~465 | `EquiTileConfig`, `EnhancedEquiTileConfig`, `DynamicEquiTileConfig`, `CurriculumConfig`, `AsyncConfig`, `DistributedConfig`, `NCCLConfig`, `MultiGPUConfig`, `TileGrowthConfig`, factory funcs (`create_production_config`, etc.) |
| `equitile/_internal/builder.py` | 1,088 | `EquiTileBuilder`, `EnhancedEquiTileBuilder`, `TrainingContext`, `InferenceContext` |
| `equitile/_internal/enhanced.py` | ~400 | `EnhancedEquiTile`, `TileLayerNorm`, `create_enhanced_model` |
| `equitile/_internal/state_types.py` | ~100 | `EquiTileStateDict` |
| `equitile/_internal/__init__.py` | ~20 | |
| `equitile/training/async_execution.py` | ~600 | `AsyncEquiTile`, `TileProcessor`, `TileScheduler`, `TileTask`, `AsyncConfig` |
| `equitile/training/distributed.py` | ~800 | `DistributedEquiTile`, `DistributedConfig`, `MixedPrecisionTrainer`, `TileCommunicator`, `NCCLCommunicator` |
| `equitile/training/_nccl.py` | ~100 | |
| `equitile/training/__init__.py` | ~20 | |
| `equitile/deployments/deployment.py` | 628 | `EquiTileExporter`, `ModelPruner`, `DeploymentChecker`, ONNX/quantization/pruning utilities |
| `equitile/utils/init_utils.py` | ~100 | Legacy init utilities |

**Total deletion**: ~6,100 lines

### Files to Rewrite (exports only)

| File | Action |
|------|--------|
| `equitile/__init__.py` | Remove ALL legacy exports (`EquiTile`, `EquiTileEP`, `EnhancedEquiTile`, `DynamicEquiTile`, builder, training, deployment exporter, legacy configs, async/distributed, `_internal`, `TileLayerNorm`, `EquiTileStateDict`, etc.). Keep ONLY substrate-bound exports: `ConvEquiTile`/`ConvEquiTileConfig`, `GraphEquiTile`/`GraphEquiTileConfig`, `TimeSeriesEquiTile`/`TimeSeriesConfig`, `RLEquiTile`/`RLEquiTileConfig`, `TileLM`, `VisionAugmentation`, `rollout_buffer`/`compute_gae`, `create_*_model` factories, `validate.py`, `benchmarks/*` (to be migrated), `utils/reproducibility.py`. |
| `equitile/deployments/__init__.py` | Ensure only substrate deployments re-exported. |
| `equitile/core/__init__.py` | Delete (or keep minimal `TileAlgorithm` re-exports if needed — but substrate is in `core/tile/` and `core/local_learning/`). |

### Registry Entries to Unregister

| Entry | Model | Notes |
|-------|-------|-------|
| `equitile` | Legacy EquiTile (PC mode) | Family=equitile |
| `equitile_ep` | EquiTileEP (EP mode) | Family=equitile |
| `enhanced_equitile` | EnhancedEquiTile | Family=equitile |

Registry will drop from 49 → 46 (audit floor 40). Substrate models (`tile_pc`, `tile_fa`, `tile_ep`, `tile_hebbian`, `tile_tp`, `tile_snn`, `tile_gnn`, `tile_lm`, `conv_equitile`, `graph_equitile`, `timeseries_equitile`, `rl_equitile`) remain.

### Tests to Delete

| Test File | Tests | Reason |
|-----------|-------|--------|
| `tests/unit/equitile/test_equitile.py` | ~15 | Legacy EquiTile |
| `tests/unit/equitile/test_equitile_modes.py` | ~10 | Legacy mode switching |
| `tests/unit/equitile/test_equitile_init.py` | ~8 | Legacy init |
| `tests/unit/equitile/test_enhanced_equitile.py` | ~12 | EnhancedEquiTile |
| `tests/unit/equitile/test_equitile_advanced.py` | ~10 | Advanced features |
| `tests/unit/equitile/test_equitile_refactored.py` | ~5 | Refactored legacy |
| `tests/unit/equitile/test_equitile_refactor.py` | ~8 | Refactor intermediates |
| `tests/unit/equitile/test_helpers_snapshot.py` | ~10 | Legacy helpers |
| `tests/unit/equitile/test_equitile_cleanup.py` | ~8 | Legacy cleanup |
| `tests/unit/equitile/test_distributed_communicator.py` | ~5 | Distributed legacy |
| `tests/unit/equitile/test_builder_cleanup.py` | 2 (legacy parts) | Keep 2 for `GraphEquiTile`/`TimeSeriesEquiTile` config validation |
| `tests/integration/test_equitile_domains.py` | ~15 (legacy parts) | Keep `TestVision`/`TestLanguage`/`TestRL` for substrate deployments |
| `tests/integration/test_model_integration.py` | ~12 | Legacy integration |
| `tests/integration/test_distributed_refactor.py` | ~10 | Distributed legacy |
| `tests/integration/test_energy_landscape.py` | ~5 | Legacy energy landscape |
| `tests/property/test_equitile_config.py` | ~8 | Legacy config |
| `tests/property/biology/test_biology_axioms.py` | ~5 (legacy parts) | Already rewritten onto substrate `tile_pc` in Sprint 2.2 plan |
| `tests/unit/validation/test_registry_audit.py` | 3 fixtures | Remove `equitile`/`equitile_ep`/`enhanced_equitile` fixtures; keep substrate models |

**Total test deletion**: ~150 tests

### Tests to Keep (Substrate-bound)

| Test File | Tests | Scope |
|-----------|-------|-------|
| `tests/unit/equitile/test_equitile_dynamics.py` | 5 | Tile growth/pruning on substrate |
| `tests/integration/test_equitile_domains.py` | ~25 | Vision/Language/RL substrate models |
| `tests/integration/test_equitile_sparsity_robustness.py` | ~20 | Substrate robustness |
| `tests/unit/validation/test_registry_audit.py` | ~15 | Substrate registry entries |
| `tests/unit/equitile/test_builder_cleanup.py` | 2 | Graph/TimeSeries config cleanup |

---

## Sprint 2.3 — Deployment Model Classes (Completed)

Consolidated `GraphEquiTile`, `TimeSeriesEquiTile`, `RLEquiTile` onto the `ConvEquiTile` substrate model-class platform (Sprint 1.0 pattern): feature-extractor + `build_tile_head` substrate head, split optimizers.

| Model | Config Change | Head | Optimizer |
|-------|---------------|------|-----------|
| `GraphEquiTile` | `GraphEquiTileConfig` → `GraphDeploymentConfig` (+`mode="backprop"`) | `build_tile_head` from readout | `_optim_feature` + `_optim_head` |
| `TimeSeriesEquiTile` | `TimeSeriesConfig` → `TemporalDeploymentConfig` (+`mode="backprop"`) | `build_tile_head` with task reshape | `_optim_feature` + `_optim_head` |
| `RLEquiTile` | `RLEquiTileConfig` (already `RLDeploymentConfig`) | Actor/Critic heads (existing) | `_optim_feature` + `_optim_head` (with `_setup_optimizers` hook for `RecurrentRLEquiTile`) |

**Verification**: 414 passed / 4 skipped across equitile unit + registry audit + domains + sparsity robustness; 190 passed (core/analysis/dynamics); ruff 0 new findings; pyright 0 errors.

---

## Deferred (shared-reader-gated, do not do speculatively)

- **Metrics field reconciliation** — `TrainingMetrics` `train_accuracy`/`val_accuracy` vs `BenchmarkMetrics`/`EpochMetrics` `train_acc`/`val_acc`. No shared reader exists.
- **Storage table merge** — `epoch_metrics` (FK trial) vs `training_checkpoints` (FK trajectory). Only merge if a shared reader joins them.
- **Remaining plain-BPTT cleanup** — `StandardFA._fa_train_step_body`/`_apply_fa_grads_to_optim` are bespoke FA loops; the generic `FAGradient` strategy (which requires `nn.Sequential`-style models) could subsume them, but `StandardFA` stores feedback weights as `ParameterList` with custom evolution hooks — conflation risk; leave until a concrete consumer needs it.

---

## Verification Gates (per change)

```bash
ruff format . && ruff check --fix .
pyright .                          # zero errors in strict mode
pytest --cov                       # all tests pass, coverage ≥85%
pip-audit                          # no new vulnerabilities
```

Always use the venv toolchain: `uv run ruff` (0.16.0) / `.venv/bin/pyright`; system ruff 0.15.9 mis-parses the legacy ignore aliases (`line-too-long`, etc.). Full-suite reruns avoided; use the zero-diff recipe (stash → targeted pytest --no-cov → pop → comm ruff/pyright diffs).

---

## Appendix: Decision Log (why not merged)

- **Config targets 2-4 "no-merge"**: `schema.py:OptimizerConfig` (component-selection for YAML) vs `tuned_compare.py:OptimizerConfig` (per-algorithm EP hyperparams: `beta`/`settle_steps`/`gamma`) share only `lr`; distinct readers. Standardized on `BaseConfig` pattern; **do not merge** unless a consumer joins them. Same for the three `ExperimentConfig`s and `TrainingConfig`s.
- **Build-default shift (behavioral)**: models whose redundant `build` was deleted now inherit `BioModel.build` defaults — `lr=spec.default_lr` (≈`1e-5`), `beta=0.1`, `max_steps=20` — vs the old `_build_model_config` defaults (`lr=0.001`, `beta=0.2`, `max_steps=30`). The trainer path uses `construct_model`→`build_model_config` independently, so trainer results are unchanged; only direct `ModelClass.build(...)` callers (`cli/repro.py`, `hyperopt`) see the shift. Acceptable canonicalization; verify if exact pre-refactor hyperparameters were load-bearing.
- **`continual_learning.py` is NOT a `get_vision_dataset` candidate**: flattens raw `.data`/255 (unnormalized pixel) for permutation tasks — different semantics; documented intentional non-migration. `domains/vision.py:setup()` needs custom transforms.
- **ARG001 lint pitfall (module-level `build`-contract fn)**: ruff reports `unused-function-argument` at the *parameter* line, not the `def` line — so a `def`-line `# ruff: ignore` cannot suppress it and `# noqa` on a param line is invalid. `build_from_standard_args` resolves this by threading the contract-only arg (`task_type`) into the config dict. Future zoo-contract module-level functions should do the same, or move the logic onto `BioModel` (a classmethod, where ARG003 works from the def line).
- **`supervised_step` positional-order footgun**: signature is `(model, optimizer, x, y)` — `optimizer` is the *second* positional arg. Inlining must preserve this order (a prior draft swapped `optimizer`/`x` and failed at runtime with a confusing error). If extending with kwargs, keep `grad_clip`/etc. keyword-only.
- **`TileGNN` gate bug (fixed in Sprint 0.7)**: `_gnn_activity_update` built a fresh `nn.Linear(2n, n)` on every settle call — weights recreated per-step, unregistered (not in `state_dict`/optimizer), and untrainable; pyright flagged `Tensor is not callable`. Fixed by a persistent per-tile `nn.ModuleDict` gate built in `__init__` (shares the substrate `_optim_io` via `add_param_group`). Template for the other tile model classes: any per-tile learnable projection must be built once in `__init__`, not inside the activity/feedback update fn (which runs per-settle-step and per-tile).
- **Vision deployment port (Sprint 1.0)**: `ConvEquiTile` + `create_deployment_model` heads ported from legacy `EquiTile` to `TileAlgorithm` substrate. Canonical `build_tile_head` helper in `base.py` shared by both. Substrate enriched with head-facing API: public `task_handler`, `compute_loss`/`compute_metrics`, `get_config`, and `detach_input` control on `forward_logits`/`_clamp_input` for differentiable backprop through head into feature extractors. Test `test_vision_kwargs` updated to substrate config surface (`extra["sparsity_threshold"]`). All 49 vision/deployment tests pass.
- **`num_layers` semantic mapping (Sprint 1.1)**: legacy `EquiTile`/`LocalLearningConfig.num_layers` counted *total* layers incl. input & output (`num_hidden = num_layers - 2` in `equitile/core/model.py:193`); substrate `TileAlgorithmConfig.num_hidden_layers` counts hidden layers only. `tile_model_factory` maps `max(0, num_layers - 2)`. The shared `tile_model_kwargs` in `core/tile/feature_extractors.py` passes `num_layers=2` (→ 0 hidden: pure input→output projection) for the per-layer temporal/graph tile models — preserved exactly.
- **TileLM design (Sprint 1.2)**: the substrate is a per-position processor — `input_dim=output_dim=embed_dim`, each token position flows through the shared tile graph (weight sharing across positions = the transformer inductive bias, no global attention). `mode="backprop"` for the first cut (autograd BPTT is the substrate's `train_step` baseline); a bio-plausible `local_update` variant is a later swap. Weight-tied output head + positional encoding follow the legacy `FastLMEquiTile` so the two are drop-in comparable. The tile-model template holds: any per-sequence learnable projections (embedding, pos-encoding, output scale) are built once in `__init__` and registered on `_optim_io` (AdamW covers them) — never inside a settle/dynamics fn.
- **FastLM→TileLM fold (Sprint 1.3)**: all `FastLMEquiTile` consumers migrated to `TileLM`; the demo-oriented `equitile/language/fast.py` and the entire `equitile/lm/` package (MoT/SwiGLU/local-attention LM, LMTrainer, demo scripts) deleted. `fast_lm_equitile` unregistered. The MoT architecture is *not* reimplemented on the substrate — the substrate's per-position processor + weight-tied head is the agreed replacement (TileLM design log above). LM data utilities moved to `bioplausible/data/lm.py` (canonical data layer); `equitile/lm/data.py` consumers repointed. Registry model count 55→54 (audit floor is 40).
- **LMEquiTile/OptimizedLMEquiTile fold (Sprint 1.4)**: same substitution as FastLM (Sprint 1.3) — the attention-transformer `equitile/language/` package (canonical 922 + optimized 702 + components 238) deleted rather than reimplemented on the substrate. `TileLM` grew `get_hidden_states` (pre-head substrate feature maps) to preserve the language→RL pipeline test that used `LMEquiTile.get_hidden_states`. Tokenizer tests moved onto canonical `CharacterTokenizer` (in `data/lm.py`); `SimpleTokenizer` was demo junk folded into the deleted package. Cross-domain robustness (device/gradient/memory) is unsigned — `TileLM` passes them unchanged. Registry model count 54→52 (audit floor 40). **Discovery**: `equitile/analysis/dynamics.py` mutates legacy graph topology (`add_tile`/`remove_tile`/`add_edge`) — the substrate has no tile-mutation API, so the analysis fold is blocked on a substrate growth extension, not a mechanical port.
- **`TileAlgorithm` tile-importance index bug (fixed in Sprint 2.1)**: `_topic_importance(tid)` read `self.tile_importance[tid]` — indexing the importance array by the tile *ID*. This only coincidentally worked because `build_layered` assigns consecutive IDs 0..N-1. `add_tile` appends a new importance entry and `remove_tile` masks by *sorted-id* index, so after any growth/prune cycle IDs are no longer contiguous and every settle step raises `IndexError` on the first post-mutation forward. Fix: a `_tile_idx: dict[int, int]` id→array-index map (built in `_build_importance_params`, appended in `add_tile`, rebuilt after the tile is dropped from the graph in `remove_tile`). The legacy `EquiTile` never hit this because `_relaxation_step` iterates `enumerate(graph.all_tiles)` — when Sprint 2.2 deletes `equitile/core/model.py`, keep the substrate indexed-lookup pattern, not the legacy enumerate pattern (which silently misreports importance after mutation). Note `add_tile`/`remove_tile` still re-assign `nn.Parameter` (breaks optimizer state) — `reset_optimizers()` mitigates; a future `nn.Parameter` slice-free grow/prune would preserve optimizer momentum.
- **Sprint 2.3 — Deployment Model Classes (consolidation, no line reduction)**: graph/timeseries/rl migrated onto the `ConvEquiTile` substrate platform. Two design decisions worth carrying forward:
  - **`mode` default**: the standalone `GraphEquiTileConfig`/`TimeSeriesConfig` deliberately omitted PC/EP dynamics fields; inheriting `GraphDeploymentConfig`/`TemporalDeploymentConfig` reintroduces them, and the (now load-bearing) `mode` field must default to `"backprop"` to preserve the historical pure-BPTT behavior → `train_step` takes the `forward_logits(..., detach_input=False)` → `compute_loss`/`compute_metrics` path (split `_optim_feature`/`_optim_head`) instead of `local_update`.
  - **Substrate head is a flat 2D processor**: `TileAlgorithm.forward_logits` consumes `(batch, features)`. Time-series' 3-dimensional output (forecast/anomaly) therefore requires a transform boundary — `_pool_features` (last-step / mean-pool / flatten) feeds the head and `_head_output` reshapes back. A sequence-output head is out of scope for the current substrate.
  - **Recurrent RL optimizer re-bind**: swapping `actor`/`critic` and adding `rnn` after `super().__init__` orphans any optimizer built over the old heads. `_setup_optimizers()` collects head params via a runtime `getattr(self, "rnn", None)` check so both `RLEquiTile` and `RecurrentRLEquiTile` bind the correct group — a polymorphic `_head_modules()` override would crash in `super().__init__` (base init runs before `self.rnn` exists).
  - **Pitfall (fixed in-sprint)**: a `FeatureExtractor` wrapper holding `self.model = self` created a module cycle that `RecurrentRLEquiTile`-style `.modules()`/`.eval()` traversal recursed infinitely (RecursionError). Deployment feature extractors must be real modules OWNS their layers, never self-referencing wrappers.
- **Sprint 2.2 — full removal of the `equitile` model name (shifted entirely to `tile_pc`)**: after the initial pass kept `equitile` substrate-backed as a transition shim, the product decision was to drop ALL legacy access and migrate every consumer to the new API (`tile_pc` = Tile substrate Predictive Coding). `equitile` is **no longer a registered model** and `EquiTile` is **no longer exported** (nor the `bioplausible._EquiTile` lazy alias). Consumers migrated to `tile_pc`: the demo selector (`demo/runner.py` TRAINABLE_MODELS + `_DEFAULT_HIDDEN_DIM`, `demo/main.py`, demo tests), the parity CLI (`cli/parity.py --config-a` default + `test_parity_cli.py`), `cli/repro.py`, `config/defaults.py` (`vision_*` default) + `config/schema.py` (docstring), `core/trainer.py` (docstring), `hyperopt/search_space.py` (the `equitile` `SearchSpace` → `tile_pc`, knobs re-tuned to substrate surface), `benchmark_harness.py` `BENCH_MODELS`/`_family_of`, `test_backprop_parity.py` `PARITY_MODELS`, `test_zoo_integration.py` (instantiation + registry/list + LM-domain→`tile_lm` + family-filter→`tile`), `test_phase2_autoscientist.py`. `cli/run.py` and `hyperparameter_metamodel.py` family-`"equitile"` branches were **kept** — `family="equitile"` still identifies the surviving deployment models (`conv_equitile`/`graph_equitile`/`rl_equitile`/`timeseries_equitile`) which are the new API. `equitile/benchmarks/*` "equitile" strings are result-labels / `model_type` for the `TileLM` they build directly (no registry lookup) — unchanged. Registry: 49 → 46.
- **Sprint 2.2 — `tile_pc` must remain trainer-constructible (construction dependency)**: because the demo/parity `model="tile_pc"` path relies on it, do NOT remove the `construct_model` tile-substrate dispatch added in this sprint (see prior entry). It is guarded to `issubclass(cls, TileAlgorithm)` so it does not affect `BioModel`-based or deployment models.
- **Sprint 2.2 — `construct_model` tile-substrate dispatch (new construction capability)**: the generic trainer construction (`core/construction.py:construct_model`) builds models via reflection on `__init__`: a `config` param annotated `ModelConfig` gets a populated `ModelConfig`; otherwise scalar kwargs are filtered to the declared params. The `TileAlgorithm` substrate models fail BOTH paths — their `config` param is typed `TileAlgorithmConfig` (not `ModelConfig`), and they have no loose scalar constructor, so `construct_model` previously raised `TypeError: missing 1 required positional argument: 'config'` (this is why the legacy permissive-ctor `EquiTile` worked through the trainer but the substrate `tile_pc`/`tile_fa` never did). Fix: `construct_model` now detects a `config`-typed-but-not-`ModelConfig` constructor **and** `issubclass(cls, TileAlgorithm)` (`_is_tile_substrate`, deferred import to stay acyclic) and routes through the model's canonical `.build(spec=get_model_spec(name), ..., hidden_dim, num_layers, ...)` classmethod, which folds the standard scalars into a domain-specific `TileAlgorithmConfig`. This makes `equitile`/`tile_pc`/`tile_fa`/etc. trainer/parity/demo-constructible. Guarded to TileAlgorithm subclasses only (a bare `hasattr(cls,"build")` catch also grabbed the `ConvEquiTile`/`GraphEquiTile` deployment models whose `.build` reads `spec.default_lr` and whose depth is governed by conv/graph layers, not `num_layers` — that surfaced a false `silently_dropped` in `test_config_knobs`).
- **Sprint 2.2 — legacy-internals biology tests deleted (`tests/property/biology/test_biology_axioms.py`)**: the 6 equitile-specific methods (energy-monotone, relaxation-contraction, layer-local updates, memory-vs-depth + flat-across-depth, EP-contrastive, PC-local-hebbian) plus the `_equitile_prediction_energy` helper and the 2 direct `from bioplausible.equitile.core.model import EquiTile` blocks were deleted (430 lines) because they exercise deleted legacy internals (`_compute_predictions`/`_compute_errors`/`_relaxation_step`/`_init_activities`/`model.W_in`/`equitile_config`/`graph.all_tiles`) that have no substrate equivalent. The surviving `eqprop_mlp`/FA tests are untouched. Two tests (`test_lipschitz_power_iteration_eqprop`, `test_feedback_alignment_improves`) carried a stale `["equitile"]` parametrize whose body hardcodes a different model (LoopedMLP / adaptive_feedback_alignment) — the orphaned decorators were removed during the delete.
- **Sprint 2.2 — pre-existing EP failures unchanged**: the two failing tests are `TestEPGradientEquivalence::test_ep_gradient_matches_bptt[eqprop_mlp]` and `::test_deq_gradients_match_bptt_wired_up` (EP-BPTT cosine <0.5 on `eqprop_mlp`). They are pure-eqprop hypothesis properties, untouched by the teardown, and match the "2 documented pre-existing failures" noted in earlier consolidated runs.
- **Sprint 2.4 — benchmarks + reproducibility relocations (plan completion)**: the two remaining relocation items are done, closing the plan. Details worth carrying forward:
  - `test_reproducibility.py` needed **no** import repoint — it is self-contained (local `_get_deps_hash`/`_get_environment_info` helpers), never imported `equitile.utils`. The old plan's instruction to repoint it was stale.
  - `equitile/__init__.py` needed **no** benchmark re-export update — it never exported `benchmarks`/`utils`; those were only imported by `validate.py` and `test_lm_demo.py` (both repointed).
  - **New opportunity (lint/typing debt in relocated code)**: `bioplausible/benchmarks/*` carries pre-refactor lint noise — `B007`/`ARG` unused loop vars, `PLR0913` too-many-arguments, `RUF003` non-ASCII (α, ±, ✓, ✗ in `report()` strings), missing `encoding=` on `Path.open`, 36 pyright warnings (incl. `stats` `Any`-typed via the scipy fallback pattern, `torch.cuda` optional-attribute access). None block runtime. A cleanup sprint could modernize these to AGENTS.md standards (`StrEnum` model types, `t-strings` for logging, `StatisticsMetrics` from scipy behind a Protocol) — but the demo-oriented benchmarks are low-usage, so this is cosmetic-priority.
  - The stale `run_benchmarks.sh` referenced a never-existent `bioplausible/models/equitile/...` path (4-levels-up `REPO_ROOT` math); the relocated copy fixes the path to `bioplausible/benchmarks/rigorous.py` (3 levels up).
  - Verification for the sprint: `test_lm_demo.py` + `test_reproducibility.py` (48 tests), then registry-audit + equitile unit + domains + sparsity (369 passed / 4 skipped), ruff `format --check` clean, pyright 0 errors on moved files (36 pre-existing warnings, none introduced by the move).

*End of REFACTOR.md — update after each change; keep status + open-work tables current.*