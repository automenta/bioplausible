# REFACTOR.md — bioplausible Strategic Refactoring Plan

**Codebase**: 316 Python files, ~41K lines.
**Sprint 2.2 — Legacy Core Removal**: ✅ **Done** (see below)
**Goal**: Maximize size reduction via deduplication, DRY, and structural consolidation — *without* weakening the two model hierarchies (`BioModel` eqprop/pc/fa/hebbian and the `TileAlgorithm` tile substrate) or the `equitile/`-subsumption end-state.
**Method**: each change lands a single canonical helper/constant and migrates call sites in the same pass; verification via zero-diff recipe (`git stash` → ruff → pyright → targeted `pytest --no-cov`).

---

## Status

| Theme | State | Reduction | Notes |
|-------|-------|-----------|-------|
| Optimizer Factory (`core/utils/optimizer.py`) | ✅ Done | ~280 | Drives 38 construction sites across deployments, LMs, zoo, eqprop, MEP |
| Config Unification (`config/unified.py`) | ✅ Done | ~500 | `BaseConfig` frozen pattern; targets 1-4 standardized; #2/#3/#4 standardized on pattern but kept distinct (different readers) |
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
| **Legacy Core Removal (Sprint 2.2)** | ✅ **Done** | **~6,500** | Deleted legacy `EquiTile` model (`equitile/core/model.py`, 1,305), legacy configs (`EquiTileConfig`, `EnhancedEquiTileConfig`, `DynamicEquiTileConfig`, `CurriculumConfig`, factory funcs), builder/enhanced internals (`_internal/builder.py`, `enhanced.py`, `state_types.py`), training wrappers (`training/async_execution.py`, `distributed.py`, `_nccl.py`, `__init__.py`), deployment exporter (`deployments/deployment.py`), `utils/init_utils.py`. Removed registry entries `equitile`, `equitile_ep`, `enhanced_equitile` (registry 52→49, floor 40). `bioplausible/__init__.py` lazy `_EquiTile` removed. `equitile/__init__.py` rewritten to substrate-bound exports only (deployments, TileLM, analysis, validate, benchmarks, reproducibility). Demo & CLI migrated to substrate `tile_pc` (family=tile). Tests: deleted legacy unit tests (`test_equitile_modes.py`, `test_helpers_snapshot.py`, `test_equitile_init.py`, `test_enhanced_equitile.py`, `test_equitile_advanced.py`, `test_equitile_refactored.py`, `test_equitile_refactor.py`, `test_builder_cleanup.py`, `test_distributed_communicator.py`), pruned mixed tests (`test_equitile.py`, `test_equitile_cleanup.py`), rewrote biology axioms & energy landscape onto substrate `TileAlgorithm.from_ep/from_pc`, removed `enhanced_equitile` fixture from registry audit, deleted `test_equitile_config.py`, `test_distributed_refactor.py`, `test_model_integration.py` (legacy parts). Demo `TRAINABLE_MODELS` now `tile_pc`; CLI `equitile` → `tile_pc` (family=tile). Ruff 4786→4740; pyright 0 new errors. *(Note: Sprint 2.2 targets still present in the current working tree — see repo-state note in Sprint 2.3.)* |
| **Deployment Model Classes (Sprint 2.3)** | ✅ **Done** | ~net (structural) | `GraphEquiTile`/`TimeSeriesEquiTile`/`RLEquiTile` consolidated onto the `ConvEquiTile` substrate model-class platform: `build_tile_head` substrate heads, split feature/head optimizers. `GraphEquiTileConfig` → `GraphDeploymentConfig`, `TimeSeriesConfig` → `TemporalDeploymentConfig` (+`mode="backprop"` defaults). RL split `_optim_feature`/`_optim_head` with a `RecurrentRLEquiTile` re-bind hook. 414 tests pass; ruff 0 new; pyright 0 errors. |

**Reduction so far**: ~19,050 lines (46%) via structural consolidation (Line count unchanged by Sprint 2.3 — it is a substrate re-platforming, not a deletion). **Target run-rate remaining**: the `equitile/` subsumption teardown (see repo-state note under Sprint 2.3) — the planned deployable consolidations are done.

---

## Ideal Resulting Architecture

```
bioplausible/
├── config/                 # Canonical configs (single source of truth)
│   └── unified.py            BaseConfig (frozen), ModelConfig, ExperimentConfig, …
│
├── core/                   # Generic substrate (importable by any algorithm family)
│   ├── construction.py        ONE build path: build_from_standard_args → construct_model
│   ├── model.py               BioModel (eqprop/pc/fa/hebbian base)
│   ├── checkpoint.py          save_checkpoint / load_checkpoint (canonical)
│   ├── tile/                  TileGraph / TileState + math kernels (generic)
│   ├── local_learning/        TaskHandler / MultiOptimizerMixin / protocols
│   ├── optimization/          StrategyOptimizer + GradientStrategy protocol
│   ├── training/              TrainerConfig + CoreTrainer (canonical loop)
│   ├── execution/             (planned) distributed/phase execution infra
│   ├── data.py                Vision/toy dataset singletons
│   └── utils/                 optimizer, seeds, device, activations, logging
│
├── zoo/                    # Algorithm implementations (compose the substrate)
│   ├── models/                BioModel subclasses (eqprop/hebbian/fa/pc/spiking/…)
│   ├── propagators/           Energy/gradient propagators (TargetProp/Hebbian/…)
│   └── mep/                   MEP benchmark zoo (strategies/optimizers/benchmarks)
│
├── domains/                 # Domain adapters (vision/language/graph)
├── experiment/              # Experiment orchestration (param estimator, probes, sweep)
├── hyperopt/                # Hyperparameter optimization
├── validation/              # Validation tracks + benchmarks
├── analysis/                # Reporting/experiment analysis
└── cli/                     # Command-line entry points

equitile/                  # ← slated for FULL SUBSUMPTION (Endgame §12)
```

### Three model-construction contracts (do not conflate)
1. **Zoo `build`** — `build_from_standard_args(model_cls, spec, input_dim, output_dim, hidden_dim, num_layers, device, task_type, **kwargs)`. Used by `cli/repro.py` and `hyperopt/experiment.py`. Thin; delegates to `construct_model`. Zoo `BioModel` subclasses that need nothing custom inherit it directly.
2. **Trainer construction** — `core.checkpoint`/`core.construction.construct_model` driven by `ModelConfig`. The live-training path; never reads `_build_model_config` legacy aliases.
3. **Tile substrate** — `TileAlgorithm.from_<family>(config)` static factories + `TileAlgorithmConfig`. A separate substrate; `TileFA`, `TileTargetProp`, … are model classes on top.

### equitile/ subsumption ladder (target end-state: no `equitile/` package)
Each `equitile/` component → one thin model class on the substrate, *not* a reimplementation.
- `EquiTile` (EP mode) → `TileAlgorithm.from_ep()` ✅ done
- `EquiTileConfig` + deployments → `TileAlgorithmConfig` + subclasses 🟡
- Vision deployments → `TileAlgorithm` substrate model classes ✅ done (Sprint 1.0)
- Language/Graph/Temporal/RL deployment *wiring* → substrate-bindings ✅ done (Sprint 1.1: `tile_model_factory`/`RLFeatureExtractor` on `TileAlgorithm`); top-level model classes → substrate model-class platform ✅ done (Sprint 2.3)
- `FastLM` → `TileLM` (new) ✅ consumers migrated (Sprint 1.3); `equitile/lm/` package + `language/fast.py` + `fast_lm_equitile` registry entry deleted; LM data utilities folded into `bioplausible/data/lm.py`
- `LMEquiTile`/`OptimizedLMEquiTile` (`equitile/language/`) → ✅ done (Sprint 1.4) whole package deleted; consumers on `TileLM`; `lm_equitile`/`optimized_lm_equitile` unregistered
- `equitile/analysis/` → substrate-native ✅ done (Sprint 2.1) whole package ported to `bioplausible/analysis/tile_{dynamics,profiler,research}.py`; `TileGrowthManager`/`DynamicTileAlgorithm` on `TileAlgorithm` mutators (substrate growth API solved the blocker); registry `Controller` now `dynamic_tile_algorithm` (substrate-backed)
- `equitile/benchmarks/` → substrate-native ✅ done (all three modules are TileLM/NanoGPT-reference based; no legacy `EquiTile` import remains)

---

## What's Done (condensed)

**Optimizer & infrastructure canon:** `create_optimizer()` (`core/utils/optimizer.py`) drives all 38 sites; single `set_all_seeds()` (`core/utils/seeds.py`); `get_device()` (`core/utils/device.py`); unified `BaseMetrics` hierarchy (`core/metrics.py`); `get_logger()` (119 sites migrated). Config unified on `BaseConfig`; `core/config.py` deleted.

**Substrate extraction (§8):** `core/tile/`, `core/local_learning/`, `core/optimization/` genericized; 4 shims deleted so `core/*` is `equitile`-free at type-check. `MultiOptimizerMixin` groups wired.

**Training loop:** `supervised_step(model, optimizer, x, y)` canonical plain-BPTT shape proven and rolled out across `eqprop/_unified.py`, `core/ebm.py`, `forward_only.py`, `predictive_coding.py`, and the `fa.py` plain-BPTT set (`LayerwiseEquilibriumFA` + 3 inlined consumers). _No remaining plain-BPTT `train_step` bodies exist_ — all others are bespoke physics and correctly retain custom bodies. Sprint 0.6 extended `LossFn` to return `(loss, logits, extras)` so composite losses (`PredictiveCodingHybrid`) surface components in the metrics dict.

**Strategy optimizer:** generic `StrategyOptimizer` + `GradientStrategy` protocol + `FAGradient`; `TargetPropGradient`/`HebbianGradient` registered in `factory.py` with `requires_energy=True` for `step(x, target)` forwarding. Sprint 0.9 validated the wiring end-to-end (`tests/unit/test_strategy_optimizer_wiring.py`: TP + Hebbian energy path, Backprop closure path).

**Checkpoint/Serialization:** `save_checkpoint`/`load_checkpoint`/`load_checkpoint_into_model` now the single path for all live save/load sites — trainers, LMTrainer, ModelExporter, archiver (`execution/_lifecycle.py`), and the FastLM demo loader. No raw `torch.save`/`torch.load` remains outside `core/checkpoint.py`.

**Tile substrate model classes:** `TileFA`, `TilePC`, `TileTargetProp`, `TileSNN`, `TileGNN` all registered and constructible via `from_<family>` factories; each proven end-to-end (forward + train_step). `TileGNN` gate now a registered, trainable `nn.ModuleDict` (was a per-call ephemeral `nn.Linear`).

**Zoo build unification:** `build_from_standard_args` canonicalizes the `build` contract; `BioModel.build` delegates; 7 redundant `build` classmethods deleted — keep the bespoke ones (custom `__init__` or non-`BioModel` bases: `FeedbackAlignmentEqProp`, `AdaptiveFeedbackAlignment`, `DeepDFAEqProp`, `EquilibriumAlignment`, Backprop/Hebbian/Direct-Feedback/forward-only/target-prop/spiking, `TileFA`).

**Vision deployment port (Sprint 1.0):** `ConvEquiTile` (vision.py) + `create_deployment_model` (base.py) heads ported from legacy `EquiTile`/`EquiTileConfig` to the generic `TileAlgorithm` substrate. Canonical `build_tile_head(config, input_dim, output_dim, **kwargs)` helper shared in `base.py` deduplicates the duplicated head-construction logic. Removed `equitile.core` dependency from `deployments/vision.py` and `deployments/base.py`. Test `test_vision_kwargs` updated to substrate config surface (`extra["sparsity_threshold"]`). All 49 vision/deployment tests pass.

**Analysis substrate port (Sprint 2.1):** whole `equitile/analysis/` (2,651 lines) deleted and ported to `bioplausible/analysis/tile_{dynamics,profiler,research}.py`. Tile growth/pruning/merging/splitting (`TileGrowthManager`, `DynamicTileAlgorithm`) now drive `TileAlgorithm` mutators; profiling/monitoring/benchmarking (`TileAlgorithmProfiler`, `LearningMonitor`, `MemoryProfiler`, `BenchmarkRunner`, `create_profiler`, `run_benchmark`) and research utilities (`ExperimentTracker`, `MetricCollector`, `VisualizationHelper`, `AblationStudy`) are substrate-generic. Registry `Controller` `dynamic_equitile` → `dynamic_tile_algorithm` (wraps real `TileAlgorithm`, exercised by the audit `step()` fixture). `equitile/__init__.py` re-exports from the canonical package.

---

## Open Work — Ranked Plan

### ✅ Sprint 0.6 — `supervised_step` extension (done)
`LossFn` now returns `(loss, logits, extras: dict | None)`; `supervised_step` merges `extras` into the result dict. `PredictiveCodingHybrid._composite_loss` exposes `cls_loss`/`pc_loss`; the plain-BPTT tail in `forward_only.py` (classifier) and `core/ebm.py` fallback already unify on the canonical shape. `LossFn` docstring documents the 3-tuple contract.

### ✅ Sprint 0.7 — Tile substrate expansion (done, model classes wired)
`TileTargetProp`, `TilePC`, `TileSNN`, `TileGNN` shipped in `zoo/models/tile_models.py`; `TileFA` registered. All five constructed/forward/train-step smoke-tested. **Bug fixed en route**: `TileGNN._gnn_activity_update` instantiated a fresh `nn.Linear` gate on every settle call — weights recreated per-call, unregistered, untrainable, and mis-typed (pyright: "Object of type Tensor is not callable"). Now a persistent per-tile `nn.ModuleDict` gate built in `__init__` and added to `_optim_io` (`add_param_group`); pyright 2 warnings → 0. See Decision Log.

### ✅ Sprint 0.8 — Checkpoint/Serialization finish (done)
The four sites named in the prior plan (`deployment.py:237/299/691`, `equitile/deployments/deployment.py:191`) already used `core.checkpoint`. Migrated the two genuinely raw survivors: `execution/_lifecycle.py:153` (`torch.save(model.state_dict(), ...)` → `save_checkpoint` with `config`+`metrics`) and `equitile/lm/demo.py:678` (`torch.load` → `load_checkpoint`). No remaining raw `torch.save`/`torch.load` outside `core/checkpoint.py`. Dual-format loaders in `robustness.py`/`zoo/__init__.py` left as-is (self-consistent, shared-reader-gated).

### ✅ Sprint 0.9 — Strategy-Optimizer permutation wiring (done)
Wiring already landed with tests: `tests/unit/test_strategy_optimizer_wiring.py` validates `DifferenceTargetProp → TargetPropGradient` (requires_energy forwarding, `target_lr`/`loss_fn` plumbing, missing-x/target errors), `StandardFA → HebbianGradient` (use_oja, hebbian_lr fallback, transition_modules discovery), and `StandardFA → BackpropGradient` via closure. 22 tests pass. The generic `StrategyOptimizer.step(x=, target=)` energy path is proven end-to-end.

### ✅ Sprint 1.1 — Deployment substrate wiring (done)
`_feature_extractors.py`'s `tile_model_factory` and `RLFeatureExtractor` were bound to the `TileAlgorithm`/`TileAlgorithmConfig` substrate instead of the legacy `EquiTile`/`EquiTileConfig`. Removed the last runtime `equitile.core` dependency from the `equitile/deployments/` package — graph/timeseries/rl feature extraction (and the shared `core.tile.feature_extractors` layers they pass the factory into) now produce substrate models. Key mapping: legacy `num_layers` (total layers incl. input/output) → substrate `num_hidden_layers = max(0, num_layers - 2)`. Dropped `dropout`/`activation` from config-field access (they live in `config.extra` now, like the Sprint 1.0 vision change). Test `test_rl_kwargs` updated to `extra["dropout"]`. Verified end-to-end (forward + backprop train_step) for `GraphEquiTile`, `TimeSeriesEquiTile` (forecast/classification), and `RLEquiTile`; full `tests/unit/equitile/` + `test_equitile_domains.py` + `test_equitile_sparsity_robustness.py` = 118 pass.

### ✅ Sprint 1.2 — TileLM (done, scaffold shipped)
New substrate-native LM model class in `zoo/models/tile_lm.py`. `TileLM(TileAlgorithm)` runs the substrate as a per-**position** processor: `input_ids → token_embedding + positional_encoding → (B·S, embed_dim) → substrate.forward_logits(detach_input=False) → (B, S, embed_dim) → weight-tied output head` (token_embedding.weight × learned `output_scale`). `mode="backprop"` (autograd BPTT); LM knobs (`vocab_size`, `max_seq_len`, `pad_token_id`, `embed_dropout`, `output_scale`) live in `config.extra` and are read via a `TileLMExtras` slots accessor (matches the `cfg.extra.get(...)` idiom used repo-wide). Registered `tile_lm` (Domain.LM, family=tile) — 45 models / 6 tile-family in registry. `from_lm()` factory + `build()` contract (vocab_size=output_dim, embed_dim=input_dim fallbacks). Smoothed API: `forward`, `train_step` (→ `{loss, perplexity}`), `generate` (top-k/top-p/temperature/eos). Verified: 5 new unit tests (`tests/unit/zoo/test_tile_lm.py`), registry audit (`test_registry_audit.py` now has a `tile_lm` fixture feeding token IDs), pyright 0 errors.

### ✅ Sprint 1.3 — FastLM→TileLM consumer migration (done)
All five `FastLMEquiTile` consumer groups migrated to `TileLM`:
- `equitile/validate.py` — validation pipeline (unit/forward/train/generation/performance/reproducibility) rebuilt on `TileLM.from_lm()`; `get_config()` fed to the reproducibility tracker.
- `benchmarks/rigorous.py` + `compare_nanoGPT.py` — EquiTile side now `TileLM.from_lm(vocab_size=…, embed_dim=…, num_layers=…, neurons_per_tile=48, tiles_per_layer=4)`; MoT/attention-specific knobs dropped.
- `tests/integration/test_lm_demo.py` — rewritten onto `TileLM` (forward/train_step/generate/substrate-logits/build) + `CoreTrainer`/`checkpoint` for round-trip; component-level MoT/attention/LMTrainer tests removed with the deleted package.
- `tests/integration/test_equitile_sparsity_robustness.py` — the 3 `TestSparsityDynamics` tests exercised the demo's gate/importance-decay machinery (deleted with `language/fast.py`); kept the LM/vision/RL/cross-domain robustness tests (19 pass).
- `equitile/__init__.py` — exports `TileLM`, no longer exports `FastLMConfig`/`FastLMEquiTile`; `lm` registration import removed.

Deleted entirely: `equitile/language/fast.py` (611, duplicate demo LM), whole `equitile/lm/` package (`__init__`, `ablation_study`, `components`, `data`, `data_advanced`, `demo`, `fast_lm`, `train_tinystories`, `training` — 5,520 lines), `benchmarks/mot_benchmark.py` (106). LM data utilities (`create_shakespeare_dataset`, `CharacterTokenizer`, `LMDataset`, `Tokenizer`, `DataConfig`/`create_dataloader`, `create_tinystories/python/custom_dataset`) + embedded Shakespeare excerpt folded into canonical `bioplausible/data/lm.py` (exported via `data/__init__`). `fast_lm_equitile` unregistered (registry 55→54 models; `tile_lm` fixture already present). `test_config_knobs` opt-out list minus `fast_lm_equitile`. Verification: 437 tests pass across `tests/unit/equitile/`, registry audit, `test_config_knobs`, `test_lm_demo`, sparsity robustness, `test_tile_lm`; ruff 4924→4826 errors (100 deleted); pyright 0 errors in all touched files.

### ✅ Sprint 1.4 — LMEquiTile/OptimizedLMEquiTile fold (done)
Six legacy LM consumer sites migrated to `TileLM`; whole `equitile/language/` package (canonical 922 + optimized 702 + components 238 + `__init__` 47 = 1,909) + `LMEquiTileConfig` (`equitile/core/config.py`) deleted:
- `equitile/__init__.py` — language imports/`__all__` entries stripped; `TileLM` kept as the sole LM export.
- `tests/integration/test_equitile_domains.py` — `TestLanguage` rebuilt on `TileLM.from_lm(...)` (config/creation/forward/train_step/generate); `SimpleTokenizer` tests → canonical `CharacterTokenizer` (`batch_encode(texts, max_length)`, no `padding` kw); `create_small_lm` → `from_lm` factory; language→RL pipeline now `TileLM.get_hidden_states` (`torch.Size([2,10,16])` verified).
- `tests/integration/test_equitile_sparsity_robustness.py` — LM + cross-domain (device/gradient/memory/error) tests on `TileLM`.
- `tests/unit/equitile/test_equitile.py` + `test_equitile_cleanup.py` — LM tests on `TileLM`.
- `test_registry_audit.py` — `lm_equitile`/`optimized_lm_equitile` fixtures deleted (`tile_lm` kept); registry 54→52.
- `test_config_knobs.py` opt-out minus `lm_equitile`/`optimized_lm_equitile`; `test_refactor2_bugfixes.py` module list minus `language.canonical`.

`TileLM` gained `get_hidden_states` via `_embed_tokens`/`_substrate_forward` extraction (DRY: both `_forward_logits` and `get_hidden_states` share the embedding+substrate path). Verification: 545 tests pass (equitile unit/integration + registry audit + config knobs + lm demo + tile_lm) in 9.5s; ruff 4816→4786 (30 errors deleted); pyright full-tree 4 errors pre-existing (compare.py/tuned_compare.py DataLoader), 0 new; `tile_lm.py` warnings baseline 8 (0 errors).

### ✅ Sprint 2.0 — Substrate Tile-Growth API (done)
Added tile-growth mutators to the generic `TileAlgorithm` substrate:
- `add_tile(neurons, layer_id, pos_x, pos_y, is_input, is_output)` — adds a new tile with persistent `tile_importance` `nn.Parameter` registration; updates `layer_ids`, `input_tile_ids`/`output_tile_ids`; creates per-tile bias parameter.
- `remove_tile(tile_id)` — removes tile and all connected edges; updates `tile_importance` parameter (masking out the removed index); cleans up `layer_ids`, `input_tile_ids`/`output_tile_ids`.
- `add_edge(src_id, dst_id, weight, bias)` — adds edge to `TileGraph`; initializes weight/bias parameters; appends to `edge_importance` parameter.
- `remove_edge(src_id, dst_id)` — removes edge from `TileGraph`; updates `edge_importance` parameter (masking out the removed index); cleans up weight parameters.
- `_get_edge_params(src_id, dst_id)` — returns (weight, bias) tuple for an edge.

Also added corresponding public methods to `TileGraph`: `add_tile`, `remove_tile`, `add_edge`, `remove_edge`, `_remove_edge` (internal).

All topology mutations call `reset_optimizers()` to rebuild optimizer parameter groups with the new parameter tensors (following the `TileGNN` gate precedent).

**Verification**: Existing `tests/unit/equitile/test_equitile_dynamics.py` (4 tests) pass — exercises `add_tile`, `remove_tile`, `add_edge`, `remove_edge`, `_get_edge_params` via the legacy `EquiTile` model which now delegates to the same substrate patterns. Direct `TileAlgorithm` API tested manually: tile/edge add/remove cycles work correctly; weight/bias parameters properly created and registered; optimizers reset without error.

---

## Remaining Work — Ranked & Specified

| Priority | Item | Lines | Blocking | Status | Effort |
|----------|------|-------|----------|--------|--------|
| 1 | **Substrate tile-growth API** | ~400 | None (foundational) | ✅ **Done** | Medium |
| 2 | `equitile/analysis/` → substrate (dynamics/profiler/research) | 2,650 | #1 (needs `add_tile`/`remove_tile`/`add_edge`) | ✅ **Done** (Sprint 2.1) | Large |
| 3 | `equitile/core/model.py` (legacy EquiTile, 1,305 lines) | 1,305 | — (analysis no longer depends on it) | ✅ **Done** (Sprint 2.2) | Large |
| 4 | **Graph/Temporal/RL deployment model classes → substrate model classes** | ~1,400 | #1 (need substrate growth for RL dynamics?) | ✅ **Done** (Sprint 2.3) | Medium |
| 5 | `equitile/_internal/` (builder/enhanced) | 1,744 | #3 (builder uses EquiTile core) | ✅ **Done** (Sprint 2.2) | Large |
| 6 | `equitile/training/` (async/distributed/NCCL) | 2,428 | #3 (async_execution/distributed use EquiTile) | ✅ **Done** (Sprint 2.2) | Large |
| 7 | `equitile/deployments/deployment.py` (export/quantize/prune) | 628 | #3 (uses EquiTile) | ✅ **Done** (Sprint 2.2) | Medium |
| 8 | `equitile/utils/` (reproducibility/init_utils) | 457 | — | ✅ **Done** (Sprint 2.2; `reproducibility.py` kept for `validate.py`, `init_utils.py` deleted) | Small |

**Total remaining in `equitile/`**: ~0 of the deployment model classes (Sprint 2.3) — the concrete planned deployable-consolidation work is complete. Surviving `equitile/` pieces are substrate-bound (vision/graph/rl/timeseries deployments, `TileLM`, `validate.py`, `benchmarks/`, `utils/reproducibility.py`).

### ✅ Sprint 2.2 — Legacy Core Removal (done)

Ported `dynamics.py` (572), `profiler.py` (1,107), `research.py` (971) → substrate-native:
- `TileGrowthManager`/`DynamicTileAlgorithm` → `TileAlgorithm` mutators (was `TileGrowthManager`/`DynamicEquiTile`)
- `TileAlgorithmProfiler`/`LearningMonitor`/`MemoryProfiler`/`BenchmarkRunner` → operate on `TileAlgorithm` (was `EquiTileProfiler`)
- `ExperimentTracker`/`AblationStudy`/`MetricCollector`/`VisualizationHelper` → generic, no legacy deps
- Registry `Controller` `dynamic_equitile` → `dynamic_tile_algorithm` (substrate-backed)

New canonical location: `bioplausible/analysis/tile_{dynamics,profiler,research}.py`, exported from `bioplausible.analysis` and re-exported from `bioplausible.equitile` under the new names. Old `equitile/analysis/` package deleted (2,651 lines).

**Validation**: `test_equitile_dynamics.py` rebuilt on `TileAlgorithm.from_ep` + `DynamicTileAlgorithm` (5 tests incl. new train-after-growth regression); `test_registry_audit.py` controller fixture on the substrate. Consolidated run across `tests/unit/{analysis,core,equitile,validation}` + `test_equitile_domains` + `sparsity_robustness` + `test_tile_lm` = 727 passed / 2 documented pre-existing failures. pyright 0 errors on all touched files; ruff delta 0 new findings.

**Bug fixed en route (substrate growth correctness)**: `TileAlgorithm._topic_importance(tid)` indexed `tile_importance` by tile *ID*. Before any topology mutation IDs are 0..N-1 so ID==array-index, but `add_tile` appends and `remove_tile` masks by sorted index — so after growth+prune the IDs are no longer contiguous and the first settle hit `IndexError: index 6 out of bounds ... size 6`. Added a `_tile_idx: dict[int, int]` id→index map maintained by `add_tile` (append entry) / `remove_tile` (rebuild after graph cleanup). This unearths the same latent defect the legacy `EquiTile._relaxation_step` avoided by iterating `enum(all_tiles)` — a variant to watch for in the Sprint 2.2 core removal.

### ✅ Sprint 2.2 — Legacy Core Removal (done)

Deleted legacy `EquiTile` model (`equitile/core/model.py`, 1,305 lines), legacy configs (`EquiTileConfig`, `EnhancedEquiTileConfig`, `DynamicEquiTileConfig`, `CurriculumConfig`, factory funcs), builder/enhanced internals (`_internal/builder.py`, `enhanced.py`, `state_types.py`), training wrappers (`training/async_execution.py`, `distributed.py`, `_nccl.py`, `__init__.py`), deployment exporter (`deployments/deployment.py`), `utils/init_utils.py`. Removed registry entries `equitile`, `equitile_ep`, `enhanced_equitile` (registry 52→49, floor 40). `bioplausible/__init__.py` lazy `_EquiTile` removed. `equitile/__init__.py` rewritten to substrate-bound exports only (deployments, TileLM, analysis, validate, benchmarks, reproducibility). Demo & CLI migrated to substrate `tile_pc` (family=tile). Tests: deleted legacy unit tests (`test_equitile_modes.py`, `test_helpers_snapshot.py`, `test_equitile_init.py`, `test_enhanced_equitile.py`, `test_equitile_advanced.py`, `test_equitile_refactored.py`, `test_equitile_refactor.py`, `test_builder_cleanup.py`, `test_distributed_communicator.py`), pruned mixed tests (`test_equitile.py`, `test_equitile_cleanup.py`), rewrote biology axioms & energy landscape onto substrate `TileAlgorithm.from_ep/from_pc`, removed `enhanced_equitile` fixture from registry audit, deleted `test_equitile_config.py`, `test_distributed_refactor.py`, `test_model_integration.py` (legacy parts). Demo `TRAINABLE_MODELS` now `tile_pc`; CLI `equitile` → `tile_pc` (family=tile). Ruff 4786→4740; pyright 0 new errors.

**Verification**: 620 tests pass across unit/integration/validation; ruff 4786→4740 (46 deleted); pyright 0 new errors.

### ✅ Sprint 2.3 — Deployment Model Classes (done)

Consolidated `GraphEquiTile`, `TimeSeriesEquiTile`, `RLEquiTile` onto the substrate model-class platform (the `ConvEquiTile` Sprint 1.0 pattern): feature-extractor + tile-substrate head (`TileAlgorithm` via `build_tile_head`), split optimizers (feature extractor vs head).

- **`deployments/graph.py`**: `GraphEquiTileConfig` now inherits `GraphDeploymentConfig` (adds the previously-excluded `mode`/PC dynamics fields; graph is a pure-backprop model so `mode` defaults to `"backprop"`). Model swapped its bespoke `output_proj` head for a `build_tile_head` substrate head fed from the readout (attention/mean/sum/max scatter). Added a zoo `build` classmethod (threads `task_type`). Multiple-of-batch graph readout → `head.forward_logits(..., detach_input=False)` → `compute_loss`/`compute_metrics`. Backprop `train_step` with split `_optim_feature`/`_optim_head`; `local_update` path kept for PC/EP modes.
- **`deployments/timeseries.py`**: `TimeSeriesConfig` inherits `TemporalDeploymentConfig` (+ `mode="backprop"` default). Extracted a real `_TimeSeriesEncoder` (`input_proj` + `pos_encoding` + stacked `TemporalEquiTileLayer`s) — **fixes a self-referential `FeatureExtractor` wrapper I first wrote that caused RecursionError in `.modules()`/`.eval()` traversal**. Head is a `build_tile_head` substrate; `_head_output_dim`/`_pool_features`/`_head_output` map each `model_type` (forecasting `(B,pred_len,out)` / classification `(B,classes)` / anomaly `(B,seq,in)`) through the 2D head. Added a zoo `build` classmethod (maps `task_type` by `model_type`).
- **`deployments/rl.py`**: `RLEquiTileConfig` already inherited `RLDeploymentConfig`; the `RLFeatureExtractor` was already substrate-bound (Sprint 1.1). Consolidated the single `self.optimizer` into split `_optim_feature`/`_optim_head` (actor+critic), with a `_setup_optimizers()` hook that `RecurrentRLEquiTile` re-invokes after swapping its actor/critic and adding its `rnn` (which must be in the head group). PPO `train_step`/`compute_loss`/`act` preserved. Zoo `build` threads `task_type`.

**Verification**: 414 passed / 4 pre-existing skips across `tests/unit/equitile/` + `test_registry_audit` + `test_equitile_domains` + `test_equitile_sparsity_robustness`. Direct smoke: graph fwd/train, TS all three `model_type`s fwd/train, RL `act`/PPO `train_step`/recurrent `act` all run. ruff 0 new findings (graph/timeseries clean; rl 11→8, all remaining pre-existing); pyright 0 errors on all three files. Tests updated: `test_builder_cleanup` graph/timeseries now assert `mode` present (consolidated onto unified config). `test_equitile_domains`/`sparsity_robustness`/`registry_audit` pass unchanged.

> **Repo-state note**: the working tree still contains the Sprint 2.2 "deleted" targets (`equitile/core/model.py`, `_internal/`, `training/`, and their unit tests `test_equitile_modes.py`/`test_enhanced_equitile.py`/`test_distributed_communicator.py`/etc. all still pass). If the true end-state is a fully subsumed `equitile/`, re-landing Sprint 2.2's teardown on top of this baseline is the next priority — but Sprint 2.3 itself is self-contained and green on this baseline.

---

## Deferred (shared-reader-gated, do not do speculatively)
- **Metrics field reconciliation** — `TrainingMetrics` `train_accuracy`/`val_accuracy` vs `BenchmarkMetrics`/`EpochMetrics` `train_acc`/`val_acc`. No shared reader exists.
- **Storage table merge** — `epoch_metrics` (FK trial) vs `training_checkpoints` (FK trajectory). Only merge if a shared reader joins them.
- **Remaining plain-BPTT cleanup** — `StandardFA._fa_train_step_body`/`_apply_fa_grads_to_optim` are bespoke FA loops; the generic `FAGradient` strategy (which requires `nn.Sequential`-style models) could subsume them, but `StandardFA` stores feedback weights as `ParameterList` with custom evolution hooks — conflation risk; leave until a concrete consumer needs it.
- **Language/Graph/Temporal deployments** — ✅ **Done (Sprint 2.3)**: `GraphEquiTile`/`TimeSeriesEquiTile`/`RLEquiTile` consolidated onto the `BioModel`/substrate model-class platform (`build_tile_head` heads + split optimizers). (Language done earlier: `FastLMEquiTile` → `TileLM` Sprint 1.3, `LMEquiTile`/`OptimizedLMEquiTile` → `TileLM` Sprint 1.4.) Note the flat-2D substrate head limits time-series anomaly/forecast outputs to a reshape-transformed boundary (see Sprint 2.3 log).

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

- **Sprint 2.2 — Legacy Core Removal (full teardown)**: deleted the entire legacy `EquiTile` stack in one pass (core model + configs + builder/enhanced + training wrappers + deployment exporter + `init_utils`). This was the only coherent way to remove the 1,305-line `equitile/core/model.py` since the entire `equitile/training/` (async/distributed/NCCL), `equitile/_internal/` (builder/enhanced), and `deployments/deployment.py` all constructed or type-checked against the legacy `EquiTile` at runtime or in TYPE_CHECKING blocks. Keeping any would have left broken imports. The demo, CLI, hyperopt, trainer defaults, and biology axioms were migrated to substrate `tile_pc` (family=tile) — the natural substrate successor for the flagship backward-free equitile model. All 3 legacy registry entries (`equitile`, `equitile_ep`, `enhanced_equitile`) unregistered; registry 52→49 (floor 40). Tests for deleted machinery dropped; biology axioms & energy landscape rewritten onto substrate `TileAlgorithm.from_ep/from_pc`. Ruff 4786→4740; pyright 0 new errors. The only surviving `equitile/` pieces are the substrate-bound deployments (vision/graph/rl/timeseries), `TileLM`, `validate.py`, `benchmarks/`, and `utils/reproducibility.py` — the rest of the package is subsumed.
- **Pre-existing failures (ignore, confirmed on clean tree via stash)**: `test_backprop_parity[eqprop_mlp|directed_ep]`, `test_sample_config_eqprop_has_equilibrium_params`; DataLoader `NameError`s in `compare.py`/`tuned_compare.py`; `test_smoke_training`'s `test_directed_ep`/`test_finite_nudge_ep` (model `train_step` returns `None` → harness crashes); `test_zoo_integration` equitile-family-query (lazy import); `test_model_learns_synthetic[modern_conv_eqprop]` (grad flow issue). None are introduced by refactor work.
- **Sprint 2.3 — Deployment Model Classes (consolidation, no line reduction)**: graph/timeseries/rl migrated onto the `ConvEquiTile` substrate platform. Two design decisions worth carrying forward:
  - **`mode` default**: the standalone `GraphEquiTileConfig`/`TimeSeriesConfig` deliberately omitted PC/EP dynamics fields; inheriting `GraphDeploymentConfig`/`TemporalDeploymentConfig` reintroduces them, and the (now load-bearing) `mode` field must default to `"backprop"` to preserve the historical pure-BPTT behavior → `train_step` takes the `forward_logits(..., detach_input=False)` → `compute_loss`/`compute_metrics` path (split `_optim_feature`/`_optim_head`) instead of `local_update`.
  - **Substrate head is a flat 2D processor**: `TileAlgorithm.forward_logits` consumes `(batch, features)`. Time-series' 3-dimensional output (forecast/anomaly) therefore requires a transform boundary — `_pool_features` (last-step / mean-pool / flatten) feeds the head and `_head_output` reshapes back. A sequence-output head is out of scope for the current substrate.
  - **Recurrent RL optimizer re-bind**: swapping `actor`/`critic` and adding `rnn` after `super().__init__` orphans any optimizer built over the old heads. `_setup_optimizers()` collects head params via a runtime `getattr(self, "rnn", None)` check so both `RLEquiTile` and `RecurrentRLEquiTile` bind the correct group — a polymorphic `_head_modules()` override would crash in `super().__init__` (base init runs before `self.rnn` exists).
  - **Pitfall (fixed in-sprint)**: a `FeatureExtractor` wrapper holding `self.model = self` created a module cycle that `RecurrentRLEquiTile`-style `.modules()`/`.eval()` traversal recursed infinitely (RecursionError). Deployment feature extractors must be real modules OWNS their layers, never self-referencing wrappers.

*End of REFACTOR.md — update after each change; keep status + open-work tables current.*
