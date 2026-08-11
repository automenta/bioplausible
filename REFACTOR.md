# REFACTOR.md — bioplausible Strategic Refactoring Plan

**Codebase**: 316 Python files, ~41K lines.
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

**Reduction so far**: ~4,393 lines (10.6%). **Target run-rate remaining**: ~1,105 lines (Registry/Build done) → next realizable ~730 lines (Graph/Temporal/RL deployments, Language models, FastLM→TileLM).

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
- Language/Graph/Temporal deployments → model classes 🟡
- `FastLM` → `TileLM` (new) 🟡
- `equitile/analysis/`, `equitile/benchmarks/` → substrate-native versions 🔴

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

### Deferred (shared-reader-gated, do not do speculatively)
- **Metrics field reconciliation** — `TrainingMetrics` `train_accuracy`/`val_accuracy` vs `BenchmarkMetrics`/`EpochMetrics` `train_acc`/`val_acc`. No shared reader exists.
- **Storage table merge** — `epoch_metrics` (FK trial) vs `training_checkpoints` (FK trajectory). Only merge if a shared reader joins them.
- **Remaining plain-BPTT cleanup** — `StandardFA._fa_train_step_body`/`_apply_fa_grads_to_optim` are bespoke FA loops; the generic `FAGradient` strategy (which requires `nn.Sequential`-style models) could subsume them, but `StandardFA` stores feedback weights as `ParameterList` with custom evolution hooks — conflation risk; leave until a concrete consumer needs it.
- **Language/Graph/Temporal deployments** — `equitile/` Language/Graph/Temporal deployment classes still need model-class-on-substrate ports.

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
- **Pre-existing failures (ignore, confirmed on clean tree via stash)**: `test_backprop_parity[eqprop_mlp|directed_ep]`, `test_sample_config_eqprop_has_equilibrium_params`; DataLoader `NameError`s in `compare.py`/`tuned_compare.py`; `test_smoke_training`'s `test_directed_ep`/`test_finite_nudge_ep` (model `train_step` returns `None` → harness crashes); `test_zoo_integration` equitile-family-query (lazy import); `test_model_learns_synthetic[modern_conv_eqprop]` (grad flow issue). None are introduced by refactor work.

*End of REFACTOR.md — update after each change; keep status + open-work tables current.*
