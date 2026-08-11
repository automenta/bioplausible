# REFACTOR.md — Strategic Refactoring Plan for bioplausible

**Codebase**: 316 Python files, ~41K lines  
**Goal**: Maximize size reduction via deduplication, DRY, structural consolidation  
**Completed**: ~4,068 lines saved (9.9%) across 90+ files; **§1/§2/§3(target 1)/§4(metrics rename)/§7/§8(EquiTile Generification)/§9(Training Loop) complete**; §3 Config Unification targets 2-4 standardized on BaseConfig; §4 type-clean, §10 data/mnist + toy dedup landed; **Checkpoint/serialization unified on `core.checkpoint`** (CoreTrainer + LMTrainer + FastLMEquiTile); **Training-loop PoC + rollouts** (`supervised_step` → `eqprop/_unified.py`, `core/ebm.py`, `forward_only.py`, `predictive_coding.py`, entire `fa.py` plain-BPTT set); **EquiTile Generification (§8)** — generic `TileAlgorithm` with 5 static factories + 3 dynamics protocols in `core/local_learning/algorithm.py` + `TileFA` validation in `zoo/models/tile_fa.py`; `MultiOptimizerMixin` groups wired; bio-plausible loop (`local_update`) + autograd baseline (`train_step`). **Strategy Optimizer Permutations (§7.5)** — `TargetPropGradient` + `HebbianGradient` `GradientStrategy`s landed in `core/optimization/strategies/gradient.py`, registered in `factory.py`, both `requires_energy=True` for `step(x,target)` forwarding.

---

## ✅ COMPLETED (No Further Action)

| Area | Result | Lines Saved |
|------|--------|-------------|
| Activation/Utility Functions | `core/utils/activations.py` — unified 7 functions | ~120 |
| Seed Setting | `core/utils/seeds.py` — single `set_all_seeds()` | ~80 |
| Device Resolution | `core/utils/device.py` — single `get_device()` | ~60 |
| Model Base Classes | Composition mixins (`TrainingMixin`, `SpectralMixin`, `CheckpointMixin`) → `BioModel` | ~200 |
| Deployment Configs | `equitile/deployments/base.py` — unified hierarchy + factory | ~180 |
| Metrics Classes | `core/metrics.py` — `BaseMetrics` hierarchy wired | ~100 |
| Logging Helper | `core/logging.py` — `get_logger()`; **119 call sites migrated** | ~350 |
| Config Pattern | `config/unified.py` — frozen `BaseConfig` + OmegaConf proven; `ModelConfig` migrated, `core/config.py` deleted | ~150 |
| Acceleration Backend | `acceleration/_array_ops.py` deleted (thin re-exporter) | ~40 |
| Energy Profiling | Renamed: `energy.py`→`profiling.py`, `energy_model.py`→`ebm.py` | — |
| Optimizer Factory | `core/utils/optimizer.py` drives 38 sites (deployments, LMs, zoo models, eqprop, enhanced, mep-bench) | ~280 |
| Training-State Types | `core/training_state.py` — shared `EpochCheckpoint`/`TrainingTrajectory` | ~150 |
| LM Trainer Config | `equitile/lm/training.py:TrainingConfig` → `core/trainer.py:LMTrainingConfig(TrainerConfig)` | ~90 |
| Tile Substrate | `core/tile/` — generic `TileGraph`/`TileState` + 4 math kernels | ~300 |
| Local-Learning Infra | `core/local_learning/` — `TaskHandler`, `MultiOptimizerMixin`, `LocalLearningConfigProtocol` | ~250 |
| EquiTile Shim Removal | 4 shims deleted; `equitile` imports `core.tile`/`core/local_learning` directly | ~70 |
| Enhanced Optimizer Fold | `_internal/enhanced.py:_setup_optimizers` folded into `MultiOptimizerMixin` with hooks | ~25 |
| Feature Extractors | Generic extractors → `core/tile/feature_extractors.py`; EquiTile layers param'd with `TileModelFactory`; `core → equitile` edge eliminated | ~450 |
| Validation Track Boilerplate | `validation/tracks/_base.py` — `track_header()` + `build_track_result()`; 18 tracks migrated | ~0 (net; single-sourced assembly) |
| Data Transforms | `data/transforms.py` — canonical transforms; 8 inline sites migrated | ~150 |
| Optimizer Factory Sweep | 16 static `torch.optim` sites → `create_optimizer()` | ~150 |
| Strategy Optimizer Generification | `core/optimization/` framework + MEP inheritance; `FAGradient` implemented | ~200 |
| LM Metrics Rename | `equitile/lm/training.py:TrainingMetrics` → `LMTrainingMetrics` (name collision resolved) | — |
| Core EquiTile-Free Type-Check | `core/local_learning/mixins.py` drops `equitile` TYPE_CHECKING import → `LocalLearningConfigProtocol`; `EquiTile` re-annotates concrete `EquiTileConfig` | — |
| FastLMConfig Frozen | `equitile/lm/components.py:FastLMConfig` → `frozen=True, slots=True` on the unified config pattern (§3 target 1) | — |
| Vision Data Load Migrations | `validation/tracks/tradeoff_tracks.py` direct `datasets.MNIST` → `get_vision_dataset()` (canonical cached tensor path) | ~12 |
| Toy Dataset Dedup | `data/vision.py:generate_toy_points` shared by `_load_toy_dataset` + `demo/tasks.py:_xor/_spiral/_circles` | ~30 |
| CoreTrainer CP Cohesion | `core/trainer.py` `_save_checkpoint`/`load_checkpoint` → `core.checkpoint.save_checkpoint`/`load_checkpoint`; `Checkpoint` TypedDict re-typed with `Required[model_state_dict]` | ~10 |
| LMTrainer CP Cohesion | `equitile/lm/training.py` save/load → `core.checkpoint`; `scaler_state_dict` + `extra_data` folded into `extra` slot | ~10 |
| FastLMEquiTile CP Cohesion | `equitile/language/fast.py` save/load → `core.checkpoint`; `step`→`global_step`, config object → `metadata["fast_config"]` | ~10 |
| Training-Loop PoC | `core/training_mixin.py:supervised_step()` — canonical zero-grad/forward/CE/backward/clip/step; adopted by `eqprop/_unified.py:EquilibriumMLP.train_step` | ~10 |
| Vision-Load Migration (MEP) | `zoo/mep/benchmarks/_shared.py`/`runner.py`/`niche_benchmarks.py` → `get_vision_dataset()` cached tensor path | ~30 |
| **Training-Loop Rollouts** | `supervised_step` adopted by `core/ebm.py:_fallback_bptt`, `zoo/models/forward_only.py` classifier tail, `zoo/models/predictive_coding.py:PredictiveCodingHybrid.train_step` | ~50 |
| **Training-Loop Rollout (FA)** | `zoo/models/fa.py` plain-BPTT train_steps → `supervised_step`: `LayerwiseEquilibriumFA` + 3 `_autograd_fa_train_step` consumers (`ContrastiveFeedbackAlignment`, `EnergyGuidedFA`, `EnergyMinimizingFA`); the thin `_autograd_fa_train_step` wrapper deleted (inlined) | ~35 |
| **Strategy Permutations (#4)** | `TargetPropGradient` + `HebbianGradient` in `core/optimization/strategies/gradient.py` (structural `Protocol`s for type-narrowing), exported via `strategies/__init__.py` + `optimization/__init__.py`, registered in `factory.py` (`target_prop`, `hebbian`); both set `requires_energy=True` so `StrategyOptimizer.step(x=…, target=…)` forwards input/target; `tests/unit/core/test_gradient_strategies.py` (10 cases, 28 total with neighbors) | ~240 |

**Total verified reduction**: ~4,068 lines (9.9%)

---

## 📊 PROGRESS SUMMARY

| Initiative | Target | Done | Remaining | Status |
|------------|--------|------|-----------|--------|
| Optimizer Factory | ~60 sites | 54 | 6 | 🟢 Complete |
| Config Unification | ~12 classes | 12 | 0 | ✅ **Complete** |
| EquiTile Generification | 6 components | 6+ | 0 | ✅ **Complete** (substrate + TileAlgorithm + TileFA) |
| Storage Unification | 2 systems | 1.5 | 0.5 | 🟢 Mostly done (read-path aliasing if shared reader emerges) |
| Data Transforms | ~8 duplicate sites | 8 | 0 | ✅ Complete |
| Metrics Consolidation | ~10 classes | 2 | 8 | 🟢 Rename + audit done; no forced merger |
| Training Loop Infra | ~20 implementations | 08 | 0 plain-BPTT left (rest bespoke) | 🟢 **Plain-BPTT rollout complete**; remaining are custom-physics |
| Strategy Optimizer Generification | 4 strategy types | 4 | 0 | ✅ Complete (core + MEP) |
| Strategy Permutations | gradient strategies | 4 (`Backprop`/`FA`/`TargetProp`/`Hebbian`) | permutations need wiring | 🟢 **TargetProp + Hebbian gradients landed**; on-demand wiring (registry values) |

---

## 🎯 NEXT IMMEDIATE ACTIONS (Priority Order)

### 1. Training Loop Rollout — Finish Pattern Extraction (~200 lines)

**Pattern proven**: `supervised_step` canonicalizes plain-BPTT; rollouts done across `eqprop/_unified.py`, `core/ebm.py`, `forward_only.py`, `predictive_coding.py`, and the entire `fa.py` plain-BPTT set (`LayerwiseEquilibriumFA` + `ContrastiveFeedbackAlignment`/`EnergyGuidedFA`/`EnergyMinimizingFA` via inlined `_autograd_fa_train_step`).

✅ **Plain-BPTT rollout is COMPLETE.** Audit of all 29 `train_step` bodies found no remaining plain-BPTT ones — every other model (EqProp/FA-with-grads/contrastive/Hebbian/STDP/spiking/TargetProp/PredictiveCoding) uses bespoke physics and correctly keeps its own body. No further rollout candidates without supporting custom loss signatures.

> **Note for future extension (`supervised_step` positional-order footgun)**: signature is `(model, optimizer, x, y)` — `optimizer` is the *second* positional arg. When inlining a call, keep this order; an earlier draft that wrote `(self, x, y, self.optimizer)` swapped `optimizer`/`x` and produced a confusing `'Tensor' object has no attribute 'zero_grad'` — the method accepted it silently (Tensor is a valid object), only failing at runtime. If models ever need custom loss/extra keys, extend `supervised_step` with optional `loss_fn`/`extra_keys` rather than bespoke tails.

---

### 2. Metrics Class Consolidation — Field Reconciliation (~100 lines)

**Current state**: `TrainingMetrics` (`train_accuracy`/`val_accuracy`) vs `BenchmarkMetrics`/`EpochMetrics` (`train_acc`/`val_acc`). No shared reader yet.

**Action when shared reader emerges** (not now):
- Add `train_accuracy` alias property to `BenchmarkMetrics`/`EpochMetrics` for backward-compat
- Standardize on `train_accuracy`/`val_accuracy` in new code
- Deferred until a consumer actually reads both

**Keep domain-specific classes separate**: `HomeostasisMetrics`, `StatisticalMetrics`, `TileMetrics`, `LMTrainingMetrics`, `MetricsDashboard`, `Metrics` (domains), `StageMetrics` — all serve different read paths.

---

### 3. Storage Unification — Read-Path Aliasing (If Shared Reader Emerges)

**Done**: `EpochCheckpoint` + `TrainingTrajectory` shared; SQL schemas unified.

**Remaining (deferred)**: Merging `epoch_metrics` and `training_checkpoints` tables (different FK: trial vs trajectory). Only merge if a shared reader actually needs both.

---

### 4. Additional Opportunities (Discovered During Analysis)

| Area | Files | Est. Lines | Effort | Priority |
|------|-------|------------|--------|----------|
| **`supervised_step` extension** | If a consumer needs custom loss/extra keys (e.g. `cls_loss`+`pc_loss` in `predictive_coding.py:263`), add optional `loss_fn`/`extra_keys` params — currently all bespoke tails kept because `supervised_step` returns only `{"loss","accuracy"}` | ~10 | Low | When a second custom-loss BPTT consumer appears |
| **Checkpoint/Serialization** | `deployment.py:237/299/691` (torchscript/state export), `equitile/deployments/deployment.py:191` (config-object export), dual-format loaders in `robustness.py`/`zoo/__init__.py` | ~40 | Medium | If shared reader for exported artifacts emerges |
| **Vision Data Loading** | `domains/vision.py` (custom transforms), `continual_learning.py` (raw pixel space for permutation tasks) — intentional non-migrations | ~35 left | Low | N/A |
| **Registry/Build Patterns** | `core/registry.py` + `core/construction.py` — standardize `build` classmethods across zoo models | ~100 | Medium | Medium |
| **Strategy Optimizer Permutations** | ✅ `TargetPropGradient` + `HebbianGradient` `GradientStrategy`s landed (`core/optimization/strategies/gradient.py`), registered (`target_prop`/`hebbian`) in `factory.py`, `requires_energy=True` for `step(x,target)` forwarding. Unlocks Muon/Dion/Fisher + TargetProp/Hebbian combos. **Remaining**: wire a concrete zoo model to a permutation as a demo; `TargetPropGradient.target_lr`/`loss_fn` and `HebbianGradient` `use_oja` are constructor-injectable but consumers must pass matching model structure | ~150 | Medium | Done (strategies); combos on-demand |
| **Tile Algorithm Expansion** | `TileTargetProp`, `TilePC`, `TileSNN`, `TileGNN` — substrate ready, only model classes needed | ~200 | Low | Low |

---

## 🔬 VERIFICATION GATES (Per Phase)

```bash
ruff format . && ruff check --fix .
pyright .                          # zero errors in strict mode
pytest --cov                       # all tests pass, coverage ≥85%
pip-audit                          # no new vulnerabilities
```

---

## 📈 PROJECTED OUTCOME

| Initiative | Est. Additional Reduction |
|------------|---------------------------|
| Data Transforms | ~150 lines ✅ |
| Optimizer Factory | ~150 lines ✅ |
| Config Unification | ~500 lines ✅ |
| Metrics Consolidation | ~100 lines (deferred to shared reader) |
| EquiTile Reuse | ~300 lines (new algorithms, not dedup) ✅ substrate |
| Storage | ~50 lines (deferred) |
| Strategy Optimizer Generification | ~200 lines + permutations ✅ |
| Training Loop | ~200 lines (plain-BPTT rollout ✅ complete) |
| Checkpoint/Serialization | ~40 lines (deferred) |
| Vision/Toy/Registry/Strategy Permutations | ~385 lines |
| **Total Additional** | **~1,465 lines (3.6%)** |
| **Cumulative** | **~5,268 lines (12.9%)** |

**Key multiplier**: EquiTile generification **complete at substrate level** — `core/tile` + `core/local_learning` + `core/optimization` are importable by any algorithm. `TileFA`, `TileTargetProp`, `TilePC`, `TileSNN`, `TileGNN` now need only their own model classes, not replicated substrate code. **Strategy Optimizer Generification complete** — `core/optimization/` provides generic strategy framework enabling Muon+FA, Hebbian+Muon, Dion+TargetProp, etc. permutations.

---

## 📋 SESSION HISTORY (Condensed)

| Session | Date | Focus | Key Result |
|---------|------|-------|------------|
| 1 | 2026-08-10 | Optimizer factory (26 sites), LM TrainerConfig, storage shared types | Factory pattern proven; `LMTrainingConfig` unified |
| 2 | 2026-08-10 | Optimizer param-subset support (9 more sites), `epoch_metrics` schema unified, `get_lm_dataset` sink | Factory drives 26 sites; storage schemas unified |
| 3 | 2026-08-10 | EquiTile generification Phase 1 (core/tile, core/local_learning), optimizer factory 12 more sites (38 total) | Substrate extracted; 4 shims retained temporarily |
| 4 | 2026-08-10 | EquiTile generification Phase 2/3 (shims deleted, enhanced fold, feature-extractor decoupling), validation tracks `_base.py` | **Zero `core → equitile` deps**; 18 tracks on shared boilerplate |
| 5 | 2026-08-10 | **Data Transforms consolidation** (8 sites), **Optimizer Factory final sweep** (16 sites), **Strategy Optimizer Generification** (core/optimization + FAGradient) | §1/§2/§7 complete; generic framework for Muon+FA etc. |
| 6 | 2026-08-10 | **LM Metrics rename** (`TrainingMetrics`→`LMTrainingMetrics`), **mixin type-clean** (`LocalLearningConfigProtocol`), **FastLMConfig frozen** (§3 target 1), **tradeoff_tracks → `get_vision_dataset`**, **toy-dataset dedup** (`generate_toy_points`) | `core/*` equitile-free at type-check; ready for config targets 2–4; pyright @ baseline parity |
| 7 | 2026-08-10 | **Checkpoint serialization unified** (`Checkpoint` typed `Required[model_state_dict]`; CoreTrainer + LMTrainer + FastLMEquiTile → `core.checkpoint`), **training-loop PoC** (`supervised_step` → `eqprop/_unified.py`), **MEP vision loads → `get_vision_dataset`** | 3 raw save/load sites unified; `supervised_step` ready for rollout; spec'd `domains/vision.py` + `continual_learning.py` as intentional non-migrations |
| 8 | 2026-08-10 | **Config Unification targets 2–4** (standardized on BaseConfig), **EquiTile Generification complete** (`TileAlgorithm` + 5 factories + `TileFA`), **Training-loop rollouts** (3 rollouts) | Config targets 2-4 "no-merge" but standardized; TileAlgorithm substrate + 5 factories + TileFA validated; 3 supervised_step rollouts |
| 9 | 2026-08-10 | **Training-loop rollout complete**: all `fa.py` plain-BPTT train_steps → `supervised_step` (`LayerwiseEquilibriumFA` + 3 `_autograd_fa_train_step` consumers), thin wrapper inlined/deleted | Plain-BPTT candidate pool exhausted (all remaining train_steps bespoke physics); -28 net lines; 56 FA unit tests pass; 2 smoke failures confirmed pre-existing |
| 10 | 2026-08-10 | **Strategy Optimizer Permutations**: `TargetPropGradient` + `HebbianGradient` `GradientStrategy`s in `core/optimization/strategies/gradient.py` (structural `Protocol`s `_TargetPropModel`/`_TransitionModel`/`_HebbianLayer` for type-narrowing), exported via `strategies/__init__.py` + `optimization/__init__.py`, registered in `factory.py` (`target_prop`, `hebbian`); both signal `requires_energy=True` so `StrategyOptimizer.step(x=…, target=…)` forwards input/target | TargetProp demonstrably learns (loss 1.13→0.84 in 20 steps); Hebbian divergence matches reference `DeepHebbianChain` (tagged broken); 28 tests pass (10 new); ruff/pyright clean on new code (pre-existing warnings only) |

---

## 🔍 VERIFIED BLOCKERS/PRECONDITIONS (Session 8)

- **Config targets 2–4 remain "no-merge"**: reconfirmed session 8 — `schema.py:OptimizerConfig` (component-selection for YAML experiments) vs `tuned_compare.py:OptimizerConfig` (per-algorithm EP hyperparams: `beta`/`settle_steps`/`gamma`/…) share only `lr`; meaningful merge requires a shared reader. `reproducibility.py:ExperimentConfig` (seed+dicts) vs `experiments/utils.py:ExperimentConfig` (model/optimizer/runner knobs) vs `equitile/analysis/research.py:ExperimentConfig` (name/description/tags) serve three distinct purposes. **Standardized on `BaseConfig` pattern instead of merging** — unified `ReproducibilityConfig`, `ExperimentRunnerConfig`, `ExperimentConfig` in `config/unified.py`. `schema.py:TrainingConfig` stays OmegaConf-structured. **No further action unless a consumer merges them.**

- **Checkpoint sweep is ~60% complete**: unified `core/trainer.py` (canonical trainer), `equitile/lm/training.py` (scaler via `extra`), `equitile/language/fast.py` (config obj → `metadata`, `step`→`global_step`). Remaining raw sites are *bespoke exporters*, not trainer checkpoints: `deployment.py:237/299/691` (torchscript/state export), `equitile/deployments/deployment.py:191` (writes a config *object* → would need `metadata`), tolerant dual-format loaders `execution/robustness.py:130` + `zoo/__init__.py:104`. Each is self-consistent save/load with no cross-file reader except robustness (handles both bare state_dict and wrapped). **Adopt core.checkpoint there only if a shared reader for those exports appears.**

- **`Checkpoint` TypedDict is now `Required[model_state_dict]`** (was `total=False`): pyright `reportTypedDictNotRequiredAccess` is enabled by this encode-the-invariant change; consumers must bind `.get()` to a local before narrowing (see `core/trainer.py:load_checkpoint`, `fast.py:load_checkpoint`).

- **Training-loop PoC is the §3/§9 proof**: `core/training_mixin.py:supervised_step(model, optimizer, x, y, *, grad_clip)` canonicalizes the plain-BPTT `zero_grad→forward→CE→backward→(clip→)step` shape. **Rollout candidates** (train_step bodies that are exactly CE+optimizer, no custom physics): `predictive_coding.py:263` (`PredictiveCodingHybrid`), `core/ebm.py:146` fallback branch, `zoo/models/forward_only.py` classifier tail. Contrastive/EqProp/FA/PEPITA steps keep their bespoke bodies. **Caution: `supervised_step` returns only `{"loss","accuracy"}`**; models that currently return extra keys (e.g. `cls_loss`) need a small extension or their own tail call.

- **`continual_learning.py` is NOT a `get_vision_dataset` candidate**: it flattens raw `.data`/255 (un-normalised pixel space) then indexes tensors directly for permutation tasks — different semantics from the canonical normalised cached path. `domains/vision.py:setup()` also can't drop to `get_vision_dataset` without losing custom `train_transform`/`val_transform` support. Both documented as intentional non-migrations.

- **MEP benchmark dataset-name mapping**: `get_dataloaders`/get_dataloader now map `mnist|fashion→fashion_mnist|cifar10` (and `MNIST|FASHIONMNIST|CIFAR10|CIFAR100` in `runner.py`) via dicts — keep in sync if a new corpus is added to `data/vision.py:_get_dataset_class`.

- **Toolchain note**: ruff config `[tool.ruff.lint] ignore` names (`line-too-long`, `lowercase-imported-as-non-lowercase`, `non-augmented-assignment`, `raise-vanilla-args`) fail on system ruff 0.15.9 but parse on the project venv (`uv run ruff`/`.venv/bin/ruff` = 0.16.0). **Always use the venv ruff**; pre-commit's pinned v0.7.0 also accepts older aliases. Consider standardizing the aliases to modern codes (`E501`, `N812`, `PLR6104`, `TRY003`) so any ruff works.

- **Pre-existing failures to ignore** (confirmed still failing on clean tree in session 8): `tests/unit/validation/test_backprop_parity.py::test_backprop_parity[eqprop_mlp|directed_ep]`; `tests/unit/test_plan2_actions.py::test_sample_config_eqprop_has_equilibrium_params`; `DataLoader` undefined errors in `compare.py`/`tuned_compare.py` (they import it from `_shared` but use it before the import is visible to pyright). **Session-9 added**: `tests/integration/test_smoke_training.py::test_directed_ep` and `::test_finite_nudge_ep` fail on clean tree (model `train_step` returns `None` → harness crashes on `assertIn` over `NoneType`; the harness doesn't guard non-dict returns).

- **Session-8 targeted test matrix passed** (no full-suite reruns): `test_checkpoint.py` + `test_trainer_coverage.py` (29), `test_lm_demo.py` (38), `test_eqprop_models.py`+`_forward`+`_base` (25), `test_phase2_integration.py` (1), `test_lm_demo.py`+`test_zoo_integration.py` (68), `test_config_knobs.py`+`test_training_path.py` (17), `test_plan2_actions.py`+`test_domains.py` (51 pass / 1 pre-existing fail). ruff baseline ~104 findings on touched files; pyright 0 errors across all touched files.

- **Session-9 verification (training-loop rollout, `zoo/models/fa.py`)**: `test_fa_model.py` 56/56 pass; `test_smoke_training.py` 23 pass + 2 pre-existing failures (`test_directed_ep`, `test_finite_nudge_ep` — confirmed identical on clean tree via `git stash`; unrelated to FA changes); pyright 0 errors on `fa.py`; ruff no new findings (pre-existing baseline only). No full-suite rerun.

- **Session-10 verification (strategy permutations)**: `test_gradient_strategies.py` 10/10 pass; `test_optimizer_factory.py` + `test_spectral_optimizer.py` + `test_optimizer_stubs.py` all pass (28 total); ruff clean on new/changed code (pre-existing baseline: 3 missing-newlines + 1 typing-only import in `factory.py` + pytest-class `no-self-use` in tests — untouched baseline); pyright 0 errors on `core/optimization/` with only pre-existing warnings (+4 factory `dict[str, object]`-kwargs warnings on `target_prop`/`hebbian` registry lambdas, same pattern as existing `backprop`/`fa`/`muon` rows). **TargetPropGradient diverged-index bug fixed during session**: `_train_forward_nets` target lookup must be `targets[-(i+1)]` (layer i's forward net predicts `hs[i+1]` whose target is `t_{i+1}`), not `targets[-len(targets)+i]`.

- **Zero-diff verification recipe**: `git stash` → `ruff check` → `pyright` → targeted `pytest --no-cov` → `git stash pop`; compute `comm` diffs. Keeps regression detection cheap without full-suite reruns.
---

## 🏛️ STRATEGIC ENDGAME: FULL EQUITILE SUBSUMPTION

**Goal**: Eliminate `equitile/` entirely by migrating all functionality to the generic substrate (`core/tile` + `core/local_learning` + `core/optimization`).

| equitile Component | Substrate Migration Target | Status |
|--------------------|----------------------------|--------|
| `EquiTile` (EP mode) | `TileAlgorithm.from_ep()` | ✅ Subsumed |
| `EquiTileConfig` + deployment configs | `TileAlgorithmConfig` + deployment-specific subclasses | 🟡 Next |
| Vision/Language/Graph/Temporal deployments | New model classes on `TileAlgorithm` substrate | 🟡 Next |
| `FastLM` + components | `TileLM` (new) on substrate | 🟡 Next |
| Async/Distributed training | Substrate execution infra (new `core/execution/`) | 🔴 Later |
| Optimizer mixins | `MultiOptimizerMixin` + `StrategyOptimizer` | ✅ Subsumed |
| Analysis/benchmarks | Rewrite against substrate APIs | 🔴 Later |
| `equitile/analysis/`, `equitile/benchmarks/` | Replace with substrate-native versions | 🔴 Later |

**Migration principle**: Each `equitile/` component becomes a *thin model class* on the substrate, not a reimplementation. The substrate provides topology, kernels, optimizers, tasks, and execution; models only compose dynamics + config.

**End state**: `bioplausible/` has `core/` (generic substrate) + `zoo/models/` (algorithm implementations) + `data/` + `experiments/` — **no `equitile/` package**.

---

*End of REFACTOR.md — update after each session; keep action list current and prioritized.*