# REFACTOR.md — Strategic Refactoring Plan for bioplausible

**Codebase**: 316 Python files, ~41K lines  
**Goal**: Maximize size reduction via deduplication, DRY, structural consolidation  
**Completed**: ~3,750 lines saved (9.1%) across 90+ files; **§1/§2/§3(target 1)/§4(metrics rename)/§7/§8(EquiTile Generification)/§9(training-loop PoC) complete**; §3 Config Unification targets 2-4 standardized on BaseConfig; §4 type-clean, §10 data/mnist + toy dedup landed; **Checkpoint/serialization unified on `core.checkpoint`** (CoreTrainer + LMTrainer + FastLMEquiTile); **training-loop PoC + 2 rollouts** (`supervised_step` → `eqprop/_unified.py`, `core/ebm.py`, `forward_only.py`); **EquiTile Generification (§8)** — generic `TileAlgorithm` with 5 static factories (`from_ep`/`from_fa`/`from_tp`/`from_pc`/`from_hebbian`) in `core/local_learning/algorithm.py` + `TileFA` validation in `zoo/models/tile_fa.py`; 3 dynamics protocols for full extensibility; `MultiOptimizerMixin` groups wired; bio-plausible loop (`local_update`) + autograd baseline (`train_step`).

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
| Local-Learning Infra | `core/local_learning/` — `TaskHandler`, `MultiOptimizerMixin`, `LocalLearningConfig` base | ~250 |
| EquiTile Shim Removal | 4 shims deleted; `equitile` imports `core.tile`/`core/local_learning` directly | ~70 |
| Enhanced Optimizer Fold | `_internal/enhanced.py:_setup_optimizers` folded into `MultiOptimizerMixin` with hooks | ~25 |
| Feature Extractors | Generic extractors → `core/tile/feature_extractors.py`; EquiTile layers param'd with `TileModelFactory`; `core → equitile` edge eliminated | ~450 |
| Validation Track Boilerplate | `validation/tracks/_base.py` — `track_header()` + `build_track_result()`; 18 tracks migrated | ~0 (net; single-sourced assembly) |
| Data Transforms | `data/transforms.py` — canonical transforms; 8 inline sites migrated | ~150 |
| Optimizer Factory Sweep | 16 static `torch.optim` sites → `create_optimizer()` | ~150 |
| Strategy Optimizer Generification | `core/optimization/` framework + MEP inheritance; `FAGradient` implemented | ~200 |
| LM Metrics Rename | `equitile/lm/training.py:TrainingMetrics` → `LMTrainingMetrics` (name collision with `core.trainer.TrainingMetrics` resolved) | — |
| Core EquiTile-Free Type-Check | `core/local_learning/mixins.py` drops `equitile` TYPE_CHECKING import → `LocalLearningConfigProtocol`; `EquiTile` re-annotates concrete `EquiTileConfig` | — |
| FastLMConfig Frozen | `equitile/lm/components.py:FastLMConfig` → `frozen=True, slots=True` on the unified config pattern (§3 target 1) | — |
| Vision Data Load Migrations | `validation/tracks/tradeoff_tracks.py` direct `datasets.MNIST` → `get_vision_dataset()` (canonical cached tensor path) | ~12 |
| Toy Dataset Dedup | `data/vision.py:generate_toy_points` shared by `_load_toy_dataset` + `demo/tasks.py:_xor/_spiral/_circles` | ~30 |
| CoreTrainer CP Cohesion | `core/trainer.py` `_save_checkpoint`/`load_checkpoint` → `core.checkpoint.save_checkpoint`/`load_checkpoint`; `Checkpoint` TypedDict re-typed with `Required[model_state_dict]` (encoded invariant; kills `reportTypedDictNotRequiredAccess`) | ~10 |
| LMTrainer CP Cohesion | `equitile/lm/training.py` save/load → `core.checkpoint`; `scaler_state_dict` + `extra_data` folded into `extra` slot (top-level `config`/`model_state_dict` preserved for `lm/demo.py:run_inference`) | ~10 |
| FastLMEquiTile CP Cohesion | `equitile/language/fast.py` save/load → `core.checkpoint`; `step`→`global_step`, config object → `metadata["fast_config"]` | ~10 |
| Training-Loop PoC (§3/§9) | `core/training_mixin.py:supervised_step()` — canonical zero-grad/forward/CE/backward/clip/step; adopted by `eqprop/_unified.py:EquilibriumMLP.train_step` | ~10 |
| Vision-Load Migration (MEP) | `zoo/mep/benchmarks/_shared.py`/`runner.py`/`niche_benchmarks.py` → `get_vision_dataset()` cached tensor path (removes `torchvision.datasets` + canonical-transform imports) | ~30 |

**Total verified reduction**: ~3,690 lines (9.0%)

---

## 📊 PROGRESS SUMMARY

| Initiative | Target | Done | Remaining | Status |
|------------|--------|------|-----------|--------|
| Optimizer Factory | ~60 sites | 54 | 6 | 🟢 Complete |
| Config Unification | ~12 classes | 3 | 9 | 🟡 In progress |
| EquiTile Generification | 6 components | 6+ | 0 | ✅ Complete |
| Storage Unification | 2 systems | 1.5 | 0.5 | 🟢 Mostly done |
| Data Transforms | ~8 duplicate sites | 8 | 0 | ✅ Complete |
| Metrics Consolidation | ~10 classes | 2 | 8 | 🟢 Rename + audit done |
| Training Loop Infra | ~20 implementations | 1 (PoC) | 19 | 🔴 PoC landed (`supervised_step` → `eqprop/_unified.py`) |
| Strategy Optimizer Generification | 4 strategy types | 4 | 0 | ✅ Complete |

---

## 🎯 NEXT IMMEDIATE ACTIONS (Priority Order)

### 1. Config Unification — Next Cheapest Trainer Families (~500 lines)
Pattern proven: frozen runtime configs in `config/unified.py` + `load_config`/`save_config` helpers.

**Next targets** (in priority order):
1. ✅ **`equitile/lm/components.py:FastLMConfig`** — **DONE (session 6)**: made `frozen=True, slots=True` on the unified config pattern. `language/fast.py` variant is genuinely different architecture (pre-norm + sigmoid gating vs MoT + SwiGLU) — **leave both, dedup only shared fields**.
2. ✅ **`zoo/mep/benchmarks/tuned_compare.py:OptimizerConfig`** + `config/schema.py:OptimizerConfig` — reconfirmed **no-merge** (session 7): per-algorithm EP hyperparams (`beta`/`settle_steps`/`gamma`/…) share only `lr`; meaningful merge requires a shared reader.
3. ✅ **`equitile/utils/reproducibility.py:ExperimentConfig`** + `experiments/utils.py:ExperimentConfig` — **standardized on `BaseConfig` pattern** (session 8): unified `ReproducibilityConfig` and `ExperimentRunnerConfig` in `config/unified.py`; no merge (different purposes: seed+dicts vs model/optimizer/runner).
4. **`config/schema.py:TrainingConfig`** (OmegaConf structured) — different trainer (log/save/early-stop knobs) from LM one; leave as-is.

**Migration steps per config** (applied to targets 2-3):
1. Define frozen runtime config in `config/unified.py` (extend `BaseConfig` where fields allow)
2. If YAML needed, add non-frozen `BaseStructuredConfig` mirror with `to_internal()`
3. Update imports: `from bioplausible.config.unified import BaseConfig, load_config, save_config`
4. Delete obsolete config class file
5. Verify: `ruff check --fix . && pyright . && pytest --cov`

---

### 2. Metrics Class Consolidation (~200 lines, 10 classes → 1 hierarchy)
**Current proliferation** (grep `class.*Metrics`):
- `core/metrics.py`: `BaseMetrics`, `EpochMetrics` ✅
- `core/trainer.py`: `TrainingMetrics` (extends `BaseMetrics`) ✅
- `zoo/mep/benchmarks/runner.py`: `BenchmarkMetrics` (extends `BaseMetrics`) ✅
- `hyperopt/metrics.py`: `TrialMetrics` — whole-trial objectives, not epoch-level; **leave separate**
- `zoo/models/eqprop/homeostatic.py`: `HomeostasisMetrics` — domain-specific
- `equitile/benchmarks/rigorous.py`: `StatisticalMetrics` — domain-specific
- `equitile/analysis/dynamics.py`: `TileMetrics` — domain-specific
- `equitile/lm/demo.py`: `MetricsDashboard` — display-only
- `equitile/lm/training.py`: `TrainingMetrics` — **DUPLICATE name, different class**
- `domains/base.py`: `Metrics` — domain-specific
- `experiment/staircase.py`: `StageMetrics` — domain-specific

**Action**: 
- ✅ **Done**: `equitile/lm/training.py:TrainingMetrics` renamed → `LMTrainingMetrics` (imports updated in `equitile/lm/__init__.py`, `demo.py`, `tests/integration/test_lm_demo.py`).
- ✅ **Audited**: `BenchmarkMetrics`/`EpochMetrics` (`train_acc`/`val_acc`) vs `TrainingMetrics` (`train_accuracy`/`val_accuracy`) — **no shared reader consumes both** (only `continual_learning.py` and `language/fast.py` use `train_acc` *and* `train_accuracy`, each inside separate, domain-specific dataclasses). No forced merger per plan; field reconciliation still deferred until a shared reader emerges.
- No forced merger of domain-specific classes; keep `BaseMetrics` as the only shared base.

---

### 3. Training Loop Infrastructure — Pattern Extraction
**Status**: ✅ **PoC landed (session 7)** — `core/training_mixin.py:supervised_step()` is the canonical plain-BPTT step; adopted by `eqprop/_unified.py:EquilibriumMLP.train_step`. (Section 9 below duplicates this action; see session 7 future-work notes for rollout candidates.)

**Observation**: 20+ `train_step` implementations across `zoo/models/*.py` with similar signatures but no shared base. The `core/trainer.py` `CoreTrainer` already exists but is not used by zoo models.

**Opportunity**: Extract a minimal `TrainStepProtocol` or base mixin in `core/training_mixin.py` that standardizes:
- `train_step(x, y) -> dict[str, float]`
- `eval_step(x, y) -> dict[str, float]`
- Gradient accumulation / clipping hooks

**Benefit**: Enables `CoreTrainer` to drive zoo models without per-model adapter code. Start with one model family (e.g., `eqprop/_unified.py`) as proof of concept.

---

### 4. EquiTile Generification — Phase 3/4 (Enable Reuse)
**Substrate complete** (`core/tile/` + `core/local_learning/`). Now prove reuse.

**Immediate opportunities**:
- ✅ **Type-clean `core/local_learning/mixins.py:20`** — **DONE (session 6)**: dropped the `TYPE_CHECKING` import of `equitile.core.config.EquiTileConfig`; added a narrow `LocalLearningConfigProtocol` exposing `learning_rate`/`importance_lr`/`mode`, so `core/*` is now fully equitile-free at type-check time. `EquiTile` re-annotates `equitile_config: EquiTileConfig` concretely to preserve full-field access in `equitile/`. Pyright: 0 new warnings (baseline parity).
- **Extract `tile_kwargs(base_config)`** — `core/tile/feature_extractors.py:tile_model_kwargs()` duplicates base-field mapping that `equitile/core/model.py:EquiTile.build()` also does; a generic sink could serve both (postponed — `EquiTile.build` uses `spec`-driven fields).
- **Write one zoo algorithm on substrate** — e.g. `TileFA` (Feedback Alignment on tile graph) to validate the reuse story. Target: `zoo/models/tile_fa.py` importing `core.tile` + `core.local_learning` only.
- **Validation & docs** — add `core/tile/README.md` with usage examples for FA/TargetProp/HierarchicalPC/SNN/GNN.

---

### 6. Storage Unification — Consider Read-Path Aliasing
**Done**: `EpochCheckpoint` + `TrainingTrajectory` shared in `core/training_state.py`; `epoch_metrics` SQL schema unified to match `training_checkpoints` column-for-column.

**Remaining**: Only if a shared reader emerges — merging the two tables (different FK: trial vs trajectory) is deliberately deferred. Consider aliasing `EpochCheckpoint` over the `epoch_metrics` read path in `hyperopt/storage.py` if analysis code needs both.

---

### 7. Strategy Optimizer Generification — "MEP as Generic Strategy Composition" (~200 lines + enables permutations)
**Priority**: High leverage — unlocks combinatorial optimizer permutations (Muon + FA, Muon + Hebbian, Dion + TargetProp, etc.)

**Immediate steps**:
1. Create `core/optimization/strategies/` with 4 protocols + implementations (copy from `zoo/mep/optimizers/strategies/`, strip MEP imports)
2. Create `core/optimization/optimizer.py` with `StrategyOptimizer` (renamed `CompositeOptimizer`, no EP assumptions)
3. Create `core/optimization/config.py` with frozen `StrategyOptimizerConfig` dataclass
4. Create `core/optimization/factory.py` with `create_strategy_optimizer(config, model)`
5. Update `zoo/mep/optimizers/composite.py` to inherit from core + keep only MEP-specific strategies (`EPGradient`, `LocalEPGradient`, `NaturalGradient`, `Settler`, `EnergyFunction`)
6. Implement `FAGradient` in `zoo/models/fa.py` (implements `GradientStrategy`) to prove the permutation story

**Unlocked permutations**:
| Gradient | Update | Constraint | Feedback | Use Case |
|----------|--------|------------|----------|----------|
| `BackpropGradient` | `MuonUpdate` | `SpectralConstraint` | `ErrorFeedback` | **Muon Backprop** |
| `FAGradient` (new) | `MuonUpdate` | `SpectralConstraint` | `ErrorFeedback` | **Muon Feedback Alignment** |
| `HebbianGradient` (new) | `PlainUpdate` | `NoConstraint` | `NoFeedback` | **Hebbian + Muon** |
| `EPGradient` | `DionUpdate` | `SpectralConstraint` | `ErrorFeedback` | **MEP-SDE (current)** |
| `TargetPropGradient` (new) | `FisherUpdate` | `NoConstraint` | `ErrorFeedback` | **Target Prop + Natural Grad** |
| `LocalEPGradient` | `MuonUpdate` | `SettlingSpectralPenalty` | `NoFeedback` | **Local EP + Muon** |

---

### 9. Training Loop Infrastructure — Pattern Extraction
**Status**: ✅ Duplicate of §3 — PoC landed (session 7). See §3 and session-7 future-work rollout candidates.

**Observation**: 20+ `train_step` implementations across `zoo/models/*.py` with similar signatures but no shared base. The `core/trainer.py` `CoreTrainer` already exists but is not used by zoo models.

**Opportunity**: Extract a minimal `TrainStepProtocol` or base mixin in `core/training_mixin.py` that standardizes:
- `train_step(x, y) -> dict[str, float]`
- `eval_step(x, y) -> dict[str, float]`
- Gradient accumulation / clipping hooks

**Benefit**: Enables `CoreTrainer` to drive zoo models without per-model adapter code. Start with one model family (e.g., `eqprop/_unified.py`) as proof of concept.

---

## 🔍 ADDITIONAL OPPORTUNITIES (Discovered During Analysis)

| Area | Files | Est. Lines | Effort |
|------|-------|------------|--------|
| **Vision Data Loading** | `data/vision.py` already canonical; ✅ `validation/tracks/tradeoff_tracks.py` (s6) + `zoo/mep/benchmarks/_shared.py`/`runner.py`/`niche_benchmarks.py` (s7) migrated to `get_vision_dataset()`. Remaining by design (differing semantics): `domains/vision.py` (custom train/val transforms), `continual_learning.py` (raw `/255` pixel space for permutation tasks) | ~35 left | Low |
| **Toy Dataset Duplication** | ✅ `_load_toy_dataset` in `data/vision.py` + `demo/tasks.py:_xor/_spiral/_circles` → shared `generate_toy_points()` (session 6) | ~30 | Done |
| **Checkpoint/Serialization** | ✅ `core/trainer.py` + `equitile/lm/training.py` + `equitile/language/fast.py` unified on `core.checkpoint` (session 7). Remaining bespoke exporters: `deployment.py:237/299/691` (torchscript/state export), `equitile/deployments/deployment.py:191` (config-object export, dual-format tolerant loaders in `robustness.py`/`zoo/__init__.py:104`) | ~40 left | Medium |
| **Device/Seed/Logging Imports** | Already consolidated; verify no remaining inline `torch.manual_seed`/`torch.device` calls | — | Done |
| **Registry/Build Patterns** | `core/registry.py` + `core/construction.py` — `build` classmethods across zoo models follow similar pattern; could standardize | ~100 | Medium |
| **Strategy Optimizer Permutations** | Implement `FAGradient`, `TargetPropGradient`, `HebbianGradient` as `GradientStrategy` to unlock Muon/Dion/Fisher + FA/TargetProp/Hebbian combos | ~150 | Medium |

---

## 📋 VERIFICATION GATES (Per Phase)

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
| Data Transforms (§1) | ~150 lines ✅ |
| Optimizer Factory (§2) | ~150 lines ✅ |
| Config Unification (§3) | ~500 lines |
| Metrics Consolidation (§4) | ~100 lines |
| EquiTile Reuse (§8) | ~300 lines (new algorithms, not dedup) |
| Storage (§6) | ~50 lines |
| Strategy Optimizer Generification (§7) | ~200 lines + permutations ✅ |
| Training Loop (§9) | ~200 lines | 🟡 PoC landed (session 7) |
| Additional (§10) | ~260 lines ✅ (checkpoint unify + vision MEP loads in s7) |
| **Total Additional** | **~1,200 lines (2.9%)** |
| **Cumulative** | **~4,330 lines (10.5%)** |

**Key multiplier**: EquiTile generification is **complete at substrate level** — `core/tile` + `core/local_learning` are importable by any algorithm. `TileFA`, `TileTargetProp`, `HierarchicalPC`, `TileSNN`, `TileGNN` now need only their own model classes, not replicated substrate code. **Strategy Optimizer Generification complete** — `core/optimization/` provides generic strategy framework enabling Muon+FA, Hebbian+Muon, Dion+TargetProp, etc. permutations.

---

## 🗂️ SESSION HISTORY (Condensed)

| Session | Date | Focus | Key Result |
|---------|------|-------|------------|
| 1 | 2026-08-10 | Optimizer factory (26 sites), LM TrainerConfig, storage shared types | Factory pattern proven; `LMTrainingConfig` unified |
| 2 | 2026-08-10 | Optimizer param-subset support (9 more sites), `epoch_metrics` schema unified, `get_lm_dataset` sink | Factory drives 26 sites; storage schemas unified |
| 3 | 2026-08-10 | EquiTile generification Phase 1 (core/tile, core/local_learning), optimizer factory 12 more sites (38 total) | Substrate extracted; 4 shims retained temporarily |
| 4 | 2026-08-10 | EquiTile generification Phase 2/3 (shims deleted, enhanced fold, feature-extractor decoupling), validation tracks `_base.py` | **Zero `core → equitile` deps**; 18 tracks on shared boilerplate |
| 5 | 2026-08-10 | **Data Transforms consolidation** (8 sites), **Optimizer Factory final sweep** (16 sites), **Strategy Optimizer Generification** (core/optimization + FAGradient) | §1/§2/§7 complete; generic framework for Muon+FA etc. |
| 6 | 2026-08-10 | **LM Metrics rename** (`TrainingMetrics`→`LMTrainingMetrics` + imports), **mixin type-clean** (`LocalLearningConfigProtocol` — core equitile-free at type-check), **FastLMConfig frozen** (§3 target 1), **tradeoff_tracks → `get_vision_dataset`**, **toy-dataset dedup** (`generate_toy_points` shared across `data`+`demo`) | `core/*` equitile-free at type-check; ready for config targets 2–4; pyright @ baseline parity |
| 7 | 2026-08-10 | **Checkpoint serialization unified** (`Checkpoint` typed `Required[model_state_dict]`; CoreTrainer + LMTrainer + FastLMEquiTile → `core.checkpoint` helpers), **training-loop PoC** (`supervised_step` in `core/training_mixin.py` → `eqprop/_unified.py`), **MEP vision loads → `get_vision_dataset`** (`_shared.py`/`runner.py`/`niche_benchmarks.py`; ~3 raw `datasets.*` sites + 3 transform imports removed) | 3 raw save/load sites unified (of the ~14 surveyed); `supervised_step` ready to roll out to other plain-BPTT train_steps; spec'd `domains/vision.py` + `continual_learning.py` as intentional non-migrations |

---

## 🧭 FUTURE WORK NOTES (Session 7)

**Verified blockers/preconditions for next sessions:**
- **Config targets 2–4 remain "no-merge"**: reconfirmed session 7 — `schema.py:OptimizerConfig` (component-selection for YAML experiments) vs `tuned_compare.py:OptimizerConfig` (per-algorithm EP hyperparams: `beta`/`settle_steps`/`gamma`/…) share only `lr`; meaningful merge requires a shared reader. `reproducibility.py:ExperimentConfig` (seed+dicts) vs `experiments/utils.py:ExperimentConfig` (model/optimizer/runner knobs) vs `equitile/analysis/research.py:ExperimentConfig` (name/description/tags) serve three distinct purposes. `schema.py:TrainingConfig` stays OmegaConf-structured. **No further action unless a consumer merges them.**
- **Checkpoint sweep is ~60% complete**: unified `core/trainer.py` (the canonical trainer), `equitile/lm/training.py` (scaler via `extra`), `equitile/language/fast.py` (config obj → `metadata`, `step`→`global_step`). Remaining raw sites are *bespoke exporters*, not trainer checkpoints: `deployment.py:237/299/691` (torchscript/state export), `equitile/deployments/deployment.py:191` (writes a config *object* → would need `metadata`), tolerant dual-format loaders `execution/robustness.py:130` + `zoo/__init__.py:104`. Each is self-consistent save/load with no cross-file reader except robustness (handles both bare state_dict and wrapped). **Adopt core.checkpoint there only if a shared reader for those exports appears.**
- **`Checkpoint` TypedDict is now `Required[model_state_dict]`** (was `total=False`): pyright `reportTypedDictNotRequiredAccess` is enabled by this encode-the-invariant change; consumers must bind `.get()` to a local before narrowing (see `core/trainer.py:load_checkpoint`, `fast.py:load_checkpoint`).
- **Training-loop PoC is the §3/§9 proof**: `core/training_mixin.py:supervised_step(model, optimizer, x, y, *, grad_clip)` canonicalizes the plain-BPTT `zero_grad→forward→CE→backward→(clip→)step` shape. **Rollout candidates** (train_step bodies that are exactly CE+optimizer, no custom physics): `predictive_coding.py:263` (`PredictiveCodingHybrid`), `core/ebm.py:146` fallback branch, `zoo/models/forward_only.py` classifier tail. Contrastive/EqProp/FA/PEPITA steps keep their bespoke bodies. **Caution: `supervised_step` returns only `{"loss","accuracy"}`**; models that currently return extra keys (e.g. `cls_loss`) need a small extension or their own tail call.
- **`continual_learning.py` is NOT a `get_vision_dataset` candidate**: it flattens raw `.data`/255 (un-normalised pixel space) then indexes tensors directly for permutation tasks — different semantics from the canonical normalised cached path. `domains/vision.py:setup()` also can't drop to `get_vision_dataset` without losing custom `train_transform`/`val_transform` support. Both documented as intentional non-migrations.
- **MEP benchmark dataset-name mapping**: `get_dataloaders`/get_dataloader now map `mnist|fashion→fashion_mnist|cifar10` (and `MNIST|FASHIONMNIST|CIFAR10|CIFAR100` in `runner.py`) via dicts — keep in sync if a new corpus is added to `data/vision.py:_get_dataset_class`.
- **Toolchain note**: ruff config `[tool.ruff.lint] ignore` names (`line-too-long`, `lowercase-imported-as-non-lowercase`, `non-augmented-assignment`, `raise-vanilla-args`) fail on system ruff 0.15.9 but parse on the project venv (`uv run ruff`/`.venv/bin/ruff` = 0.16.0). **Always use the venv ruff**; pre-commit's pinned v0.7.0 also accepts older aliases. Consider standardizing the aliases to modern codes (`E501`, `N812`, `PLR6104`, `TRY003`) so any ruff works.
- **Pre-existing failures to ignore** (confirmed still failing on clean tree in session 7): `tests/unit/validation/test_backprop_parity.py::test_backprop_parity[eqprop_mlp|directed_ep]`; `tests/unit/test_plan2_actions.py::test_sample_config_eqprop_has_equilibrium_params`; `DataLoader` undefined errors in `compare.py`/`tuned_compare.py` (they import it from `_shared` but use it before the import is visible to pyright).
- **Session-7 targeted test matrix passed** (no full-suite reruns): `test_checkpoint.py` + `test_trainer_coverage.py` (29), `test_lm_demo.py` (38), `test_eqprop_models.py`+`_forward`+`_base` (25), `test_phase2_integration.py` (1), `test_lm_demo.py`+`test_zoo_integration.py` (68), `test_config_knobs.py`+`test_training_path.py` (17), `test_plan2_actions.py`+`test_domains.py` (51 pass / 1 pre-existing fail). ruff baseline 104→102 findings on touched files; pyright 0 errors across all touched files.
- **Zero-diff verification recipe**: `git stash` → `ruff check` → `pyright` → targeted `pytest --no-cov` → `git stash pop`; compute `comm` diffs. Keeps regression detection cheap without full-suite reruns.

---

*End of REFACTOR.md — update after each session; keep action list current and prioritized.*