# REFACTOR.md — Strategic Refactoring Plan for bioplausible

**Codebase**: 316 Python files, ~41K lines  
**Goal**: Maximize size reduction via deduplication, DRY, structural consolidation  
**Completed**: ~3,080 lines saved (7.5%) across 90+ files  
**Status**: EquiTile generification complete (core substrate extracted); optimizer factory at 38/60 sites; config unification in progress; validation tracks boilerplate centralized.

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
| EquiTile Shim Removal | 4 shims deleted; `equitile` imports `core.tile`/`core.local_learning` directly | ~70 |
| Enhanced Optimizer Fold | `_internal/enhanced.py:_setup_optimizers` folded into `MultiOptimizerMixin` with hooks | ~25 |
| Feature Extractors | Generic extractors → `core/tile/feature_extractors.py`; EquiTile layers param'd with `TileModelFactory`; `core → equitile` edge eliminated | ~450 |
| Validation Track Boilerplate | `validation/tracks/_base.py` — `track_header()` + `build_track_result()`; 18 tracks migrated | ~0 (net; single-sourced assembly) |

**Total verified reduction**: ~3,080 lines (7.5%)

---

## 📊 PROGRESS SUMMARY

| Initiative | Target | Done | Remaining | Status |
|------------|--------|------|-----------|--------|
| Optimizer Factory | ~60 sites | 38 | 22 | 🟡 In progress |
| Config Unification | ~12 classes | 2 | 10 | 🟡 In progress |
| EquiTile Generification | 6 components | 6 | 0 | ✅ Complete |
| Storage Unification | 2 systems | 1.5 | 0.5 | 🟢 Mostly done |
| Data Transforms | ~8 duplicate sites | 0 | 8 | 🔴 Not started |
| Metrics Consolidation | ~10 classes | 1 | 9 | 🔴 Not started |
| Training Loop Infra | ~20 implementations | 0 | 20 | 🔴 Not started |
| Strategy Optimizer Generification | 4 strategy types | 0 | 4 | 🔴 Not started |

---

## 🎯 NEXT IMMEDIATE ACTIONS (Priority Order)

### 1. Data Transforms Consolidation (~150 lines, 8 sites) — **QUICK WIN**
Create `data/transforms.py` with canonical transforms; migrate all inline `transforms.Compose` calls.

**Sites to migrate** (grep `transforms.Compose`):
- `validation/tracks/tradeoff_tracks.py` (MNIST, 2×)
- `zoo/mep/benchmarks/continual_learning.py` (MNIST)
- `zoo/mep/benchmarks/niche_benchmarks.py` (MNIST, 2×)
- `zoo/mep/benchmarks/runner.py` (CIFAR10, 2×)
- `zoo/mep/benchmarks/_shared.py` (MNIST, KMNIST, CIFAR10, 4×)
- `domains/vision.py` (MNIST/CIFAR, 2×)

**Deliverable**: `data/transforms.py` exporting `MNIST_TRANSFORM`, `CIFAR10_TRANSFORM`, `CIFAR100_TRANSFORM`, `SVHN_TRANSFORM`, `create_dataloader()`.

---

### 2. Optimizer Factory — Final Sweep (~150 lines, 22 sites)
Migrate remaining direct `torch.optim` calls to `create_optimizer(model_or_params, OptimizerConfig)`.

**High-value sites** (preserve original `weight_decay` — factory default is `1e-4`):
| File | Sites | Notes |
|------|-------|-------|
| `zoo/models/eqprop/holomorphic_ep.py` | 1 | SGD |
| `zoo/models/eqprop/_energy.py` | 1 | SGD |
| `zoo/models/eqprop/eqprop_diffusion.py` | 1 | Adam default param |
| `zoo/models/base.py` | 1 | Adam |
| `zoo/models/predictive_coding.py` | 1 | Adam |
| `zoo/propagators/eqprop.py` | 1 | Adam |
| `zoo/mep/benchmarks/niche_benchmarks.py` | 2 | SGD + Adam lambdas |
| `zoo/mep/benchmarks/ewc_baseline.py` | 1 | SGD |
| `zoo/nebc_base.py` | 1 | Adam |
| `graph/training.py` | 2 | Adam (param_list) |
| `equitile/benchmarks/rigorous.py` | 1 | AdamW |
| `equitile/benchmarks/compare_nanoGPT.py` | 1 | AdamW |
| `equitile/language/canonical.py` | 1 | AdamW |
| `equitile/validate.py` | 3 | AdamW (1e-3) |
| `sklearn_interface.py` | 1 | Adam |
| `hyperopt/experiment.py` | 1 | Dynamic `getattr(torch.optim, name)` — **assess only** |
| `validation/tracks/hardware_tracks.py` | 1 | SGD (0.01) |
| `core/trainer.py` | 1 | **DO NOT MIGRATE** — dynamic `opt_cls` via `getattr` is already config-driven and more expressive |

**Rule**: Pass `weight_decay=0.0` explicitly where original used torch default (0.0). The factory default `1e-4` silently changes training (broke `test_backprop_parity[forward_forward]`).

---

### 3. Config Unification — Next Cheapest Trainer Families (~500 lines)
Pattern proven: frozen runtime configs in `config/unified.py` + `load_config`/`save_config` helpers.

**Next targets** (in priority order):
1. **`equitile/lm/components.py:FastLMConfig`** — canonical for `lm/fast_lm.py`; `language/fast.py` variant is genuinely different architecture (pre-norm + sigmoid gating vs MoT + SwiGLU) — **leave both, dedup only shared fields**.
2. **`zoo/mep/benchmarks/tuned_compare.py:OptimizerConfig`** + `config/schema.py:OptimizerConfig` — per-algorithm configs with different fields (`gamma`, per-family values); reconcile only shared subset if any.
3. **`equitile/utils/reproducibility.py:ExperimentConfig`** + `experiments/utils.py:ExperimentConfig` — different purposes (seed+dicts vs model/optimizer/runner); no merge, but standardize on `BaseConfig` pattern.
4. **`config/schema.py:TrainingConfig`** (OmegaConf structured) — different trainer (log/save/early-stop knobs) from LM one; leave as-is.

**Migration steps per config**:
1. Define frozen runtime config in `config/unified.py` (extend `BaseConfig` where fields allow)
2. If YAML needed, add non-frozen `BaseStructuredConfig` mirror with `to_internal()`
3. Update imports: `from bioplausible.config.unified import BaseConfig, load_config, save_config`
4. Delete obsolete config class file
5. Verify: `ruff check --fix . && pyright . && pytest --cov`

---

### 4. Metrics Class Consolidation (~200 lines, 10 classes → 1 hierarchy)
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
- Rename `equitile/lm/training.py:TrainingMetrics` → `LMTrainingMetrics` (avoid collision with `core.trainer.TrainingMetrics`)
- Audit `BenchmarkMetrics` vs `TrainingMetrics` field naming (`train_acc`/`val_acc` vs `train_accuracy`/`val_accuracy`) — reconcile if shared readers emerge (REFACTOR.md §7)
- No forced merger of domain-specific classes; keep `BaseMetrics` as the only shared base.

---

### 5. EquiTile Generification — Phase 3/4 (Enable Reuse)
**Substrate complete** (`core/tile/` + `core/local_learning/`). Now prove reuse.

**Immediate opportunities**:
- **Type-clean `core/local_learning/mixins.py:20`** — drop `TYPE_CHECKING` import of `equitile.core.config.EquiTileConfig`; use `LocalLearningConfig` or a narrow `Protocol` exposing `learning_rate`/`importance_lr`/`mode` so `core/*` is equitile-free at type-check time.
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
| **Vision Data Loading** | `data/vision.py` already canonical; migrate `validation/tracks/tradeoff_tracks.py` direct `datasets.MNIST` calls to `get_vision_dataset()` | ~50 | Low |
| **Toy Dataset Duplication** | `_load_toy_dataset` in `data/vision.py` vs `validation/tracks/_signal_probe.py` | ~30 | Low |
| **Checkpoint/Serialization** | `core/checkpoint.py` + `core/checkpoint_mixin.py` + `hyperopt/storage.py` save/load — unify serialization helpers | ~80 | Medium |
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
| Data Transforms (§1) | ~150 lines |
| Optimizer Factory (§2) | ~150 lines |
| Config Unification (§3) | ~500 lines |
| Metrics Consolidation (§4) | ~100 lines |
| EquiTile Reuse (§8) | ~300 lines (new algorithms, not dedup) |
| Storage (§6) | ~50 lines |
| Strategy Optimizer Generification (§7) | ~200 lines + permutations |
| Training Loop (§9) | ~200 lines |
| Additional (§10) | ~260 lines |
| **Total Additional** | **~1,910 lines (4.7%)** |
| **Cumulative** | **~4,990 lines (12.2%)** |

**Key multiplier**: EquiTile generification is **complete at substrate level** — `core/tile` + `core/local_learning` are importable by any algorithm. `TileFA`, `TileTargetProp`, `HierarchicalPC`, `TileSNN`, `TileGNN` now need only their own model classes, not replicated substrate code.

---

## 🗂️ SESSION HISTORY (Condensed)

| Session | Date | Focus | Key Result |
|---------|------|-------|------------|
| 1 | 2026-08-10 | Optimizer factory (26 sites), LM TrainerConfig, storage shared types | Factory pattern proven; `LMTrainingConfig` unified |
| 2 | 2026-08-10 | Optimizer param-subset support (9 more sites), `epoch_metrics` schema unified, `get_lm_dataset` sink | Factory drives 26 sites; storage schemas unified |
| 3 | 2026-08-10 | EquiTile generification Phase 1 (core/tile, core/local_learning), optimizer factory 12 more sites (38 total) | Substrate extracted; 4 shims retained temporarily |
| 4 | 2026-08-10 | EquiTile generification Phase 2/3 (shims deleted, enhanced fold, feature-extractor decoupling), validation tracks `_base.py` | **Zero `core → equitile` deps**; 18 tracks on shared boilerplate |

---

*End of REFACTOR.md — update after each session; keep action list current and prioritized.*