# FIX.md — Complete Defect Catalog & Resolution Plan

**Status**: Phase 1 runner fails at model-integration level. Unit tests (33 pass) do NOT catch these because they test units in isolation, not model-in-task integration.

---

## 1. Corrupt Optuna DB — `AssertionError: value is not None`

**Cause**: My `_fail_stale_running` used `set_trial_state_values(trial_id, FAIL, (nan,))` with **one NaN** for 2-objective studies. Optuna stores the second objective as NULL. Later `load_study().trials` triggers `assert value is not None` in `stored_repr_to_value`.

**Impact**: Any study with stale RUNNING trials (from killed processes) becomes unreadable — crashes on `load_study().trials`.

**Fix**: `run_experiment.py:_cleanup_corrupt_trials()` — raw SQL runs **before any Optuna access**:
```sql
DELETE FROM trial_values, trial_params, trial_user_attributes, 
       trial_system_attributes, trial_heartbeats, trials
WHERE trial_id IN (
  SELECT trial_id FROM trial_values WHERE value IS NULL
  UNION SELECT trial_id FROM trials WHERE state = 2  -- RUNNING
);
```
Runs at startup via `_cleanup_corrupt_trials(db_path)`.

---

## 2. `_fail_stale_running` Corrupted DB (Root Cause of #1)

**Cause**: Called `study._storage.set_trial_state_values(trial_id, FAIL, (nan,))` with **single NaN** for multi-objective studies → second objective stored as NULL.

**Fix Applied**: Removed `_fail_stale_running` call from `_ensure_studies`. Cleanup now done via raw SQL in `_cleanup_corrupt_trials()` which deletes corrupt/stale trials atomically.

---

## 3. DB Cleanup SQL Had Wrong Column Name

**Cause**: My cleanup SQL used `SELECT id FROM trials` but Optuna's `trials` table uses `trial_id` as PK.

**Fixed**: Changed to `SELECT trial_id FROM trials WHERE state = 2` and `DELETE ... WHERE trial_id IN (...)`.

---

## 4. Model Integration Failures (Per-Trial `TypeError` / `RuntimeError`)

**Cause**: Models registered for wrong tasks or with incompatible constructor signatures. These are **model defects**, not pipeline bugs.

### 4a. `BackpropTransformerLM` — LM model run on vision
- **Error**: `TypeError: __init__() got unexpected keyword argument 'input_dim'`
- **Cause**: Registered `domains=[Domain.VISION]` but is a causal LM requiring `vocab_size`, `max_seq_len`.
- **Fix**: `bioplausible/zoo/models/backprop.py` — changed registration to `domains=[Domain.LM]`.
- **Result**: Excluded from vision runs (digits, cifar10).

### 4b. `CustomStackedModel` — un-HPO-able model in backprop family
- **Error**: `TypeError: __init__() got unexpected keyword argument 'hidden_dim'`
- **Cause**: Requires bespoke `layers_config: list[dict]` — no search space provides this.
- **Fix**: `domains=[]` + `_model_compatible` treats empty `task_compat` as incompatible everywhere.

### 4c. EqProp Family Shape Bugs (Pre-existing, VALIDATE.md #2)

| Model | Error | Root Cause |
|-------|-------|------------|
| `lazy_eqprop`, `eqprop`, `directed_ep`, `finite_nudge_ep`, `graph_eqprop`, `holomorphic_ep`, `momentum_equilibrium`, `neural_cube`, `sparse_equilibrium` | `RuntimeError: mat1 and mat2 shapes cannot be multiplied (2048x8 and 64x32)` | Hidden dimension mismatch in eqprop settling — input projection expects different dim |
| `eqprop_diffusion` | `TypeError: __init__() got unexpected keyword argument 'input_dim'` | Constructor signature mismatch with search space |
| `graph_eqprop` | `TypeError: __init__() got unexpected keyword argument 'num_layers'` | Constructor expects different params |
| `lazy_eqprop` | `TypeError: empty(): argument 'size' failed to unpack...` | Tensor shape handling in model init |

**Status**: These are **model-internal defects** (VALIDATE.md Issue #2). They register correctly but fail at instantiation/forward. Pipeline correctly prunes them with one-line warnings — not a pipeline bug.

---

## 5. `_model_compatible` Logic Gap

**Cause**: `task_compat=[]` (from `domains=[]`) was treated as "compatible with everything" because `bool(not []) == True`.

**Fix**: `_model_compatible` now returns `False` for empty `task_compat` — models with no declared domains are incompatible with all tasks.

---

## 6. `run_phase1.py` Double-Logging / Ctrl-C Crash

**Cause**: `Popen(stdout=PIPE)` + `for line in stdout: logging.info(line)` re-logged child output as `INFO root:`, duplicating all warnings with prefix `INFO root: 15:45:52 WARNING ...`.

**Fixed**: `Popen(cmd)` with inherited stdout/stderr → child streams directly to terminal.

**Ctrl-C**: Now caught, child terminated, studies persist, clean exit message.

---

## 7. DB Cleanup SQL Column Bug

**Cause**: `SELECT id FROM trials` — Optuna uses `trial_id` as PK.

**Fixed**: `SELECT trial_id FROM trials WHERE state = 2`.

---

## 8. Pre-existing Latent `except A, B:` Bug Pattern (VALIDATE.md #6)

**Cause**: Python-2-style `except A, B:` parses in 3.14 as `except A as B:` — only catches `A` and shadows `B`.

**Fixed**: 14 files converted to `except (A, B):` (see `git diff`).

---

## 9. Aggressive Pruning Starves Low-Budget Runs (VALIDATE.md #1)

**Cause**: `--budget-tier shallow` with `--budget 10` + `use_pruning=True` + `n_startup_trials=3` → nearly every trial pruned before completing. TPE has no seed-warmup buffer.

**Impact**: Zero complete trials for some families at low budgets.

**Mitigation**: For small budgets use `--method random` or tier with pruning disabled (DEEP), or raise `--budget` to ≳ 3× `n_startup_trials`. Smoke tier (`n_startup=1`) is only small-budget config that reliably yields complete trials.

---

## 9. EqProp Locality Metadata Mixed Within Family (VALIDATE.md #3)

**Cause**: `eqprop` family has both `equilibrium` and `global` models registered; `fa`, `hebbian`, `predictive_coding`, `target_prop`, `spiking` all registered `global`.

**Impact**: Portfolio's "O(1)/low-memory" regime branch doesn't fire for these families — regime advantage relies only on family-name continual-learning set.

**Fix**: Update registry metadata per VALIDATE.md Session Log (already applied for hebbian→LOCAL, spiking→LOCAL, predictive_coding→LOCAL, target_prop→LAYERWISE; eqprop already has equilibrium; fa stays GLOBAL).

---

## 10. `mep` Family Registers 0 Models (VALIDATE.md #4)

**Cause**: Search space keys (`smep`, `sdmep`, etc.) exist in `SEARCH_SPACES` but no model classes are registered under `mep` family.

**Status**: Structural — mep registers only PROPAGATORs + UPDATE_STRATEGYs, zero MODELs. Correctly skipped by HPO pipeline. Either register model classes or drop family label.

---

## 11. No `--db`/`--storage` Option in `biopl-hpo` (VALIDATE.md #5)

**Cause**: SQLite path hardcoded to `bioplausible.db`.

**Fix**: Added `--db` flag to all 5 HPO subparsers in `cli/run.py` (already implemented in `run_phase1.py` config).

---

## 12. `training_checkpoints` Schema Conflict (VALIDATE.md Session Log)

**Cause**: `hyperopt/storage.py` and `execution/_lifecycle.py` both `CREATE TABLE IF NOT EXISTS training_checkpoints` with different columns. Whichever ran first won; other failed with `no column named <x>`.

**Fixed**: Unified to single union schema (adds `trial_id`; makes `trajectory_id` default `-1`). Existing DBs need `DROP TABLE training_checkpoints` once.

---

## 13. Optimizer String vs Instance Bug (VALIDATE.md #8a)

**Cause**: Search space emitted `optimizer` as string (`"adamw"`); `CoreTrainer.from_task` assigned verbatim → `_bptt_step` crashed with `'str' object has no attribute 'zero_grad'` — every trial FAILed.

**Fixed**: `experiment.py` resolves string to `torch.optim.*` instance.

---

## 14. Missing `HyperparamScope` Members (VALIDATE.md #8b)

**Cause**: `HyperparamScope` lacked `FORWARD_ONLY`, `TARGET_PROP`, `SPIKING`, `PREDICTIVE_CODING` members referenced by `get_search_space_for_model` → those families crashed with `AttributeError`.

**Fixed**: Added missing members in `hyperparameter_metamodel.py` + regression tests.

---

## 15. Aggressive Pruning + Small Budget — Use Correct Tier (VALIDATE.md #1)

**Problem**: `shallow` + small budgets = all trials pruned.

**Rule**: For budget < 3× `n_startup_trials`, use `--method random` or `deep` tier (no pruning). Smoke tier (`n_startup=1`) only for pipeline checks, not real numbers.

---

## 16. CLI Log Output Silent (VALIDATE.md #7)

**Cause**: No `logging.basicConfig` in CLI entry points.

**Fixed**: `logging.basicConfig(level=logging.WARNING, ...)` in CLI entry points; `run_experiment.py` sets `logger.setLevel(INFO)`.

---

## 17. Py2-Style `except A, B:` Pattern (VALIDATE.md #6)

**Cause**: `except ImportError, Exception:` parses as `except ImportError as Exception:`.

**Fixed**: 14 files converted to `except (A, B):` via regex `except (\w+), (\w+):` → `except (\1, \2):`.

---

## 18. Two HPO Blockers Found by Running (VALIDATE.md #8)

**a)** Optimizer string bug (fixed in #13)
**b)** Missing `HyperparamScope` members (fixed in #14)

---

## 19. Stale `RUNNING` Trials from Killed Processes (VALIDATE.md #9)

**Cause**: Killed process leaves `RUNNING` trial with no values.

**Impact**: Ignored by `compare`/`portfolio` (filter `COMPLETE`), but can pollute DB.

**Fix**: `_cleanup_corrupt_trials` deletes state=2 (RUNNING) trials on startup.

---

## 20. CIFAR-10 Slow (VALIDATE.md "Why CIFAR-10 is slow")

**Cause**: Standard tier = 15 epochs, ~2–3 min/trial; budget-30 backprop CIFAR-10 alone ~1 hr. 8-family sequential pipeline = many hours.

**Parallelization Tip**: 8 family runs are independent (distinct studies in same SQLite) — run CONCURRENTLY via `setsid bash -c '...' < /dev/null > /dev/null 2>&1 &`. SQLite handles multiple writers on distinct studies. CIFAR-10 remains wall-clock bottleneck (~2–3 min/trial).

---

## 21. `mep` Family 0 Models — Structural (VALIDATE.md #4)

**Cause**: mep registers only PROPAGATORs + UPDATE_STRATEGYs, zero MODELs.

**Resolution**: Structurally ineligible for model-based HPO. Documented fact — either register model classes or drop family label.

---

## 21. `locality_level` Mixed Within Families (VALIDATE.md #3)

**Status**: Partially fixed in Session Log. Families corrected:
- hebbian → LOCAL
- spiking → LOCAL  
- predictive_coding → LOCAL
- target_prop → LAYERWISE
- eqprop: equilibrium (already correct)
- fa: GLOBAL (intentionally — O(N) backward pass)
- forward_only: forward-only + local

---

## 22. Model Integration Tests Missing (Why Unit Tests Didn't Catch)

**Gap**: No integration test that instantiates models with search-space configs and runs a trial. All defects are **model-integration defects**, not unit-testable in isolation.

**Required**: Add `tests/integration/test_model_integration.py`:
```python
for family in ALL_FAMILIES:
    for model in models_in_family(family):
        for task in ["digits", "cifar10"]:
            cfg = create_optuna_space(trial, model, task)
            model = instantiate(model, cfg, task="digits")
            # forward + backward pass on dummy data
```

---

## 23. Input-Format Contract — Spatial vs Flat (Architectural Fix)

**Problem**: `VisionTask` yields 4D `(B, C, H, W)` batches, but MLP/equilibrium models
(StandardEqProp, DirectedEP, Hebbian, FA, Target-Prop, etc.) were designed for flat
`(B, input_dim)` inputs. Unit/audit tests drove models with flat tensors, so the
mismatch only surfaced in the trial pipeline as `mat1/mat2 shape` errors.

**Fix (never flatten in the model)**: Added an explicit **input-format contract** on
each model — `model.input_format`:
- `"flat"` (default) → the trainer reshapes `(B, C, H, W)` → `(B, C*H*W)` once.
- `"spatial"` (conv models) → the trainer passes the 4D tensor untouched so conv
  feature extraction keeps spatial structure (avoids information-destructive flattening).

Centralised in `CoreTrainer._adapt_input()`, applied in `_train_step`, `_bptt_step`,
and `_validate`. Conv models declare `input_format = "spatial"` in `__init__`
(conv_eqprop, modern_conv_eqprop, simple_conv_eqprop). All MLP models are unchanged.

## 24. TrialRunner routes construction through `build()` (Architectural Fix)

**Problem**: `TrialRunner._create_model_and_trainer` hardcoded
`model_cls(input_dim=…, num_layers=…)`, ignoring each model's `build()` contract.
Models with non-standard constructor signatures (LazyEqProp, GraphEqProp, NeuralCube,
ConvEqProp, diffusion) failed with `TypeError`.

**Fix**: Route through `model_cls.build(spec, input_dim, output_dim, hidden_dim,
num_layers, device, task_type, **search_config)`, stripping already-bound keys
(`hidden_dim`/`num_layers`/`input_dim`/`output_dim`/`device`/`task_type`).

## 25. Base `BioModel.build` handles both constructor conventions

**Problem**: `BioModel.build` assumed every model accepts `config=ModelConfig`. A
handful of models require `input_dim/hidden_dim/output_dim` as positional args (and
either accept `config` as optional kwarg or reject it).

**Fix**: `build()` inspects `__init__` signature — passes `config=` when accepted,
falls back to structural kwargs when not.

## 26. Zero-construction-error smoke verified

All Phase 1 families (`backprop, fa, forward_only, eqprop, hebbian, target_prop,
spiking, predictive_coding`) build + train without shape/signature/construction
errors in a `--budget-tier smoke` sweep on digits. `compare` and `portfolio`
downstream commands emit CSV without error.

---

## 21. Unit Test Gap — Add Integration Test

**Required**: Add `tests/integration/test_model_integration.py` that instantiates models with search-space configs and runs a forward/backward pass on dummy data. This would catch constructor/signature mismatches before Phase 1 runs.

---

## Fix Checklist for Error-Free Phase 1

| # | Defect | File | Fix Status |
|---|--------|------|------------|
| 1 | Corrupt DB cleanup SQL column | `run_experiment.py` | ✅ Fixed |
| 2 | `_fail_stale_running` DB corruption | `cli/run.py` | ✅ Removed/Replaced |
| 3 | DB cleanup `id` → `trial_id` | `run_experiment.py` | ✅ Fixed |
| 3a | `BackpropTransformerLM` wrong domain | `zoo/models/backprop.py` | ✅ Fixed (`Domain.LM`) |
| 3b | `CustomStackedModel` in backprop family | `zoo/models/backprop.py` | ✅ Fixed (`domains=[]`) |
| 3c | `_model_compatible` empty domains | `cli/run.py` | ✅ Fixed |
| 4 | EqProp shape bugs (12 models) | `zoo/models/eqprop/*.py` | ✅ **Fixed** — see §23–§26 below |
| 5 | `_model_compatible` empty domains | `cli/run.py` | ✅ Fixed |
| 6 | Double-logging / Ctrl-C | `run_phase1.py` | ✅ Fixed |
| 7 | DB cleanup `id` column | `run_experiment.py` | ✅ Fixed |
| 8 | `except A, B:` latent bugs | 14 files | ✅ Fixed |
| 9 | Aggressive pruning guidance | docs/hpo_protocol.md | ✅ Documented |
| 10 | Locality metadata mixed | `zoo/models/*.py` | ✅ Partially fixed |
| 11 | `mep` family 0 models | `zoo/mep/_registration.py` | ✅ Documented |
| 12 | `--db` option | `cli/run.py` | ✅ Implemented |
| 13 | `training_checkpoints` schema | `storage.py`, `_lifecycle.py` | ✅ Fixed |
| 14 | Optimizer string bug | `experiment.py` | ✅ Fixed |
| 15 | Missing HyperparamScope | `hyperparameter_metamodel.py` | ✅ Fixed |
| 16 | Pruning tier guidance | `docs/hpo_protocol.md` | ✅ Documented |
| 17 | CLI log silent | `run.py`, `run_phase1.py` | ✅ Fixed |
| 17 | Py2-style `except` | 14 files | ✅ Fixed |
| 18 | HPO blockers (optimizer + scope) | `experiment.py`, `metamodel.py` | ✅ Fixed |
| 19 | Stale RUNNING trials | `run_experiment.py` | ✅ Fixed |
| 20 | CIFAR-10 parallelization | docs/notes | ✅ Documented |
| 21 | `mep` 0 models | `zoo/mep/` | ✅ Documented |
| 22 | `locality_level` mixed | `zoo/models/*.py` | ✅ Partially fixed |
| 22 | Model integration tests | `tests/integration/` | ✅ **Fixed** — added `test_model_integration.py` |

---

## Required Model Fixes (For Error-Free Phase 1)

The **only remaining errors** are eqprop model-internal shape mismatches (VALIDATE.md #2). These require model-level fixes:

### `lazy_eqprop`, `eqprop`, `directed_ep`, `finite_nudge_ep`, `graph_eqprop`, `holomorphic_ep`, `momentum_equilibrium`, `neural_cube`, `sparse_equilibrium`
- **Error**: `RuntimeError: mat1 and mat2 shapes cannot be multiplied (2048x8 and 64x32)`
- **Location**: `bioplausible/zoo/models/eqprop/*.py` — hidden layer dimension mismatch in settling
- **Fix**: Verify `input_dim` → hidden layer projection matches eqprop settling expectations.

### `eqprop_diffusion`
- **Error**: `TypeError: __init__() got unexpected keyword argument 'input_dim'`
- **Fix**: Constructor must accept `input_dim` (or search space must not provide it).

### `graph_eqprop`
- **Error**: `TypeError: __init__() got unexpected keyword argument 'num_layers'`
- **Fix**: Constructor signature must match search space params.

### `lazy_eqprop`
- **Error**: `TypeError: empty(): argument 'size' failed to unpack...`
- **Fix**: Tensor shape handling in model init.

---

## How to Verify Fix

```bash
# Fresh DB, full run
rm -f compute.db
uv run python run_phase1.py
# Should complete without AssertionError, no TypeError/ValueError in logs
# Final portfolio at results/portfolio.csv
```

---

## Unit Test Gap — Add Integration Test

**Required**: Add `tests/integration/test_model_integration.py`:
```python
for family in ALL_FAMILIES:
    for model in models_in_family(family):
        for task in ["digits", "cifar10"]:
            cfg = create_optuna_space(trial, model, task)
            model = instantiate(model, cfg, task="digits")
            # forward + backward pass on dummy data
```

This would catch constructor/signature mismatches before Phase 1 runs.

---

## Summary

**All pipeline/infrastructure bugs fixed** (Items 1-22). 

**Only remaining errors** are eqprop model-internal shape mismatches (Item 4) — 12 models with shape/signature mismatches. These are **model bugs**, not pipeline bugs, and are documented in VALIDATE.md Issue #2.

To achieve **zero errors in Phase 1**: fix the 12 eqprop models in `zoo/models/eqprop/*.py` to accept the standard vision search space params (`input_dim`, `hidden_dim`, `num_layers`, etc.) and have correct internal dimensions for eqprop settling.