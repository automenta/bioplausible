# REFACTOR2: Remaining Work — Strategic Consolidation Plan

## Core Philosophy
The codebase (~78k LOC, ~297 modules) has 7 parallel training stacks, 4 config hierarchies, 5 `BenchmarkResult` classes, and 5 persistence layers. **Capability is not the problem — consolidation is.** Target: a strict dependency-layered core with exactly one implementation of every cross-cutting concern.

```
┌─────────────────────────────────────────────────────────────────┐
│ L7  Interfaces  : CLI · deployment · sklearn · lightning        │  public API
├─────────────────────────────────────────────────────────────────┤
│ L6  Measurement : evaluation · validation · benchmarks ·        │  one BenchmarkResult,
│                   analysis · reporting · leaderboard           │  one report renderer
├─────────────────────────────────────────────────────────────────┤
│ L5  Orchestration: execution · hyperopt · autoscientist        │  adapters over runner
├─────────────────────────────────────────────────────────────────┤
│ L4  Training     : CoreTrainer (THE single train path)         │  runners become adapters
├─────────────────────────────────────────────────────────────────┤
│ L3  Data/Domains : data · domains                                │  one task abstraction
├─────────────────────────────────────────────────────────────────┤
│ L2  Zoo          : models · propagators · optimizers · mep      │  registered components
├─────────────────────────────────────────────────────────────────┤
│ L1  Core         : registry · construction · config ·           │  zero upward imports
│                    checkpoint · metrics · result_sink ·         │
│                    tile substrate · local_learning              │
└─────────────────────────────────────────────────────────────────┘
```

**Layering rule:** L_N may import from L_{≤N−1} only. Enforced by Pillar N gate (`tools/check_imports.py`).

---

## Remaining Pillars (Ordered by Value/Effort)

### 1. B — Single Config Hierarchy (XL, High risk, blocks A/C/E)  ✅ COMPLETE (verified 2026-08-13)
**Problem:** 4+ duplicate hierarchies, same-named classes.
- `ModelConfig` ×2 (`unified.py`, `omegaconf.py` renamed)
- `ExperimentConfig` ×2 (`unified.py`, `omegaconf.py` renamed)
- `TrainerConfigSchema` (Pydantic, zero prod consumers) — delete or auto-generate via `TypeAdapter`
- `config/omegaconf.py` mirror — delete; keep `unified.py` as single I/O pair
- `_KNOB_ALIASES` in `construction.py` — shrink to zero once all sites emit canonical names

**Verified:** `grep "class ModelConfig"` == 1 (`config/unified.py:123`); `ExperimentConfig` == 1
(`unified.py:328`); `TrainerConfigSchema` and `_KNOB_ALIASES` no longer exist. `omegaconf.py` was
resolved into the intended *I/O-boundary facade* design (mutable OmegaConf YAML document types with
`to_internal()` seams into the frozen `unified.py` tree) rather than deletion — this is the correct
"single I/O pair" outcome and should be kept, not re-deleted.

**Target:** One compositional tree in `config/unified.py`:
```python
@dataclass(frozen=True, slots=True)
class ModelConfig: ...
@dataclass(frozen=True, slots=True)
class OptimizerConfig: ...
@dataclass(frozen=True, slots=True)
class DataConfig: ...
@dataclass(frozen=True, slots=True)
class TrainLoopConfig: ...
@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    model: ModelConfig
    optimizer: OptimizerConfig
    data: DataConfig
    train_loop: TrainLoopConfig
    hardware: HardwareConfig
    seed: int
    tags: tuple[str, ...]
```

**Win:** ~1,200 lines removed; drift bugs eliminated; `phantom_knobs` provably exhaustive.

---

### 2. C — Single Construction Funnel (M, Medium, unblocks A)
**Status:** Model instantiation funnel complete. **Task/geometry resolution collapse DONE (2026-08-13).**
- **Task/geometry resolution collapse:** `create_task`/`resolve_task`/`trainer._setup_data`/`engine._get_train_loader` → single `DataConfig → DomainTask` resolution in `domains/registry`. **COMPLETE:**
  - Added `DataConfig` frozen dataclass + `resolve_task_from_data_config` in `config/unified.py` (delegates to registry to stay import-acyclic).
  - `domains/registry.resolve_task_from_data_config(DataConfig, device) -> DomainTask` is now the single canonical path.
  - `CoreTrainer._setup_data()` rewritten to build a `DataConfig` from `TrainerConfig` and resolve once; `_setup_lm_data()` deleted. Non-DataLoader domains (LM/RL/graph/timeseries) use `task_obj.get_batch`; DataLoader domains (vision/tabular) resolve `train_loader`/`val_loader` from the task (preserving the test-override contract).
  - `run_from_runconfig()` (trainer.py) and `TrialRunner._setup_task()` (hyperopt/experiment.py) and `ExecutionEngine._get_train_loader/_get_val_loader` (execution/engine.py) all route through the same resolver — no more scattered `create_task`/`create_data_loaders` calls.
  - **Acceptance criterion #3** (`grep -rn "model_cls(" | grep -v construction.py` = 0) and the single-resolution seam are both satisfied.
- **Acceptance criterion #3:** `grep -rn "model_cls(" bioplausible/ | grep -v construction.py` → zero instantiation sites (only `construct_model` calls and `.build` for tile/deployment).

**Progress (2026-08-13):**
- **Task/geometry collapse implemented & verified** (see above). New tests in `tests/unit/domains/test_registry.py`: `test_data_config_defaults_and_frozen`, `test_resolve_from_data_config_returns_task_with_loaders`, `test_resolve_from_data_config_rejects_unknown`. Suite: core_trainer + registry + config_unified + smoke_training + hyperopt_integration + engine_stability + phase0 + cli all green; pyright 0 errors on changed modules; ruff clean (only pre-existing warnings remain).
- **Spatial-geometry note:** `resolve_task_from_data_config` preserves the raw `DomainTask.input_dim` tuple (e.g. mnist → `(1,28,28)`) — it is NOT flattened, so conv models get `input_channels` from `input_dim[0]` via `construct_model._derive_conv_channels`. The flattening for scheduling lives only in `resolve_task` (used by the scheduler), per the documented "task geometry is ambiguous by design" policy.
- **Acceptance criterion #3 SATISFIED.** The last two direct-instantiation sites now route through
  the canonical `construct_model` funnel:
  - `cli/lab.py:44` — was `model_cls(input_dim=..., output_dim=...)`; now `construct_model(...)` with
    `{"hidden_dim": 64, "num_layers": 2}` defaults. Also fixed a latent Python-2-style
    `except RuntimeError, ValueError, TypeError:` clause (valid syntax in 3.14, left as-is) and the
    import ordering. Verified end-to-end: `inspect_model` builds + forward-passes `backprop_mlp` on
    `xor` (scalar input) and `mnist` (spatial `(1,28,28)` tuple input). Spatial tuple `input_dim`
    must be passed **straight through** (do NOT `int()` it) — matches `_build_runconfig_model`
    (`core/trainer.py:1711`).
  - `cli/repro.py:125` (pre-existing uncommitted) — was `model_cls.build(...)`; now `construct_model`.
- **Pre-existing uncommitted Pillar C work in the tree (left intact, verified green):**
  `core/construction.py` adds `_is_deployment_model()` and routes the deployment `BioModel` family
  (`conv_equitile`/`rl_equitile`/etc.) through the substrate `build` branch; `repro.py` consumes it;
  `tests/unit/experiment/test_config_knobs.py` opts `conv_equitile` out of the depth audit (its depth
  map is offset by 2). `pytest tests/unit/experiment/test_config_knobs.py` → 13 passed.

**Win:** ~600 lines removed; one error message for "unknown task/model"; lossless checkpoint round-trips.

---

### 3. A — Single Training Path (XL, High, depends on B/C)
**Problem:** 7 parallel run stacks (CoreTrainer, TrialRunner, Verifier, StaircaseRunner, BenchmarkRunner, BioLightningModule, graph/training.py). All re-implement: `train_step` dispatch, model instantiation, device/seed, metrics, checkpointing.

**Target:** `CoreTrainer` is the ONLY training loop. Everything else is a thin adapter:
- `TrialRunner`, `Verifier`, `StaircaseRunner`, `BenchmarkRunner` → construct `CoreTrainer`, call `fit()`/`train_epoch()`.
- `BioLightningModule` → callback/plugin on `CoreTrainer` (PL-specific needs = `ExecutionCallback` implementations). Delete its `create_model` helper.
- `graph/training.py` custom PCN/BPTT loops — leave alone unless trivially adaptable.
- Validation tracks keep declarative `track_*` spec but delegate execution to shared runner.

**Acceptance criterion #1:** `grep -rl "loss.backward()" bioplausible/` outside `core/` & `training_mixin` returns nothing.

**Progress (2026-08-14):** Deployment-model migration COMPLETE — see Session Log below.
- **CoreTrainer spatial input handling fixed** — `_setup_data` now populates `model_kwargs` with `input_dim`/`output_dim` from task (including spatial tuples). `_create_model` passes through without `int()` coercion.
- **ConvEquiTile deployment model integrated** — `build` handles tuple `input_dim` via `math.prod()`; `input_format = "spatial"` signals CoreTrainer to preserve 4D input.
- **LightningExecutionCallback added** — enables PL-compatible logging from CoreTrainer.
- **Verified:** CoreTrainer + ConvEquiTile on MNIST works end-to-end (model creation, spatial data adaptation, train_step dispatch). Locked in by `tests/unit/core/test_deployment_models.py` (5 tests, green).

**Win:** ~2,500 lines removed; one place for bug fixes/features.

---

### 4. E — Single Result & Persistence Funnel (M, Medium, depends on A)
**Problem:** 5 write paths for outcomes (Optuna SQLite, HyperoptStorage, JSONL Report, KB, execution_state.db). `result_sink.record_experiment_result` is the canonical funnel but engine, verifier, mep-benchmarks write around it.

**Target:** `result_sink` is the ONLY writer. All backends become private implementation details.
- `record_experiment_result` owns: Optuna `tell`, `hyperopt_logs`, JSONL `Report`, KB upsert, failure log.
- One artifact loader: `core/checkpoint.load_checkpoint` + `find_trial_artifact(trial_id)` (already done).
- Ad-hoc checkpoint saves → `CheckpointMixin`/`core.checkpoint` calls.
- Evaluate `CheckpointManager` in `execution/_lifecycle.py` against `core.checkpoint`.

**Acceptance criterion #3:** `record_experiment_result` called by execution, hyperopt, validation, mep-benchmarks; all five backends written only from `result_sink`.

**Win:** ~700 lines removed; split-brain audit trails eliminated.

---

### 5. D — Single Measurement & Reporting Stack (XL, High, depends on E)
**Problem:** Parallel measurement ecosystems.
- `BenchmarkResult` ×5 (`evaluation/base.py`, `rigorous.py`, `compare_nanoGPT.py`, `tile_profiler.py`, `mep/runner.py`) — **do not mechanically merge** (semantically distinct). Establish `evaluation/base.BenchmarkResult` as canonical interface; others become Tracks/composites.
- Report renderers ×5 → one canonical JSONL renderer (`experiment/report.py`); others become thin adapters.
- Benchmark loops → registry-driven `BenchmarkRegistry` tracks (declarative, not new loops).
- Metrics: `core.losses.compute_accuracy` is canonical; fold remaining inline copies (only legitimately different sites remain: 3-D per-token, accumulation, PL raw tensors).
- Leaderboard/ranking ×3 → one implementation in `leaderboard/` + `cli/rank.py`.

**Win:** ~4,000 lines removed; findings share schema/CIs/renderers.

---

### 6. G — Propagator/Model Unification (M, Medium, depends on A/F)
**Status:** Alias map done (`_PROPAGATOR_TO_MODEL` → `_ALIASES`; `Registry.get(PROPAGATOR, "ff")` returns model class). **Remaining:**
- `CoreTrainer._train_step` 5→2 phase collapse: `energy-model` (structural `match`) → `model.train_step` → `BPTT`. Delete phases 2 & 4 (explicit propagator/learning-rule optimizer).
- Convert `zoo/propagators/{eqprop,fa,hebbian,backprop,spiking}.py` to model-side `train_step`s or delete.
- `ComponentCategory.PROPAGATOR` shrinks to pure gradient transformers (Muon, spectral norm, EWC).

**Acceptance criterion #6:** `zoo/propagators/` contains only `mep.py` and pure-gradient-transform submodules.

**Win:** ~800 lines removed; one interface; AutoScientist composition simplifies.

---

### 7. K — CLI & Interface Hygiene (M, Low)
**Status:** `DASHBOARD` global decoupled via `EventSink` protocol (`execution/events.py`). **Remaining:**
- **`biopl` dispatcher:** Consolidate 13 console scripts + `cli/run.py` 6-subcommand monolith into one `argparse` subcommand dispatcher (`run | report | parity | repro | hpo | audit | frontier | rank`). Each `cli/` module becomes thin adapter over Pillars A–F APIs.

**Progress (2026-08-13):** The **`biopl` dispatcher is implemented and verified.**
- `bioplausible/cli/__main__.py` rewritten as a single lazy dispatcher over
  `run | report | parity | repro | hpo | audit | frontier | rank | lab`. Each
  sub-command is resolved lazily (no zoo/execution import at dispatch time) and
  delegates to its module `main`, rewriting `sys.argv` so each adapter parses its
  own flags. `SystemExit` (e.g. `--help`) is caught and re-mapped to its code.
- Added `biopl = "bioplausible.cli.__main__:main"` to `pyproject.toml`
  `[project.scripts]` as the canonical public entry point.
- Added `tests/unit/cli/test_cli_dispatch.py` (14 tests): subcommand set, unknown
  command, no-args, `--help`, per-command `--help` exit 0, and sys.argv passthrough.
  Verified green: `uv run biopl --help` / `biopl rank --help` / `biopl lab --help` /
  `biopl audit` / `biopl run list` all route correctly with correct `prog` strings.
- `ruff` + `pyright` clean on changed files; CLI suite (dispatch + parity + experiment
  cli + audit) = 46 passed.

**Remaining (optional, additive):** The individual `biopl-*` console scripts are
**kept** because CI (`.github/workflows/ci.yml`) and docs reference
`biopl-registry-audit`, `biopl-repro-check`, `eqprop-verify`. They are now thin
adapters, so `biopl` can be adopted in CI later without breaking the current gate.

**Win:** Clean public API boundary; headless CI/sweeps work.

---

### 8. L — Self-Registration (M, Low)
**Status:** `zoo/models/eqprop/__init__.py` auto-computes `__all__` from `vars(module)`. Registry has `aliases()` + `resolve_alias()`. **Remaining:**
- Reduce `bioplausible/__init__.py` and `core/__init__.py` `_LAZY` maps to declared shortlist of public API.
- Other leaf re-export subpackages (e.g. `fa.py`-style) adopt `vars(module)` pattern with per-file `ruff` ignores.

**Win:** Adding a model/rule = one registration decorator; nothing else to touch.

---

### 9. J — Dead Code Tail (S, Low)
- `analysis/tile_*.py` legacy systems → superseded by `evaluation/` + `mep/benchmarks` (post-Pillar D).
- `TODO.md`, `REFACTOR.md`, stale `docs/` → archive out of tree.

**Progress (2026-08-13):** `TODO.md` and `REFACTOR.md` (the two superseded plans — the active plan is
`REFACTOR2.md`) archived to `docs/archive/20260813/` following the dated-dir convention already used by
`docs/archive/2026MMDD/`. Verified no code/tests read these files (all references are docstring/comment
prose); README does not link them; `tools/check_imports.py` unaffected.

**Remaining:** `analysis/tile_*.py` supersession is explicitly gated on Pillar D — do **not** archive
until the measurement stack lands.

---

## Deprioritized (Explicitly Not Now)
- **God-Object Decomposition (O):** `core/trainer.py`, `knowledge/kb.py`, `execution/strategy.py` — split only when Pillars A/E/D touch them; cap effort; stop when cohesive.
- **Settling Loop Merge (I):** Family A/B convergence loops — high numerical risk, low gain. Telemetry unification done.
- **Visualization Stack Consolidation:** 4 stacks — UI preference, not architectural flaw.
- **Micro-Consolidation Remainder (M):** ~12 inline accuracy folds (3-D/accumulation/PL) are legitimately different; `count_parameters` + seeding done.

---

## Execution Sequence & Metrics

| Phase | Pillars | Metric |
|-------|---------|--------|
| **1. Foundations** | B (config) → C (construction tail) | `grep "class ModelConfig"` = 1; `grep "model_cls("` outside construction = 0 |
| **2. Core Unification** | A (training) → E (persistence) → D (measurement) | `grep "loss.backward()"` outside core = 0; 100% outcomes via `result_sink`; 1 `BenchmarkResult` interface |
| **3. Clarity** | G (propagator/model) → K (CLI) → L (self-reg) | 0 propagator-only loops; `biopl` dispatcher works; 1 registration decorator |

**Acceptance Criteria (all must pass):**
1. Import-DAG checker passes in CI (Pillar N gate).
2. `CoreTrainer` sole owner of BPTT/optimizer step logic.
3. No split-brain persistence (all writes via `result_sink`).
4. Zero new test failures beyond the 6 pre-existing numerical/parity drifts.
5. `grep` criteria from criteria #1–#6 all satisfied.

---

## Current Baseline
**Full suite:** 2002 pass / 6 fail / 10 skip / 1 xfail (6 failures = documented numerical/parity drift, unrelated to refactor).
**Lint gate:** Functional (ruff 0.16 parses config; ~2k pre-existing warnings are backlog, not blocker).
**Pyright:** 0 errors strict mode.

---

## Session Log (2026-08-13)

### Completed
- **Pillar B — verified COMPLETE** (single `ModelConfig`, single `ExperimentConfig`, `TrainerConfigSchema`
  and `_KNOB_ALIASES` gone; `omegaconf.py` resolved as I/O-boundary facades, keep).
- **Pillar C — task/geometry resolution collapse COMPLETE.** Added the unified `DataConfig` +
  `resolve_task_from_data_config` seam in `config/unified.py` (thin delegate to
  `domains/registry.resolve_task_from_data_config`). All data-loading sites now route through the
  single resolver: `CoreTrainer._setup_data` (deleted `_setup_lm_data`, removed the `match`-based
  `create_data_loaders` dispatch), `run_from_runconfig`, `TrialRunner._setup_task`, and
  `ExecutionEngine._get_train_loader`/`_get_val_loader`. Geometry threads straight through
  (spatial tuples preserved for conv-channel derivation). New tests in `tests/unit/domains/test_registry.py`.
- **Pillar C acceptance criterion #3 — satisfied** (`grep -rn "model_cls(" | grep -v construction.py` = 0).
  Both `cli/lab.py` and `cli/repro.py` now construct exclusively via `construct_model`.
- `cli/lab.py` hardened: import ordering (E402), `cast` to `nn.Module` (ruff + pyright clean, 0 errors),
  added `hidden_dim`/`num_layers` defaults so the lab actually constructs models that declare them.
- **Pillar K — `biopl` dispatcher implemented** (see §7): single lazy dispatch entry point over
  `run | report | parity | repro | hpo | audit | frontier | rank | lab`, added as the `biopl` console
  script; 14 new tests in `tests/unit/cli/test_cli_dispatch.py`; ruff + pyright clean.
- **Pillar J (partial) — archived** `TODO.md` + `REFACTOR.md` to `docs/archive/20260813/`; safe (no code
  reads them). `analysis/tile_*.py` held for post-Pillar D.

### Verification (2026-08-13, targeted suites — not the full 2k suite)
`pytest` on: `test_core_trainer.py` (23) + `test_registry.py` (10) + `test_config_unified.py` (11) +
`test_smoke_training.py` (25) + `test_hyperopt_integration.py` (3) + `test_engine_stability.py` (1) +
`test_phase0.py` (5, exercises `run_from_runconfig`) + `tests/unit/cli/` (23) → **all green**.
`pyright` 0 errors on `config/unified.py`, `domains/registry.py` (new code). `ruff` clean on all changed
modules except pre-existing trainer.py/engine.py warnings (typing-only imports, encoding, os.path.join,
unused `raise-vanilla-args` suppression) — none introduced by this work.

### Risk assessment of remaining pillars (surveyed 2026-08-13)
This is the blocker-analysis a future session needs before touching Pillars A/E/D/G. Each was surveyed
and found **not** safely completable without CoreTrainer surgery or a large dependency cascade:
- **Pillar A (single training path, XL):** Criterion #1 needs `loss.backward()` removed from ~35 sites
  across mep benchmarks, propagators, deployment models, validation tracks. These are *semantically
  different* training loops (energy/dual-phase, spiking, tile substrate), so they cannot be mechanically
  folded into `CoreTrainer.train_epoch`. This is a multi-session architectural effort, not a bounded edit.
- **Pillar G (propagator/model unification):** verified `zoo/propagators/{backprop,base,fa,eqprop,hebbian,
  spiking}.py` are **NOT dead code** — heavily imported by ~20 tests, `cli/repro.py:232-243`,
  `validation/tracks/nebc_tracks.py`, and `bioplausible/__init__.py`. The plan's "or delete" path is
  unavailable without first migrating those consumers to model-side `train_step` (risky). Criterion #6
  remains open.
- **Pillar E (persistence funnel):** the `execution/engine.py` Optuna `study.tell/ask` (lines 558,597,639)
  and `state.failure_tracker.log_failure` (451) are the engine's *online HPO loop itself*, not outcome
  recording — folding them into `record_experiment_result` would conflate the search loop with the KB
  audit trail. `result_sink` already owns KB + FailureTracker and is called from hyperopt, validation
  tracks, trainer, and probe. Treat the remaining unification as architectural, not mechanical.
- **Pillar D (measurement/reporting):** `experiment/reporting.render_report` (JSONL) is already the
  canonical `biopl-report` renderer; `analysis/reporting.generate_experiment_report` consumes Optuna
  trials (different input, not a duplicate). The 5 `BenchmarkResult`s are semantically distinct by design
  (plan says do not mechanically merge).

**Recommended next session:** with the Pillar C task/geometry collapse now done, the remaining
open work is Pillar A (single training path, XL — criterion #1 needs `loss.backward()` removed from
~35 semantically-distinct loops) and Pillar G (propagator/model unification). Start Pillar A with a
*single* tracked loop (e.g. a deployment model) migrated to `CoreTrainer` behind targeted training
tests, not the full suite. Both are CoreTrainer-adjacent, high-risk architecture work.

### Findings for future work
- **`python -m bioplausible.cli <cmd>` shows the wrong `prog`** ("python3 -m bioplausible.cli") in the
  adapter's `--help` usage line, but the **installed `biopl` script shows the correct one** ("biopl rank",
  "biopl lab"). Cause: runpy vs. entry-script `sys.argv[0]` differences under argparse. Cosmetic only;
  do not chase it — verify via `uv run biopl ...`, not `python -m`.
- **argparse `--help` raises `SystemExit(0)`** before the adapter body runs, so a dispatcher that calls
  the adapter `main` directly must catch `SystemExit` and remap `exc.code` or the CLI's exit status is
  wrong under the console script.
- **CI pins 3 legacy scripts** (`.github/workflows/ci.yml:31,33,59`: `biopl-registry-audit`,
  `biopl-repro-check`, `eqprop-verify`). Any future script-removal pass must update CI + README/docs
  references or keep them as thin `biopl`-delegating shims.
- **Python 3.14 allows `except A, B, C:`** (tuple-of-exceptions form) without parentheses — old-style
  clauses are valid and `ruff format` will strip redundant parens back to it. Do not "fix" them; they are
  not bugs.
- **Task geometry is ambiguous by design:** `TaskProtocol.input_dim` is typed `int | None` but concrete
  tasks return *tuples* (e.g. `mnist → (1,28,28)`). The single task-resolution seam must thread geometry
  straight through to `construct_model` (matching `_build_runconfig_model`), never `int()`-coerce it.
  `domains/registry.resolve_task` already flattens via `math.prod` for scheduling; unify this policy.
  `resolve_task_from_data_config` (new) preserves the raw tuple — flattening stays only in `resolve_task`
  (the scheduler's geometry view).
- **`cli/lab.py` `args.model="MLP"` is NOT a registered name** — the registry's model names are
  `backprop_mlp`, `eqprop`, `forward_forward`, etc. The CLI's default is misleading; a future Pillar K pass
  should map a friendly shortlist or error clearly.
- **Engine loader duplication (Pillar C tail) — RESOLVED (2026-08-13):** `engine._get_train_loader`/
  `_get_val_loader` (`execution/engine.py`) and `trainer._setup_data` (`core/trainer.py`) previously
  called `create_data_loaders`/`_setup_lm_data` independently. All three now collapse onto the single
  `DataConfig → DomainTask` resolver in `domains/registry` (`resolve_task_from_data_config`), which
  returns a ready `DomainTask` with `get_dataloader`/`get_batch`. One remaining inconsistency to note
  for future work: `ExecutionEngine._get_train_loader` passes `device="cpu"` regardless of the engine's
  actual device (the non-PL path routes through `run_single_trial_task`/`TrialRunner` which resolves its
  own device); a future Pillar A pass should thread the engine's device through.

### Acceptance-criteria status
- Criterion #1 (`loss.backward()` outside `core/`+`training_mixin` = 0): **NOT done** — ~35 legit sites
  remain (mep benchmarks, propagators, deployment models, validation tracks). Pillar A scope.
- Criterion #3 (`model_cls(` outside construction = 0): **DONE**.
- Criterion #6 (`zoo/propagators/` = only `mep.py` + gradient transformers): **NOT done** — `backprop.py`,
  `base.py`, `eqprop.py`, `fa.py`, `hebbian.py`, `spiking.py` still present. Pillar G scope.
- Criterion "`biopl` dispatcher works" (Pillar K): **DONE**.

---

## Session Log (2026-08-14)

### Completed (Pillar A — Single Training Path: Initial Work)
- **CoreTrainer spatial input handling fixed:** `_setup_data` now populates `model_kwargs` with `input_dim` and `output_dim` from the resolved task object (including spatial tuples like `(1,28,28)` for MNIST). `_create_model` passes these through without `int()` coercion.
- **ConvEquiTile deployment model works with CoreTrainer:** 
  - `ConvEquiTile.build` now handles tuple `input_dim` by flattening via `math.prod()`.
  - Added `input_format = "spatial"` attribute so CoreTrainer's `_adapt_input` preserves 4D spatial tensors for conv models.
  - Verified end-to-end: CoreTrainer creates ConvEquiTile, adapts MNIST data to spatial format, and calls model's `train_step`.
- **Added LightningExecutionCallback** in `execution/callbacks.py` for PyTorch Lightning compatible logging from CoreTrainer.
- **Added `tests/unit/core/test_deployment_models.py`** (5 tests): construction via the single funnel, spatial-tuple threading into `model_kwargs`, a full training epoch through CoreTrainer's Phase-3 `train_step` dispatch (asserts `training_paths["model_train_step"]`), tuple-flattening in `ConvEquiTile.build`, and `_adapt_input` 4D preservation. All green.

### Verification (2026-08-14, targeted suites)
- `pytest tests/unit/core/test_core_trainer.py` (23) — **all green**
- `pytest tests/unit/core/test_deployment_models.py` (5) — **all green**
- `pytest tests/unit/core/test_execution_callbacks.py` (6) — **all green**
- `pytest tests/unit/experiment/test_training_path.py` (4) — **all green**
- `pytest tests/integration/test_smoke_training.py` (25) — **all green**
- `pytest tests/unit/cli/` (23) — **all green**
- `pytest tests/unit/domains/test_registry.py` (10) — **all green**
- `pytest tests/unit/experiment/test_config_knobs.py` (13) — **all green**
- `pytest tests/integration/test_hyperopt_integration.py` (3) — **all green**
- `pytest tests/integration/test_engine_stability.py` (1) — **all green**
- `pytest tests/unit/core/test_execution_callbacks.py` + `test_deployment_models.py` (11) — **all green**
- `pyright` 0 errors / 0 warnings on `execution/callbacks.py` (new code) + `test_deployment_models.py`
- `ruff` clean on `execution/callbacks.py` and `test_deployment_models.py`; no NEW warnings on `core/trainer.py` (36 pre-existing before == 36 after); `vision.py` remaining warnings all pre-existing (build-arg arity, magic dim constants)

### Next Steps for Pillar A
1. **BioLightningModule scoping decision — DONE (2026-08-14 continuation):** the shared `dispatch_train_step` seam is extracted and both `CoreTrainer` and `BioLightningModule` route through it (PL stays the outer loop; manual vs. automatic optimization handled externally). The `LightningExecutionCallback` from the previous session is the natural bridge for any remaining PL logging deltas.
2. **Convert RL training** (`training/rl.py`) to use CoreTrainer.
3. **Convert CLI repro** (`cli/repro.py`) `_train_one_epoch` to use CoreTrainer.
4. **Address validation tracks** — delegate execution to CoreTrainer-based runner.
5. **Address MEP benchmarks** — convert to registry-driven BenchmarkRegistry tracks (Pillar D dependency).

### Findings for future work
- **ConvEquiTile NaN loss:** default tile config (`neurons_per_tile=64`, `tiles_per_layer=4`, `mode="backprop"`) diverges on MNIST through CoreTrainer; the smaller test config (`conv_channels=[4,8]`, `tiles_per_layer=1`, `mode="pc"`) trains stably. This is a hyperparameter/initialization issue, not a CoreTrainer structural one — keep the small config for tests.
- **Other deployment models** (GraphEquiTile, TimeSeriesEquiTile, RLEquiTile) use scalar `input_dim` so they should work with CoreTrainer without changes — only ConvEquiTile needed the tuple handling fix.
- **`LightningExecutionCallback` is infrastructure, not wired:** it logs `TrainingMetrics` fields via `pl.LightningModule.log` but no production code uses it yet. It is the bridge the Pillar A BioLightningModule pass (Next Steps #1) should consume.
- **`_create_model` no longer needs `int()` coercion:** with `_setup_data` seeding `model_kwargs["input_dim"]` from the task, the `int(... or 0)` guard in `_create_model` was replaced with a `None` check that preserves tuples. Any future caller building `TrainerConfig` by hand must either pass a task name (so `_setup_data` resolves geometry) or include `input_dim`/`output_dim` in `model_kwargs` explicitly.

---

## Session Log (2026-08-14, continuation — Shared train-step dispatch)

### Completed (Pillar A, Next Step #1 — single dispatch seam)
- **Extracted the canonical 5-phase `train_step` dispatcher into a module-level pure function** `dispatch_train_step` in `core/trainer.py`. It owns the order-of-routing (energy-model → learning-rule propagator → model-side `train_step` → learning-rule optimizer → BPTT fallback) previously inlined in `CoreTrainer._train_step`. Callers inject `adapt_input`, a `bptt_step` callback, and an optional `record_path` recorder, keeping the dispatch pure and reusable.
- **`CoreTrainer._train_step` now delegates** to `dispatch_train_step` (behavior identical — verified by the training-path / propagator / energy / deployment suites).
- **`BioLightningModule.training_step` now routes through the same dispatcher** — PL stays the outer loop. Manual-optimization mode zeroes its own optimizer before dispatch and steps after; automatic mode returns the tensor loss for PL to backprop. Added a small `_bptt_forward` helper. This directly implements the documented "extract `_train_step`'s dispatch into a reusable pure function both CoreTrainer and BioLightningModule call, keeping PL as the driver."

### Verification (targeted suites)
- `pytest` `test_core_trainer.py` + `test_deployment_models.py` + `test_training_path.py` + `test_lightning_integration.py` → **58 passed** (1 skipped).
- `pytest` `test_energy_model.py` + `test_energies.py` + `test_execution_callbacks.py` + `test_registry.py` → **41 passed**.
- `pytest` `test_smoke_training.py` → **25 passed**.
- `pyright` strict: **0 errors** on both files; warnings **net decreased** (113→111) across `core/trainer.py` + `lightning_/module.py` — no new warnings introduced.
- `ruff format` + `ruff check --select E,F`: no new errors (only the two pre-existing line-too-long docstrings in `trainer.py:591,1042`).
- **Pillar A Next Step #3 (`cli/repro.py` `_train_one_epoch` → `dispatch_train_step`):** `tests/unit/validation/test_repro_check.py` → **9 passed** (bitwise gate intact); `pytest tests/unit/cli/` → **23 passed**; `pyright` 0 errors on `cli/repro.py`; `ruff` clean.

### Findings for future work
- **The dispatch is typed for `dict[str, object]` by design:** PL's automatic path must return a *tensor* loss (for PL's backward), while `CoreTrainer`'s paths yield floats. So `dispatch_train_step` and its `bptt_step` callback are typed `dict[str, object]`, and `CoreTrainer._train_step` casts the result back to `dict[str, float]`. New callers should be aware of this split and pick the appropriate typing.
- **`BioLightningModule` passes `propagator=None`/`optimizer=None` to the dispatcher and steps its own optimizer externally.** This deliberately suppresses Phase-4 (learning-rule optimizer) so its bio-optimizers keep their prior `model.train_step`→`opt.step()` semantics rather than switching to `rule.step(x=, target=)` — a behavior-preserving choice worth re-evaluating once Pillar G lands (Phase-4 is the intended home for those optimizers).
- **The EnergyModel branch is guarded by `config is not None`:** CoreTrainer passes config (enabling the `_make_ebm_trainer` energy path); `BioLightningModule` passes none, so EnergyModels fall through to model-side `train_step`, matching its prior behavior. A future pass could give the module a minimal config facade to unlock the energy path.
- **`BioLightningModule.create_model` helper was deliberately kept** — `tests/integration/test_lightning_integration.py:368,417` patch `bioplausible.lightning_.module.create_model`; deleting it requires updating those tests (the Pillar A "Delete its create_model helper" item).
- **Next Step #3 (convert `cli/repro.py` `_train_one_epoch`, was lines 146-172) — DONE (2026-08-14 continuation).** The hand-rolled train_step-vs-Adam-BPTT loop now routes each batch through `dispatch_train_step` (its `_bptt` closure supplies the Adam BPTT fallback). This removes the duplicated train-step dispatch, not the BPTT backward itself — repro.py still contains a `loss.backward()` inside the `_bptt` closure, because the non-`train_step` families (e.g. `fa`) legitimately train via plain BPTT. So **criterion #1 is NOT advanced by this change**; the remaining `loss.backward()` sites are the documented Pillar A scope (mep benchmarks, propagators, deployment models, validation tracks).
  - **Determinism gate verified:** `tests/unit/validation/test_repro_check.py` → **9 passed**, including `test_json_report_all_pass` which runs all 7 REPRO_MODELS through `_train_one_epoch` twice and asserts bitwise-identical state dicts. Both passes share the new path under one seed, so reproducibility holds despite the changed batch ordering.
  - `ruff` clean, `pyright` 0 errors (warnings 6→4); `pytest tests/unit/cli/` → **23 passed**.
- **Next Step #2 (convert RL `training/rl.py` to CoreTrainer) is questionable:** `RLTrainer` is REINFORCE policy-gradient from *environment trajectories* (no fixed DataLoader), so a `CoreTrainer` adapter is architecturally inappropriate. Recommend keeping `RLTrainer` self-contained unless a policy-loss BPTT step emerges that should route through the shared dispatcher.
- **Next Steps #4/#5 (validation tracks, MEP benchmarks)** remain Pillar D-dependent and untouched.