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

## Status Summary

| Pillar | Scope | Status | Remaining |
|--------|-------|--------|-----------|
| B — Single Config Hierarchy | XL | **DONE** | none |
| C — Single Construction Funnel | M | **DONE** | none (tail folded into A) |
| A — Single Training Path | XL | **IN PROGRESS** | criterion #1 (`loss.backward()` in ~40 files); validation tracks; MEP benchmarks; RL decision |
| E — Single Result & Persistence Funnel | M | **OPEN** | architectural (not mechanical) |
| D — Single Measurement & Reporting Stack | XL | **OPEN** | blocked on E |
| G — Propagator/Model Unification | M | **OPEN** | blocked on A; criterion #6 |
| K — CLI & Interface Hygiene | M | **DONE** | optional: adopt `biopl` in CI; fix `lab.py` default |
| L — Self-Registration | M | **OPEN** | lazy-map shortlist; `vars(module)` adoption |
| J — Dead Code Tail | S | **PARTIAL** | `analysis/tile_*.py` gated on D |

---

## Remaining Pillars (Ordered by Value/Effort)

### 1. A — Single Training Path (XL, High) — IN PROGRESS
**Problem:** 7 parallel run stacks (CoreTrainer, TrialRunner, Verifier, StaircaseRunner, BenchmarkRunner, BioLightningModule, graph/training.py) re-implement `train_step` dispatch, model instantiation, device/seed, metrics, checkpointing.

**Target:** `CoreTrainer` is the ONLY training loop. Everything else is a thin adapter over `fit()`/`train_epoch()`.

**Completed so far:**
- **`dispatch_train_step` shared seam exists** (`core/trainer.py`): a module-level pure function owning the 5-phase routing (energy-model → learning-rule propagator → model `train_step` → learning-rule optimizer → BPTT fallback). `CoreTrainer._train_step` and `BioLightningModule.training_step` both route through it; PL stays the outer loop. **All future loop conversions should route through this seam, not hand-roll dispatch.**
- Deployment-model migration: `ConvEquiTile` trains through CoreTrainer (spatial input preserved; `input_format="spatial"`); `LightningExecutionCallback` added.
- `cli/repro.py` `_train_one_epoch` now uses `dispatch_train_step` (dedup'd the loop). **Note: this did NOT advance criterion #1** — the `loss.backward()` in its `_bptt` fallback remains (non-`train_step` families like `fa` legitimately use BPTT).

**Acceptance criterion #1:** `grep -rln "loss.backward()" bioplausible/` (outside `core/` + `training_mixin`) currently returns **~40 files** — semantically-distinct training loops (energy/dual-phase, spiking, tile substrate), so they cannot be mechanically folded into `CoreTrainer.train_epoch`. This is a **multi-session architectural effort**, not a bounded edit. Biggest clusters:
- `zoo/mep/benchmarks/*` + `zoo/mep/*` (6+), `zoo/propagators/*` (5+), `zoo/models/deployments/*` + `target_prop`/`tile_lm`/`forward_only` (7+)
- `validation/tracks/*` + `validation/utils.py` (7)
- `analysis/dynamics.py`, `analysis/energy_landscape.py`
- `execution/{robustness,interpretability,_guards}.py`, `graph/training.py`, `sklearn_interface.py`, `training/rl.py`

**Remaining concrete steps (ordered by tractability):**
1. **Validation tracks** → delegate execution to a CoreTrainer-based runner (declarative `track_*` spec stays). Partially Pillar-D-dependent.
2. **MEP benchmarks** → convert to registry-driven `BenchmarkRegistry` tracks. **Blocked on Pillar D.**
3. **RL training (`training/rl.py`) — recommend NOT converting.** `RLTrainer` is REINFORCE policy-gradient from *environment trajectories* (no fixed DataLoader); a `CoreTrainer` adapter is architecturally inappropriate. Keep it self-contained unless a policy-loss BPTT step emerges (then route it through `dispatch_train_step`).
4. **Thread engine device:** `ExecutionEngine._get_train_loader` passes `device="cpu"` regardless of the engine's real device (the non-PL path re-resolves via `TrialRunner`). A future pass should thread the engine's device through.

**Findings that will block/guide the work:**
- `BioLightningModule.create_model` helper was kept because `tests/integration/test_lightning_integration.py:368,417` patch `bioplausible.lightning_.module.create_model`. Deleting it (the Pillar A target) requires updating those tests first.
- `BioLightningModule` passes `propagator=None`/`optimizer=None` to the dispatcher and steps its own optimizer externally — this suppresses Phase-4 so its bio-optimizers keep `model.train_step → opt.step()` semantics. Re-evaluate once Pillar G lands (Phase-4 is their intended home).
- The EnergyModel dispatch branch is guarded by `config is not None`: CoreTrainer enables `_make_ebm_trainer`; `BioLightningModule` (no config) lets EnergyModels fall through to model `train_step`. Give the module a minimal config facade to unlock the energy path if needed.
- `LightningExecutionCallback` (`execution/callbacks.py`) is infrastructure, **not wired** — no production consumer yet. It is the bridge any remaining PL-logging work should consume.
- ConvEquiTile default tile config (`neurons_per_tile=64`, `tiles_per_layer=4`, `mode="backprop"`) diverges on MNIST (NaN); the small test config (`conv_channels=[4,8]`, `tiles_per_layer=1`, `mode="pc"`) trains stably — hyperparameter issue, not structural. Keep the small config for tests. Other deployment models (Graph/TimeSeries/RL EquiTile) use scalar `input_dim` and work without changes.
- `dispatch_train_step` is typed `dict[str, object]` by design: PL's automatic path must return a *tensor* loss; CoreTrainer's paths yield floats (and cast back to `dict[str, float]`). New callers must pick the right typing.

**Win:** ~2,500 lines removed; one place for bug fixes/features.

---

### 2. E — Single Result & Persistence Funnel (M, Medium) — OPEN
**Problem:** 5 write paths for outcomes (Optuna SQLite, HyperoptStorage, JSONL Report, KB, execution_state.db). `result_sink.record_experiment_result` is the canonical funnel but engine, verifier, mep-benchmarks write around it.

**Target:** `result_sink` is the ONLY writer. `record_experiment_result` owns Optuna `tell`, `hyperopt_logs`, JSONL `Report`, KB upsert, failure log. One artifact loader: `core/checkpoint.load_checkpoint` + `find_trial_artifact(trial_id)` (done). Ad-hoc saves → `CheckpointMixin`/`core.checkpoint`. Evaluate `CheckpointManager` (`execution/_lifecycle.py`) against `core.checkpoint`.

**Acceptance criterion #3:** `record_experiment_result` called by execution, hyperopt, validation, mep-benchmarks; all five backends written only from `result_sink`.

**Risk assessment (surveyed — do NOT do mechanically):** the `execution/engine.py` Optuna `study.tell/ask` (lines 558,597,639) and `state.failure_tracker.log_failure` (451) are the engine's **online HPO loop itself**, not outcome recording — folding them into `record_experiment_result` would conflate the search loop with the KB audit trail. `result_sink` already owns KB + FailureTracker and is called from hyperopt, validation tracks, trainer, and probe. **Treat the remaining unification as architectural, not mechanical.**

**Win:** ~700 lines removed; split-brain audit trails eliminated.

---

### 3. D — Single Measurement & Reporting Stack (XL, High) — OPEN (depends on E)
**Problem:** Parallel measurement ecosystems.
- `BenchmarkResult` ×5 (`evaluation/base.py`, `rigorous.py`, `compare_nanoGPT.py`, `tile_profiler.py`, `mep/runner.py`) — **do not mechanically merge** (semantically distinct). Establish `evaluation/base.BenchmarkResult` as canonical interface; others become Tracks/composites.
- Report renderers ×5 → one canonical JSONL renderer (`experiment/report.py`); others become thin adapters.
- Benchmark loops → registry-driven `BenchmarkRegistry` tracks (declarative, not new loops).
- Metrics: `core.losses.compute_accuracy` is canonical; fold remaining inline copies (only legitimately-different sites remain: 3-D per-token, accumulation, PL raw tensors).
- Leaderboard/ranking ×3 → one implementation in `leaderboard/` + `cli/rank.py`.

**Risk assessment:** `experiment/reporting.render_report` (JSONL) is already the canonical `biopl-report` renderer; `analysis/reporting.generate_experiment_report` consumes Optuna trials (different input, not a duplicate). The 5 `BenchmarkResult`s are semantically distinct by design.

**Note:** unblocks Pillar A steps #2 (MEP benchmarks) and Pillar J (tile archive).

**Win:** ~4,000 lines removed; findings share schema/CIs/renderers.

---

### 4. G — Propagator/Model Unification (M, Medium) — OPEN (depends on A)
**Status:** Alias map done (`_PROPAGATOR_TO_MODEL` → `_ALIASES`; `Registry.get(PROPAGATOR, "ff")` returns model class).

**Remaining:**
- Collapse `CoreTrainer._train_step` (via the shared `dispatch_train_step`) 5→2 phases: `energy-model` → `model.train_step` → `BPTT`. Delete phases 2 & 4 (explicit propagator/learning-rule optimizer).
- Convert `zoo/propagators/{eqprop,fa,hebbian,backprop,spiking}.py` to model-side `train_step`s, or delete.
- `ComponentCategory.PROPAGATOR` shrinks to pure gradient transformers (Muon, spectral norm, EWC).

**Acceptance criterion #6:** `zoo/propagators/` contains only `mep.py` and pure-gradient-transform submodules.

**Risk assessment (blocking):** `zoo/propagators/{backprop,base,fa,eqprop,hebbian,spiking}.py` are **NOT dead code** — heavily imported by ~20 tests, `cli/repro.py` `_gradient_gate`, `validation/tracks/nebc_tracks.py`, and `bioplausible/__init__.py`. The "or delete" path is unavailable without first migrating those consumers to model-side `train_step` (risky). Criterion #6 remains open.

**Win:** ~800 lines removed; one interface; AutoScientist composition simplifies.

---

### 5. L — Self-Registration (M, Low) — OPEN
**Status:** `zoo/models/eqprop/__init__.py` auto-computes `__all__` from `vars(module)`. Registry has `aliases()` + `resolve_alias()`.

**Remaining:**
- Reduce `bioplausible/__init__.py` and `core/__init__.py` `_LAZY` maps to a declared shortlist of public API.
- Other leaf re-export subpackages (e.g. `fa.py`-style) adopt the `vars(module)` pattern with per-file `ruff` ignores.

**Win:** Adding a model/rule = one registration decorator; nothing else to touch.

---

### 6. J — Dead Code Tail (S, Low) — PARTIAL
**Done:** `TODO.md` + `REFACTOR.md` archived to `docs/archive/20260813/` (the active plan is this file).

**Remaining:** `analysis/tile_*.py` legacy systems are superseded by `evaluation/` + `mep/benchmarks`. **Do NOT archive until Pillar D lands** (gated on the measurement stack).

---

### 7. K — CLI & Interface Hygiene (M, Low) — DONE (optional additive)
`biopl` lazy dispatcher implemented over `run | report | parity | repro | hpo | audit | frontier | rank | lab`; `DASHBOARD` global decoupled via `EventSink` (`execution/events.py`); 14 dispatch tests green.

**Optional remaining:**
- Adopt `biopl` in CI. The individual `biopl-*` console scripts are kept because `.github/workflows/ci.yml` (lines 31,33,59) references `biopl-registry-audit`, `biopl-repro-check`, `eqprop-verify`; they are now thin `biopl`-delegating shims. Any future script-removal must update CI + README/docs or keep the shims.
- `cli/lab.py` `args.model="MLP"` is **not a registered name** (registry uses `backprop_mlp`, `eqprop`, `forward_forward`, …). The default is misleading; map a friendly shortlist or error clearly.

**Win:** Clean public API boundary; headless CI/sweeps work.

---

## Deprioritized (Explicitly Not Now)
- **God-Object Decomposition (O):** `core/trainer.py`, `knowledge/kb.py`, `execution/strategy.py` — split only when Pillars A/E/D touch them; cap effort; stop when cohesive.
- **Settling Loop Merge (I):** Family A/B convergence loops — high numerical risk, low gain. Telemetry unification done.
- **Visualization Stack Consolidation:** 4 stacks — UI preference, not architectural flaw.
- **Micro-Consolidation Remainder (M):** ~12 inline accuracy folds (3-D/accumulation/PL) are legitimately different; `count_parameters` + seeding done.

---

## Execution Sequence & Metrics

| Phase | Pillars | Metric | Status |
|-------|---------|--------|--------|
| **1. Foundations** | B → C | `grep "class ModelConfig"` = 1; `grep "model_cls("` outside construction = 0 | **DONE** |
| **2. Core Unification** | A → E → D | `grep "loss.backward()"` outside core = 0; 100% outcomes via `result_sink`; 1 `BenchmarkResult` interface | A in progress; E/D open |
| **3. Clarity** | G → K → L | 0 propagator-only loops; `biopl` dispatcher works; 1 registration decorator | K done; G/L open |

**Acceptance Criteria (all must pass):**
1. Import-DAG checker passes in CI (Pillar N gate).
2. `CoreTrainer` sole owner of BPTT/optimizer step logic. **Not done** — `loss.backward()` in ~40 files (Pillar A).
3. No split-brain persistence (all writes via `result_sink`). **Not done** — Pillar E.
4. Zero new test failures beyond the 6 pre-existing numerical/parity drifts.
5. `grep` criteria from criteria #1–#6 all satisfied. **Criteria #3 (construction) and K (dispatcher) done; #1 (A) and #6 (G) open.**

---

## Current Baseline
**Full suite:** 2002 pass / 6 fail / 10 skip / 1 xfail (6 failures = documented numerical/parity drift, unrelated to refactor).
**Lint gate:** Functional (ruff 0.16 parses config; ~2k pre-existing warnings are backlog, not blocker).
**Pyright:** 0 errors strict mode.

---

## Consolidated Guidance for Future Sessions

### Architecture invariants
- **Task geometry is ambiguous by design:** `TaskProtocol.input_dim` is typed `int | None` but concrete tasks return *tuples* (e.g. `mnist → (1,28,28)`). The single task-resolution seam (`domains/registry.resolve_task_from_data_config`) must thread geometry **straight through** to `construct_model` (matching `_build_runconfig_model`), never `int()`-coerce it. Flattening (`math.prod`) lives only in `domains/registry.resolve_task` (the scheduler's geometry view).
- **`_create_model` needs no `int()` coercion:** `_setup_data` seeds `model_kwargs["input_dim"]` from the task; `_create_model` uses a `None` check that preserves tuples. Any caller hand-building `TrainerConfig` must pass a task name (so geometry resolves) or include `input_dim`/`output_dim` in `model_kwargs`.
- **Shared `dispatch_train_step` is the single train-step seam** (`core/trainer.py`). New loops must route through it.

### Toolchain / environment gotchas
- **`python -m bioplausible.cli <cmd>` shows the wrong `prog`** ("python3 -m bioplausible.cli") vs the installed `biopl` script ("biopl rank"). Cosmetic runpy vs. entry-script difference; verify via `uv run biopl ...`, not `python -m`.
- **argparse `--help` raises `SystemExit(0)`** before the adapter body runs — a dispatcher calling adapter `main` directly must catch `SystemExit` and remap `exc.code`, or the CLI exit status is wrong under the console script.
- **Python 3.14 allows `except A, B, C:`** (tuple-of-exceptions form) without parentheses — old-style clauses are valid; `ruff format` strips redundant parens back to it. Do not "fix" them.
- **Engine loader device:** `ExecutionEngine._get_train_loader` passes `device="cpu"` regardless of the engine's actual device (the non-PL path re-resolves via `TrialRunner`). A Pillar A pass should thread the engine's device through.

### Acceptance-criteria status (as of latest)
- Criterion #1 (`loss.backward()` outside `core/`+`training_mixin` = 0): **NOT done** — ~40 files (Pillar A scope).
- Criterion #3 (`model_cls(` outside construction = 0): **DONE**.
- Criterion #6 (`zoo/propagators/` = only `mep.py` + gradient transformers): **NOT done** — `backprop.py`, `base.py`, `eqprop.py`, `fa.py`, `hebbian.py`, `spiking.py` present (Pillar G scope).
- Criterion "`biopl` dispatcher works" (Pillar K): **DONE**.