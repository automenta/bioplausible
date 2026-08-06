# Bioplausible — Experiment Layer: Status & Improvement Plan

**Status (2026-08-06): BUILT.** The thin experiment layer described in the (now-retired)
`EXPERIMENT_ARCHITECTURE.md` is implemented, tested, and verified end-to-end. This file is the
successor revision: it records precise current status, the decision rationale worth keeping, and a
prioritized set of concrete **improvement opportunities** for the next phase. It is *not* a build
runbook — the code is the source of truth for what each module does
(`bioplausible/experiment/*.py`, `validation/statistics.py`, `validation/gradient_check.py`,
`domains/registry.py` all carry module + public-API docstrings).

---

## 1. Current status (grounded, not claimed)

The full experiment process is wired and green:

| Step | Verified |
|------|----------|
| Tooling gate | `ruff format --check` + `ruff check` clean on new code; `pyright` 0 errors on `experiment/` |
| `biopl-run validate` | rejects unknown task, `seeds<10`/missing `matched_by`/missing dual-`energy` on evidence stages |
| `biopl-run plan` | exact in-budget probe count + `estimate_total_time`; matches `run` (budget-filtered) |
| `biopl-run run` | trains + appends JSONL; budget filter rejects over-`max_params` configs pre-training |
| Re-run | true no-op (skips finished probes before any model construction; ~1.4s demo) |
| `biopl-report` | parity table (mean±bootstrap-CI, params, epoch_s), Cohen's d / Cliff's δ, Pareto, failure manifesto |
| `biopl-repro-check` | `--gradient` gate on 8 aligned families; `--resume-check` verifies no-op |
| Overnight config | `smoke` gate fixed (15 epochs / 0.60 on xor) so all 9 models advance; 6 in-budget parity models verified training 1-epoch cifar10 `ok` |
| `--producer bayes` | **NEW**: `OptunaBayesProducer` (TPESampler) behind `ConfigProducer`; `plan run --producer grid|bayes --candidates N` (**B.1**) |

Target test suite: **86 passing** (`tests/unit/experiment/` 43 + `validation/test_statistics.py` 27
+ `domains/test_registry.py` 7 + `integration/test_gradient_equivalence.py` 9). Staircase
survivor-gate coverage 90%.

**The one thing not yet done** is the literal overnight parity run:
`nohup biopl-run run examples/parity_cifar10_mlp.yaml` with a long wall-clock budget. Prereqs
met: `cifar10` on disk (`./data`), network not required, all in-budget models construct and
train on cifar10. Expected budget ~37.5h (equilibrium probes ~30–70s each).

---

## 2. Decision ledger worth preserving (non-obvious "why")

Gotchas and scope decisions that a future reader cannot recover from the code alone:

- **Grid must be enumerated once per stage** (Optuna `GridSampler`, one study/stage), not per
  (stage, model); the objective must call `trial.suggest_categorical` for every grid column or
  Optuna emits empty `{}` params. `producer.py:72-93`
- **Budget filter is schedule-time and pre-training**: a `(model, config)` whose training-free
  `estimate_param_count` exceeds its arm's `max_params` is dropped before any compute;
  `plan`'s count comes from the same enumeration so it matches `run`. `staircase.py:229-250`, `cli.py:75-114`
- **Resume no-op ordering matters**: a config whose seeds are all already recorded must be
  skipped *before* the param count (otherwise a re-run rebuilds models). Report resume gotcha in
  tests: unlink the report file *before* constructing `Report` (its in-memory `_finished` index
  loads at construction). `staircase.py:258-297`, `test_experiment.py:378-418`
- **The driver threads the campaign `compute` block** into `TrainerConfig`; CoreTrainer computes
  flops+memory+energy under one `track_energy` gate, so the driver enables it when *any* of
  flops/memory/energy is requested. `probe.py:186-201`, `test_probe.py:67-70`
- **Task geometry is derived from the concrete task** via the domain factory (`resolve_task`),
  never hardcoded; `SUPPORTED_TASKS` excludes network-fetching sets (cifar100/svhn, graph).
  `registry.py:23-45`, `registry.py:62-92`
- **Bayes vs grid is a pure sampler swap** (`producer.py`): both `HyperoptGridProducer` and
  `OptunaBayesProducer` expose the same `configs_for(stage) -> list[dict]` seam, so the probe
  count is sampler-defined — `grid` = grid cardinality, `bayes` = `n_candidates` (capped at
  cardinality so TPE never degenerates into an exhaustive grid). `plan` and `run` consume the
  same producer so the count stays 1:1 under either flag (`cli.py:_producer`).

---

## 3. Improvement opportunities (prioritized)

### A. Overnight execution (highest value; run it)
- **Run `examples/parity_cifar10_mlp.yaml` overnight**; then `biopl-report` and `biopl-repro-check
  --gradient` + `--resume-check` on the result.
- **Smoke-gate calibration**: the 15-epoch/0.60 xor gate is empirically defensible but tuned to
  one measurement; re-check thresholds if the model set changes (aim: pass all healthy trainers,
  reject chance-level non-learners). `examples/parity_cifar10_mlp.yaml:18-23`
- **Per-probe `wall_time_s` / `peak_memory_mb` are not populated on CPU** — **DONE (A.1)**:
  the driver now derives `wall_time_s` from the summed `epoch_time_s` so
  `matched_by.reported: [wall_time_s, ...]` is meaningful on CPU
  (`probe.py:214-223`). `peak_memory_mb` remains CUDA-only (CoreTrainer only
  reports it when the EnergyTracker profile runs on a CUDA device).

### B. Zoo / engine breadth (deferred scope, now re-openable)
- **Bayesian sampling**: swap `GridSampler` for `TPESampler`/`NSGAIISampler` behind the
  `ConfigProducer` interface (no other change needed). `producer.py:49-57`, `producer.py:60-93`
- **Concurrent probe scheduling**: parallelize within/across probes behind `ProbeDriver`/
  `ConfigProducer`, once determinism (seeding, report append) is proven. `probe.py:100-113`, `producer.py:49-57`
- **Non-CoreTrainer engines** (NumPy/Triton/spiking): a `ProbeDriver` impl, no core change.
  `probe.py:100-113`
- **Conv / language / cross-domain parity** (full §0.1 breadth): new Stages + registry tasks.
  Currently 3 of 37 registered models don't construct at MLP geometry (`backprop_transformer_lm`,
  `conv_eqprop`, `custom_stacked_model`) because they need LM/conv/spatial ctor kwargs that the
  MLP-geometry `build_model_kwargs` doesn't map — a concrete starting point for LM/conv parity.
  `param_estimator.py:107-148`, `registry.py:23-45`

### C. Measurement & instrumentation honesty
- **CoreTrainer `compute.track` gating quirk** (residual): CoreTrainer's EnergyTracker computes
  all of flops/memory/energy under the single `track_energy` gate; `track_flops`/`track_memory`
  are otherwise ignored. The experiment layer bridges this, but CoreTrainer's own flag semantics
  are inconsistent and worth normalizing (gate profiling on any-of). `trainer.py:126-128`, `trainer.py:776-803`, `probe.py:186-201`
- **`peak_memory_mb` CUDA-only / `wall_time_s` unset** — see A.
- **loky semaphore/resource_tracker warning** on shutdown from `hyperopt`/`execution` imports;
  benign on small runs, surface observed on a long run — worth a look before a long overnight.
  `hyperopt/__init__.py` imports, `execution/__init__.py` imports

### D. Cleanup / refinement
- **Model-constructibility census** — **DONE (D.1)**: a process-scoped memo
  (`_PARAM_COUNT_CACHE` keyed by frozen `(model, dims, config)`) reuses the
  exact estimator, so `plan` and each `run` stage count a given triple once.
  `param_estimator.py`, `test_param_estimator.py`
- `test_experiment.py` still carries tolerated `N802`/`ARGu`-style test-double nits (relaxed
  baseline); splitting the large file is optional, low priority.
- Consider a `docs/` home for the decision ledger if this file grows; keep the code as the sole
  design authority.

---

## 4. Concrete implementation paths (new section)

### A.1 Derive `wall_time_s` from `epoch_time_s` in driver — DONE
**Commit date**: 2026-08-06  
**Files**: `bioplausible/experiment/probe.py`, `tests/unit/experiment/test_probe.py`
**Why**: CoreTrainer never populates `TrainingMetrics.wall_time_ms` on CPU
(memory/flops remain CUDA-only EnergyTracker profile values), so every
`ProbeResult.wall_time_s` defaulted to 0 — making
`matched_by.reported: [wall_time_s]` in the parity contract vacuous.

**What was done**:
1. `CoreTrainerDriver.train` (`probe.py:214-223`) now returns
   `"wall_time_s": total_time` (the summed per-epoch time), alongside the
   existing `epoch_time_s`.
2. `run_probe` (`probe.py:283-284`) reads `metrics.get("wall_time_s", 0.0)`
   into `ProbeResult.wall_time_s`.
3. `test_probe.py::test_driver_threads_compute_settings_into_trainer_config`
   asserts `out["wall_time_s"] == 1.0`.

**Gate**: `ruff check` + `ruff format` clean; `pyright` 0 errors; tests green.

### A.2 Smoke-gate threshold validation
**File**: `examples/parity_cifar10_mlp.yaml:18-23`
**Action**: After overnight run, check `biopl-report` output for:
- All 9 models PASS smoke stage (acc ≥ 0.60 on xor, 15 epochs, 1 seed)
- Any false positives (chance-level learners passing) → raise threshold
- Any false negatives (healthy trainers failing) → lower threshold or add warmup epochs

### A.3 Overnight-run reliability hardening — DONE
**Commit date**: 2026-08-06
**Files**: `staircase.py`, `report.py`, `cli.py`,
`tests/unit/experiment/{test_experiment,test_cli}.py`
**Why**: three failure modes could silently kill a 37.5h run and lose the
resume contract (architecture §6.7). Each is now isolated so the run either
continues or surfaces a clear resume path.

**What was done**:
1. **Per-config param-count isolation** (`staircase.py:289-303`): a `(model,
   config)` whose static constructor raises is recorded as an errored probe
   (seat in the failure manifesto) instead of aborting the whole cascade. The
   other models/configs still train.
2. **Torn report tail** (`report.py:37-46`): `_load_existing` skips a
   truncated final line (crash between appends) instead of failing resume.
3. **CLI resume safety net** (`cli.py:200-215`): an unexpected exception or
   `KeyboardInterrupt` inside `runner.run()` returns a clean non-zero exit and
   prints the report path + "rerun to resume" (never a bare traceback losing
   the resume path).
4. **Resilient `plan`** (`cli.py:107-118`): `_in_budget_pairs` skips a
   non-constructible (model, config) (logged as a warning) instead of raising,
   so the overnight pre-flight `plan` always succeeds and matches `run`'s
   per-config isolation (configs in both back-ends stay 1:1).

**Gate**: `ruff check` + `ruff format` clean; `pyright` 0 errors;
`tests/unit/experiment/` 47 passing.

### B.1 Bayesian sampling behind `ConfigProducer` — DONE
**Commit date**: 2026-08-06
**Files**: `bioplausible/experiment/producer.py`, `cli.py`,
`tests/unit/experiment/{test_experiment,test_cli}.py`
**Why**: the grid is exhaustive (grid-cardinality probes) which can waste
compute on uninformative configs; a TPE-backed producer samples the same space
so `plan`/`run` shrink to a configurable candidate budget.
**What was done**: `OptunaBayesProducer` presents each grid column as a
categorical over its declared choices (identical space/budget to the grid) to
a `TPESampler`, and `configs_for` returns up to `n_candidates` distinct, seeded
in-grid points (`n_candidates` capped at cardinality so TPE can't degenerate to
a full grid). CLI gains `--producer grid|bayes --candidates N` on `plan`/`run`;
`_producer` builds either behind the same `ConfigProducer` seam. Decision ledged
in §2. `--candidates` affects only the bayes producer; `grid` ignores it.
**Gate**: `ruff check` clean on `experiment/`; `pyright` 0 errors; 93 passing
(experiment + statistics + registry + gradient).

### B.2 Concurrent probe scheduling
**File**: `bioplausible/experiment/staircase.py:252-297` (`_collect_probes`)  
**Approach**: Replace sequential `for seed in pending:` loop with `asyncio.TaskGroup`:
- Each probe is independent (different seed, same config)
- `Report.append` is thread-safe for JSONL (append-only, one line per probe)
- Need deterministic seed ordering for resume index consistency

### C.1 Normalize CoreTrainer profiling gates
**File**: `bioplausible/core/trainer.py:126-128`, `trainer.py:776-803`  
**Change**: In `_train_epoch`, replace `if self.config.track_energy:` with:
```python
track_any = self.config.track_energy or self.config.track_flops or self.config.track_memory
if track_any:
    with EnergyTracker(...) as et:
        ...
    if self.config.track_flops: metrics["forward_flops"] = et.profile.forward_flops
    if self.config.track_memory: metrics["peak_memory_mb"] = et.profile.peak_memory_mb
```
Then experiment layer `probe.py:186-201` can drop the `core_train_flag` bridging logic.

### C.2 Fix loky/resource_tracker warning
**File**: `bioplausible/hyperopt/__init__.py`, `bioplausible/execution/__init__.py`  
**Investigation**: The warning originates from `multiprocessing.resource_tracker` on
`loky` backend shutdown. Options:
- Set `LOKY_MAX_CPU_COUNT=1` for single-threaded runs (already `num_workers=0`)
- Explicitly `loky.get_reusable_executor().shutdown(wait=True)` at process exit
- Switch to `threading` backend for Optuna (`storage="sqlite:///..."` avoids multiprocessing)

---

## 5. Working agreement

- No backwards compatibility, ever (AGENTS.md).
- Modify only the layer + its tests; repo-wide legacy lint (~2560 errors) is a separate phase,
  not a blocker.
- Keep this file lean and forward-looking; if a non-obvious decision is made, append it to §2
  with a one-line rationale in the same commit as the code.

(End of file)