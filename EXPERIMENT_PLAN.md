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
| `--time-budget` | **NEW**: `plan/run --time-budget 1h` calibrates epoch time on the configured device, then auto-scales epochs/configs/seeds to fit (proportional planning: ~1% of budget) |

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
- **Device resolution is duplicated and must stay in sync**: `cli._resolve_device` resolves
  `auto` → `cuda:0`/`cpu`; `StaircaseRunner._resolve_device` must do the same. Until 2026-08-06
  the runner hardcoded `auto` → `cpu`, so `biopl-run run` with `compute.device: auto` ran on CPU
  while `plan --time-budget` calibrated on GPU — a silent 100%-CPU/0%-GPU mismatch. Fix in
  `staircase.py` (device auto-resolves to CUDA). `examples/parity_digits.yaml` is a faster parity
  sibling of `parity_cifar10_mlp` on the sklearn 8x8 `digits` task.

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

### A.2 Smoke-gate threshold validation**File**: `examples/parity_cifar10_mlp.yaml:18-23`
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

### B.3 Auto-scale to a time budget (`--time-budget`) — NEW
**Commit date**: 2026-08-06
**Files**: `bioplausible/experiment/cli.py`, `bioplausible/experiment/probe.py`,
`examples/parity_quick_smoke.yaml`, `tests/unit/experiment/test_cli.py`
**Why**: a fixed campaign (e.g. cifar10 parity) is ~37h, but a "useful
preliminary run" inside a wall-clock budget needs the schedule to fit the
operator's available time. Previously the only knobs were manual epoch/config
edits in the YAML.

**What was done**: `plan`/`run` accept `--time-budget 1h|30m|3600s`:
1. **Calibrate** (`_calibrate_epoch_times`): runs 1-epoch probes on the
   *configured* device (`_resolve_device` turns `auto` → cuda/cpu) for the
   bottleneck stage (max epochs), using 2 representative models (backprop + one
   equilibrium/FA) on a tiny config and a reduced number of batches (scaled
   with the budget), then extrapolates to a full 100-batch epoch.
2. **Proportional planning**: calibration batch count scales with the budget
   (~1% of it, clamped to a 10–100 batch floor/ceiling) so a 1h plan calibrates
   in seconds, not minutes — keeping planning ≪ budget (≤~1min for 1h).
3. **Auto-scale** (`_auto_scale_campaign` + `_reduce_stage_*` helpers):
   iteratively reduce epochs → configs → seeds (respecting minimums: 1 for
   smoke, 10 for evidence/baseline stages) until the calibrated estimate fits
   the budget; if the minimums can't fit it, print an honest infeasibility
   warning rather than silently overshooting.
4. The calibrated wall-time estimate is printed (vs the heuristic tier table).

`CoreTrainerDriver` gains an optional `batches_per_epoch` knob (probe.py) so
calibration can run fewer batches through the exact same training path.

**Gate**: `ruff check` + `ruff format` clean; `pyright` 0 errors; 97 passing
(experiment + statistics + registry + gradient).

### B.4 Device-resolution fix + run-time status + faster parity example — DONE
**Commit date**: 2026-08-06
**Files**: `staircase.py`, `cli.py`, `examples/parity_digits.yaml`
**Why**: three gaps blocked an actual overnight/quick parity run worth analyzing:
1. `StaircaseRunner._resolve_device` hardcoded `auto` → `cpu`, so a `device: auto`
   campaign trained on CPU even when CUDA was available ("100% CPU / 0% GPU").
2. `biopl-run run` printed nothing until the whole cascade finished — no live
   probe status on a long run.
3. No faster parity campaign than the ~37.5h `parity_cifar10_mlp`.
**What was done**:
1. `staircase.py:_resolve_device` now mirrors `cli._resolve_device` (`auto` →
   `cuda:0` if available else `cpu`); `cli._cmd_run` prints the effective device.
2. `staircase.py:_collect_probes` prints `[running]`/`[done]` per probe (model,
   seed, config, acc, elapsed) for live status.
3. `examples/parity_digits.yaml`: same evidence structure as
   `parity_cifar10_mlp` but on the sklearn 8×8 `digits` task (64-dim input vs
   cifar10's 3072) — network-free, far faster.
**Gate**: `ruff check` + `ruff format` clean; `pyright` 0 errors on changed files;
`tests/unit/experiment/test_cli.py` + `test_experiment.py` 44 passing.

### B.5 15-minute smoke run working end-to-end — DONE
**Commit date**: 2026-08-06
**Files**: `cli.py` (`_estimate_total_wall_time`, `_reduce_epochs_keeping_gates`,
`_MIN_EVIDENCE_EPOCHS`), `reporting.py` (`parity_table` n<2 guard),
`tests/unit/experiment/test_cli.py`
**Why**: the first `--time-budget` run was degenerate: it produced a single-survivor
report that could not even render. Two distinct bugs caused it.
**What was done**:
1. **Per-probe overhead overestimate** (`cli.py:_estimate_total_wall_time`): a flat
   8s per-probe overhead dominated ~0.1s GPU probes (~80x overestimate), so the
   auto-scaler concluded it "couldn't fit" and ground epochs to the floor of 1.
   Now bounded to the epoch cost (`min(8, max(1, per_epoch))`), keeping the
   estimate a conservative upper bound without swamping fast probes.
2. **Gate reachability** (`cli.py:_reduce_epochs_keeping_gates`): epoch scaling now
   applies only to evidence (baseline) stages, floored at `_MIN_EVIDENCE_EPOCHS=5`
   so parity accuracy stays analyzable. Gating (non-baseline) stages keep their
   full epochs — a pass rule unreachable in 1 epoch (e.g. xor `acc >= 0.60`)
   was rejecting 8/9 models and collapsing to a single survivor.
3. **Reporter n<2 crash** (`reporting.py:parity_table`): parity_table computed
   Cohen's d / Cliff's delta for every stage, but a gating stage runs 1 seed per
   model → `cohens_d` raised "requires at least 2 observations per group" and
   aborted the whole report. Effect sizes for n<2 are undefined and are now
   skipped (each `(model, n>=2)` pair guarded), not raised.
**Result**: `biopl-run run examples/parity_digits.yaml --time-budget 15m` completes in
~2.5 min (148.7s), all 9 models pass both stages, and `biopl-report --baseline
backprop_mlp` renders a full parity table + Pareto frontier + effect sizes
(e.g. eqprop d=-6.1, neural_cube d=-7.2, deep_hebbian d=-7.0 vs backprop on
digits; three_factor/pepita/forward_forward fall well below backprop).
`biopl-repro-check --resume-check parity_digits.report.jsonl`: resume no-op
verified for all 99 probes; repro 7/7 reproducible.
**Open residuals**: `_calibrate_epoch_times` still overestimates wall time for the
`--time-budget` estimate (calibrates a single warmup-inflated 1-epoch probe and
defaults non-bottleneck task cost to factor 1.0, so xor/smoke is charged as if it
cost the same as the bottleneck). The run still fits comfortably under budget
because scaling is conservative; tightening the calibration is optional polish,
not a correctness blocker.

### B.6 Calibration/estimator accuracy + a thorough parity run — DONE
**Commit date**: 2026-08-06
**Files**: `cli.py` (`_calibrate_epoch_times`, `_measure_epoch`,
`_estimate_total_wall_time`, `_reduce_stage_configs`, gradual `_auto_scale`),
`reporting.py` (zero-variance guard), `tests/unit/experiment/test_cli.py`,
`examples/parity_digits_thorough.yaml`
**Why**: the `--time-budget` estimator was ~80x too conservative, which under-filled
generous budgets and made it impossible to author a genuinely thorough run.
**What was done**:
1. **Per-task calibration** (`_calibrate_epoch_times`): every distinct task is now
   measured (not just the bottleneck), so cheap `xor`/smoke gets a real small
   figure instead of the old factor-1.0 assumption. `_task_cost_factor` removed.
2. **Warm-up amortization** (`_measure_epoch`): calib probes run `_CALIB_EPOCHS=3`
   epochs and average, removing the single-epoch CUDA-kernel-compile spike that
   inflated per-epoch cost ~3x.
3. **Per-model estimate** (`_estimate_total_wall_time`): totals now sum each model
   at its *own* calibrated per-epoch time instead of charging every model at the
   slowest one; setup floor lowered to a realistic 0.5s.
4. **Grid config reduction fix** (`_reduce_stage_configs`): the old code re-read
   the producer's original grid every call and could never shrink a 3x3 grid
   (the 6 smallest configs still span every value, reconstructing all 9). It now
   greedily drops the highest-cost value choice per key.
5. **Gradual auto-scaling**: per-pass reduction is clamped (max 30%) so the scaler
   converges to a schedule that *fills* the budget instead of jumping straight to
   the epoch floor.
6. **Reporter zero-variance guard** (`reporting.py`): `cohens_d` raises on a pair
   with zero variance (e.g. a smoke model scoring 1.0 on every config); such
   undefined effect sizes are now skipped, not fatal.
**Result**: `examples/parity_digits_thorough.yaml` (smoke xor full + parity digits
15 epochs × hidden [64,128,256] × 10 seeds) completes in **1200s (20 min)** on the
RTX 3080, all 9 models pass both stages, 288/288 probes, report renders with
effect sizes and a 3-point Pareto frontier. `resume-check`: 288-probe no-op
verified; repro 7/7. **Scientific takeaway corrected vs the 5-epoch smoke run**:
at 5 epochs the bio models spuriously "beat" backprop; at 15 converged epochs
backprop (0.965) is statistically tied-to-superior vs eqprop (0.954),
neural_cube (0.985, d=-2.5 vs backprop), while deep_hebbian/standard_fa/pepita/
three_factor/forward_forward underperform and diff_target_prop is high-variance.

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