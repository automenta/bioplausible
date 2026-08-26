# Evaluation Fairness Contract (PR-6)

> Pre-registered evaluation rules for any benchmark-paper claim from this
> codebase. Register an instance **before** running: copy
> `docs/preregistration_template.md`, fill every `MUST` field, commit the JSON
> to `configs/preregistrations/`, then run. Working example:
> `configs/preregistrations/eqprop_mnist_80pct.json`.

## 1. Scope

Applies to all empirical claims entering RESEARCH3 deliverables: benchmark
paper coordinates (L1–L3.5), Z3 flagship, ICL bridge, edge/green comparisons,
and discovery-campaign final reports. Claims made before this contract fall
under it at rerun time.

## 2. Compute budgets — GPU-hours, not epochs

Epochs are not comparable across rules (settling depth differs per dynamics).
Every registration declares a **GPU-hour budget per coordinate**, hard-capped:

| Rule family | Budget | Notes |
|---|---|---|
| Energy minimization (EqProp-class) | 2 h | settling loop dominates |
| Instantaneous (feedforward/backprop-class) | 0.5 h | |
| Predictive settling | 1.5 h | |
| Spike integration | 1.5 h | |
| Diffusion variants | 1.5 h | |

Budget consumption is measured with `ResourceUsage` (`core/profiling.py`) and
reported per run. A run that hits its budget stops early; its best checkpoint
under §3 stands as the result. No budget extensions after first results are
observed.

## 3. Early stopping & model selection

- Selection metric declared up front (default: validation accuracy).
- Checkpoint selection = **best validation value over the whole budget**
  (early stopping), with the final-epoch value reported alongside.
- Both numbers enter publications when they differ by more than noise (§4);
  reporting only the flattering one is a violation.
- Stopping/selection uses validation data only; test data is touched once,
  at the end, by the registered evaluation script.

## 4. Seeds & statistics

- Minimum **5 seeds** (`MIN_SEEDS` in `validation/preregistration.py`);
  seed list declared in the registration.
- Headline comparisons use `paired_comparison` from the same module:
  bootstrap CI (95%), permutation p-value, Cohen's dz/d reported together.
- "Improvement" requires the CI to exclude zero AND the effect size to be
  stated. Noise band for §3's difference report = half the CI width.
- Threshold claims (e.g., "reaches ≥80%") use `ThresholdRegistration.passes`
  semantics: the registered claim wording decides whether best-so-far or
  final-epoch counts.

## 5. Data splits

- MNIST-style datasets: 50k train / 10k val / 10k test, fixed permutation
  seed 42, splits committed as generated indices (no re-shuffling per run).
- Synthetic tasks (shakedown suites): generator seed equals episode/run seed;
  generators are constructed per-run so runs are independent.
- No test-set-informed hyperparameter changes; violations invalidate the run.

## 6. ICL-bridge scale-matching

Cross-paradigm comparisons (in-context vs weight-based learning) qualify a
small-system rule for scale-up only if it reaches **≥95% of the reference
task performance** on every task in the qualification battery. Qualification
runs obey the same budgets (§2), seeds (§4), and splits (§5). Scale-matched
pairs declare parameter count AND settling compute per step; matching is on
measured FLOPs/step (`ResourceUsage.compute`), not layer counts.

## 7. Deviations

Any deviation (budget change, split change, added seed, metric change)
requires a new registration file referencing the old one (`supersedes`
field) with the reason, committed before the new results exist. Post-hoc
changes are recorded in the results JSON under `deviations` and flagged in
any publication.

## 8. Artifacts per claim

1. `configs/preregistrations/<claim>.json` (committed pre-run)
2. Raw results JSON(s) with per-seed records incl. `resources`
3. Statistics summary (CI, p, effect sizes) produced by the PR-4 kit
4. This contract version hash in the results metadata
