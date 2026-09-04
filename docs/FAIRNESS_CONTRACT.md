# PR-6 — Evaluation Fairness Contract (Draft v0.1)

> **TODO11 R11.4.2 / RESEARCH3 PR-6.** One pre-registered document,
> written once, consumed by four items: the benchmark paper, the Z3
> discovery pre-registration, the edge comparisons, and the ICL bridge.
> Zero compute. Amendments after any consumer's pilot promotion require a
> new section (append-only), never edits to what a consumer already used.

The contract operationalizes RESEARCH3's E-4 (baseline protection) for
every comparison this repository publishes. Its scope is *methodology
fairness*: identical treatment of arms on data, budget, stopping, seeds,
and reporting. It is not a claim about outcomes.

## F-1 Tuning budgets

- Budgets are **wall-clock GPU-hours, never epochs** — epoch counts reward
  rules with cheap steps (Euclidean) over expensive-per-step rules
  (contrastive settling, orthogonalized updates).
- Every arm in a comparison receives an identical budget from E-9's class
  envelope for that item, fixed before the first arm trains.
- A budget's spend is logged per arm (config, device, walltime); the log is
  part of the run's manifest (E-3).

## F-2 Early stopping

- One stopping rule per experiment, stated in its pre-registration and
  applied identically to every arm (E-6 plateau rule: windowed relative
  improvement below the registered threshold over the registered window).
- No per-arm hand-tuned patience. A rule that cannot reach the plateau
  criterion within its F-1 budget is reported as walled-at-budget, with
  its best checkpoint — not silently extended, not silently dropped.

## F-3 Seeds and splits

- Minimum seed counts per experiment class: toy suites ≥ 3, flagship ≥ 5,
  campaign/frontier ≥ 5 with seed-stratified task order (E-5: task order
  randomized across seeds where order could matter).
- Splits are declared in the task config and versioned with the run;
  adaptation and evaluation streams stay disjoint (E-5 checklist), and
  every split boundary is asserted in the harness, not assumed.
- Seeds are recorded per arm in the run manifest; a figure's data layer is
  regenerable from seeds + config hash alone (E-3).

## F-4 Capacity and cost disclosure

- Parameter counts, adaptation FLOPs, and peak memory are reported per arm
  (proxy tier: `ResourceUsage` / `core/profiling.py`, R11.3.2b) so residual
  scale asymmetry stays visible next to every comparison.
- Comparisons are performance-gated, not parameter-matched (RESEARCH3
  ICL-bridge scale-matching rule): each mechanism qualifies at whatever
  scale reaches its gate within the equal budget; unmatched capacity is
  disclosed, never normalized away.
- Structure-vs-capacity claims (e.g. D8's conv retention) assert the
  capacity ordering explicitly in the demonstrating test.

## F-5 Multiple comparisons

- When more comparisons than the pre-registered headline are reported, all
  of them are shown unfiltered, or a correction is applied and named.
  Cherry-picking is a triage failure (E-7), not an editorial choice.

## F-6 Pipelines

- Training fairness mirrors export fairness: the same quantizer, the same
  calibrations, and the same evaluation pipeline for every arm (local-rule
  exports and quantized baselines alike).
- Control arms from E-10 (lr=0 planted control, chance-line floor) run in
  the same process and device regime as the arms they validate.

## Consumers

| Item | What it takes from this contract |
|------|----------------------------------|
| Z3 flagship (R11.3.5) | F-1 budgets, F-3 seed floor, ICL-bridge gating |
| Benchmark paper (20-rules) | F-1/F-2 across all rules, F-4 disclosure table |
| P-axis frontier (R11.3.4) | F-4 𝒞-vector reporting, F-5 pipeline parity |
| Task-family generalization (R11.3.7) | F-3 splits, F-5 identical pipelines |

## Status

Draft (v0.1, 2026-09-03). Each consumer's pre-registration cites this file
and pins its concrete numbers (hours, seed counts, thresholds) at pilot
promotion; this document fixes the *rules*, consumers fix the *values*.
