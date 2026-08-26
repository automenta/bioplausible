# Pre-registration Template (PR-4)

Every empirical claim that gates a decision (paper, campaign gate,
benchmark contract) must be registered here **before** the confirming run.
Copy this template into `configs/preregistrations/<claim-slug>.json`,
commit it, then evaluate with
`computronium.validation.preregistration.paired_comparison` +
`PairedComparison.passes(ThresholdRegistration.load(...))`.

## Contract

| Field | Meaning | Rules |
|-------|---------|-------|
| `claim` | Statement under test | Falsifiable, one sentence |
| `metric` | Metric name | Must match the key written to results JSON |
| `threshold` | Minimum material effect | Treatment − control, metric units |
| `alpha` | Family-wise error rate | Declared once; no peeking |
| `min_seeds` | Per-arm seed floor | ≥ `MIN_SEEDS` (= 5) |
| `created` | ISO date committed | Must precede the confirming run |

## Decision rule

Confirm iff **both** hold on matched-seed paired results:

1. bootstrap 95% CI lower bound > `threshold`
2. sign-flip permutation `p_value` < `alpha`

(`PairedComparison.passes` implements exactly this.)

## Tuning budget

Per-rule tuning budgets are denominated in **GPU-hours**, not epochs
(PR-6 fairness contract). Record the budget spent alongside the claim
result; exceeding a registered budget voids the comparison.

## Worked example

See `configs/preregistrations/eqprop_mnist_80pct.json` — the TODO4 §7.2
EqProp ≥80% MNIST accuracy claim, evaluated as a one-arm threshold check
against chance (control arm = chance-level accuracies).
