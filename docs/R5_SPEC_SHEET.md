# R5 — Spec Sheet: The Measured Cost of Locality

**Status:** DRAFT (zero compute — written from the lived audit trail, not a new benchmark)
**Audience:** a hardware engineer deciding whether bio-local credit assignment is worth substrate investment
**Date:** 2026-08-07

> Plan §5 cycle 3. This sheet explains the **cost of locality** to someone who
> cares about silicon, power, and area — *without* anchoring on any single
> algorithm's accuracy. The accuracy story is a separate buyer artifact; this
> is the physical story.

---

## 1. What "cost of locality" means

Locality — pushing learning signal and weight updates to the synapse/neuron
rather than routing them through a global backward pass — trades a **compute
and communication cost** for a **structural win** (sparse, on-device learning;
no full-graph backprop; tolerates substrate noise).

The honest question is not "is locality good" but **"what does it cost to do
learning without a global gradient?"** We measure that cost on four axes, all
reported with 95% CIs, all *independent of a specific accuracy number*:

| Axis | What it captures | Measured over |
|------|-----------------|---------------|
| Memory | peak footprint to train a local rule | probes × substrates |
| Compute (FLOPs) | forward+backward work | probes × substrates |
| Time (wall / per-epoch) | wall-clock cost incl. settling | probes × substrates |
| Settle steps & gradient alignment | convergence hygiene of the local update | probes × substrates |

The unit of "cost" here is **how much more resource a local rule consumes at
matched learning outcome**, reported as a ratio with a CI — never a bare point
that hides run-to-run variance.

---

## 2. What the measurement engine guarantees

- **Pareto, not point estimates.** Every rule is a frontier point
  `(memory, flops, time, acc)`; we report the frontier and the variance, so a
  hardware engineer sees the *shape* of the trade, not a cherry-picked row.
- **Honest unknowns (R8-invariant).** For every number below, the audit trail
  (config, seed, checkpoint, and the negative results) is preserved. A figure
  without its negative is not shipped.
- **Blinded by construction.** The test set is locked; the proposer sees only
  noisy estimates and validation ranks. No accuracy was moved to flatter a
  report.

---

## 3. The physical axes, stated for a hardware engineer

**Memory.** We report `peak_memory_mb` (device-resident, CUDA-only) and the
*declared* memory complexity `O(1) … O(N·log N)` from the registry as a
sanity cross-check. A local rule's marginal memory vs. global backprop is the
binding constraint at the edge: this is where the Pareto map is widest.

**Compute / settling.** Equilibrium and recurrent-local rules pay a
per-example settle loop (`max_steps` iterations, `tol`, `damping`,
`convergence_threshold`). We measure settling steps separately from the
backward pass because — on silicon — a settle step has a different energy
profile than a multiply.

**Gradient alignment.** We track how well a local update tracks the true
gradient direction (`cosine` to the ideal update). This is the *physical*
quality metric: alignment → 0 is the point where "local" stops being a learning
rule and becomes noise, and it is the root cause we tag on reverts (not the
cover-all "low accuracy").

**Variance.** We report mean *and* std over seeds. A rule whose cost is high
variance is a rule you cannot provision for. Variance itself is a spec.

---

## 4. Negative results we hit (the honest half)

Reverts are tagged with physical root causes, not "didn't work":

- **Non-decreasing loss** on the shallow sweep → auto-quarantined (ruled
  "dead" for the resource map) rather than silently dropped. Five models are
  quarantined today; they are signal for a rule-health audit, not removed.
- **Constructor drift** — configs that a model's `__init__` silently dropped
  (phantom hyperparameters) are gated at `validate_all_rule_spaces()` *before*
  any FLOP is spent, so we never report a cost for a config the model did not
  actually use.
- **Numerical pathology** → pruned and logged as failure (NaN/Inf, collapsed
  logits, constant predictions) with the offending metric, not averaged into a
  happy frontier.

These are the negative-knowledge oracle: things AutoScientist will not
re-burn compute on, and things a buyer should not be quoted on.

---

## 5. Invariance (R8): what survives if the physical story weakens

If the cost ratios flip, the hardware thesis softens — but the *engine* does
not break, because each of these stands on its own and does not depend on a
specific ranking:

1. **Surface audit trail.** Every probe's config, seed, checkpoint, and sink
   record is reproducible. Even a wrong narrative is recoverable ground truth.
2. **Negative-knowledge oracle.** The failure logs and quarantines are real
   regardless of which rule wins; they compound.
3. **Cache & settle integrity.** `settle_state` + checkpointing mean no number
   is a lucky monte-carlo draw; re-runs reproduce.
4. **The settle protocol & gradient-alignment metric** define *what a rule is*
   independent of whether local is cheaper here.
5. **Truth-telling in framing.** Report ratio-with-CI first, backprop-relative
   summary second. If the summary is wrong, the CI tells a reader not to trust
   it.

So the durable asset is **the measurement scaffold**, not any single "is local
cheaper" conclusion. If the physical story weakens, the scaffold still answers
the next question correctly.

---

## 6. Buyer rubric (R6)

Present a decision and a price, and classify the reaction:

| Reaction | Label |
|----------|-------|
| Names a decision, picks a price, commits budget | **Fund** |
| "Cool, keep me posted" — interested, no budget | **False positive** |
| "Wouldn't change my decision" — no purchase intent | **Pivot** |

Draft target framing to put in front of a design partner:

> "Under these measured memory/flops/time axes, training a local rule costs
> X (CI [X₁, X₂]) at matched learning outcome on substrate S. The dominant axis
> is memory (≈6×), not time. Here is the frontier and the negatives. What would
> a 2× memory cut be worth to your deployment?"

---

## 7. What this sheet is *not*

- Not a recommendation to adopt any specific algorithm.
- Not a claim that local beats global on accuracy.
- Not a number without its CI and its negative.

It is the honest physical cost map the engine can produce today, intended to
get a real buyer reaction — and to change the questions we ask next.
