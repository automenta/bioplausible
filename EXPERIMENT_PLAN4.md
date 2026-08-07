# EXPERIMENT_PLAN4.md — The Relaxed Discovery Loop

**Status.** PLAN3's build items are landed. The suite is green (2008 pass / 15 skip / 5 xfail, ~60% coverage). The measurement tooling exists. The remaining work is **running the loop**, not building more gates.

---

## The Core Thesis (from the Manifesto)

The product is the **Epistemic Engine** (AutoScientist + Knowledge Base + honest measurement), not any single bio-plausible algorithm. If we pick a "flagship" now, we bias the engine to validate that choice. The engine must be able to evaluate *any* algorithm fairly.

---

## The 3-Step Loop (Repeat Until Conviction)

### 1. Broad Sweep (Low Compute)
**Goal:** Map the territory, not pick winners.

- Run **shallow sweeps** (1–2 epochs, 3–5 probes/family) across **all registered families**: EqProp, FA, Hebbian, Forward-Forward, Predictive Coding, STDP, etc.
- **Metrics:** variance in memory, compute time, settling steps, gradient alignment — *not* accuracy.
- **Output:** a coarse Pareto landscape showing where each family lives in resource space.
- **No flagship selection.** No "honest 30-probe search" yet. Just breadth.

```
uv run python scripts/broad_sweep.py --epochs 1 --probes-per-rule 3 --families all
```

### 2. Engine Audit (Zero Compute)
**Goal:** Make the AutoScientist smarter, not the models more accurate.

- **Bias check:** Is the proposer over-optimizing for accuracy? Force it to propose for *memory efficiency*, *settling speed*, or *hardware noise robustness*.
- **Compositionality:** Can it generate hypotheses that combine families? (e.g., "EquiTile topology + Forward-Forward credit assignment")
- **KB ingestion:** Drop in a random external paper's algorithm — does the KB + proposer handle it without code changes?
- **Failure manifesto quality:** Are reverts tagged with physical root causes (Lipschitz > 1, gradient cosine → 0) or just "low acc"?

### 3. Market Reality Check (Zero Compute)
**Goal:** Validate the thesis with buyers, not benchmarks.

- Draft **R5 Spec Sheet**: "Here is the measured cost of locality under these physical constraints, with 95% CIs, and the negative results we hit."
- **Buyer rubric (R6):** Name a decision/price → Fund / "cool, keep me posted" (False positive) / "wouldn't change my decision" (Pivot).
- **R8 Invariance paragraph:** What survives if the physical story weakens? (Surface audit trail, negative-knowledge oracle, cache integrity, settle protocol.)

---

## What We Explicitly Defer (No Longer Gates)

| Item | Reason |
|------|--------|
| 85% test coverage | ~60% is enough to run experiments. Chase it later if CI demands it. |
| Fixing 5 non-converging models | Quarantine them (`xfail`). They're signal for later rule-health audit, not blockers now. |
| "Honest" 30-probe flagship search | Premature. Do it *after* the engine can propose across the whole landscape. |
| Substrate P4-lite/P4-full | Branch A is recorded (facades faithful on LoopedMLP). Run when a buyer asks for it. |
| Roofline / Memory Wall / Hardware Tax benchmarks | Build when a specific buyer conversation needs them. |

---

## Standing Discipline (Minimal)

- **Never measure before the suite is green.** (Already satisfied.)
- **Sink everything.** Wins and reverts go to `result_sink` with structured tags.
- **Blinded trials by default.** Proposer sees validation ranks / noisy estimates only. Test set locked.
- **No backwards compatibility.** Delete dead code, move fast.

---

## Current State (One Source of Truth)

| Asset | Status |
|-------|--------|
| Suite | Green (2008 pass, ~60% cov) |
| P0a Integrity Gate | Live (`validate_all_rule_spaces()`) |
| P1 Settle Protocol | Live (`settle_state` + checkpointing) |
| P2 Read-Half | Live (`query_conditionals` + `avoid_characterized`) |
| P3a `select_flagship()` | Live (KB query, geomean-cost rank) |
| P3b Memory Lever | Live (checkpointed settle) |
| P4-lite Verdict | **Branch A** — facades faithful on LoopedMLP (`scripts/p4lite_surrogate_sanity.py`) |
| Market Probe (R5–R8) | **▶ READY TO DRAFT** — zero compute |

---

## Next Action (Right Now)

1. **Run the first Broad Sweep** (1 epoch, 3 probes, all families) → log to KB.
2. **Draft the R5 Spec Sheet** — explain "cost of locality" to a hardware engineer without citing a specific algorithm's accuracy.
3. **Audit the Proposer** — force one proposal cycle optimizing for *memory efficiency* instead of accuracy.

That's it. No 10-phase gauntlet. Just the loop.