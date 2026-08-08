# EXPERIMENT_PLAN7.md — Empirical Loop Plan (Post-6.0)

## Executive Summary

**Goal**: Give every bio-plausible component the room to demonstrate what it can actually do — by running the engine, finding the defects and hyperparameter gaps that hold each model back, and fixing them.

**Strategy**: Don't build infrastructure speculatively. Apply the 30-second EqProp fix, run the sweep, observe, fix the biggest gap, sweep again. Close the credibility gap in 45 minutes (the validation infra already exists). Defer every spec-compliance item until it actually blocks a result.

**Operating principle**: Every underperforming result is a defect or an un-tuned hyperparameter, not a verdict on the algorithm. The loop's job is to find the defect and fix it — giving each component its best shot. Plan-6 history proves this: every family that looked "dead" (FA all-skipped, Hebbian flat-loss, Target Prop 10%, Spiking random, EqProp zero-gradient) turned out to have a fixable bug, not a fundamental ceiling.

---

## Reality Check: What Already Exists (Don't Rebuild)

| Component | Status | Notes |
|-----------|--------|-------|
| **Backprop Parity Suite** (synthetic) | ✅ 24 pass | `tests/unit/validation/test_backprop_parity.py`, per-model YAML thresholds, FLOPs/memory checks |
| **Gradient Equivalence** (7 families) | ✅ 9 pass | `tests/integration/test_gradient_equivalence.py` + `bioplausible/validation/gradient_check.py` |
| **Registry Metadata Audit** | ✅ 286 pass | `tests/unit/validation/test_registry_audit.py` — instantiates, forwards, determinism, smoke all components |
| **Statistics Utilities** | ✅ 27 pass | `bioplausible/validation/statistics.py` — bootstrap CI, Cohen's d, Cliff's δ, BH-FDR, power |
| **Reproducibility tests** | ⚠️ 16 pass / 6 fail | Failing only on `equitile` (not a registered model name) |
| **CLI Parity** | ✅ 7 pass | `tests/unit/cli/test_parity_cli.py` |
| **Broad Sweep** | ✅ works | `scripts/broad_sweep.py` — all families, ~12 models/min on GPU |

**The validation infrastructure is mostly done.** The credibility gap is 45 minutes of polish, not weeks of new code. **Don't conflate spec-compliance with credibility.**

---

## Remaining from EXPERIMENT_PLAN6.md

All Plan-6 items are **DONE** except three:

| # | Item | Effort | Priority |
|---|------|--------|----------|
| 1 | **EqProp search space fix** (§7.6 / §8.1) | 1 line, 30 sec | **DO FIRST** — unblocks honest sweep on 6 models |
| 2 | **FabricPC over-budget** (§7.12 / §8.5) | 1 day | Mark budget-incompatible; flag in sweep; not a blocker |

---

## The Plan: A Single Empirical Loop

### Phase 0 — Unstick (45 minutes)

Do these in one sitting, in this order:

| Step | Action | Time | Why |
|------|--------|------|-----|
| 0.1 | **Apply EqProp search space fix** — `bioplausible/hyperopt/search_space.py:398-413`: `learning_rate (1e-2, 5e-1, "log")`, `beta (1e-3, 1e-1, "log")` | 30 sec | 6 eqprop models stuck at 10-14% because sampled β is 10x too large and lr is 100x too small (§8.6) |
| 0.3 | **Fix reproducibility test model names** — replace `equitile` (not registered) with `eqprop_mlp` in `tests/unit/validation/test_reproducibility.py` | 15 min | Closes 6 failing reproducibility tests; no code change |
| 0.4 | **Run P0 validation suite** — `uv run pytest tests/unit/validation/ tests/integration/test_gradient_equivalence.py -q --no-cov` | 2 min | Confirm all green; the credibility floor is now solid |

**After Phase 0**: Zero failures in `tests/unit/validation/`. Sweep infra unchanged. We have an honest baseline.

---

### Phase 1 — First Empirical Sweep (15 minutes GPU)

```bash
uv run python scripts/broad_sweep.py \
  --families eqprop \
  --probes-per-rule 3 \
  --epochs 2 \
  --device cuda \
  --max-params 32000
```

**What we're looking for** (from §8.6 diagnostic + §12.0 results table):

| Model | Plan-6 (broken space) | Expected after fix | Signal to act on |
|-------|----------------------|--------------------|------------------|
| `eqprop`, `directed_ep`, `finite_nudge_ep`, `lazy_eqprop` | 10-14% | **40-60%** | If still <30%, β or lr range still wrong |
| `momentum_equilibrium` | 10-11% | **40-60%** | If still flat, velocity buffer bug returned |
| `sparse_equilibrium` | 5-11% | **30-50%** | Sparse updates may need higher lr or longer epochs |
| `graph_eqprop`, `eqprop_mlp` (implicit-equilibrium path) | 70-93% | **70-93%** (unchanged) | If drops, we broke the implicit path |

**Decision gate after Phase 1:**

| Outcome | Next Action |
|---------|-------------|
| ≥4/6 fundamental models > 40% | EqProp family is unblocked → Phase 2 broadens sweep to all families |
| 3/6 > 40% | Widen β/lr range, re-sweep those 3 — they likely need finer hyperparameter tuning, not a redesign |
| ≤2/6 > 40% | **Defect-hunt** on the 3-4 worst models: instrument gradient norms, verify energy gap is non-zero, check weight update sign. Plan-6 found the energy-contrastive engine had a detached-tensor bug that zeroed every gradient — assume a similar defect until proven otherwise. |

---

### Phase 2 — Broad Sweep, Honest Pareto (30-60 min GPU)

Only after Phase 1 confirms EqProp fundamental models are not crippled by the search space:

```bash
uv run python scripts/broad_sweep.py \
  --families fa,hebbian,forward_only,predictive_coding,spiking,target_prop,eqprop \
  --probes-per-rule 2 \
  --epochs 2 \
  --device cuda \
  --max-params 32000 \
  --max-epoch-time 15
```

This produces the **first honest Pareto across all 7 bio families** with the corrected hyperparams. Each family's result becomes a concrete next task:

| Family | Likely status (post-fix) | Next task if poor |
|--------|--------------------------|-------------------|
| `fa` | ✅ Working (89-94% acc) | None — already a reference family |
| `forward_only` | ✅ Working (FF 76%, Pepita 47%) | None — known low acc, documented in `parity_gaps.md` |
| `eqprop` | 🟡 Now learning | **Real-task parity** if 50%+ — this is the flagship family |
| `target_prop` | 🟡 Slow (was 11%, fix §10.2 → 63%) | Widen `target_lr` range again if still <40% |
| `hebbian` | 🟡 Mixed (`deep_hebbian` 7%, `three_factor` 13%) | **Debug modulator instability** if NaN recurs |
| `spiking` | 🟡 STDP 29% (fixed §10.3) | Widen epochs if still <25% — 3-factor STDP may need more steps |
| `predictive_coding` | 🟡 Hybrid 87% (working), FabricPC 621k (over-budget) | Just `# noqa` the FabricPC over-budget flag — it's documented |

**After Phase 2**: We have a *real* ranking of 7 bio families on the same task, same 32k param budget, same epochs. This is the first honest measurement the framework has produced.

---

### Phase 3 — Empirical Fix Loop (iterative, days not weeks)

Each iteration:
1. Identify the family/model with the **biggest gap between current accuracy and what it should reach**
2. **Assume a defect or un-tuned hyperparameter**, not an algorithm ceiling. Plan-6 found fixable bugs in every "dead" family; this one will too.
3. Diagnose: hyperparam range (1-line fix), silent update drop (instrument `param.grad` norms), gradient flow (instrument energy gap), routing (verify `train_step` not BPTT fallback), or device placement
4. Fix the *specific* defect
5. Re-sweep that one model on GPU (2-5 min)
6. Commit only if accuracy improves — **every commit changes a number**
7. Repeat — *don't batch*, don't plan 6 tasks ahead

**Concrete candidates** (ordered by likely impact, will be reordered by Phase 2 results):

| Candidate fix | Triggering signal | Effort | Probability of win |
|---------------|-------------------|--------|-------------------|
| **Widen `target_prop` LR range** | `diff_target_prop` <40% | 1 line | High (fix moved 10%→63% in §10.2, full range may push higher) |
| **Fix `deep_hebbian` modulator NaN on MNIST** | NaN/divergence in sweep | 1 line (normalize by max-abs) | High (§10.4 already diagnosed root cause) |
| **Extend `spiking_stdp` epochs** in rule space | `spiking_stdp` <25% | 1 line | Medium (3-factor STDP needs more steps to show effect) |
| **Tune `momentum_equilibrium` momentum** | flat loss | extend `RULE_SPACES["eqprop"]` with `momentum` knob | Low — may need algorithm-level fix |

Each candidate is a *result-driven* task, not a speculative one. If Phase 2 shows `target_prop` already hits 50%, skip that candidate entirely.

---

### Phase 4 — Real-Task Compute-Matched Parity (1-2 weeks, post-sweep stability)

**Only after Phase 3 stops yielding cheap wins** (i.e. the sweep Pareto is stable for a few iterations).

**Trigger**: We have a single-paper-figure-worthy question — "Is family X within Y% of backprop on MNIST at matched compute?" — and we need to answer it with CI, effect sizes, and a CLI command.

**File**: `bioplausible/validation/backprop_parity.py` (production module — *not* the test file, which already exists for synthetic validation)

**MVP**:
- MNIST, MLP arch, 5 seeds, 2 epochs, backprop vs top-3 bio families from Phase 2
- Metrics: accuracy (mean ± BCa 95% CI), FLOPs/sample, peak memory, wall-time/epoch
- Output: JSON + markdown table + one Pareto plot
- CLI: `biopl-parity --family eqprop --task mnist --seeds 5`

**Do NOT spec-build the full RESEARCH.md §0.1 suite** (CIFAR-10, Tiny Shakespeare, n≥10 seeds, energy model). That's spec creep. Ship the MVP, see if anyone needs more, expand incrementally.

---

## What's deliberately deferred (spec-compliance, not result-blockers)

| Item | Original priority | New priority | Why |
|------|-------------------|--------------|-----|
| Extend `ComponentMetadata` with `bio_plausibility_score`, `memory_complexity`, `provides/requires` | P0 | **P2** | Spec compliance. Doesn't change what the sweep finds. Add when AutoScientist actually reads these fields. |
| Gate ALL 18+ propagators in `test_gradient_equivalence.py` | P0 | **P2** | Currently 7/18 gate; the missing 11 are FA variants (already tested by 7) + non-gradient families (skip by design). Marginal coverage gain. |
| `bioplausible/utils/reproducibility.py` global seed manager module | P0 | **P2** | Tests already validate determinism. The module would be refactor candy, not a fix. |
| `biopl-repro-check` CI gate | P0 | **P2** | Tests already run in CI. New binary adds packaging work, no new signal. |
| Analysis toolkit (dynamics, scaling, pareto, ablation modules) | P2 | **P3** | The sweep JSON already has the data. Plot when we have a paper draft. |
| AutoScientist v1 (CoT + KB synthesis + campaign) | P2 | **P3** | Compounds only after KB has enough entries from sweeps. We have ~100 right now; need ~500+ to be useful. |
| EquiTile flaky test fixes, gradient checkpointing, mixed-precision | P2 | **P3** | Don't touch EquiTile until a sweep actually uses it as a flagship. |
| Progressive Locality hybrid | P1 (flagship) | **conditional** | Only build if Phase 2 shows EqProp fundamental models plateau <60% AND analysis says "annealing would close the gap." Don't spec-build an algorithm before the data justifies it. |

**The hierarchy is simple**: things that change the numbers on the next sweep > things that change the numbers on a hypothetical future sweep > things that make the spec look prettier.

---

## Decision rules (so we don't re-plan every day)

1. **If a sweep result contradicts the plan, the result wins.** Don't update the plan — update the code.
2. **1-line hyperparam fixes beat 2-week algorithm redesigns.** Always try the cheap fix first.
3. **Never build infrastructure for an experiment you haven't run yet.** Run the experiment with the existing tools first.
4. **Spec compliance is not credibility.** The existing tests validate what matters (parity, gradients, determinism). Filling in metadata fields adds nothing until *something reads them*.
5. **Commit only when accuracy improves.** No "infrastructure-only" commits. Every commit changes a number or fixes a failing test.
6. **Defer is not delete.** Every deferred item stays in `RESEARCH.md`; we'll pick it up when it becomes the bottleneck.

---

## File/Module Map for *Actual* Changes

Nothing built speculatively. Only what the loop demands:

```
Phase 0 (45 min):
  bioplausible/hyperopt/search_space.py   # fix lr/beta range for "eqprop" (1 line)
  tests/unit/validation/test_reproducibility.py  # replace "equitile" → "eqprop_mlp" (6 occurrences)

Phase 1-3 (loop, no new files unless loop demands):
  scripts/broad_sweep.py                  # only if a sweep flag needs adding
  bioplausible/hyperopt/search_space.py   # only if another family's range needs fixing
  bioplausible/zoo/models/*              # only if a model defect needs fixing (last resort)
  bioplausible/experiment/param_estimator.py  # FabricPC budget-incompatible flag (when sweep trips on it)

Phase 4 (only when sweep is stable):
  bioplausible/validation/backprop_parity.py  # NEW — production parity, MVP only
```

No `hybrid/progressive_locality.py`. No `analysis/*.py`. No `utils/reproducibility.py`. No metadata field extension. **Build those when the loop asks for them, not before.**

---

## Verification: Cheap, Specific, After Every Loop Iteration

```bash
# Phase 0: sweep infra unchanged, validate floor
uv run pytest tests/unit/validation/ tests/integration/test_gradient_equivalence.py -q --no-cov

# Phase 1/3: every iteration, run the *one* family that changed
uv run python scripts/broad_sweep.py --families eqprop --probes-per-rule 3 --epochs 2 --device cuda --max-params 32000

# Nightly: full regression to catch silent breakage
uv run pytest tests/unit/ -q --no-cov
```

No new CI gates, no new binaries, no new test files (unless the loop demands them).

---

## What This Plan *Doesn't* Do (and why that's intentional)

- **Doesn't rule out components.** The sweep diagnoses defects and un-tuned hyperparameters — it never condemns an algorithm. If a model underperforms, we find the bug and fix it; Plan-6 did this for 5 families and the loop will do it again.
- **Doesn't speculatively build new algorithms.** Progressive Locality stays on the shelf *until the loop's diagnostics say it's the right tool* — e.g. "EqProp plateaus at 60% even with corrected hyperparams AND gradient norms suggest annealing would close the gap." Build when the data justifies it, not before.
- **Doesn't satisfy every RESEARCH.md checkbox.** The roadmap is a wishlist; this plan is a sprint. Items move from deferred → active *when they become the bottleneck*, not when the spec says they're P0.
- **Doesn't add up to a fixed 8-week calendar.** The loop continues as long as it keeps yielding improvements. When a family hits a stable accuracy that matches its theoretical capacity, we move to the next family — not declare it done and abandon it.
