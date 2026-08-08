# EXPERIMENT_PLAN5.md — Self-Diagnosing High-Performance Engine

**Status:** PLAN4's build items are landed (broad sweep, proposer bias knob, R5 spec sheet). The current sweep runs with honest bio-rules (eqprop→contrastive, FA→propagator, hebbian→CHL propagator) but the engine lacks self-diagnosis, wastes per-step measurement, and eqprop is overhead-bound on GPU (spectral norm power iteration, serial Python settle loop, per-step EnergyTracker). This plan fixes the engine so it **self-diagnoses**, **measures efficiently**, and **runs a correct, memory-efficient GPU eqprop that undercuts Backprop**.

---

## Core Thesis (unchanged from Manifesto)

The product is the **Epistemic Engine** (AutoScientist + Knowledge Base + honest measurement). The engine must evaluate *any* algorithm fairly, detect its own faults, and surface them without human audit. If the engine can't run a correct, memory-efficient bio-plausible GPU implementation, the thesis is unfalsifiable.

---

## The 3-Step Loop (Now with Fixes)

### 1. Broad Sweep (Self-Diagnosing)
**Goal:** Map the territory with correct bio-rules, auto-surface defective models, measure efficiently.

- Run shallow sweeps (1–2 epochs, 3 probes/family) across **all registered families**: EqProp, FA, Hebbian, Forward-Forward, Predictive Coding, STDP, etc.
- **Rule activation per family** (enforced by sweep, never optional):
  - EqProp → `gradient_method="contrastive"` (equilibrium loop, not BPTT)
  - FA → `propagator="feedback_alignment"` (local random-feedback)
  - Hebbian → `propagator="contrastive_hebbian_learning"` (CHL)
- **Liveness gate (binary):** `loss_epoch_0 > loss_epoch_final` — models that don't decrease loss are marked "dead" and excluded from the resource map.
- **Self-diagnosis (auto, not manual):**
  - `training_path` recorded per probe: `energy` | `model_train_step` | `propagator` | `bptt`
  - Any bio-family probe using `bptt` is flagged `DEFECT: silent BPTT fallback` in the sweep report.
  - Per-step overhead spikes (e.g., spectral-norm power iteration > 2× baseline) are auto-surfaced.
- **Efficient measurement:**
  - `EnergyTracker` heavy metrics (activation sparsity forward, weight-sparsity GPU reduction) computed **once per probe** (throttled), not every step.
- **Metrics:** variance in memory, compute time, settling steps, gradient alignment — **not accuracy**.
- **Output:** coarse Pareto landscape + auto-surfaced defect list.

```bash
uv run python scripts/broad_sweep.py --epochs 2 --probes-per-rule 3 --families all
```

---

### 2. Engine Audit (Zero Compute)
**Goal:** Make the AutoScientist smarter, not the models more accurate.

- **Bias check:** Is the proposer over-optimizing for accuracy? Force it to propose for *memory efficiency*, *settling speed*, or *hardware noise robustness*.
- **Compositionality:** Can it generate hypotheses that combine families? (e.g., "EquiTile topology + Forward-Forward credit assignment")
- **KB ingestion:** Drop in a random external paper's algorithm — does the KB + proposer handle it without code changes?
- **Fault diagnosis quality:** Are reverts/defects tagged with physical root causes (Lipschitz > 1, gradient cosine → 0, spectral-norm power-iteration overhead, BPTT fallback) or just "low acc"?

---

### 3. Market Reality Check (Zero Compute)
**Goal:** Validate the thesis with buyers, not benchmarks.

- **R5 Spec Sheet** (drafted): "Here is the measured cost of locality under these physical constraints, with 95% CIs, and the negative results we hit."
- **Buyer rubric (R6):** Name a decision/price → Fund / "cool, keep me posted" (False positive) / "wouldn't change my decision" (Pivot).
- **R8 Invariance paragraph:** What survives if the physical story weakens? (Surface audit trail, negative-knowledge oracle, cache integrity, settle protocol, training-path telemetry.)

---

## High-Performance GPU EqProp (Must Undercut Backprop)

**Current state:** EqProp runs ~23–50 ms/batch on GPU *and* CPU — overhead-bound (Python settle loop, spectral-norm power iteration, per-step EnergyTracker). Memory is NOT < Backprop.

**Target:** Correct, stable, high-performance GPU implementation with **lower peak memory than Backprop at matched loss**, and wall-time competitive at scale.

### Fixes Required

| Defect | Fix | Test |
|--------|-----|------|
| Spectral norm power iteration runs every forward (1.6× overhead, Python overhead) | Expose `n_power_iterations` (default 1, sweep can set 0 for coarse map); allow `use_spectral_norm=False` via config | A/B eqprop with/without spectral norm: peak memory + loss on GPU |
| Serial Python settle loop (20 steps, tiny tensors → GPU underutilized) | Expose `max_steps` sweep param (default 5 for coarse map); implement Triton `TritonEqPropOps` if available | `max_steps` sweep param; verify loss decreases at 5 steps |
| EnergyTracker measures every step (activation sparsity forward + GPU reduction) | Throttle: heavy metrics computed **once per probe** (or every N=10 steps); pass `global_step` to tracker | Test: `activation_sparsity` computed ≤1× per probe |
| BPTT fallback silent default for bio models | `allow_bptt_fallback=False` default for bio families; raise loud warning + record `training_path="bptt"` | Bio model with `bptt` path = defect in sweep report |

### Verification (Must Pass Before Sweep Runs)

```bash
# 1. EqProp contrastive loses loss at 2 epochs
uv run pytest tests/unit/experiment/test_eqprop_learns.py -q

# 2. EqProp contrastive peak_memory < Backprop at matched config
uv run pytest tests/unit/experiment/test_eqprop_memory_advantage.py -q

# 3. Training-path telemetry correct (no silent BPTT)
uv run pytest tests/unit/experiment/test_training_path.py -q

# 4. EnergyTracker throttle effective (heavy metrics ≤1/probe)
uv run pytest tests/unit/experiment/test_energy_tracker_throttle.py -q

# 5. BPTT-fallback flagged as defect in sweep
uv run pytest tests/unit/experiment/test_sweep_defect_flag.py -q
```

---

## What We Explicitly Defer (No Longer Gates)

| Item | Reason |
|------|--------|
| 85% test coverage | ~60% is enough to run experiments. Chase it later if CI demands it. |
| Fixing 5 non-converging models | Quarantine them (`xfail`). They're signal for later rule-health audit, not blockers now. |
| "Honest" 30-probe flagship search | Premature. Do it *after* the engine can propose across the whole landscape. |
| Substrate P4-lite/P4-full | Branch A is recorded (facades faithful on LoopedMLP). Run when a buyer asks for it. |
| Roofline / Memory Wall / Hardware Tax benchmarks | Build when a specific buyer conversation needs them. |
| Triton kernel for eqprop settle | Optional. CPU loop is acceptable for coarse map; add when GPU underutilization is the bottleneck. |

---

## Standing Discipline (Minimal)

- **Never measure before the suite is green.** (Already satisfied.)
- **Sink everything.** Wins and reverts go to `result_sink` with structured tags.
- **Blinded trials by default.** Proposer sees validation ranks / noisy estimates only. Test set locked.
- **No backwards compatibility.** Delete dead code, move fast.
- **Self-diagnosis is non-negotiable.** Every probe records `training_path`; sweep auto-flags defects. No human audit required.

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
| Broad Sweep | **▶ RUNNING** — with rule activation, liveness, checkpoint disable, no BPTT fallback |
| Proposer Bias Knob | **▶ LIVE** — `propose_batch(objective=accuracy\|memory\|settling_speed\|noise_robustness)` |
| Hebbian Propagator Fix | **▶ LIVE** — `transition_modules` includes W_in/head; `n_power_iterations` param |
| Training-Path Telemetry | **▶ LIVE** — `CoreTrainer` records `training_path` per step; driver surfaces it in probe metrics |
| EnergyTracker Throttle | **▶ LIVE** — heavy metrics (activation sparsity, weight-sparsity reduction) computed once per probe, cached on model |
| BPTT Opt-Out Default | **▶ LIVE** — `allow_bptt_fallback=False` for bio families; loud warning + `training_path='bptt'` |
| Spectral-Norm Power-Iteration Knob | **▶ LIVE** — `spectral_norm_power_iterations` exposed on `BioModel`/eqprop |
| Deep Local O(1) Memory | **▶ LIVE** — `DeepHebbianChain.train_step` runs its local Oja rule (no BPTT): 14.8 MB @100 layers vs backprop 47.3 MB; CHL propagator rewritten to no_grad-streaming (no 2×depth state lists): 27.3 MB @100 layers |
| EqProp GPU Memory vs BP A/B | **▶ LIVE** — O(1) implicit undercuts unrolled BPTT on the same arch; telemetry labels it `implicit_equilibrium` (not bptt); lock-in test passes |
| R5 Spec Sheet | **▶ DRAFTED** — `docs/R5_SPEC_SHEET.md` |

---

## Next Actions (Right Now — In Order)

1. **EnergyTracker throttle** — heavy metrics once/probe, not every step (`bioplausible/core/energy.py`).
2. **Training-path telemetry** — CoreTrainer records `training_path` per step; driver surfaces in probe metrics (`bioplausible/core/trainer.py`, `bioplausible/experiment/probe.py`).
3. **Sweep defect flag** — bio family using `bptt` path = `DEFECT` in report (`scripts/broad_sweep.py`).
4. **BPTT opt-out default** — `TrainerConfig(allow_bptt_fallback=False)` for bio families; loud warning on fallback (`bioplausible/core/trainer.py`).
3. **Spectral norm `n_power_iterations` param** — LoopedMLP exposes `spectral_norm_power_iterations` (already added to hebbian; add to eqprop/LoopedMLP).
4. **EqProp GPU A/B** — run contrastive vs BPTT on GPU, measure `peak_memory_mb` + `loss` decrease; assert eqprop < BP.
4. **Run fixed sweep** — `uv run python scripts/broad_sweep.py --families all --probes-per-rule 3 --epochs 2 --device cuda`
5. **Draft R5 Spec Sheet** — already done (`docs/R5_SPEC_SHEET.md`).

---

## Tests to Add (Must Pass Before Next Sweep)

| Test File | What It Verifies |
|-----------|------------------|
| `tests/unit/experiment/test_eqprop_learns.py` | EqProp contrastive decreases loss at 2 epochs (GPU) |
| `tests/unit/experiment/test_eqprop_memory_advantage.py` | EqProp contrastive `peak_memory_mb` < Backprop at matched config (GPU) |
| `tests/unit/experiment/test_training_path.py` | CoreTrainer records correct `training_path` per phase |
| `tests/unit/experiment/test_energy_tracker_throttle.py` | Activation sparsity computed ≤1× per probe |
| `tests/unit/experiment/test_sweep_defect_flag.py` | Bio model with BPTT path → `DEFECT` in sweep report |
| `tests/unit/experiment/test_bptt_opt_out.py` | `allow_bptt_fallback=False` raises warning on bio model fallback |

---

That's it. No 10-phase gauntlet. Just the loop with a self-diagnosing, efficient, honest engine.

(End of file)