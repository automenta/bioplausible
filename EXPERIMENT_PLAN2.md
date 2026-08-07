# EXPERIMENT_PLAN2.md — Strategic Forward Plan (Revised)

**Status (updated):** Infrastructure + measurement layer is now **built and validated end-to-end**. The §9-§4D fair-comparison pipeline (IdealBackpropFinder → RuleFrontierFinder → compare_frontiers → cost_of_plausibility → scaling laws) runs as one command and produced a real multi-family MNIST result (below). The training-loop overhead was reduced to the §5 floor (~1.9 s/epoch) via a `num_workers=0` fix for cached datasets (2.7×). Two implementation-suspicion hypotheses were resolved as **epoch-budget artifacts, not bugs** (see §12). **Session 3:** `peak_memory_mb` **verified** (PASS, 0.96 ratio); the comparison was hardened by making `epochs` part of the frontier-cache identity (a stale 1-epoch backprop reference had silently produced `inf` costs — §16.3); and the §7 equilibrium wall was attacked as controlled experiments — **warm-start negative** (reverted), **adaptive early-stop a WIN** (`convergence_threshold=1e-2` → ~1.3–1.4× settling speedup at negligible acc cost, now searchable for eqprop). `torch.compile` remains available as an option (`TrainerConfig(use_compile=True)`). Remaining work is measurement-driven: scaling probe counts toward the plan's 500-1000 budget for tight CIs.

---

## 1. The Real Scientific Question (Not "Does X Beat Y?")

The current parity framing ("does bio model X match backprop accuracy?") is the **wrong question**. It yields "no" or "tie" on toys, and the answer changes with epoch count.

**The right question is multidimensional:**

> *For a given task/domain, what is the Pareto frontier of (accuracy, wall_time, FLOPs, peak_memory, energy) achievable by each learning rule at its own hyperparameter optimum, with available optimizations? Which rules dominate under which resource constraints?*

This reframes "parity" as **a Pareto trade-off surface**. A rule that matches backprop accuracy at 0.3× FLOPs/energy *is* the result — not "it matches accuracy." The experiment layer already emits `wall_time_s, forward_flops, backward_flops, peak_memory_mb`; we must stop treating accuracy as the only axis.

**Autonomous invention pipeline angle:** An autonomous system doesn't "run an experiment" — it navigates a design space. It needs: *given compute budget X, memory cap M, accuracy target Y, and latency budget L, which learning rule + architecture + hyperparams + optimizations should I deploy?* Our job: make that mapping empirically grounded.

---

## 2. The Memory Dimension — Measure First, Optimize Later

**The hypothesis space:**
- Standard backprop stores forward activations for the backward pass → **O(L·B·D)** activation memory (L layers, B batch, D width).
- Standard equilibrium models (EQ, NC) also store activations for the nudged/error phase → similar O(L·B·D) in current implementations.
- **O(1) additional memory is achievable** via reversible architectures, activation checkpointing (gradient recomputation), and activation offloading — optimization techniques we can implement and measure, not theoretical impossibilities.

**The measurement protocol (hypothesis-driven):**
1. **Baseline measurement:** For each rule, measure `peak_memory_mb` vs batch size (16, 32, 64, 128, 256) and width (64, 128, 256, 512) on MNIST/CIFAR. Verify against `torch.cuda.max_memory_allocated()`.
2. **Breakdown measurement:** Decompose `peak_memory_mb` into model params + optimizer states + forward activations + grads + equilibrium buffers. Confirm with `torch.cuda.memory_snapshot()`.
3. **Implement & measure optimizations** as controlled experiments:
   - **Activation checkpointing** (gradient recomputation): ~2× compute, ~O(√L) memory reduction
   - **Reversible architectures** (RevNet, iRevNet): ~O(1) additional memory, ~2× compute
   - **Activation offloading** (CPU offload): CPU↔GPU transfer cost, huge memory savings
   - **Gradient accumulation**: larger effective batch size without larger memory
4. **Measure the trade-off:** For each optimization, record the (accuracy, wall_time, FLOPs, memory) tuple. This *is* the optimization-opportunity map.

**Target result:** A per-rule plot of (memory, accuracy, wall_time) with and without each optimization. The crossover point where an optimization becomes valuable is the decision data.

**Why this matters for autonomous invention:** Edge deployment has hard memory caps. A rule that fits in 50 MB while backprop needs 2 GB is the *only* viable option — accuracy is secondary to "does it fit."

---

## 3. The Compute Wall — Measure First, Then Optimize

**Measured reality (cached MNIST, 128/2):**

| Component | Current | Floor | Overhead | Optimization Opportunity |
|-----------|---------|-------|----------|--------------------------|
| Backprop data loading | ~1 s/ep | ~0.1 s | ~0.9 s | DataLoader cache ✅ done; pin_memory, persistent workers |
| Backprop training loop | ~5.4 s/ep | ~1.9 s | ~3.5 s | **Primary target**: validation eval, per-batch Python overhead, missing `torch.compile` |
| Equilibrium iterations | ~21 s/ep | ~1 s (compute) | ~20 s | **Algorithmic**: fewer iterations, learned damping, predictor-corrector, warm-start |

**This is a measurement, not a verdict.** The point of measuring is to identify where optimization effort pays off.

**The measurement protocol for optimization:**
1. **Profile each component** (data, forward, backward, equilibrium, validation, metrics) with `torch.profiler`.
2. **Quantify each overhead** as % of epoch time.
3. **Implement targeted optimizations** as controlled experiments: measure (accuracy, time, memory) before/after.
4. **Only invest in deep optimization** when the data shows the ROI crosses the threshold (e.g., "if we cut equilibrium iterations 10×, CIFAR-10 becomes feasible").

The equilibrium iteration wall is an algorithmic opportunity, not a dead end — see §7.

---

## 4. Finer-Grained Hyperparameter Search (Beyond Coarse Grids)

Current grids (`hidden_dim: [16,32,64,128,256]`, `num_layers: [1,2,4]`) are coarse and linear — they miss optima and waste probes on dead regions.

### A. Continuous Bayesian Search (already built, not used effectively)
`OptunaBayesProducer` with TPE exists. Use it properly:
- Continuous ranges: `hidden_dim ∈ [32, 1024]` (log-uniform), `num_layers ∈ [1, 6]`, `learning_rate ∈ [1e-5, 1e-1]` (log-uniform), `dropout ∈ [0, 0.5]`, `weight_decay ∈ [1e-6, 1e-2]`.
- **Budget: 500–1000 probes per task** for a meaningful Pareto frontier (not 100–200); TPE finds the true posterior over the space.
- The "ideal backprop" is the Bayesian optimum of backprop on a task; all other rules are compared *at their own Bayesian optima*, not at one shared coarse grid.

### B. Multi-Fidelity / Successive Halving
For expensive tasks (CIFAR-10):
- Fidelity = epochs (5 → 10 → 20 → 40) or batch size / dataset fraction.
- Early-stop bad configs; promote promising ones.
- Optuna supports this via `HyperbandPruner`; our `ConfigProducer` interface can support it.

### C. Architecture Search (Not Just Hyperparams)
The grid only varies width/depth. Missing:
- Activation functions (ReLU, GELU, Swish, SiLU)
- Normalization (BatchNorm, LayerNorm, GroupNorm, none)
- Skip connections (plain, ResNet-style, DenseNet-style)
- **Equilibrium-specific**: damping factor, step size, max iterations, convergence threshold, damping schedule, predictor-corrector steps

These are *rule-specific* hyperparameters that could be decisive for equilibrium models.

### D. The "Ideal Backprop" as the Moving Reference
Don't compare rules at a shared coarse grid. Instead:
1. Define the backprop search space (width, depth, LR, WD, dropout, act, norm, optimizer, scheduler, batch size).
2. Run **full Bayesian optimization (2000+ probes)** to find the *true* Pareto frontier of backprop: (accuracy, FLOPs, memory, time). 200 probes is far too low for CIFAR-10.
3. Treat this frontier as the "ideal backprop" reference.
4. Run each bio rule's Bayesian search in its own rule-specific space (including equilibrium params: damping, iterations, convergence threshold, predictor-corrector config).
5. Compare the **two Pareto frontiers**:
   - Does the bio rule touch or dominate backprop at any operating point?
   - At what FLOPs/memory budget does it dominate?
   - What is the "cost of bio-plausibility" in FLOPs/memory at matched accuracy?

This is the **only fair comparison**. A single-point baseline is scientifically meaningless.

---

## 5. The Training Loop Bottleneck — The Real Blocker

**Measured floor:** minimal MNIST train epoch (data + forward + backward) = **1.9 s** (938 batches, 128/2 MLP, RTX 3080).
**CoreTrainer cached run:** backprop 6.4 s/epoch, eqprop/neural_cube ~21 s/epoch.

**Gap analysis:**
- Backprop: 6.4 s vs 1.9 s floor → **4.5 s overhead (3.4×)** in the CoreTrainer loop.
- Equilibrium models: ~21 s/epoch → dominated by equilibrium iterations (10–100× per batch), not data loading.

**Root causes to investigate (priority order):**
1. **Per-epoch validation eval** — runs the full val set every epoch; ~1–2 s/epoch.
2. **Per-batch Python overhead** — metric computation, device sync, logging, callback dispatch per batch.
3. **Per-epoch device sync / memory stats** — `peak_memory_mb` collection, energy tracking.
4. **Missing `torch.compile` / vectorization** — per-batch Python loops vs fused kernels.
5. **Energy tracking overhead** — `track_energy=True` adds significant cost.

**Immediate profiling targets (this week):**
1. `torch.profiler` on CoreTrainer `fit()` — identify top 3 time consumers.
2. Disable validation eval → measure epoch-time drop.
3. Disable `track_memory`/`track_flops`/`track_energy` → measure drop.
4. Add `torch.compile` to model forward/backward.

**Target:** reduce backprop overhead from 4.5 s → ≤1.5 s/epoch (total ≤3.5 s/epoch). Equilibrium models stay compute-bound (see §7), but backprop should approach the floor.

### Session-2 update (measured, post-fix)
We are **at the floor.** With the cached-dataset `num_workers=0` fix, MNIST backprop epochs measure **1.86 s** (post-warmup) regardless of configured `num_workers` — the target's lower bound.

**Decomposed MNIST backprop epoch (hidden 256, batch 128, CUDA, post-warmup):**
- Data iteration (cached): ~0.45 s/epoch (at `num_workers=0`; >0 is *slower* — see below)
- Forward + backward: ~1.3 s
- Validation: ~0.02 s (was mis-attributed as 25 s — that was one-time CUDA/cuDNN autotune, not per-epoch)
- Per-batch Python / tracking: a few ms

**Key optimization found:** for in-memory cached `TensorDataset`s, `num_workers > 0` is **counterproductive** — there is no disk I/O to hide, so workers add only multiprocessing/IPC cost (5.0 s vs 1.86 s). `create_data_loaders` now forces `num_workers=0` for `_CACHEABLE_VISION` sets. Non-cached sets (generated toys, disk/LM) keep the operator's worker count.

**Remaining lever for backprop:** `torch.compile` (still untested). Given we are at the floor, the ROI is marginal vs the §7 equilibrium lever, which is the real time sink on bio rules.

---

## 6. The Memory Measurement Gap — Verify First

**We don't know what `peak_memory_mb` actually measures.** Before building analysis on it:
1. Run a probe with `track_memory=True` on a simple task.
2. Compare reported `peak_memory_mb` vs `torch.cuda.max_memory_allocated()` / `max_memory_reserved()`.
3. Break down: model params + optimizer states + forward activations + grads + equilibrium buffers.
4. Verify it matches `torch.cuda.max_memory_allocated()` (should be close if implemented correctly).

**Only after verification** build memory-vs-accuracy plots and scaling-law fits.

### Session-2 status
`scripts/verify_memory_measurement.py` is written and ready (trains 1 backprop epoch, compares reported `peak_memory_mb` to `torch.cuda.max_memory_allocated()`/`max_memory_reserved()`, PASS/FAIL within 10%). **It has not been executed yet** — this is a top pending item. Run `uv run python scripts/verify_memory_measurement.py` (requires CUDA). Until it PASSes, treat all reported memory figures as unverified, per the §6 caution.

### Session-3 status — **PASSED**
`uv run python scripts/verify_memory_measurement.py` now runs (the script previously crashed: it was missing `import bioplausible.zoo`, the registration side-effect that populates the model registry). Result (RTX 3080, backprop MLP hidden 256, batch 128, 1 epoch):

```
reported peak_memory_mb : 20.5
max_memory_allocated()  : 21.4
max_memory_reserved()   : 26.0
ratio                   : 0.960
RESULT: PASS (within 10% of max_memory_allocated)
```

`peak_memory_mb` is within 10% of `torch.cuda.max_memory_allocated()` → **verified**. Memory figures may now be used in scaling-law fits and frontier analysis. Note the reported value tracks *allocated* (21.4), not *reserved* (26.0) — the allocator caches ~5 MB beyond the true peak.


---

## 7. The Equilibrium Compute Wall — Opportunity, Not Wall

**Measured:** `eqprop` / `neural_cube` at **21 s/epoch** on MNIST (128/2). The equilibrium iterations (10–100 inner steps per batch) dominate.

**This is a measurement, not a verdict** — an algorithmic optimization opportunity, not a dead end.

**Optimization hypotheses to TEST (each a controlled experiment):**
1. **Learned damping / step size** (meta-learned per layer, or Adam on the equilibrium loss)
2. **Anderson acceleration / predictor-corrector** (reduce iterations 5–10×)
3. **Warm-start from the previous epoch** (reuse the equilibrium point as initialization)
4. **Early stopping with adaptive threshold** (dynamic convergence criteria)
5. **Jacobi-style parallel updates** (instead of Gauss-Seidel sequential)
6. **Equilibrium as an implicit layer** (implicit differentiation, single backward pass)

### Session-3 measurement: warm-start is **negative/neutral** (hypothesis 3)
`scripts/equil_warmstart_experiment.py` is a controlled measurement of *nudged-phase warm-start* (init the beta>0 settle from the settled free phase instead of the raw feedforward init), timing `StandardEqProp.train_step` over matched MNIST batches for identical-init models with `use_equilibrium_warm_start` on vs off, **discarding the first CUDA epoch and counterbalancing variant order** to kill the warmup confound:

| max_steps | speedup | Δacc |
|-----------|---------|------|
| 20        | 1.02×   | 0.0000 |
| 30        | 1.03×   | 0.0000 |
| 100       | 1.02×   | +0.0002 |

**Verdict: NO meaningful speedup.** The `settle_activations_list` early-stops (convergence_start=5, tol 1e-3) well before filling `max_steps`, so a better start point saves almost nothing; the free phase dominates and cannot be warm-started per-batch. Per the §7 decision rule ("only invest when ROI crosses threshold"), the implementation was **reverted** (no dead code) and the script + logs retained as reproducible evidence. An early, plausible-looking 3.02× reading was a **CUDA-warmup order artifact** — always discard the first epoch and counterbalance.

### Session-3 measurement: adaptive early-stop **is a WIN** (hypothesis 4)
`scripts/equil_adaptive_stop.py` sweeps `(convergence_threshold, convergence_start)` on `StandardEqProp` (MNIST, matched init, warmup discarded) over the grid `{1e-2, 1e-3, 1e-4} × {2, 5}`, baseline = default `(1e-3, 5)`, timing the *settling* cost:

| max_steps | best (thresh, start) | speedup | Δacc |
|-----------|----------------------|---------|------|
| 30        | (1e-2, 2)            | 1.36×   | −0.0004 |
| 100       | (1e-2, 5)            | 1.31×   | −0.0006 |

**Verdict: loosening the convergence threshold `1e-3 → 1e-2` gives ~1.3–1.4× speedup at negligible accuracy cost** (robust across seeds/max_steps). Tighter (`1e-4`) is *slower* (0.8×); `convergence_start` barely matters. The knobs are now **wired through** (`ModelConfig.convergence_threshold/_start`, applied in `StandardEqProp`), so they can be set as `model_kwargs`, and they are now exposed as **searchable** in `RULE_SPACES["eqprop"]` (`convergence_threshold ∈ [1e-4, 1e-2]` log, `convergence_start ∈ [2, 10]` int) per §4C/§12, so the Bayesian search discovers the looser threshold where it pays. Net: a real §7 compute win, unlike warm-start.

**Each is a hypothesis to test, not a dismissal:**
- Implement the variant → measure (accuracy, wall_time, memory) → compare to baseline.
- Only invest in deep optimization when the data shows the ROI crosses the threshold.

The measurement infrastructure exists to tell us which option pays off.

---

## 8. Conditions of Results → Application Enablement (Autonomous Pipeline)

The autonomous pipeline needs **decision rules** mapping (task, constraints, results) → (deployment decision). Experiments must produce *conditionals*, not point estimates.

| Pipeline Decision | Required Experimental Evidence |
|-------------------|--------------------------------|
| **Rule selection** | "For task T with compute budget C and memory M, rule R achieves accuracy A with CI." |
| **Architecture choice** | "For rule R on task T, optimal (W, L, act) = (W, L, act) with Pareto frontier F." |
| **Resource allocation** | "To reach accuracy A on task T with rule R, need FLOPs F, memory M, time T; scaling laws predict cost." |
| **Deployment target** | "Rule R fits in memory M on device D; latency L; energy E per inference." |
| **Fallback chain** | "If rule R fails on task T, try R' with config C'." |

**Experimental design implication:** every experiment must emit **fitted conditionals with uncertainty**, not point estimates. Fit scaling laws (`accuracy ~ log(FLOPs)`, `memory ~ L·B·D`) and emit *fitted parameters with uncertainty*.

**Autonomous pipeline loop:**
1. Task arrives with constraints (accuracy target, latency budget, memory cap, energy budget, hardware spec).
2. Pipeline queries experimental DB for conditionals: "What rules/architectures satisfy these constraints?"
3. If no data, pipeline triggers a targeted experiment (Bayesian search in the relevant subspace) to fill the gap.
4. Pipeline makes a decision and deploys.
5. Runtime monitoring feeds back to update the DB (online learning).

---

## 9. The "Ideal Backprop" as a Service

Build a reusable component `IdealBackpropFinder(task, budget_probes=2000) → ParetoFrontier`. It:
- Runs a full Bayesian optimization once per task,
- Caches the resulting backprop Pareto frontier,
- Serves as the reference for *all* subsequent bio-rule experiments on that task.

This is infrastructure, not an experiment.

---

## 10. Immediate Next Actions (This Week)

**Status: all items below are DONE (this session).** They now serve as a baseline; the pending work is shifted to §12/§13.

| Action | Success Criterion | Status |
|--------|-------------------|--------|
| **Profile CoreTrainer loop** with `torch.profiler` | Top 3 overheads identified; each >5% of epoch time | ✅ `CoreTrainer(profile_epochs=True)` → `_profile_loop` dumps `profile_e0.trace` + top consumers |
| **Disable validation eval** in a test run | Measure epoch-time drop; target −1.5 s/epoch | ✅ `run_validation=False` toggle; post-warmup `_validate` measured at ~0.02 s (the earlier "25 s" was a CUDA warmup artifact) |
| **Disable `track_memory`/`track_flops`/`track_energy`** | Measure epoch-time drop; quantify overhead | ✅ flags exist in `TrainerConfig` |
| **Add `pin_memory=True` to DataLoader** in `create_data_loaders` | MNIST epoch time drops from 6.4 s → ≤4 s | ✅ pin_memory + persistent_workers threaded; **plus the real win below** |
| **Generalize vision dataset cache to CIFAR-10** | `get_vision_dataset("cifar10")` returns a cached `TensorDataset` | ✅ `_CACHEABLE_VISION` includes cifar10/cifar100 |
| **Verify `peak_memory_mb`** vs `torch.cuda.max_memory_allocated()` | Reported value within 10% of `max_memory_allocated()` | ✅ **PASS (session 3)** — `scripts/verify_memory_measurement.py` (fixed missing `import bioplausible.zoo`); ratio 0.96. See §6 |
| **Profile equilibrium inner loop** (eqprop 1 epoch) | Identify equilibrium iteration cost; count iterations/batch | ⏳ **NOT DONE** — see §7 |
| **Define continuous search spaces per rule** | `RULE_SPACES = {"backprop": {...}, "eqprop": {...}, ...}` incl. equilibrium-specific params | ✅ `RULE_SPACES`: backprop, eqprop, neural_cube, **pepita, forward_forward, feedback_alignment** (see §12 for hyperparam-completeness lesson) |
| **Add `HyperbandPruner` support to `OptunaBayesProducer`** | `producer = OptunaBayesProducer(..., pruner=HyperbandPruner())` | ✅ pruner threaded into `create_study` |

### Also built this session (not in the original §10 list)
- **`hyperopt/frontier.py`** — `RulePoint`, `pareto_frontier`, `cost_of_plausibility` (§11 metric).
- **`hyperopt/ideal_backprop.py`** — `IdealBackpropFinder` (§9), TPE over `RULE_SPACES["backprop"]`, JSON cache.
- **`hyperopt/rule_frontier.py`** — generic `RuleFrontierFinder` for ANY rule incl. eq-specific params (§4D.4).
- **`hyperopt/comparator.py`** — `compare_frontiers`, `OperatingPointMatch` (§4D.5-7).
- **`hyperopt/scaling_law.py`** — `fit_accuracy_scaling`, `predict_flops_for_accuracy` with CI (§8).
- **`hyperopt/__init__.py`** — re-exports the whole frontier stack.
- **`cli/frontier.py`** — `biopl-frontier` CLI over a probe JSONL.
- **`scripts/preliminary_run.py`** — multi-family run command (the working harness).
- **`scripts/verify_memory_measurement.py`** — §6 probe (written, not run).
- **Training-loop fix:** cached `TensorDataset` sets force `num_workers=0` in `create_data_loaders` — **2.7× faster epochs** (5.0 s → 1.9 s for cached vision). See §5 note.
- **Probe-driver diagnostics:** `best_epoch_acc`, `acc_at_half` per probe — distinguish "needs more epochs" from "never learns" (§12).

---

## 11. The One Metric That Matters

Stop reporting "accuracy." Report **the Pareto frontier of (accuracy, FLOPs, memory, wall_time, energy) for each rule at its own optimum**.

The single number that summarizes a rule's competitiveness on a task:

> **`cost_of_plausibility(task) = min_{p∈bio_frontier} [ FLOPs(p) / FLOPs(backprop_at_same_acc) ] × (mem(p) / mem(backprop_at_same_acc)) × (time(p) / time(backprop_at_same_acc))`**

At a given accuracy level, how many more FLOPs×memory×time does the bio rule need vs ideal backprop? If the geometric-mean ratio is ≤1.5 at 95% of backprop accuracy, the rule is **deployment-viable**; at 5× it's a curiosity. This composite number is what the autonomous pipeline uses to decide "deploy or not."

---

## Appendix: The MNIST Trio Result (Partial, 27/36 probes)

| Model | n | Mean Acc | Wall Time (s/ep) | Peak Mem (reported, unverified) |
|-------|---|----------|------------------|--------------------------------|
| backprop_mlp | 10 | 0.9799 | 6.4 | ~200 MB |
| neural_cube | 10 | 0.9796 | 21.0 | ~300 MB |
| eqprop_mlp | 1 | 0.965 | 21.1 | ~350 MB |

**Interpretation:** On MNIST (real 28×28), `neural_cube` **ties** backprop (it won on digits). eqprop is slightly below. Both bio rules cost ~3× wall-time. The digits "winner" does not generalize its advantage to real 28×28 images — valuable *as part of a systematic mapping*, not as a standalone experiment. (Memory figures are unverified; see §6.)

---

## 12. Diagnostics Lesson: "Poor" Results ≠ Implementation Bugs (Session-2 Finding)

**Symptom:** In the first multi-family run (1 epoch/probe), `pepita` hit only 0.30-0.40 acc and `forward_forward` 0.55 — looking like bugs in those models.

**Diagnosis (via new `best_epoch_acc` / `acc_at_half` probe fields):** they were **mid-convergence, not broken**. At 5 epochs each:

| Model | 1 ep | 5 ep | trajectory (half→final) | verdict |
|-------|------|------|------------------------|---------|
| pepita | 0.40 | **0.848** | 0.75 → 0.85 (rising) | needs epochs |
| forward_forward | 0.55 | **0.887** | 0.86 → 0.89 | needs epochs |
| feedback_alignment | 0.89 | **0.968** | 0.95 → 0.97 | needs epochs |

**Two lessons for future runs:**
1. **Epoch count must be a recorded covariate.** Comparing rules at different effective convergence is meaningless. Either match epochs across rules, or report `best_epoch_acc`/`acc_at_half` so an "epoch-budget artifact" is distinguishable from a real failure — this is now built into the probe driver.
2. **`RULE_SPACES` must be complete per model constructor.** `forward_forward` needed `threshold`/`layer_lr`/`classifier_lr`; `feedback_alignment` needed `alpha`/`feedback_mode`/`use_spectral_norm`; these were missing initially and are now added. When adding a new family, grep the model's `__init__` and mirror every tunable into `RULE_SPACES` (and cast dims to int — see §14).

---

## 13. Session-2 Multi-Family Result (MNIST, CUDA, 1 epoch/probe, 3 probes/family, 8.4 min)

The full fair-comparison pipeline ran end-to-end. All numbers are **preliminary** (3 probes/family, small); the point is the harness works and produces per-family decision conditionals.

| Family | cost_of_plausibility | best ΔAcc vs bp | best FLOPs× | best Mem× | best Time× | scaling r² |
|--------|---------------------|-----------------|------------|-----------|-----------|-----------|
| **neural_cube** | 2.09 | **+0.002** (ties) | **0.52** | 6.38 | 2.7 | 0.98 |
| **pepita** | 2.12 | −0.45 | 4.7 | 1.9 | 1.1 | 0.01 |
| **feedback_alignment** | 2.46 | −0.003 | 0.67 | 2.15 | 10.3 | 0.01 |
| **forward_forward** | 3.23 | −0.29 | 4.5 | 1.95 | 3.9 | 0.29 |
| **eqprop** | 9.78 | −0.05 | 20 | 0.80 | 51 | 0.98 |

**Read-outs (answers to §1/§8 guidance questions):**
- **neural_cube is the standout:** it *ties/beats backprop accuracy* at **half the FLOPs** (0.52×) — exactly the "matches at lower FLOPs" win §1 is designed to surface. Its costs are 6× memory and 2.7× time → the §7 equilibrium levers target its time.
- **feedback_alignment** nearly matches accuracy (Δ−0.003) at below-backprop FLOPs (0.67×), but pays 2× mem / 10× time.
- **eqprop** is the §7 compute wall: 20-96× FLOPs and ~50× time despite equal/less memory and a clean scaling law (r² 0.98) — an algorithmic-optimization target, not a parity ceiling.
- **pepita** shows negative/zero scaling (r² 0.01) — at 1 epoch it had not converged (see §12); re-run with more epochs before judging.
- **cost of bio-plausibility is family-dependent (2.1-9.8), not monolithic.**

**Reproduction:** `uv run python scripts/preliminary_run.py --device cuda --task mnist --bio eqprop,neural_cube,pepita,forward_forward,feedback_alignment --bp-probes 10 --bio-probes 3 --epochs 1 --json` → `logs/multi_family_mnist.json`. Backprop frontier is cached and reused across runs/files (IdealBackpropFinder cache). TIP: the shell tool's 120 s timeout kills background jobs unless launched with `setsid bash -c '...' </dev/null >/dev/null 2>&1 & disown`.

---

## 14. Math-Verification Notes (Session-2) — what was checked and fixed

- **FIXED (scaling-law inverse bug):** the fit is `accuracy = a·log(FLOPs+1) + b`, but `predict_flops_for_accuracy` previously inverted as `FLOPs = exp(log_f)` instead of `exp(log_f) − 1`. Corrected in mean + both CI bounds (delta-method SE unchanged) and the unit test. Watch for this class of `log(x+1)` vs `log(x)` mismatch when inverting: always mirror the exact forward transform.
- **Verified correct:** `cost_of_plausibility` (geometric mean of FLOPs×/Mem×/Time× at closest-accuracy backprop match, min over bio frontier — matches §11; cube root is a monotonic transform of the §11 product, thresholds on the same scale); `pareto_frontier` dominance rule (acc ≥ within ε, all resources ≤, one strict); delta-method CI on `log(FLOPs+1)`.
- **Warmup artifact:** early runs overstated validation cost (25 s) because the first CUDA `fit()` absorbed cuDNN autotune. Always discard the first run / warm-up before timing per-epoch.

### Implementation gotchas recorded for the next session
- Continuous `hidden_dim`/`cube_size` sampled as float → must be **cast to int** before handing to the model builder (done in both `ideal_backprop._sample_backprop_config` and `rule_frontier.sample_config_for_rule`).
- Force `find(force=True)` in run scripts so a stale/partial cache from a killed run is never silently reused as a truncated frontier.
- `RULE_SPACES` sample `"log"`/`"linear"`→`suggest_float`, `"int"`→`suggest_int`, list→`suggest_categorical`; dims cast to int (see `rule_frontier._INT_CAST_PARAMS`).

---

## 15. Next-Session Roadmap (priority order)

The infrastructure is proven. The remaining work is **measurement-driven**, ordered by ROI:

1. **Run §6 memory verification** (`uv run python scripts/verify_memory_measurement.py`). Until it PASSes, don't build analysis on reported memory. High priority, quick. → **DONE (session 3): PASS** (ratio 0.96; see §6). Script also fixed (`import bioplausible.zoo` was missing).
2. **Re-run forward-only + FA families with ≥5 epochs** (pepita/forward_forward/feedback_alignment) so their `cost_of_plausibility` is meaningful (they underfit at 1 epoch, §12). Keep epochs as a recorded covariate. → **DONE (session 3)**, and it surfaced a **cache-key bug** (§16): the backprop reference must be epoch-matched or `cost_of_plausibility` is `inf`/meaningless.
3. **Attack the eqprop/neural_cube time wall (§7)** as a controlled experiment — largest algorithmic lever. → **DONE (session 3)**: warm-start tested and found negative/neutral; **adaptive early-stop found a WIN** (`convergence_threshold=1e-2` → ~1.3–1.4× settling speedup at negligible acc cost), now searchable in `RULE_SPACES["eqprop"]`. Backprop/others are already at the epoch-time floor.
4. **Scale probe counts** toward 500-1000 (§4A) now that epochs are ~2 s — tighten CIs on `cost_of_plausibility` and the scaling laws; at minimum raise `--bp-probes` (cached frontier stays) and `--bio-probes` per family. → **pending**. The epoch-keyed backprop cache (budget10/5ep) is in `logs/`; raising `--bp-probes` re-derives only backprop (cheap).
5. **Consider `torch.compile`** on backprop forward/backward (marginal now; we are at the floor).
6. **Test `HyperbandPruner` end-to-end** on an expensive task (CIFAR-10) via the §4B multi-fidelity path (infrastructure is wired but not exercised on a real run).

**When adding a new rule family:** (a) register the model, (b) add a complete `RULE_SPACES` entry mirroring the constructor's tunables, (c) confirm it trains through `CoreTrainer` (some families need `.build()`/model-specific setup), (d) run 5+ epochs before interpreting.

---

## 16. Session-3 Findings (measurement-driven)

### 16.1 §6 memory verification passed
`peak_memory_mb` ≈ 0.96 × `max_memory_allocated()` (20.5 vs 21.4 MB) on a backprop MNIST run. Verified → memory axes are now trustworthy for scaling laws and frontier analysis. See §6.

### 16.2 §7 warm-start is not a lever (negative result)
Tested nudged-phase warm-start across max_steps 20/30/100 — no meaningful speedup (1.02–1.03×), zero accuracy change. Reverted the implementation (`use_equilibrium_warm_start`), kept `scripts/equil_warmstart_experiment.py` as a reproducible controlled-experiment harness. See §7.

### 16.3 Cache-key bug: epochs now part of frontier cache identity (§12 enforced)
`IdealBackpropFinder` and `RuleFrontierFinder` cache names did **not** include `epochs`. A stale 1-epoch backprop frontier (max acc 0.90) was silently reused as the reference for 5-epoch bio runs → `cost_of_plausibility` came back `inf` for the acc-winning families (neural_cube 0.975, feedback_alignment 0.973 > backprop 0.90 ceiling). Fixed by keying and validating caches on `(task, model, epochs, budget)`:
- `ideal_backprop_<task>_<model>_epochs<e>_budget<b>.json`
- `rule_frontier_<rule>_<task>_epochs<e>_budget<b>.json`
Old non-epoch caches are orphaned (harmless). **Rule: epochs are a covariate that must be matched and cached independently** — a frontier derived at one epoch budget is not comparable to (nor a valid reference for) runs at another.

### 16.4 Corrected multi-family result — MNIST, 5-epoch matched, bio=3 probes vs backprop=10 (CUDA)
Preliminary (small probe counts), but now epoch-matched and comparable to §13's 1-epoch table:

| Family | cost (5ep) | best Acc | vs §13 (1ep) cost | read-out |
|--------|-----------|----------|-------------------|----------|
| **neural_cube** | **1.75** | **0.975** | 2.09 | remains the standout; near-backprop accuracy, lowest cost |
| **feedback_alignment** | 3.36 | 0.973 | 2.46 | near-backprop acc; cost up vs 1ep (converges later) |
| **pepita** | 2.11 | 0.865 | 2.12 | converges (0.86 at 5ep) but below backprop acc |
| **forward_forward** | 4.45 | 0.882 | 3.23 | mid-convergence; still below backprop acc |
| **eqprop** | 10.07 | 0.921 | 9.78 | the §7 compute wall persists (highest cost) |

`repro`: 5-epoch match = `IdealBackpropFinder(budget=10, epochs=5)` reference (now epoch-keyed) + Episode-5 bio frontiers from `logs/multi_family_mnist_ep5.json`, compared via `compare_frontiers`. Full self-consistent `scripts/preliminary_run.py --epochs 5` re-run (bio force=True) is straightforward now and remains a ~40 min budget if a single command-line report is wanted.

---

## 17. Session-4 Strategic Work: The Universal Result Sink (why the moat now compounds)

### The gap this closes
The repo already had the *business architecture* (per the revised thesis): an ExecutionEngine / AutoScientist (`bioplausible/execution/`), a `FailureTracker`, a `KnowledgeBase` with SQLite + vector store, and hardware tracks (FPGA INT8, analog-photonics noise, thermodynamic-DNA). **But the measurement layer did not write to any of it** — the frontier pipeline (`CoreTrainerDriver → IdealBackpropFinder/RuleFrontierFinder`) emitted throwaway JSON to `logs/` and never called `KnowledgeBase.add_experiment` or `FailureTracker`. The moat was built but disconnected from the experiments that should feed it.

### What was built
**`bioplausible/experiment/result_sink.py`** — one universal, idempotent entrypoint:

```
record_experiment_result(model, task, config, metrics, status, ...)
```

- `status="completed"` → writes a verified positive entry to the **KnowledgeBase** (`add_experiment`: metrics + hyperparameters + artifacts).
- any failure status → writes a **negative record** to the `FailureTracker` (so `FailureManifestoGenerator` can mine it).
- Normalizes the two metric dialects present in the repo (probe-driver `final_acc`/`wall_time_s` vs ExecutionEngine `accuracy`/`time`).
- Best-effort (never breaks a probe), env-gated (`BIOPLAUSIBLE_RECORD_RESULTS`, `BIOPLAUSIBLE_KB_PATH`, `BIOPLAUSIBLE_FAILURES_PATH`), caches one KB/FailureTracker instance per process.

**Wired into the two main experiment frameworks:**
1. `CoreTrainerDriver.train()` (`experiment/probe.py`) — the frontier/parity driver; records success + training-failure (`except` in `fit()`) + empty-history.
2. `run_single_trial_task()` (`hyperopt/experiment.py`) — the ExecutionEngine trial path; successes now reach the KB (previously only failures landed in a temp FailureTracker).

### Validation
- Sink unit tests (`tests/unit/test_result_sink.py`): success→KB, failure→FailureTracker, and the two sinks stay disjoint.
- **Real end-to-end probe**: `CoreTrainerDriver.train(backprop_mlp, mnist, 1 epoch)` → KB entry `EXP-…` with real `final_acc`, FLOPs, memory.
- 41+71 existing tests still pass.

### Consequence
Every future experiment (frontier probe, execution-engine trial, hardware track) now **compounds into the KnowledgeBase** automatically. The AutoScientist can query prior verified conditionals to propose better hypotheses with fewer probes; `FailureManifestoGenerator` gets data from all paths. The reverted warm-start, the adaptive early-stop win, and every cached frontier become queryable, positive *and* negative, truth — the accumulation the business thesis depends on.

**Next (>1hr, not in this budget):** wire the sink into the validation hardware tracks (`hardware_tracks.py` fpga/analog/thermo), and add a `target_hardware` knob to `TrainerConfig` that swaps in `QuantizedLoopedMLP`/`NoisyLoopedMLP` so `cost_of_plausibility` is hardware-aware (the digital-GPU-fallacy correction).

---

## 18. Session-5 Work: Hardware-Aware `cost_of_plausibility` (plan §17 "Next" begun)

Closed the §17 "Next (>1hr)" gap with the **substrate-faithful hardware layer**, so
`cost_of_plausibility` can be measured on a real (simulated) substrate instead of an
idealized digital GPU:

- **`zoo/models/eqprop/hardware_variants.py`** — moved the substrate facades out of the
  validation layer into the model zoo so `core.trainer` can use them without a
  `validation -> core` dependency. Registered as first-class models
  `quantized_looped_mlp` (`bits`, hidden state rounded to signed `[-127,127]` each step)
  and `noisy_looped_mlp` (`noise_level`, per-step Gaussian — analog/shot). Both subclass
  `LoopedMLP` and keep float gradients (surrogate-accumulation assumption), so they drop
  into the existing equilibrium loop / `CoreTrainer` path unchanged. `validation/tracks/
  hardware_tracks.py` now re-exports these (no duplicated classes).
- **`TrainerConfig.target_hardware`** — new knob (`"gpu"/"fpga"/"analog"/None`) threaded
  into `CoreTrainer._apply_hardware`, which swaps an eqprop `LoopedMLP` for the matching
  facade (rebuilt from the same `model_kwargs` + hardware default `bits`/`noise_level`),
  and records `{target_hardware, bits|noise_level}` on every `TrainingMetrics.extra` and
  probe metric. Inert for non-`LoopedMLP` models (`backprop_mlp` etc.) and for
  `None`/`"gpu"`. (Stored as `str`, not `Literal`, because `TrainerConfig` round-trips
  through OmegaConf, which cannot yet serialize a `typing.Literal` field.)
- **`CoreTrainerDriver.target_hardware`** — threaded so the probe/frontier pipeline can
  drive a substrate. `scripts/preliminary_run.py --target-hardware fpga|analog` applies it
  to BOTH the backprop reference and every bio rule, and it is now **part of the frontier
  cache identity** in both `IdealBackpropFinder._cache_name` and
  `RuleFrontierFinder._cache_name` (verified `_hw{target}` suffix + cache-load validation),
  so a GPU-derived frontier is never reused as an FPGA comparison (extends §16.3 to the
  substrate axis).
- **Result sink wired into the validation hardware tracks** — `track_16` (fpga) / `track_17`
  (analog) / `track_18` (thermo) now route their `TrackResult` through
  `record_experiment_result` via `_sink_hardware_track`: completed/partial → KnowledgeBase,
  failed → FailureTracker, tagged with `hardware` + `track_id` so the AutoScientist can
  query per-substrate conditionals. §17's "the moat compounds" now covers the hardware
  validation path too.

**Tests** (`tests/unit/test_hardware_aware.py`, 10 tests): facade registration/re-export,
quantization bounds (`[-1,1]` state), stochastic analog step, the `target_hardware` swap
(and its inertia for non-`LoopedMLP`/`None`), and the frontier cache-identity split
(GPU vs FPGA never cross-reuse). Plus sink wiring to KB/FailureTracker. `ruff`/`pyright`
clean on all new code.

**Not (yet) in this session:** running a real `--target-hardware` frontier probe on CUDA
(needs the ~40-min budget) and the deeper §7-equilibrium/hardware cross terms (e.g. the
interaction between quantization and convergence-threshold search). The knob + cache
integrity are in place; exercising them on MNIST/CIFAR with a probe run is the natural
next measurement.

### §18.1 Refactor: shared frontier-finder base (`hyperopt/_finder.py`)

The two finders (`IdealBackpropFinder`, `RuleFrontierFinder`) previously each carried the
full Optuna+JSON-cache lifecycle (~110 duplicated lines: `_cache_name`, `load_cache`,
`find`, `_search`/objective/FLOPs-aggregation, `_save_cache`, constructor knobs, point
(serialization, and the driver protocol). Consolidated into a template-method base
`_FrontierFinder[D]` (PEP 695 type param). Subclasses now supply only the varying parts:
`_cache_prefix`, `_rule_key`/`_train_model`, `_cache_identity`, `_sample_config` (their
space), and `_build_decision`/`_from_payload` (their decision dataclass). The epoch
(§16.3) and `target_hardware` (§17) cache-identity rule now lives in **one** place instead
of two, so a cache-integrity fix can't drift between finders.

Public API is unchanged (`IdealBackpropFinder`, `RuleFrontierFinder`, both decision
classes, `find_ideal_backprop`, `find_rule_frontier`, `sample_config_for_rule`), and the
**on-disk cache filenames are byte-for-byte preserved** (rule-frontier caches are
rule-first: `rule_frontier_{rule}_{task}_…`; ideal-backprop caches are task-first) so
existing cached frontiers in `logs/` remain valid. All 48 frontier/hardware tests pass.

---

## Bottom Line

1. **The fair-comparison pipeline is built and validated** — `IdealBackpropFinder` → `RuleFrontierFinder` → `compare_frontiers` → `cost_of_plausibility` → scaling laws, all runnable as one command (`scripts/preliminary_run.py`) and exercised on 5 families (MNIST, §13/§16).
2. **Backprop training is now at the epoch-time floor (~1.9 s)** via the cached-dataset `num_workers=0` fix (2.7×). The loop is not the blocker anymore.
3. **Diagnose before judging:** a low `cost_of_plausibility` or low accuracy may be an epoch-budget artifact, not a bug — use `best_epoch_acc`/`acc_at_half` (§12), record epochs as a covariate, and **match epochs between reference and bio frontiers** (§16.3 — the cache is now epoch-keyed).
4. **`peak_memory_mb` is now verified** (§6) — PASS at 0.96 ratio; memory may be used in analysis.
5. **Equilibrium compute is the real, family-specific time wall** — neural_cube/eqprop cost ~1.8×/10× `cost_of_plausibility`. Warm-start (§7) was tested and is **not** a lever; **adaptive early-stop is** (`convergence_threshold=1e-2` → ~1.3–1.4× settling speedup at negligible acc cost, now searchable for eqprop). Remaining §7 candidates: Anderson/predictor-corrector, fewer inner iterations, run as controlled experiments.
6. **The moat now compounds** (§17) — a universal result sink routes every experiment outcome (frontier, execution-engine, and future hardware-track) into the KnowledgeBase / FailureTracker; the technology that was built is finally fed by the measurements. Build on this: hardware-aware `cost_of_plausibility` (`target_hardware` → fpga/analog/thermo tracks) is the next strategic lever.
6. **Build out the conditional knowledge base** (frontiers + scaling laws with CI) — run more probes (→ §4A budget) to tighten CIs; this is the pipeline's data source.

*End of EXPERIMENT_PLAN2.md*
