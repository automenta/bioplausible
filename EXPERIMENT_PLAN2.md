# EXPERIMENT_PLAN2.md — Strategic Forward Plan (Revised)

**Status**: Post-MNIST-trio partial run. Data-loader cache is in. CoreTrainer loop overhead is the dominant cost. MNIST trio (partial: 27/36 probes) shows `neural_cube` ties backprop on MNIST (~0.98), eqprop slightly below; both bio rules cost ~3× wall-time. Experiment-layer plumbing is largely solved. Mission: **ask better questions, measure to find optimization opportunities, then build optimized implementations where the data shows value.**

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

---

## 6. The Memory Measurement Gap — Verify First

**We don't know what `peak_memory_mb` actually measures.** Before building analysis on it:
1. Run a probe with `track_memory=True` on a simple task.
2. Compare reported `peak_memory_mb` vs `torch.cuda.max_memory_allocated()` / `max_memory_reserved()`.
3. Break down: model params + optimizer states + forward activations + grads + equilibrium buffers.
4. Verify it matches `torch.cuda.max_memory_allocated()` (should be close if implemented correctly).

**Only after verification** build memory-vs-accuracy plots and scaling-law fits.

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

| Action | Success Criterion |
|--------|-------------------|
| **Profile CoreTrainer loop** with `torch.profiler` | Top 3 overheads identified; each >5% of epoch time |
| **Disable validation eval** in a test run | Measure epoch-time drop; target −1.5 s/epoch |
| **Disable `track_memory`/`track_flops`/`track_energy`** | Measure epoch-time drop; quantify overhead |
| **Add `pin_memory=True` to DataLoader** in `create_data_loaders` | MNIST epoch time drops from 6.4 s → ≤4 s |
| **Generalize vision dataset cache to CIFAR-10** | `get_vision_dataset("cifar10")` returns a cached `TensorDataset` |
| **Verify `peak_memory_mb`** vs `torch.cuda.max_memory_allocated()` | Reported value within 10% of `max_memory_allocated()` |
| **Profile equilibrium inner loop** (eqprop 1 epoch) | Identify equilibrium iteration cost; count iterations/batch |
| **Define continuous search spaces per rule** | `RULE_SPACES = {"backprop": {...}, "eqprop": {...}, "neural_cube": {...}, ...}` incl. equilibrium-specific params |
| **Add `HyperbandPruner` support to `OptunaBayesProducer`** | `producer = OptunaBayesProducer(..., pruner=HyperbandPruner())` |

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

## Bottom Line

1. **Measure first, optimize when the data shows ROI.** Every "wall" is a hypothesis about where optimization pays off.
2. **Fix the training loop** (CoreTrainer overhead) — this unlocks CIFAR-10 scale.
3. **Verify memory measurements** before building analysis on them.
4. **Equilibrium compute is an algorithmic opportunity, not a wall** — test predictor-corrector, learned damping, warm-start, etc.
5. **Build the conditional knowledge base** (fitted scaling laws + Pareto frontiers) — the pipeline is the product; experiments are just its data source. Invest in optimizations when measurements show they pay off.

*End of EXPERIMENT_PLAN2.md*
