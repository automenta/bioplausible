# EXPERIMENT_PLAN2.md — Strategic Forward Plan (Revised)

**Status**: Post-MNIST-trio partial run. Data-loader cache is in. CoreTrainer loop overhead identified as dominant cost. MNIST trio run (partial: 27/36 probes) shows `neural_cube` ties backprop on MNIST (~0.98), eqprop slightly below. Both bio rules cost ~3× wall-time. Experiment layer plumbing largely solved. Need to **ask better questions, measure to find optimization opportunities, then build optimized implementations where the data shows value.**

---

## 1. The Real Scientific Question (Not "Does X Beat Y?")

The current parity framing ("does bio model X match backprop accuracy?") is the **wrong question**. It yields "no" or "tie" on toys, and the answer changes with epoch count.

**The right question is multidimensional:**

> *For a given task/domain, what is the Pareto frontier of (accuracy, wall_time, FLOPs, peak_memory, energy) achievable by each learning rule at its own hyperparameter optimum? Which rules dominate under which resource constraints?*

This reframes "parity" as **a Pareto trade-off surface**. A rule that matches backprop accuracy at 0.3× FLOPs/energy *is* the result — not "it matches accuracy." The experiment layer already emits `wall_time_s, forward_flops, backward_flops, peak_memory_mb`; we need to stop treating accuracy as the only axis.

**Autonomous invention pipeline angle:** An autonomous system doesn't "run an experiment" — it navigates a design space. It needs: *given compute budget X, memory cap M, accuracy target Y, and latency budget L, which learning rule + architecture + hyperparams should I deploy?* Our job: make that mapping empirically grounded.

---

## 2. The Memory Dimension — Measure First, Optimize Later

**The hypothesis space:** 
- Standard backprop stores forward activations for backward pass → **O(L·B·D)** activation memory.
- Standard equilibrium models (EQ, NC) also store activations for nudged/error phase → similar O(L·B·D) in current implementations.
- **O(1) additional memory is achievable with reversible architectures, activation checkpointing, or gradient recomputation** — these are optimization techniques we can IMPLEMENT and MEASURE, not theoretical impossibilities.

**The measurement protocol (hypothesis-driven):**
1. **Baseline measurement**: For each rule, measure `peak_memory_mb` vs batch size (16, 32, 64, 128, 256) and width (64, 128, 256, 512) on MNIST/CIFAR. Verify against `torch.cuda.max_memory_allocated()`.
2. **Breakdown measurement**: Decompose `peak_memory_mb` into: model params + optimizer states + forward activations + grads + equilibrium buffers. Confirm with `torch.cuda.memory_snapshot()`.
3. **Implement & measure optimizations** as controlled experiments:
   - **Activation checkpointing** (gradient recomputation): expected ~2× compute, ~O(√L) memory reduction
   - **Reversible architectures** (RevNet, iRevNet): ~O(1) additional memory, ~2× compute
   - **Activation offloading** (CPU offload): ~CPU→GPU transfer cost, massive memory savings
   - **Gradient accumulation**: effective batch size increase without memory increase
4. **Measure the trade-off**: For each optimization, measure the (accuracy, wall_time, FLOPs, memory) tuple. This IS the optimization opportunity mapping.

**The target result:** A plot showing for each rule: (memory, accuracy, wall_time) with and without each optimization technique. The crossover point where an optimization becomes valuable IS the decision data.

**Why this matters for autonomous invention:** Edge deployment has hard memory caps. A rule that fits in 50 MB while backprop needs 2 GB is the *only* viable option — accuracy is secondary to "does it fit."

---

## 3. The Compute Wall — Measure First, Then Optimize

**Measured reality (cached MNIST, 128/2):**
- Backprop: 6.4s/epoch (floor 1.9s) → 4.5s CoreTrainer overhead
- EQ/Neural Cube: 21s/epoch → dominated by equilibrium iterations (not data)

**This is a measurement, not a verdict.** The point of measuring is to identify WHERE optimization effort pays off:

| Component | Current | Floor | Overhead | Optimization Opportunity |
|-----------|---------|-------|----------|--------------------------|
| Backprop data loading | ~1s/ep | ~0.1s | 0.9s | DataLoader cache ✅ done, pin_memory, persistent workers |
| Backprop training loop | ~5.4s/ep | ~1.9s | 3.5s | **Primary target**: validation eval, per-batch Python overhead, missing `torch.compile` |
| Equilibrium iterations | ~21s/ep | ~1s (compute) | 20s | **Algorithmic**: fewer iterations, learned damping, predictor-corrector, warm-start |

**The measurement protocol for optimization:**
1. **Profile each component** (data, forward, backward, equilibrium, validation, metrics) with `torch.profiler`.
2. **Quantify each overhead** as % of epoch time.
3. **Implement targeted optimizations** as controlled experiments: measure (accuracy, time, memory) before/after.
4. **Only invest in deep optimization** when the data shows the ROI crosses the threshold (e.g., "if we can reduce equilibrium iterations by 10×, CIFAR-10 becomes feasible").

**The equilibrium iteration wall is an algorithmic opportunity, not a dead end.** Options to measure:
- **Learned damping/step size** (meta-learned per layer)
- **Predictor-corrector / Anderson acceleration** (reduce iterations 5-10×)
- **Warm-start from previous epoch** (reuse equilibrium point)
- **Early stopping with adaptive threshold** (dynamic convergence criteria)
- **Parallel equilibrium solvers** (Jacobi-style parallel updates)

Each is a hypothesis to TEST, not a dismissal. The measurement infrastructure exists to tell us which one pays off.

---

## 1. The Real Scientific Question (Not "Does X Beat Y?")

The current parity framing ("does bio model X match backprop accuracy?") is the **wrong question**. It yields "no" or "tie" on toys, and the answer changes with epoch count.

**The right question is multidimensional:**

> *For a given task/domain, what is the Pareto frontier of (accuracy, wall_time, FLOPs, peak_memory, energy) achievable by each learning rule at its own hyperparameter optimum, WITH available optimizations? Which rules dominate under which resource constraints?*

This reframes "parity" as **a Pareto trade-off surface**. A rule that matches backprop accuracy at 0.3× FLOPs/energy *is* the result — not "it matches accuracy." The experiment layer already emits `wall_time_s, forward_flops, backward_flops, peak_memory_mb`; we need to stop treating accuracy as the only axis.

**Autonomous invention pipeline angle:** An autonomous system doesn't "run an experiment" — it navigates a design space. It needs: *given compute budget X, memory cap M, accuracy target Y, and latency budget L, which learning rule + architecture + hyperparams + optimizations should I deploy?* Our job: make that mapping empirically grounded.

---

## 2. The Memory Dimension — The Hidden Frontier (With Optimization Path)

**The hypothesis space:** 
- Standard backprop stores forward activations for backward pass → **O(L·B·D)** activation memory.
- Standard equilibrium models (EQ, NC) also store activations for nudged/error phase → similar O(L·B·D) in current implementations.
- **O(1) additional memory is achievable** via: reversible architectures, activation checkpointing (gradient recomputation), activation offloading, gradient recomputation — these are optimization techniques we can IMPLEMENT and MEASURE.

**The measurement protocol (hypothesis-driven):**
1. **Baseline measurement**: For each rule, measure `peak_memory_mb` vs batch size (16, 32, 64, 128, 256) and width (64, 128, 256, 512) on MNIST/CIFAR. Verify against `torch.cuda.max_memory_allocated()`.
2. **Breakdown measurement**: Decompose `peak_memory_mb` into: model params + optimizer states + forward activations + grads + equilibrium buffers. Confirm with `torch.cuda.memory_snapshot()`.
3. **Implement & measure optimizations** as controlled experiments:
   - **Activation checkpointing** (gradient recomputation): expected ~2× compute, ~O(√L) memory reduction
   - **Reversible architectures** (RevNet, iRevNet): ~O(1) additional memory, ~2× compute
   - **Activation offloading** (CPU offload): ~CPU→GPU transfer cost, massive memory savings
   - **Gradient accumulation**: effective batch size increase without memory increase
4. **Measure the trade-off**: For each optimization, measure the (accuracy, wall_time, FLOPs, memory) tuple. This IS the optimization opportunity mapping.

**The target result:** A plot showing for each rule: (memory, accuracy, wall_time) with and without each optimization technique. The crossover point where an optimization becomes valuable IS the decision data.

**Why this matters for autonomous invention:** Edge deployment has hard memory caps. A rule that fits in 50 MB while backprop needs 2 GB is the *only* viable option — accuracy is secondary to "does it fit."

---

## 3. Finer-Grained Hyperparameter Search (Beyond Coarse Grids)

Current grids (`hidden_dim: [16,32,64,128,256]`, `num_layers: [1,2,4]`) are coarse and linear. They miss optima and waste probes.

### A. Continuous Bayesian Search (Already Built, Not Used Effectively)
`OptunaBayesProducer` with TPE exists. Use it properly:
- Continuous ranges: `hidden_dim ∈ [32, 1024]` (log-uniform), `num_layers ∈ [1, 6]`, `learning_rate ∈ [1e-5, 1e-1]` (log-uniform), `dropout ∈ [0, 0.5]`, `weight_decay ∈ [1e-6, 1e-2]`.
- **Budget: 500–1000 probes per task** for a meaningful Pareto frontier (not 100–200). TPE finds the true posterior.
- The "ideal backprop" = Bayesian optimum of backprop on a task. All other rules compared *at their own Bayesian optima*, not at one shared coarse grid.

### B. Multi-Fidelity / Successive Halving
For expensive tasks (CIFAR-10):
- Fidelity = epochs (5 → 10 → 20 → 40) or batch size / dataset fraction.
- Early-stop bad configs; promote promising ones.
- Optuna supports `HyperbandPruner`. Our `ConfigProducer` interface can support it.

### C. Architecture Search (Not Just Hyperparams)
Grid only varies width/depth. Missing:
- Activation functions (ReLU, GELU, Swish, SiLU)
- Normalization (BatchNorm, LayerNorm, GroupNorm, none)
- Skip connections (plain, ResNet-style, DenseNet-style)
- **Equilibrium-specific**: damping factor, step size, max iterations, convergence threshold, damping schedule, predictor-corrector steps
These are *rule-specific* hyperparameters that could be decisive for equilibrium models.

### D. The "Ideal Backprop" as the Moving Reference (Corrected)
**Protocol (corrected):**
1. Define backprop search space (width, depth, LR, WD, dropout, act, norm, optimizer, scheduler, batch size).
2. Run **full Bayesian optimization (2000+ probes)** to find the *true* Pareto frontier of backprop: (accuracy, FLOPs, memory, time). **200 probes is far too low for CIFAR-10.**
3. This is the "ideal backprop" reference — a Pareto frontier, not a point.
4. For each bio rule, run its own Bayesian search in its own rule-specific space (including equilibrium params: damping, iterations, convergence threshold, predictor-corrector config).
5. Compare the **two Pareto frontiers**:
   - Does bio rule's frontier touch/dominate backprop's at any operating point?
   - At what FLOPs/memory budget does bio rule dominate?
   - What is the "cost of bio-plausibility" in FLOPs/memory at matched accuracy?

**The "Ideal Backprop" as a Service:**
Build `IdealBackpropFinder(task, budget_probes=2000) → ParetoFrontier`. Runs once per task, caches result, serves as reference for all bio-rule experiments.

---

## 4. Conditions of Results → Application Enablement (Autonomous Pipeline)

The autonomous pipeline needs **decision rules** mapping (task, constraints, results) → (deployment decision). Experiments must produce *conditionals*, not point estimates.

| Pipeline Decision | Required Experimental Evidence |
|-------------------|--------------------------------|
| **Rule selection** | "For task T with compute budget C and memory M, rule R achieves accuracy A with CI." |
| **Architecture choice** | "For rule R on task T, optimal (W, L, act) = (W, L, act) with Pareto frontier F." |
| **Resource allocation** | "To reach accuracy A on task T with rule R, need FLOPs F, memory M, time T; scaling laws predict cost." |
| **Deployment target** | "Rule R fits in memory M on device D; latency L; energy E per inference." |
| **Fallback chain** | "If rule R fails on task T, try R' with config C'." |

**Experimental design implication:** Every experiment must emit **fitted conditionals with uncertainty**, not point estimates. Fit scaling laws (`accuracy ~ log(FLOPs)`, `memory ~ L·B·D`) and emit *fitted parameters with uncertainty*.

**Autonomous pipeline loop:**
1. Task arrives with constraints (accuracy target, latency budget, memory cap, energy budget, hardware spec).
2. Pipeline queries experimental DB for conditionals: "What rules/archs satisfy these constraints?"
3. If no data, pipeline triggers targeted experiment (Bayesian search in relevant subspace) to fill gap.
4. Pipeline makes decision and deploys.
5. Runtime monitoring feeds back to update DB (online learning).

---

## 5. The "Ideal Backprop" as a Service (Corrected)

**The rigorous protocol (corrected):**
1. Define backprop search space (width, depth, LR, WD, dropout, act, norm, optimizer, scheduler, batch size).
3. Run **full Bayesian optimization (2000+ probes)** to find the *true* Pareto frontier of backprop: (accuracy, FLOPs, memory, time). **200 probes is far too low for CIFAR-10.**
3. This is the "ideal backprop" reference — a Pareto frontier, not a point.
4. For each bio rule, run its own Bayesian search in its own rule-specific space (including equilibrium params: damping, iterations, convergence threshold, predictor-corrector config).
5. Compare the **two Pareto frontiers**:
   - Does bio rule's frontier touch/dominate backprop's at any operating point?
   - At what FLOPs/memory budget does bio rule dominate?
   - What is the "cost of bio-plausibility" in FLOPs/memory at matched accuracy?

**The "Ideal Backprop" as a Service:**
Build `IdealBackpropFinder(task, budget_probes=2000) → ParetoFrontier`. Runs once per task, caches result, serves as reference for all bio-rule experiments.

---

## 5. The Training Loop Bottleneck — The Real Blocker

**Measured floor:** Minimal MNIST train epoch (data + forward + backward) = **1.9s** (938 batches, 128/2 MLP, RTX 3080).  
**CoreTrainer cached run:** backprop 6.4s/epoch, eqprop/neural_cube ~21s/epoch.

**Gap analysis:**
- Backprop: 6.4s vs 1.9s floor → **4.5s overhead (3.4×)** in CoreTrainer loop.
- Equilibrium models: ~21s/epoch → dominated by equilibrium iterations (10-100× per batch), not data loading.

**Root causes to investigate (priority order):**
1. **Per-epoch validation eval** — runs full val set every epoch. Cost: ~1-2s/epoch.
2. **Per-batch Python overhead** — metric computation, device sync, logging, callback dispatch per batch.
3. **Per-epoch device sync / memory stats** — `peak_memory_mb` collection, energy tracking.
4. **Missing `torch.compile` / vectorization** — per-batch Python loops vs fused kernels.
5. **Energy tracking overhead** — `track_energy=True` adds significant cost.

**Immediate profiling targets (this week):**
1. `torch.profiler` on CoreTrainer `fit()` — identify top 3 time consumers.
2. Disable validation eval → measure epoch time drop.
3. Disable `track_memory`/`track_flops`/`track_energy` → measure drop.
4. Add `torch.compile` to model forward/backward.

**Target:** Reduce backprop overhead from 4.5s → ≤1.5s/epoch (total ≤3.5s/epoch). Equilibrium models will still be compute-bound (equilibrium iterations), but backprop should approach floor.

---

## 5. The Memory Measurement Gap — Verify First

**We don't know what `peak_memory_mb` actually measures.** Before building analysis on it:
1. Run a probe with `track_memory=True` on a simple task.
2. Compare reported `peak_memory_mb` vs `torch.cuda.max_memory_allocated()` / `max_memory_reserved()`.
3. Break down: model params (known) + optimizer states + forward activations + grads + equilibrium buffers.
4. Verify it matches `torch.cuda.max_memory_allocated()` (should be close if implemented correctly).

**Only after verification** build memory-vs-accuracy plots and scaling law fits.

---

## 5. The Equilibrium Compute Wall — Opportunity, Not Wall

**Measured:** `eqprop`/`neural_cube` at **21s/epoch** on MNIST (128/2). 

**This is a measurement, not a verdict.** The equilibrium iterations (10-100 inner steps per batch) dominate. This is an **algorithmic optimization opportunity**, not a dead end.

**Optimization hypotheses to TEST (each is a controlled experiment):**
1. **Learned damping/step size** (meta-learned per layer, or Adam on equilibrium loss)
2. **Anderson acceleration / predictor-corrector** (reduce iterations 5-10×)
3. **Warm-start from previous epoch** (reuse equilibrium point as initialization)
4. **Early stopping with adaptive threshold** (dynamic convergence criteria)
5. **Jacobi-style parallel updates** (instead of Gauss-Seidel sequential)
5. **Equilibrium as implicit layer** (implicit differentiation, single backward pass)

**Each is a hypothesis to TEST as a controlled experiment:**
- Implement variant → measure (accuracy, wall_time, memory) → compare to baseline
- Only invest in deep optimization when the data shows ROI crosses threshold

**The equilibrium iteration wall is an algorithmic optimization opportunity, not a dead end.** The measurement infrastructure exists to tell us which one pays off.

---

## 5. The Training Loop Bottleneck — The Real Blocker

**Measured floor:** Minimal MNIST train epoch (data + forward + backward) = **1.9s** (938 batches, 128/2 MLP, RTX 3080).  
**CoreTrainer cached run:** backprop 6.4s/epoch, eqprop/neural_cube ~21s/epoch.

**Gap analysis:**
- Backprop: 6.4s vs 1.9s floor → **4.5s overhead (3.4×)** in CoreTrainer loop.
- Equilibrium models: ~21s/epoch → dominated by equilibrium iterations (10-100× per batch), not data loading.

**Root causes to investigate (priority order):**
1. **Per-epoch validation eval** — runs full val set every epoch. Cost: ~1-2s/epoch.
2. **Per-batch Python overhead** — metric computation, device sync, logging, callback dispatch per batch.
3. **Per-epoch device sync / memory stats** — `peak_memory_mb` collection, energy tracking.
4. **Missing `torch.compile` / vectorization** — per-batch Python loops vs fused kernels.
5. **Energy tracking overhead** — `track_energy=True` adds significant cost.

**Immediate profiling targets (this week):**
1. `torch.profiler` on CoreTrainer `fit()` — identify top 3 time consumers.
2. Disable validation eval → measure epoch time drop.
3. Disable `track_memory`/`track_flops`/`track_energy` → measure drop.
4. Add `torch.compile` to model forward/backward.

**Target:** Reduce backprop overhead from 4.5s → ≤1.5s/epoch (total ≤3.5s/epoch). Equilibrium models will still be compute-bound (equilibrium iterations), but backprop should approach floor.

---

## 5. The Memory Measurement Gap — Verify First

**We don't know what `peak_memory_mb` actually measures.** Before building analysis on it:
1. Run a probe with `track_memory=True` on a simple task.
2. Compare reported `peak_memory_mb` vs `torch.cuda.max_memory_allocated()` / `max_memory_reserved()`.
3. Break down: model params (known) + optimizer states + forward activations + grads + equilibrium buffers.
4. Verify it matches `torch.cuda.max_memory_allocated()` (should be close if implemented correctly).

**Only after verification** build memory-vs-accuracy plots and scaling law fits.

---

## 5. The Equilibrium Compute Wall — Opportunity, Not Wall

**Measured:** `eqprop`/`neural_cube` at **21s/epoch** on MNIST (128/2). 

**This is a measurement, not a verdict.** The equilibrium iterations (10-100 inner steps per batch) dominate. This is an **algorithmic optimization opportunity**, not a dead end.

**Optimization hypotheses to TEST (each is a controlled experiment):**
1. **Learned damping/step size** (meta-learned per layer, or Adam on equilibrium loss)
2. **Anderson acceleration / predictor-corrector** (reduce iterations 5-10×)
3. **Warm-start from previous epoch** (reuse equilibrium point as initialization)
4. **Early stopping with adaptive threshold** (dynamic convergence criteria)
5. **Jacobi-style parallel updates** (instead of Gauss-Seidel sequential)
5. **Equilibrium as implicit layer** (implicit differentiation, single backward pass)

**Each is a hypothesis to TEST as a controlled experiment:**
- Implement variant → measure (accuracy, wall_time, memory) → compare to baseline
- Only invest in deep optimization when the data shows ROI crosses threshold

**The equilibrium iteration wall is an algorithmic optimization opportunity, not a dead end.** The measurement infrastructure exists to tell us which one pays off.

---

## 6. Immediate Next Actions (This Week — Realistic)

| Action | Success Criterion |
|--------|-------------------|
| **Profile CoreTrainer loop** with `torch.profiler` | Top 3 overheads identified; each >5% of epoch time |
| **Disable validation eval** in a test run | Measure epoch time drop; target: -1.5s/epoch |
| **Disable `track_memory`/`track_flops`/`track_energy`** | Measure epoch time drop; quantify overhead |
| **Add `pin_memory=True` to DataLoader** in `create_data_loaders` | MNIST epoch time drops from 6.4s → ≤4s |
| **Generalize vision dataset cache to CIFAR-10** | `get_vision_dataset("cifar10")` returns cached TensorDataset |
| **Verify `peak_memory_mb`** vs `torch.cuda.max_memory_allocated()` | Reported value within 10% of `max_memory_allocated()` |
| **Profile equilibrium model inner loop** (eqprop 1 epoch) | Identify equilibrium iteration cost; count iterations/batch |
| **Define continuous search spaces per rule** | `RULE_SPACES = {"backprop": {...}, "eqprop": {...}, "neural_cube": {...}, ...}` with equilibrium-specific params |
| **Add `HyperbandPruner` support to `OptunaBayesProducer`** | `producer = OptunaBayesProducer(..., pruner=HyperbandPruner())` |

---

## 7. The One Metric That Matters

Stop reporting "accuracy." Report **the Pareto frontier of (accuracy, FLOPs, memory, wall_time, energy) for each rule at its own optimum**.

The single number that summarizes competitiveness:

> **`cost_of_plausibility(task) = min_{p∈bio_frontier} [ FLOPs(p) / FLOPs(backprop_at_same_acc) ] × (mem(p) / mem(backprop_at_same_acc)) × (time(p) / time(backprop_at_same_acc))`**

At a given accuracy level, how many more FLOPs×memory×time does the bio rule need vs ideal backprop? If the geometric mean ratio is ≤1.5 at 95% of backprop accuracy, the rule is **deployment-viable**. If it's 5×, it's a curiosity.

This composite number is what the autonomous pipeline uses to decide "deploy or not."

---

## Appendix: The MNIST Trio Result (Verified)

| Model | n | Mean Acc | Wall Time (s/ep) | Peak Mem (reported, unverified) |
|-------|---|----------|------------------|--------------------------------|
| backprop_mlp | 10 | 0.9799 | 6.4 s/ep | ~200 MB |
| neural_cube | 10 | 0.9796 | 21.0 s/ep | ~300 MB |
| eqprop_mlp | 1 | 0.965 | 21.1 s/ep | ~350 MB |

**Interpretation:** On MNIST (real 28×28), `neural_cube` **ties** backprop (won on digits). eqprop slightly below. Both bio rules cost ~3× wall-time. The "winner" on digits does not generalize its advantage to real 28×28 images. This is the kind of negative result that's valuable *as part of a systematic mapping* — not as a standalone "experiment."

---

## Bottom Line

1. **Measure first, optimize when the data shows ROI.** Every "wall" is a hypothesis about where optimization pays off.
2. **Fix the training loop** (CoreTrainer overhead) — this unlocks CIFAR-10 scale.
3. **Verify memory measurements** before building analysis on them.
4. **Equilibrium compute is an algorithmic opportunity, not a wall** — test predictor-corrector, learned damping, warm-start, etc.
5. **Build the conditional knowledge base** (fitted scaling laws + Pareto frontiers) — that's the product; experiments are just the data source.

The pipeline is the product; experiments are just its data source. Build the infrastructure that lets us ask and answer the right questions at scale, and invest in optimizations when the measurements show they pay off.

*End of EXPERIMENT_PLAN2.md (Final)*

---

**Appendix: The MNIST Trio Result (For the Record)**

| Model | n | Mean Acc | Wall Time (s/ep) | Peak Mem (reported, unverified) |
|-------|---|----------|------------------|----------------|
| backprop_mlp | 10 | 0.9799 | 6.4 s/ep | ~200 MB |
| neural_cube | 10 | 0.9796 | 21.0 s/ep | ~300 MB |
| eqprop_mlp | 1 | 0.965 | 21.1 s/ep | ~350 MB |

**Interpretation:** On MNIST (real 28×28), `neural_cube` **ties** backprop (won on digits). eqprop slightly below. Both bio rules cost ~3× wall-time. The "winner" on digits does not generalize its advantage to real 28×28 images. This is the kind of negative result that's valuable *as part of a systematic mapping* — not as a standalone "experiment."

---

**Bottom line:** Measure to find optimization opportunities. Don't dismiss — measure to find WHERE optimization pays off, then invest. The pipeline is the product; experiments are just its data source. Build the infrastructure that lets us ask and answer the right questions at scale, and invest in optimizations when the measurements show they pay off.

*End of EXPERIMENT_PLAN2.md (Final)*