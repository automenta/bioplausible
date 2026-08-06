# EXPERIMENT_PLAN2.md — Strategic Forward Plan (Big Picture)

**Status**: Post-MNIST-trio partial run. The data-loader cache is in. The CoreTrainer loop overhead is identified as the dominant cost. The MNIST trio run gave a clear null signal: `neural_cube` ties backprop on real 28×28 images, eqprop slightly worse. The experiment layer plumbing is largely solved. Now we need to **ask better questions and build the infrastructure that answers them**.

---

## 1. The Real Scientific Question (Not "Does X Beat Y?")

The current parity framing ("does bio model X match backprop accuracy?") is **the wrong question**. It's a yes/no that yields "no" or "tie" on toys and near-toy tasks, and the answer changes with epoch count.

**The right question is multidimensional:**

> *For a given task/domain, what is the Pareto frontier of (accuracy, wall_time, FLOPs, peak_memory, energy) achievable by each learning rule? Which rules dominate under which resource constraints?*

This reframes "parity" as **a Pareto trade-off surface**. A rule that matches backprop accuracy at 0.3× FLOPs/energy *is* the result — not "it matches accuracy." The experiment layer already emits `wall_time_s, forward_flops, backward_flops, peak_memory_mb`; we just need to stop treating accuracy as the only axis and start rendering the full trade-off.

**The autonomous invention pipeline angle:** An autonomous system doesn't "run an experiment" — it navigates a design space. It needs to know: *given a compute budget X and accuracy target Y, which learning rule + architecture + hyperparams should I deploy?* Our job is to make that mapping empirically grounded, not to produce one-off "X beats Y" blog posts.

---

## 2. The Memory Dimension — The Hidden Frontier

We've ignored `peak_memory_mb` entirely. This is a massive oversight.

**The hypothesis:** Backprop requires storing the full forward activation graph for the backward pass → **O(L·B·D)** memory (L layers, B batch, D width). Equilibrium models (EQ, NC, etc.) settle by iterating on a fixed-point; they *can* be implemented with **O(1)** additional memory (overwrite activations in place, or use reversible formulations). This is a **theoretical superpower** of bio-plausible rules that we have never measured.

**What to measure (per probe):**
- `peak_memory_mb` — already emitted by CoreTrainer when `track_memory=True`
- Break it down: model params + optimizer states + forward activations + grads + equilibrium buffers
- Plot `peak_memory_mb` vs `accuracy` vs `wall_time` for each rule

**The target result:** A plot where the X axis is memory (MB), Y is accuracy, color is rule. If equilibrium rules achieve 95% of backprop accuracy at 10% of peak memory, *that* is a publishable, deployment-relevant result. No one has this data rigorously.

**Experimental protocol:**
- Run a memory-focused sweep: vary batch size (16, 32, 64, 128, 256) and width (64, 128, 256, 512) on a fixed task (MNIST or CIFAR-10 subset).
- For each (rule, config), log `peak_memory_mb`, `accuracy`, `wall_time_s`, `forward_flops`, `backward_flops`.
- Fit the scaling laws: `memory ~ O(L·B·D)` for backprop vs `memory ~ O(D)` (or `O(L·D)`) for equilibrium rules.
- The crossover point where equilibrium rules become memory-advantageous IS the result.

**Why this matters for autonomous invention:** An autonomous agent deploying to edge devices (mobile, embedded) has hard memory caps. A rule that fits in 50 MB but backprop needs 2 GB is the *only* viable option — accuracy is secondary to "does it fit."

---

## 3. Finer-Grained Hyperparameter Search (Beyond Coarse Grids)

The current grids (`hidden_dim: [16,32,64,128,256]`, `num_layers: [1,2,4]`) are **coarse and linear**. They miss the true optima and waste probes on dead regions.

### A. Continuous Bayesian Search (Already Built, Not Used Effectively)
We have `OptunaBayesProducer` with TPE. Use it properly:
- Define continuous ranges: `hidden_dim ∈ [32, 1024]` (log-uniform), `num_layers ∈ [1, 6]` (int), `learning_rate ∈ [1e-5, 1e-1]` (log-uniform), `dropout ∈ [0, 0.5]`, `weight_decay ∈ [1e-6, 1e-2]`.
- Budget: 100–200 probes per task. TPE will find the true posterior over the space.
- The "ideal backprop" is the Bayesian optimum of backprop on a task. All other rules are compared *at their own Bayesian optima*, not at one shared coarse grid.

### B. Multi-Fidelity / Successive Halving
For expensive tasks (CIFAR-10, ImageNet), use multi-fidelity:
- Fidelity = epochs (5 → 10 → 20 → 40) or batch size.
- Early-stop bad configs; promote promising ones.
- Optuna supports this via `HyperbandPruner` or custom successive halving. Our `ConfigProducer` interface can support it.

### C. Architecture Search (Not Just Hyperparams)
The grid only varies width/depth. What about:
- Activation functions (ReLU, GELU, Swish, SiLU)
- Normalization (BatchNorm, LayerNorm, GroupNorm, none)
- Skip connections (plain, ResNet-style, DenseNet-style)
- Equilibrium-specific: damping factor, step size, max iterations, convergence threshold
These are *rule-specific* hyperparameters that could be decisive for equilibrium models.

### D. The "Ideal Backprop" as the Moving Reference
Don't compare rules at a shared coarse grid. Instead:
1. Find backprop's Bayesian optimum on the task (accuracy, FLOPs, memory, time).
2. Find each bio rule's Bayesian optimum on the *same* task (with its own rule-specific search space).
3. Compare the **Pareto fronts** of the two optima.
This is fair: each rule gets its best shot. The "gap" is the true cost of bio-plausibility.

---

## 4. Conditions of Results → Application Enablement (Autonomous Pipeline Vision)

The autonomous invention pipeline doesn't just "run experiments." It needs **decision rules** that map from (task, constraints, results) → (deployment decision). Our experimental design should produce the *conditionals* the pipeline needs.

### What conditionals does the pipeline need?

| Pipeline Decision | Required Experimental Evidence |
|-------------------|--------------------------------|
| **Rule selection** | "For task T with compute budget C and memory M, rule R achieves accuracy A with confidence interval CI." |
| **Architecture choice** | "For rule R on task T, the optimal (width, depth, activation) is (W, L, act) with Pareto frontier F." |
| **Resource allocation** | "To reach accuracy A on task T with rule R, I need FLOPs F, memory M, time T; scaling laws predict cost for higher accuracy." |
| **Deployment target** | "Rule R fits in memory M on device D; latency L; energy E per inference." |
| **Fallback/fallback chain** | "If rule R fails to converge on task T, try rule R' with config C'." |

**What this means for experiment design:**
- Every experiment must emit **full conditionals**, not just point estimates.
- Fit scaling laws (accuracy vs epochs, width, batch, FLOPs, memory) and emit the *fitted parameters* as the result, not just point estimates.
- The report should output *fitted models* (e.g., `accuracy ~ log(FLOPs)`, `memory ~ L·B·D`) that the pipeline can query.

### The autonomous invention pipeline loop:
1. **Task arrives** with constraints (accuracy target, latency budget, memory cap, energy budget, hardware spec).
2. **Pipeline queries the experimental database** for conditionals: "What rules/archs satisfy these constraints?"
3. **If no data**, pipeline triggers a targeted experiment (Bayesian search in the relevant subspace) to fill the gap.
4. **Pipeline makes decision** and deploys.
5. **Runtime monitoring** feeds back to update the database (online learning).

Our job is to build the **experimental database** with the right structure — not just a list of runs, but *fitted conditionals with uncertainty*.

---

## 5. The "Ideal Backprop" Optimization — The Reference Standard

The current paradigm: "run backprop at one config and call it the baseline." This is lazy and unfair.

### The rigorous protocol:
1. **Define the search space for backprop** (width, depth, LR, WD, dropout, act, norm, optimizer, scheduler, batch size).
2. **Run a full Bayesian optimization** (200–500 probes) to find the *true* Pareto frontier of backprop on the task: the set of configs that are not dominated in (accuracy, FLOPs, memory, time).
3. **This is the "ideal backprop" reference** — a Pareto frontier, not a single point.
4. For each bio rule, run its own Bayesian search *in its own rule-specific space* (including rule-specific params like damping, iterations, etc.).
5. Compare the **two Pareto frontiers**:
   - Does the bio rule's frontier touch or dominate backprop's at any operating point?
   - At what FLOPs/memory budget does the bio rule dominate?
   - What is the "cost of bio-plausibility" in FLOPs/memory at matched accuracy?

This is the **only fair comparison**. A single-point baseline is scientifically meaningless.

### The "Ideal Backprop" as a Service
We should build a reusable component: `IdealBackpropFinder(task, budget_probes=500) → ParetoFrontier`. It runs once per task, caches the result, and serves as the reference for *all* subsequent bio-rule experiments on that task. This is infrastructure, not an experiment.

---

## 6. The Big Picture: From Experiments to an Autonomous Invention Pipeline

The end state we're building toward is not "a set of experiments." It's **a system that autonomously explores the learning-rule/architecture/hyperparam design space, builds a conditional knowledge base, and can answer deployment queries for arbitrary tasks and constraints.**

### The pipeline components we need to build (in priority order):

| Component | Status | Next Action |
|-----------|--------|-------------|
| **Experiment layer** (config, run, report, resume) | ✅ Done | Harden, add tests |
| **Data caching / fast DataLoader** | 🟡 Partial (MNIST cache works) | Generalize to all vision tasks; add `pin_memory`, persistent workers |
| **Training loop efficiency** | 🔴 **Critical gap** (3× overhead) | Profile CoreTrainer loop; remove per-epoch DataLoader recreation; add `torch.compile`; vectorize metrics; batch `to(device)` |
| **Memory measurement** | 🟡 Emitted, not analyzed | Add memory breakdown; plot Pareto (acc vs memory); fit scaling laws |
| **Bayesian search + multi-fidelity** | 🟡 `OptunaBayesProducer` exists | Add `HyperbandPruner`; define continuous spaces per rule; multi-fidelity epochs |
| **Ideal backprop finder** | ❌ Missing | Build `IdealBackpropFinder(task, budget)` → returns cached Pareto frontier |
| **Rule-specific search spaces** | ❌ Missing | Define per-rule continuous spaces (equilibrium params, etc.) |
| **Pareto comparison engine** | ❌ Missing | Compare two Pareto frontiers; compute "cost of bio-plausibility" at each operating point |
| **Fitted conditional database** | ❌ Missing | Store fitted scaling laws (acc ~ log FLOPs, memory ~ L·B·D) with uncertainty |
| **Pipeline query engine** | ❌ Missing | Query: "given constraints C, what configs satisfy?" |
| **Autonomous loop** | ❌ Missing | Task → query DB → if gap, launch targeted search → update DB → deploy |

### The two near-term milestones that matter:

**Milestone 1 (4–6 weeks): "Feasible CIFAR-10"**
- Fix CoreTrainer loop overhead (target: ≤2.5× minimal floor).
- Cache CIFAR-10 dataset transforms (like MNIST cache).
- Run *ideal backprop finder* on CIFAR-10 (200 probes) → get backprop Pareto frontier.
- Run `eqprop` + `neural_cube` Bayesian searches (100 probes each) at their own optima.
- Output: first real Pareto comparison on a real vision task, with memory + FLOPs + time.

**Milestone 2 (8–12 weeks): "The Conditional Database"**
- Build the fitted conditional database (scaling law fits with uncertainty).
- Implement the query engine: "given constraints C, what works?"
- Run a second task (e.g., a tabular or RL task) to validate cross-domain conditionals.
- Demo: pipeline receives a synthetic task request, queries DB, launches targeted search if gap, returns a deployable config.

---

## 7. Immediate Next Actions (This Week)

| Action | Owner | Success Criterion |
|--------|-------|-------------------|
| Profile CoreTrainer loop (identify top 3 overheads) | — | `cProfile`/`torch.profiler` output; top 3 >5% each |
| Add `pin_memory=True`, `persistent_workers=True` to DataLoader | — | MNIST epoch time drops from 6.4s → ≤3s |
| Generalize vision dataset cache to CIFAR-10 | — | `get_vision_dataset("cifar10")` returns cached TensorDataset |
| Build `IdealBackpropFinder` class (Bayesian search → cached Pareto) | — | `finder = IdealBackpropFinder("cifar10", 200); finder.frontier` returns Pareto points |
| Add `peak_memory_mb` breakdown (model params / optimizer / activations / grads / eq buffers) | — | Report shows breakdown per probe |
| Define continuous search spaces per rule (backprop, eqprop, neural_cube, FA, etc.) | — | `RULE_SPACES = {"backprop": {...}, "eqprop": {...}, ...}` |
| Add `HyperbandPruner` support to `OptunaBayesProducer` | — | `producer = OptunaBayesProducer(..., pruner=HyperbandPruner())` |

---

## 8. The One Metric That Matters

Stop reporting "accuracy." Report **the Pareto frontier of (accuracy, FLOPs, memory, wall_time, energy) for each rule at its own optimum**.

The single number that summarizes a rule's competitiveness on a task:

> **`cost_of_plausibility(task) = min_{p∈bio_frontier} [ FLOPs(p) / FLOPs(backprop_at_same_acc) ]`**

At a given accuracy level, how many more FLOPs does the bio rule need vs the ideal backprop? If this ratio is ≤1.5 at 95% of backprop accuracy, the rule is **deployment-viable**. If it's 5×, it's a curiosity.

This single number, computed from the two Pareto frontiers, is what the autonomous pipeline uses to decide "deploy or not." Everything else is plumbing.

---

## Appendix: The MNIST Trio Result (For the Record)

| Model | n | Mean Acc | Wall Time (s/ep) | Peak Mem (est) |
|-------|---|----------|------------------|----------------|
| backprop_mlp | 10 | 0.9799 | 6.4 s/ep | ~200 MB |
| neural_cube | 10 | 0.9796 | 21.0 s/ep | ~300 MB |
| eqprop_mlp | 1 | 0.965 | 21.1 s/ep | ~350 MB |

**Interpretation:** On MNIST (real 28×28), `neural_cube` **ties** backprop (it won on digits). eqprop is slightly below. Both bio rules cost 3× the wall-time. The "winner" on digits does not generalize its advantage to real images. This is the kind of negative result that's valuable *if* it's part of a systematic mapping — not as a standalone "experiment."

---

**Bottom line:** Stop running small experiments. Build the infrastructure that lets us ask and answer the *right* questions at scale. The pipeline is the product; experiments are just its data source.

*End of EXPERIMENT_PLAN2.md*