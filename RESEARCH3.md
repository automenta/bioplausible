# RESEARCH3 — Research Items

> **Deliberately unordered.** Each entry is self-contained: question, falsifiable hypothesis, design, controls, targets, statistics/budget, deliverables, stretch goals, substrate, risks. Prioritization and sequencing are separate decisions made once every item is fully specified.

---

## Z3 Fixed Weights (Level 4)

**Question:** *Can frozen $\theta$ solve multiple tasks via $\psi$-mediated rule selection?* No direct analogue exists in the literature (see the ICL-bridge item for the nearest competitor and why it differs). Tests the thesis that elevating the computational rule to a dynamical variable (the M-axis) yields a qualitatively different capability.

**Framing:** the Zuse analogy (`Z3.md`) carries any introduction: fixed relays = $\theta$, punched tape = $\psi$. Switching must emerge from differentiable arithmetic masking,
$$T_t = \sum_k g_k(\psi_t)\, T_k, \qquad g_k(\psi_t) = \mathrm{softmax}(\text{controller}(\psi_t, x_t)),$$
never from weight edits.

**Falsifiable hypothesis:** after meta-training, ψ-only adaptation reaches ≥95% accuracy on all three tasks within ≤20% of the gradient steps fine-tuning needs, at exactly $\Delta\theta = 0$, while fine-tuning loses ≥10 points on previously learned tasks.

**Design (current defaults → upgrades):**
1. Current implementation: `Z3Model` (8 operators, `operator_dim=input_dim=32`, `controller_hidden=128`), meta-train 50 epochs joint over all 3 tasks (Adam lr $10^{-3}$), freeze θ (`requires_grad_(False)`), reset ψ per task (zero controller state + logits), adapt ψ online 20 epochs on fresh batches, eval hard-selection over 20 batches. Upgrade path: sweep `seq_len ∈ {10, 25, 50}`, `input_dim ∈ {32, 64}`, meta-train epochs ∈ {50, 200}; report sensitivity.
2. Tasks stay disjoint-structured: `Parity` (order-$n$ counting mod 2), `LastSymbol` (position-selective readout), `Threshold` (accumulation + comparison). No single operator solves all three; switching is mandatory.
3. Library ablation axis: full 8 operators (`Identity, Threshold, Accumulate, LastSymbol, Parity, SparseTopKRoute, SignFlip, Delay`) vs. minimal 3-operator subset vs. shuffled-operator assignment. Library size vs. adaptation speed is itself a finding.
4. Adaptation schedule study: Gumbel-Softmax temperature annealing (constant 1.0 today) vs. linear decay $1.0 \to 0.1$ vs. cyclic re-annealing per task switch.

**Metrics:** steps-to-criterion (first 100-step window ≥98% accuracy); FLOPs + wall-clock of ψ-only adaptation vs. θ-updates; operator-diversity entropy $H(g_k)$ at convergence (collapse detection: $H < \log 2$ flags single-operator reliance); exact-zero $\Delta\theta$ check ($<10^{-6}$ tolerance, `theta_invariant` flag) — headline integrity claim, reported per seed.

**Baselines:** (a) fine-tune θ, same step budget — measures the forgetting tax; (b) random-ψ init — isolates what meta-training bought; (c) frozen θ + frozen ψ — floor control proving the trunk alone can't solve tasks; (d) **new:** soft-mixture-at-eval (no hard argmax) — quantifies how much performance depends on discretization.

**Statistics:** ≥5 seeds; mean ± bootstrap 95% CI; paired vs. baseline (a) on identical task orderings; thresholds pre-registered before any full run.

**Deliverables:** Fig. 1 accuracy-vs-steps curves (Z3 vs. all baselines); Fig. 2 forgetting matrix (task × time under each method); Table 1 Δθ audit + diversity entropy per seed.

**Stretch:** >3 tasks (add SignFlip-, Delay-, TopK-defined tasks); *compositional* switching (task = ordered pair of operators); Hebbian/local controller updates replacing backprop-through-ψ (fully local Z3 — closes the loop with the locality thesis); measured Joules via `computronium/core/profiling.py` instead of FLOP proxies.

**Substrate:** `computronium/experiments/joint/z3_fixed_weights.py`, `computronium/core/plasticity/rule_state.py`; run via `comp benchmark run --suite z3_fixed_weights --seeds 5 --device cuda`.

**Risks & mitigations:** soft-to-hard selection gap (mitigate with temperature annealing + straight-through estimators); controller collapse onto one operator (entropy monitoring + entropy bonus term as fallback); optimizer state staleness across phases (rebuild Adam between meta-train and ψ-adaptation — current code carries optimizer state over θ's momentum buffers).

---

## Z3 ↔ In-Context Learning Bridge

**Question:** frozen-weights task switching looks superficially like transformer in-context learning, hypernetworks, and visual prompt tuning. Is ψ-mediated rule selection the same phenomenon, a strict generalization, or something else?

**Why it matters:** the flagship Z3 claim ("no analogue in literature") is only credible if the nearest neighbours are named and differentiated experimentally, not rhetorically. This item converts a positioning weakness into a contribution.

**Design:**
1. Implement three comparator mechanisms on identical task suites (the Z3 triple + switching stream): (a) a small transformer conditioned on task demonstrations (ICL), (b) a hypernetwork generating layer weights from a task embedding, (c) prompt-conditioned frozen MLP (prompt tuning).
2. Hold data, budget, and evaluation fixed; vary mechanism only.
3. Measure: adaptation data efficiency (demonstrations needed vs. gradient steps needed), parameter-update requirement (all comparators except ICL require some training; ICL requires none but pays attention-compute per query), OOD task generalization (tasks outside the meta-training distribution).

**Discriminating predictions:** Z3's ψ state persists across queries within a task without re-computation per token (unlike ICL); ψ capacity scales with operator-library size, not sequence length; switching latency is constant rather than growing with context.

**Deliverable:** position table + one head-to-head figure (accuracy vs. adaptation cost, four mechanisms).

**Risks:** transformer comparator needs careful scale-matching or reviewers dismiss it; scope creep — cap at one architecture per comparator family.

---

## Adaptation Efficiency Comparison (Level 1)

**Question:** *does plasticity adapt faster than Null?* The conventional workhorse: Null vs. Routing vs. FastWeight on a switching distribution.

**Hypothesis:** FastWeight cuts post-switch re-adaptation steps by ≥30% vs. Null at equal accuracy; Routing wins when the switch is categorical (identity change) rather than parametric.

**Design:**
1. `create_switching_task` stream with swept switch period ∈ {500, 1000, 2000} steps and task-gap magnitude ∈ {small, large}; these hyperparameters otherwise silently dominate results.
2. Coordinates spanning M ∈ {Null, Routing, FastWeight} at matched trainable-parameter counts (`PlasticityModulatedModel`, `CompositeState`).
3. Epochs-per-phase 50, batch 64, ≥5 seeds (suite reports mean ± std into `adaptation_efficiency_results.json`).

**Metrics:** adaptation half-life per switch (steps to recover 90% of pre-switch accuracy), final accuracy, plasticity primitive parsed from coordinate string, cumulative adaptation compute.

**Deliverables:** Pareto plot (adaptation time vs. compute cost); switch-period sensitivity curve — robustness of the conclusion to the experimental knob nobody reports.

**Stretch:** overlay RuleState as fourth arm once Z3 stabilizes; recast as *anytime adaptation* (accuracy as function of post-switch budget).

**Substrate:** `computronium/experiments/joint/adaptation_efficiency.py`; `comp benchmark run --suite adaptation_efficiency`.

**Risks:** confirmatory rather than novel — its value is as the guaranteed-clean fallback figure and as calibration for L3.5/L4 claims.

---

## Compute Efficiency (Level 2) & Structural Robustness (Level 3)

Two implemented-but-unplanned suites complete the benchmark staircase below Z3; they anchor the resource-vector story empirically.

**L2 — Question:** *does routing reduce effective ops?* Mixture-of-experts synthetic (8 experts, 1 active): dense baseline vs. sparse-routed model. Metrics: active units, gate entropy, effective matmul FLOPs (`compute_efficiency.py`). Hypothesis: routing achieves ≥5× effective-FLOP reduction at <2-point accuracy loss, with gate entropy confirming specialization (not load collapse).

**L3 — Question:** *can the system recover after damage?* Lesion suite: zeroed weights, removed nodes, dead channels, noisy memristive states; compare Null vs. Routing vs. SubstrateCoupled (`structural_robustness.py`). Metrics: accuracy-retention-vs-lesion-fraction curves, recovery steps if plasticity is active during recovery. Hypothesis: SubstrateCoupled degrades most gracefully under memristive noise specifically (its native failure mode), validating the substrate-aware ontology claim.

**Shared design:** matched parameter counts; ≥5 seeds; lesion/damage fractions swept {10, 25, 50, 75}%; identical backbone across arms.

**Deliverables:** two figures slotting directly beneath the Z3 result in the benchmark paper; L2's effective-FLOPs metric feeds the $\mathcal{C}$ vector definition used everywhere else.

**Risks:** both toy tasks have low ceilings; treat as instrumentation-validation layers for the resource vector, not headline results.

---

## Algorithm Migration (Level 3.5)

**Question:** *can ψ switch strategy without θ update?* The direct precursor to Z3: migration between exactly two strategies, A0 (classify by cumulative sum) → A1 (classify by last symbol), measuring time(A0→A1), energy(A0→A1), and enforcing $\|\theta_{after} - \theta_{before}\| = 0$ (`algorithm_migration.py`).

**Role:** Z3's minimal sibling. Runs first-in-catalog-order nowhere — it is simply the cheapest setting in which the ψ-switching machinery can be validated end-to-end, and its two-task version yields closed-form optimal switching policies that Z3's three-task setting does not.

**Hypothesis:** ψ-mediated migration beats re-training-from-A0-init on time-to-A1 by an order of magnitude at zero θ drift.

**Design:** sweep ψ-state dimensionality {32, 128, 512}; measure whether migration time scales with state size (it should, if ψ genuinely encodes strategy) or saturates immediately (suggesting it doesn't — diagnostic value either way).

**Deliverables:** migration-time table; the $\Delta\theta = 0$ audit reused verbatim in the Z3 paper.

---

## AutoScientist M-Axis Ablation Campaign

**Goal:** turn any single manual result into a frontier by sweeping the M-axis with the other five axes pinned.

**Design:**
1. Pin S/G/D/C/U at the flagship coordinate; sweep M ∈ {Null, Routing, FastWeight, RuleState}. One axis at a time — an ablation, not a search.
2. `AutoScientistCampaign.run_iteration` + `CampaignDatabase` + `CampaignCheckpointer` manage iterations, checkpoint/resume; wall-clock capped via campaign config `max_wall_hours`.
3. Per-coordinate `ResourceUsage` records aggregated post-hoc; dominance filtering never runs inline (avoids order-dependence).

**Output:** the Resource-Vector Pareto Frontier over
$$\mathcal{C} = (\text{compute}, \text{memory}, \text{energy}, \text{latency}, \text{plastic-state capacity})$$
(`frontier.py::ResourceUsage`). Deliverable figures: 2-D projections (accuracy-per-Joule, plastic-capacity-vs-forgetting) annotated with which M primitive owns each knee.

**Gate:** the flagship result sits on/near the front across seeds.

**Stretch:** two-axis sweeps (M × CreditAssignment) once single-axis frontier exists; the proposer's `ProposalObjective` non-accuracy ranking (bias-audited in `proposer.py`) can drive energy- or stability-first campaigns — a campaign whose objective is *not accuracy* is itself a demo no competing framework offers.

**Risks:** proxy metrics may not discriminate primitives at small scale (validate proxies against one measured workload first); campaign DB schema churn mid-campaign (freeze before launch).

---

## Runtime Verification Guard & Failure Manifesto

**Goal:** shift verification philosophy from offline "Proof" to online "Monitoring."

**Design:**
1. Elevate `StabilityMonitor` diagnostics — $\rho(J_F)$ via `SpectralRadiusEstimator`, Lyapunov exponents, settling behavior (`computronium/core/stability/`) — into guards inside `AutoScientistCampaign.run_iteration`.
2. Policy: rollout exhibiting $\rho(J_F) > 1.0$ or non-decreasing free energy (EqProp coordinates) → kill pre-budget-burn, append structured record to the failure manifesto (`analysis/failure_manifesto.py`), mutate hyperparameters (contractive rescaling, temperature reset), retry same iteration.
3. Calibration pass: run guard against held-out known-good and known-bad configs; choose thresholds on ROC, not intuition. `_fast_proxy` vs. full-Jacobian estimator disagreement rate reported explicitly.

**Acceptance:** false-kill rate <5% on known-good set; unstable-coordinate kill rate >95%; guard overhead <10% of iteration wall-clock.

**Deliverables:** working guard; manifesto-as-dataset — "where does the joint system go unstable?" is a standalone empirical contribution about the M-axis's stability cost.

**Stretch:** manifest-derived *a priori* instability predictor (classify configs before running) feeding back into the proposer's acceptance sampling.

---

## Continual Learning Proof (Catastrophic Forgetting)

**Problem:** backprop + SGD overwrites old knowledge; replay buffers are the dominant patch.

**Hypothesis (falsifiable):** M-axis decoupling (ψ fast states vs. θ consolidation) matches-or-beats EWC on backward transfer without any replay buffer, on Split-MNIST.

**Design:**
1. Split-MNIST (5 binary tasks); arms: FastWeightPlasticity, ElasticConsolidationUpdate, backprop+SGD control, replay buffer baseline at matched memory. Task-free variant (no task boundaries signaled) as second protocol.
2. Backward transfer matrix after each boundary; memory footprint tracked explicitly (replay pays storage; ψ pays state).
3. Only escalate to Continual RL (context-bandwidth or MazeBase-class) if Split-MNIST separates arms cleanly.

**Synergy:** baseline-(a) forgetting numbers from Z3/adaptation items seed the control arm — reuse, don't rerun.

**Kill criterion:** replay matching ψ-decoupling at equal total memory demotes this to appendix.

**Stretch:** permuted-MNIST stream (50 tasks) to test scaling of ψ retention with task count; interference-vs-capacity curve (at what plastic-state size does forgetting reappear? — this curve *is* the stability-plasticity trade-off made empirical).

---

## Physics-Informed Proof (Strict Conservation Laws)

**Problem:** networks treat conservation laws as soft penalties; violations compound over long horizons.

**Hypothesis:** EqProp coordinates whose Lyapunov function is the system's Hamiltonian conserve invariants to integrator-drift level (<1e-3 relative drift over 10⁴ steps) where PINNs violate at ≥10× that rate, at equal compute.

**Design:**
1. Systems ladder: Heat → Wave → Burgers → Navier-Stokes (2-D periodic). Each adds one conservation law; failures localize cleanly.
2. `EnergyMinimizationDynamics` configured so descent energy ≡ physical Hamiltonian; verify discrete descent property numerically before claiming physics (the Lean-scaffolded statement `E(h_{t+1}) ≤ E(h_t)` becomes an executable check).
3. Comparators at matched compute: PINN (soft penalty), Hamiltonian NN (structure-preserving but backprop-trained), vanilla integrator baseline.

**Metrics:** relative invariant drift per horizon; long-horizon rollout divergence; penalty-overhead (zero by construction for EqProp — report PINN's for contrast).

**Kill criterion:** drift exceeding PINN violation at equal compute demotes to appendix.

**Risks:** constructing exactly-conservative discretizations is its own research subfield (symplectic structure vs. generic Lyapunov descent); PINN baselines are community-tuned — unfair comparisons get caught in review, so publish configs.

---

## Theory Program (Lean + Expressivity)

**Goal:** convert scaffolded statements into checked artifacts, and add the missing Z3 expressivity piece.

**Design:**
1. Complete the existing Lean scaffold (`lean/ComputroniumFormal`): energy decrease for `EnergyMinimizationDynamics.settle` under step-size < 2/L; control-Lyapunov bound $dV/dt \le -kV$ for matched β in the nudged phase. CI integration pending Lean toolchain install (`TODO3` Phase 4.3.4 — the last planned formalization task).
2. **New statement — ψ-selection coverage:** for a finite operator library $\{T_k\}$ and softmax gating, the function class realized by $T_t = \sum_k g_k(\psi_t) T_k$ contains every deterministic selection when $g$ concentrates; prove the approximation-rate statement for Lipschitz controllers. This is the formal kernel of the Z3 claim: fixed hardware, tape-programmable behaviour.
3. Stability-plasticity frontier: state and prove the contraction-vs-plasticity trade-off for composite transition operator $z_{t+1} = F_\theta(z_t; G, S)$ — even a restricted version (RoutingPlasticity preserves spectral radius bounds; FastWeightPlasticity perturbs them by a bounded factor) gives the paper a theorem.

**Deliverables:** machine-checked Lean file in CI; one proposition per item above with Hypothesis property-test counterparts (95%-rigor-at-5%-cost policy from TODO3 stands).

**Risks:** Lean/Mathlib friction is high — hard-stop after the scaffolded statements per existing policy; expressivity statement may reduce to known mixture-of-experts results (check literature before writing; if so, cite and narrow the delta).

---

## De Facto Non-Backprop Benchmark

**Strategy:** own the evaluation of alternatives-to-backpropagation. Fairness-on-equal-footing is the product; `SystemTrainer` is the moat no single-rule codebase has.

**Design:**
1. Freeze evaluation contract *first*: per-rule tuning budget (equal GPU-hours, not equal epochs), early-stopping policy, seeds, data protocols — pre-registered, published before results exist.
2. Inventory: zoo currently ships backprop, eqprop, FA, forward-only (FF), hebbian, predictive coding, target prop, spiking, tile variants, MEP, o1memory — the "20 rules" target is reachable via propagators/optimizers/transitions combinations; register the canonical 20 and lock.
3. Report: capability matrix, accuracy-per-resource overlays, stability audits per rule, failure modes from the manifesto.

**Deliverables:** benchmark paper (*"A Fair, 6-D Evaluation of 20 Local Learning Rules"*), public leaderboard, machine-readable results release.

**Stretch:** external submission pipeline (containerized rule API) converting the benchmark from static paper to living infrastructure — this is what makes it "de facto standard" rather than "one more table".

**Risks:** house-rule bias perception — mitigate via pre-registration + external submissions; maintenance burden of 20 rules under one contract — gate new registrations on CI-green property locks.

---

## Algorithm Discovery (AI for AI)

**Question:** can the AutoScientist invent a (CreditAssignment × ParameterUpdate) combination that beats Adam+backprop on a declared task suite — and transfer beyond it?

**Design:**
1. Search substrate: compatible coordinate pairs under ontology constraints; `ExperimentProposer` generates systematic exploration proposals; `HypothesisReasoner` + `LLMHypothesisGenerator` supply hypothesis chains (`reasoner.py` scaffolding exists).
2. Novelty gate: discovered coordinate must differ structurally (not just hyperparameterially) from registry entries; automated diff against known-rule signatures.
3. Replication gate: winner replicates across ≥5 seeds and ≥2 task families beyond the discovery task; counterfactual analysis (`counterfactual.py`) attributes the win to specific axis changes, not noise.

**Pre-registration:** declare the task suite and baselines *before* discovery runs; otherwise the result is unfalsifiable selection.

**Kill criterion:** wins confined to the discovery task = negative result about search-space design; document, stop.

**Deliverables:** the discovered rule as a standalone artifact (spec + minimal implementation + repro script), plus the search methodology paper.

**Risks:** compute-hungry; LLM proposers anchor on known literature (novelty gate addresses symptom, prompt-diversity addresses cause); reviewer suspicion of overfitting-by-search — replication gates are the defense.

---

## Edge / Green AI Deployment

**Strategy:** own accuracy-per-watt under hard physical ceilings (microcontroller-class memory/compute) where global backward passes cannot exist.

**Design:**
1. Targets from `docs/hardware_targets.md`; export via ONNX/TorchScript/INT8/ternary pipelines (`docs/tutorials/export_*`).
2. Comparison set: quantized MobileNet-class baselines vs. local-rule models exported through the same pipeline (same quantizer, same calibrations — pipeline fairness mirrors training fairness).
3. Energy methodology: measured Joules on at least one physical board if available; otherwise proxy estimates labeled as such, with one measured anchor point for calibration ratio.

**Hypothesis:** local-learning-rule models beat quantized MobileNets by ≥1.3× on accuracy-per-watt at ≤256 KB RAM budgets.

**Deliverables:** deployment artifact suite (flashing-ready builds), accuracy-per-watt frontier chart, honest proxy-vs-measured error bars.

**Stretch:** Loihi 2 / FPGA pilot (see hardware co-design item); sleep-wake duty-cycled inference exploiting EqProp settling as cheap convergence.

**Risks:** physical measurement slow/noisy; without hardware access the claim rests on calibrated proxies — still publishable as methodology, weaker as headline.

---

## Hardware Co-Design Pilot

**Question:** does substrate-aware co-design (choosing the Substrate axis deliberately) beat port-after-training on a real neuromorphic/FPGA target?

**Design:**
1. One coordinate trained *with* memristive/neuromorphic substrate constraints active (IR-drop, spike sparsity) vs. the same architecture trained clean then exported.
2. Deploy both to a single concrete target (Loihi 2 preferred; FPGA via existing tutorial path); measure task accuracy, latency, energy on-device.
3. Report the co-design delta — the number that justifies (or refutes) the entire substrate-aware ontology pitch.

**Gate:** requires one physical target available; pure-simulation versions exist but weaken the point to "expected delta".

**Deliverables:** on-device benchmark table; co-design delta figure; reusable deployment recipe doc.

**Risks:** hardware availability and toolchain friction dominate schedule; scope strictly to one target, one task.

---

## Drop-in PyTorch Wrapper

**Strategy:** remove adoption friction — users swap one line, not their training loop.

**Design:**
1. `torch.nn.ComputroniumLinear` (+ conv/embedding as needed): replaces `nn.Linear` + optimizer with an EqProp or Forward-Forward coordinate; free/nudged phases, settling loops, ψ bookkeeping handled internally; `NullPlasticity`+backprov coordinate falls back to native behavior bit-for-bit.
2. Compatibility targets: DDP wrapping, LR schedulers, `torch.compile` smoke test, torchvision-style model zoo integration example.
3. Acceptance test: training script written by someone unfamiliar with Computronium internals runs unmodified except the swapped line; gradients/accuracy match hand-written loop within noise.

**Deliverables:** pip-installable module, smoke-test suite, one-line-swap README GIF.

**Stretch:** autograd-compatible hybrid mode (some layers Computronium, some native) unlocking incremental adoption in existing codebases.

**Risks:** optimizer/scheduler impedance mismatch; performance parity with hand-written loops must hold or adoption stalls (profile early, not last).

---

## Biological Twin

**Strategy:** 1:1 simulation of a documented microcircuit in the 6-D ontology; predict responses to held-out stimuli/lesions.

**Candidates:** *C. elegans* anterior touch circuit or full somatic nervous system connectome (302 neurons, wiring measured, laser-ablation literature abundant) vs. a named cortical column (higher data quality per neuron, heavier fitting burden).

**Design:**
1. Connectome → Geometry; measured cell physiology (type-specific time constants, thresholds) → StateDynamics/Substrate parameters.
2. Fit remaining free parameters to a *training* split of stimulus-response data; freeze; predict held-out stimulus conditions and ablation effects.
3. Comparators: published circuit-specific models (e.g., NeuroPAL-era dynamical models for *C. elegans*) and a generic RNN fitted to the same data.

**Claim under test:** ontology-native modeling (explicit substrate/state-dynamics separation) predicts lesion responses better than black-box RNNs fit to the same activity data.

**Deliverables:** prediction-vs-measurement tables for held-out perturbations; mapped ontology file released alongside.

**Risks:** highest-risk item — parameter identifiability, data-quality ceilings, domain-expert review barriers; payoff is interdisciplinary credibility, not ML impact. Scope discipline: one circuit, one dataset release, no "whole brain" rhetoric.

---

## Factored Prerequisites

Shared infrastructure extracted from the item catalog. Each is built once; consumers listed per row.

| ID | Prerequisite | Contents | Unblocks |
|----|--------------|----------|----------|
| **PR-0** | **Verification gate** | `docs/baseline.md` gates at-or-better (pytest/pyright strict/ruff) + TIER 0/digits campaign green | Every empirical item — no result trusted otherwise |
| **PR-1** | **Optimizer-phase hygiene** | Rebuild optimizer between meta-train and ψ-adaptation phases; `evaluate_z3` currently carries Adam momentum buffers over frozen θ into adaptation — contaminates the exact-zero Δθ claim | Z3, Algorithm Migration |
| **PR-2** | **θ-invariance audit harness** | Snapshot → freeze → run → re-snapshot → exact-diff, emitted as a reusable context manager with per-seed reports | Z3, Algorithm Migration, continual-learning claims |
| **PR-3** | **Calibrated resource instrumentation** | `ResourceUsage` + `core/profiling.py` wired into every suite runner; ≥1 *measured* Joule/FLOP anchor workload to calibrate proxies; proxy error bars reported thereafter | Z3 energy metrics, L2 effective-FLOPs, AutoScientist frontier, Edge/Green AI, Hardware pilot |
| **PR-4** | **Pre-registration & statistics kit** | Seed count (≥5), bootstrap-CI utility, paired-comparison harness, threshold-registration template checked into repo | Z3, L1–L3.5, benchmark contract, discovery replication gates |
| **PR-5** | **Calibrated stability guard** | ROC-calibrated kill thresholds (<5% false-kill on known-good set, >95% kill rate, <10% overhead); `_fast_proxy` vs. full-Jacobian disagreement rate quantified | Unattended AutoScientist campaigns, discovery |
| **PR-6** | **Evaluation fairness contract** | One pre-registered document: per-rule tuning budgets (GPU-hours, not epochs), early-stopping policy, seeds, data splits — written once, consumed by three different items | Benchmark paper, discovery pre-registration, edge comparisons |
| **PR-7** | **Switching-machinery shakedown** | L3.5 two-task migration + L1 adaptation run as *instrumentation tests* before Z3: validates ψ reset, temperature schedule, diversity entropy, Δθ audit end-to-end on the cheapest settings | Z3 (its minimal sibling de-risks it directly) |
| **PR-8** | **Export pipeline parity** | ONNX/ternary export verified round-trip (accuracy delta ≤ noise) on one representative model | Edge/Green AI, Hardware pilot |

---

## Dependency Graph

```mermaid
flowchart TD
    PR0[PR-0 Verification gate] --> CHEAP[PR-7 Shakedown: L3.5 + L1 + L2 + L3]
    PR0 --> PHYS[Physics proof]
    PR0 --> TWIN[Biological twin]

    PR1[PR-1 Optimizer hygiene] --> PR2[PR-2 θ-audit harness] --> CHEAP
    PR1 --> Z3[Z3 Flagship]
    PR3[PR-3 Resource calibration] --> Z3
    PR3 --> FRONTIER[Pareto frontier campaign]
    PR3 --> EDGE[Edge/Green AI]

    PR4[PR-4 Stats kit] --> CHEAP & Z3

    CHEAP -->|machinery validated| Z3
    CHEAP -->|known-good/bad configs| PR5[PR-5 Guard calibration]
    Z3 -->|flagship result| FRONTIER[AutoScientist M-axis campaign]
    PR5 --> FRONTIER
    FRONTIER --> MANIFESTO[Failure-manifesto dataset]
    MANIFESTO --> DISCOVERY[Algorithm discovery]
    PR6[PR-6 Fairness contract] --> BENCH[20-rules benchmark]
    PR6 --> DISCOVERY
    FRONTIER --> BENCH

    TLEAN[Lean toolchain install] --> PROOFS[Scaffolded proofs + ψ-coverage prop]
    PROOFS -.->|numeric counterparts| PHYS
    Z3 -.->|inform statement scope| PROOFS

    WRAPPER[PyTorch wrapper] -.->|adoption multiplier| BENCH

    PR8[PR-8 Export parity] --> EDGE
    EDGE --> HW[Hardware co-design pilot]
    PROCURE[Hardware procurement ⏳ lead time] --> HW

    CONT[Continual learning]
    CHEAP -->|forgetting baselines reused| CONT
    Z3 -->|baseline-a forgetting| CONT
```

Reading notes: solid arrows are hard dependencies; dashed are informational/informal. `PROCURE` starts immediately regardless of everything else — its lead time, not its difficulty, makes it critical.

---

## Critical Paths

### CP-A — Empirical Spine *(carries most of the catalog)*

`PR-0 → PR-1 → PR-2 → PR-4 → PR-7 (shakedown) → Z3 flagship → PR-5 (guard, overlapping) → AutoScientist frontier campaign → manifesto dataset`

Then fan-out, all gated only on CP-A's tail:
- **Benchmark paper** (needs frontier + PR-6 contract + locked rule registry)
- **Algorithm discovery** (needs campaign infra + manifesto priors + PR-6)
- **Continual learning proof** (needs forgetting baselines from shakedown/Z3 — largely free by then)

This is the longest chain and the one that gates the two highest-leverage strategic outputs. Its single biggest schedule risk is **Z3 non-convergence**; the built-in fallback is structural: if Z3 falsifies, L1's clean adaptation figure substitutes as the campaign seed, CP-A continues degraded-but-intact, and the negative result becomes an M-axis boundary-condition publication.

### CP-B — Verification Spine *(parallel to CP-A)*

`Lean toolchain install → complete scaffolded proofs → ψ-selection coverage proposition (scope refined by early Z3 observations) → numeric counterparts executed inside experimental suites`

Hard-stops at the existing TODO3 policy boundary (no further formalization beyond scaffolded statements). Physics-proof credibility borrows the descent-property checks from here.

### CP-C — Positioning Spine *(parallel, cheap)*

`PR-6 fairness contract draft → PyTorch wrapper v1 → wrapper acceptance test → released alongside first flagship artifact`

The wrapper has no research dependencies — only API stability — so it fills any waiting period on CP-A. Shipping it with the flagship multiplies the flagship's audience.

### CP-D — Physical Spine *(latency-gated, start earliest)*

`hardware procurement (day one) → PR-3 measured-anchor workloads double as board bring-up → PR-8 export parity → Edge/Green AI artifact → co-design pilot`

Everything except procurement is software-side and can begin immediately; the board arrives into a prepared pipeline rather than blocking one.

### CP-E — Independent Tracks

- **Physics proof:** depends only on PR-0 + scientific-domain dynamics; zero coupling to the M-axis storyline.
- **Biological twin:** depends only on ontology + public connectome data; zero coupling to everything above. Pure parallel capacity when CP-A is blocked.

---

## Bottlenecks & Single Points of Failure

| Bottleneck | Gates | Mitigation |
|------------|-------|------------|
| **PR-3 resource calibration** | Z3 energy figures, frontier quality, all edge claims | Calibrate against one measured workload before *any* campaign consumes proxies |
| **Z3 convergence** | Entire CP-A fan-out | Structural fallback to L1 seed (above); shakedown (PR-7) surfaces failure modes cheaply first |
| **PR-5 guard false-positive rate** | Unattended campaigns, discovery throughput | ROC calibration on known-good/bad sets harvested free from PR-7 runs |
| **Hardware lead time** | Co-design pilot only | Procure day one; pilot is deliberately last-in-catalog so slippage costs nothing upstream |
| **Lean/Mathlib friction** | Formal claims only | Hard-stop policy already in place; Hypothesis property tests carry rigor meanwhile |

---

## Coverage Check: does CP-A + satellites reach "most possibilities"?

| Item | Reached via |
|------|-------------|
| Z3 flagship | CP-A core |
| ICL bridge | CP-A core (uses Z3 task suites + comparators) |
| L1, L2, L3, L3.5 | CP-A shakedown stage |
| AutoScientist frontier | CP-A core |
| Runtime guard + manifesto | CP-A core (overlapped) |
| Continual learning | CP-A fan-out (baselines reused) |
| Benchmark paper | CP-A ∩ CP-C |
| Algorithm discovery | CP-A fan-out |
| Theory program | CP-B |
| Physics proof | CP-E (borrows CP-B checks) |
| Wrapper | CP-C |
| Edge/Green AI | CP-D |
| Hardware pilot | CP-D tail |
| Biological twin | CP-E |

All 15 items sit on some path. Only two require resources money can't shortcut (hardware, Lean toolchain) and both are latency-gated rather than effort-gated — hence procured/installed on day one and kept off the spine.
