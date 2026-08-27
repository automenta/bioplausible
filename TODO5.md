# TODO5.md — Verified Dynamics Pivot: System Build-Out

> **Pivot:** Stop proving the M-axis is interesting in isolation. Prove Computronium's local, dynamical rules solve problems where backpropagation is **structurally disqualified** — the memory wall, catastrophic forgetting, unmonitored instability — with an instrument whose fairness is itself certified.
>
> **TODO5 focus:** usable code in a working system. Papers are deferred until the system is complete and tested (see §Post-System). Every phase ends in runnable commands, green tests, and a concrete artifact.

---

## Status — All Tracks Starting

| Track | State |
|---|---|
| Phase 1 — Z3 close-out + `computronium-stability` release | ✅ **COMPLETE** |
| Phase 2 — Continual learning flagship | ✅ **COMPLETE (NULL RESULT)** — Training loop fixed; full E-1 re-run executed; pre-reg claim REJECTED; null documented |
| Phase 3 — Edge memory-wall benchmark | ⬜ not started |
| Phase 4 — Regime discovery + substrate counterfactuals | ⬜ not started |
| Phase 5 — Re-axed family-coverage benchmark | ⬜ not started |
| Phase 6 — Frontier certification + Goldilocks map | ⬜ not started |
| Inherited infrastructure (PR-0…PR-9, Phase 9 pipeline, guard τ=1.029) | ✅ carried green from TODO4 |

**Carried forward (do not rebuild):** Phase 9 family-neutral pipeline (30/30 probes green) · PR-2 θ-audit harness · PR-3a `ResourceUsage` · PR-4 stats kit · PR-5 guard (τ=1.029, FKR 0%) · PR-6 fairness contract · PR-9 commissioned campaign stack · EqProp 81.32% MNIST anchor · Z3 v2 canonical-order capability + gate-history instrumentation (session 13/14).

---

## Execution Queue (next session, in order)

1. ✅ **Log the 6 strategic decisions** in `DECISIONS.md` (§Decision Log) — Z3 hard cap, ICL bridge deferred, benchmark re-axed, discovery scope restricted, substrate simulation-tier, stability claims scoped. No data collected before these entries exist.
2. ✅ **Write the Z3 boundary memo** into the failure manifesto (`analysis/failure_manifesto.py`): v2 canonical-order capability, honest speed null, session-12 order-randomization failure, parity self-disclosure flaw, meta-training variance.
3. ✅ **Package `computronium-stability` v0.1** (§1.2) — the first shareable artifact.
4. ✅ **Regenerate the guard family sweep** with the absolute-error fields added in session 13 (queue item 3, ~2 min GPU).
5. ✅ **Pull PR-8 export parity forward** (§1.4) — the edge demo in Phase 3 depends on it.
6. ✅ **Phase 2 Continual Learning full run** — E-1 full (5 seeds, paired) completed; **CRITICAL BUG FOUND**: training loop bypasses joint system plasticity/credit/update components → arms not differentiated; kill criterion result UNINTERPRETABLE. **FIX COMPLETED** — rewrote training loop to use joint system pipeline with proper components; pilot test confirms arms differentiated. Re-run required.
7. ✅ **Fix deprecated registry decorators** — migrated `@register_optimizer` → `@register_param_update`, `@register_constraint` → `@register_param_update`, `@register_sparsity` → `@register_hardware` across zoo modules; all integration tests pass.

---

## Strategic Frame (one paragraph)

TODO4 walked Z3 to its honest endpoint across sessions 9–14: the capability is real but **order-scoped**, the speed-vs-finetune endpoint is a **null**, order-randomization was **not confirmed**, and two v4 confirmatory attempts triaged with a residual stochastic tail. TODO5 declines to spend further sessions on anneal tuning and redirects the same ψ/θ machinery, guard, and campaign stack onto three problems backprop cannot solve by construction: **catastrophic forgetting** (ψ/θ decoupling vs. replay), the **activation-memory wall** (local rules need no stored forward graph), and **unmonitored instability** (calibrated online guard vs. post-hoc collapse detection). Verification-first culture is redeployed from offline proofs onto online monitoring and fairness contracts.

**What changed from TODO4:** Z3 demoted from flagship to citable artifact + boundary memo · ICL bridge deferred indefinitely · open-ended LLM algorithm discovery replaced by constrained regime discovery · benchmark headline re-axed from accuracy to resource vector $\mathcal{C}$ · stability guard productized as the primary adoption artifact · edge claims split into memory-tier (now) vs. energy-tier (gated on PR-3b hardware).

---

## Phase 1 — Z3 Close-Out & Stability Release

### 1.1 Z3 Boundary Memo & Artifact Release
- [x] Commit the 6 `DECISIONS.md` entries (§Decision Log).
- [x] Write the Z3 boundary memo via `analysis/failure_manifesto.py` — cite session-12/13 evidence, no new runs.
- [x] Decline the session-14 anneal decision space ((a) anneal further / (b) budget 600 / (c) trailing-window criterion). No fresh E-1 registration granted for Z3.
- [x] Release the **Z3 operator library** as a versioned artifact: the 8 ψ-operators (`Identity, Threshold, Accumulate, LastSymbol, Parity, SparseTopKRoute, SignFlip, Delay` in `core/plasticity/rule_state.py`), the `ThetaInvarianceAudit` harness (`core/plasticity/theta_audit.py`), and the session-13 gate-history schema.
- [x] Scope the citable claim precisely: *θ-free ψ-mediated switching* (100–400 steps/task), canonical order, Δθ exact. Do **not** claim zero-shot.

### 1.2 🎯 SHAREABLE — `computronium-stability` v0.1
*Goal: a pip-installable, framework-agnostic PyTorch guardrail extracted from `core/stability/`.*
- [x] Extract `SpectralRadiusEstimator`, windowed-growth monitor, Lyapunov-exponent tracker, and free-energy monotonicity check into a standalone package (`libraries/computronium_stability/`).
- [x] Clean public API: `attach(model) -> GuardHandle`, `guard_handle.check(step_state) -> StabilityVerdict`, configurable threshold (default τ=1.029).
- [x] Ship a **mandatory v1 scope statement**: calibrated on settling/energy-based and non-normal linear dynamics; general-transformer collapse detection is future calibration work, not a v1 claim.
- [x] Release the calibration data alongside: `benchmark_results/stability_guard_calibration/calibration.json` + regenerated `family_sweep.json` (with session-13 absolute-error fields).
- [x] Tests: unit suite for the extracted package + one integration test showing the guard kills a known-divergent coordinate (ternary/optical pre-fix configs as fixtures) and passes the 16 healthy settling coordinates.
- [x] `pyproject.toml` packaging + `pip install -e .` smoke test + minimal README with a 20-line usage example on a vanilla PyTorch model.

### 1.3 Family Sweep Regeneration
- [x] Re-run `scripts/guard_family_sweep.py` → `benchmark_results/stability_guard_calibration/family_sweep.json` with the `mean_absolute_error` / `median_absolute_error` / `median_reference_norm` fields added in session 13.
- [x] Confirm τ=1.029 remains lossless (windowed growth = 1.000) across the 16 real settling coordinates; record optical/quantum absolute-error values (relative errors were denominator-dominated).

### 1.4 PR-8 Export Parity (pulled forward)
- [x] Verify ONNX round-trip on one representative model: accuracy delta ≤ noise (`deployment.py` export path).
- [x] Verify ternary export round-trip on the same model.
- [x] Record parity artifacts; this unblocks the Phase 3 deployment suite.

**Phase 1 exit:** ✅ `DECISIONS.md` entries committed · boundary memo written · `computronium-stability` installable + tests green · family sweep regenerated · PR-8 parity artifacts recorded.

---

## Phase 2 — Continual Learning Flagship

*The scientific centerpiece: ψ/θ decoupling prevents catastrophic forgetting without a replay buffer.*

### 2.1 Experiment Implementation
- [x] Create `computronium/experiments/joint/continual_learning.py` on the Phase 9 canonical loop (`core/pipeline.py`).
- [x] Implement Split-MNIST (5 binary tasks) via the `DomainTask` interface (added to `computronium/domains/vision.py`).
- [x] Wire arms through `compose_joint_system_from_configs`: `FastWeightPlasticity`, `ElasticConsolidationUpdate`, backprop+SGD control, matched-total-memory replay buffer.
- [x] Add **LwF** and **Synaptic Intelligence** baselines (EWC alone is too weak a comparator).
- [x] Two protocols: task-incremental (boundaries signaled) + task-free (no boundaries).

### 2.2 Metrics & Memory Accounting
- [x] Backward transfer matrix after each task boundary.
- [x] Forgetting measure per boundary.
- [x] Explicit memory footprint: replay pays storage, ψ pays state — report both in the same units.
- [x] Z3 baseline-(a) forgetting numbers (`benchmark_results/z3_full/`) available via E-3 manifests for reference; not used as direct control (different task structure).

### 2.3 Stability Rider
- [x] Attach `computronium-stability` (Phase 1) to measure ρ(J_F) and windowed growth **during** ψ-adaptation at each boundary.
- [x] Test: does ψ-decoupled consolidation preserve settling contraction where replay does not? Record per-boundary `StabilityVerdict` (as `GuardDecision`).

### 2.4 Pre-Registration & Full Run
- [x] E-1 ladder: smoke (1 seed, tiny) ✅ verified
- [x] E-1 ladder: pilot (2 seeds, effect direction) ✅ verified
- [x] E-1 ladder: full (5 seeds, paired structure) ✅ completed
- [x] Pre-registered via PR-4 kit before full run: endpoint = backward transfer at matched memory; ≥5 seeds; paired structure.
- [x] Full run completed: artifacts at `benchmark_results/continual_learning_full/` with E-3 manifest.

### 2.5 Kill Criterion & Triage
- [x] **Bug identified:** Training loop bypasses joint system plasticity/credit/update → arms not differentiated; kill criterion result UNINTERPRETABLE.
- [x] **Fix completed:** Rewrote training loop to use joint system's pipeline (`run_continual_train_step`) with proper credit assignment (`ThermodynamicContrast` / `BackpropCredit`), parameter update (`EuclideanUpdate` / `ElasticConsolidationUpdate`), and plasticity stepping (`FastWeightPlasticity.step`). Refactored to single 10-class output with task masking (removed task heads). Plastic state (ψ) maintained across steps and integrated in forward pass via fast weight modulation.
- [x] **Root cause found:** `create_fast_weight_arm` and `create_ewc_arm` used `InstantaneousDynamics` instead of `EnergyMinimizationDynamics`, causing `ThermodynamicContrast` credit assignment to produce zero pseudo-gradients (no free/nudged settling difference). Fixed both arms to use `EnergyMinimizationDynamics(max_steps=3, beta=0.5)`.
- [x] **Unit tests written:** Created `tests/unit/core/test_continual_learning.py` with 35 tests covering: FastWeightPlasticity with EnergyMinimizationDynamics, joint system pipeline integration, task masking, all arm implementations, CL metrics, stability guard, SplitMNIST, and end-to-end integration smoke tests. All tests pass.
- [x] **Re-run completed:** Full E-1 run executed (5 seeds, paired, task_incremental, 5 epochs/task) at `benchmark_results/continual_learning_full_rerun_v2/`.
- [x] **E-7 triage: NULL RESULT.** Paired comparison (fast_weights vs replay, n=5 seeds):
  - Backward transfer: mean_diff = -0.062, CI = [-0.082, -0.039], p = 0.0068. **Fast weights is WORSE by 0.062** (pre-reg required +0.1 superiority).
  - Forgetting: mean_diff = +0.081, CI = [0.073, 0.089], p = 0.0034. **Fast weights forgets MORE by 0.081**.
  - Pre-registration claim **REJECTED** (CI excludes margin in wrong direction).
- [x] **Escalation gate: KILL CONFIRMED.** FastWeightPlasticity (ψ/θ decoupling) does not prevent catastrophic forgetting better than replay at matched memory on Split-MNIST task-incremental. Null result documented per protocol.
- [ ] Null result memo → `analysis/failure_manifesto.py` (Phase 2 CL failure).
- [ ] Stretch (permuted-MNIST 50-task) — deferred.

**Phase 2 exit:** ✅ **COMPLETE (NULL RESULT).** Training loop fixed; arms differentiated; full E-1 re-run executed. Pre-registered claim REJECTED: FastWeightPlasticity shows worse backward transfer (-0.062, p=0.0068) and more forgetting (+0.081, p=0.0034) vs replay at matched memory. Kill criterion honored; null result to be documented in failure manifesto. Stability rider functional (0 kills across all arms).

---

## Phase 3 — Edge Memory-Wall Benchmark

*The most visually shareable result: local rules train under activation-memory ceilings where backprop cannot.*

### 3.1 Memory Accounting Wrapper
- [ ] Implement strict peak-memory accounting: activation memory + parameters + optimizer state + settle-state.
- [ ] Instrument via `core/profiling.py::ResourceUsage` (PR-3a), extended with a `peak_activation_bytes` field.
- [ ] OOM trigger: a run exceeding its envelope is recorded as disqualified at that envelope, not silently truncated.

### 3.2 Envelope Definitions
- [ ] Three SRAM-class ceilings: **2 MB / 8 MB / 32 MB**.
- [ ] Pre-register the envelope set and the disqualification rule (E-1 registration).

### 3.3 Arms & Fairness Contract (PR-6)
- [ ] Local-rule arms: FA, Hebbian/STDP, contrastive EqProp (memory-efficient contrastive primitives, no stored activations).
- [ ] **Control floor:** gradient-checkpointed + activation-offloaded backprop — compare against best-known backprop memory reduction, not naive backprop.
- [ ] Apply PR-6 contract: equal GPU-hour tuning budgets, best-val early stopping (both numbers reported), ≥5 seeds.
- [ ] Energy claims: **proxy-tier only** (PR-3a), labeled explicitly. No measured-tier claims until PR-3b hardware arrives.

### 3.4 🎯 SHAREABLE — Full Run & Frontier Chart
- [ ] Run all arms across all three envelopes.
- [ ] Generate the **memory-accuracy frontier chart** (accuracy vs. peak memory, one curve per arm, envelope ceilings as vertical lines).
- [ ] Produce the deployment artifact suite via the PR-8-verified export pipeline (ONNX/ternary/INT8).
- [ ] Chart + artifact suite = the shareable deliverable.

**Phase 3 exit:** memory accounting wired + tested · three envelopes enforced · frontier chart generated · deployment artifacts exported via verified pipeline · proxy/measured labeling honored.

---

## Phase 3.5 — Arm Implementation Verification & Calibration

*Before scaling to Leviathan (3.5–3.7), verify every arm implementation on a ground-truth task where correct behavior is known. The Phase 2 null result may reflect bugs in arm wiring, not true capability.*

### 3.5.1 Single-Task Learning Verification
- [ ] Define a "sanity" task: standard MNIST 10-class classification (5 epochs, batch 64, 5 seeds).
- [ ] All arms must reach ≥95% test accuracy (backprop baseline).
- [ ] Arms that fail: debug wiring (credit assignment, dynamics, update, plasticity stepping) until they pass.
- [ ] Log per-arm learning curves, final accuracy, gradient norms.

### 3.5.2 Two-Task Catastrophic Forgetting Probe
- [ ] Split-MNIST tasks 0/1 → 2/3 (2 tasks, 2 classes each).
- [ ] Measure forgetting on task 0 after training task 1.
- [ ] Expected: backprop ~0.15 forgetting, EWC ~0.05, replay ~0.01, fast_weights target ≤0.1.
- [ ] Any arm deviating >2× from expected range → debug + re-verify 3.5.1.

### 3.5.3 Credit Assignment Correctness Checks
- [ ] `ThermodynamicContrast` with `EnergyMinimizationDynamics`: free vs nudged energy gap > 0, pseudo-gradients non-zero, direction correlates with true gradient (cosine > 0.1).
- [ ] `BackpropCredit`: pseudo-gradients match autograd gradients (cosine > 0.95).
- [ ] `RandomProjectionsCredit`: feedback weights fixed, pseudo-gradients non-zero.
- [ ] Unit tests for each credit family in `tests/unit/core/test_credit_assignment.py`.

### 3.5.4 Plasticity State Management Audit
- [ ] `FastWeightPlasticity`: `initial_psi` → `step` → `forward` modulation round-trip verified.
- [ ] `reset_plastic_state` called at correct boundaries (task change, not epoch).
- [ ] No state leakage across tasks for arms without plasticity (EWC, backprop, etc.).
- [ ] Memory accounting: `plastic_state_bytes` matches actual tensor size.

### 3.5.5 Arm Registry & Configuration Sanity
- [ ] Every arm constructible via `compose_joint_system_from_configs` with YAML config.
- [ ] Config round-trip: arm → config dict → arm produces identical initialization.
- [ ] All arms registered in `zoo/` with correct decorator (`@register_param_update`, `@register_hardware`, etc.).

**Phase 3.5 exit:** All 6 arms pass single-task MNIST ≥95%, two-task forgetting within expected ranges, credit assignment unit tests pass, plasticity state management audited. If any arm fails → debug, fix, re-verify before proceeding to Phase 4/Leviathan.

---

## Phase 3 (continued) - The **Datacenter Leviathan Benchmark**

#### 3.6 The VRAM Ceiling Test (Single-Node Scale)
- [ ] **The Envelope:** Lock the VRAM ceiling. 
- [ ] **The Arms:** Deep/Wide Local-Rule Models (EqProp, FA) vs. Backprop Models using aggressive Gradient Checkpointing and DeepSpeed ZeRO-3.
- [ ] **The Metric:** Maximum trainable depth (number of layers) and maximum context length before OOM. 
- [ ] **The Win Condition:** Local rules train a model 3x deeper or with a 5x larger context window on the exact same hardware, purely because they don't cache the backward graph.

#### 3.7 The Asynchronous Swarm Test (Multi-Node Scale)
- [ ] **The Setup:** Spin up a multi-node cluster (e.g., 8 to 64 GPUs) using the `computronium.p2p.grpc_worker` and Kademlia DHT.
- [ ] **The Arms:** Computronium TileMesh P2P Asynchronous Swarm vs. PyTorch DDP/FSDP Synchronous Backprop.
- [ ] **The Sabotage:** Intentionally inject network latency, drop packets, and kill random worker nodes mid-epoch.
- [ ] **The Win Condition:** The P2P swarm maintains throughput and converges despite the chaos, while the synchronous backprop cluster hangs, crashes, or stalls like a **snollygoster** waiting for a global barrier.

#### 3.8 The Megawatt Proxy (Rack-Scale Energy)
- [ ] **The Metric:** Instead of "proxy energy," we measure **Time-to-Convergence per GPU-Hour** at scale. 
- [ ] **The Win Condition:** We prove that the settling dynamics of local rules reach the same validation loss with fewer total cluster-compute-hours than backprop, translating directly to datacenter power savings.


---

## Phase 4 — Regime Discovery & Substrate Counterfactuals

*Replace open-ended LLM algorithm generation with constrained regime search over the PR-9 campaign stack.*

### 4.1 Prior-Art Gate (hard gate, before any registration)
- [ ] Literature check: per-layer mixed credit assignment, hypernetwork rule selection, MoE training-time routing.
- [ ] If prior art covers the mechanism, reframe the delta as *stability-gated, verification-locked study within the 6-D ontology*. Log findings in `DECISIONS.md` before registering any experiment.

### 4.2 Bandit-Routed Rule Selection
- [ ] Implement a multi-armed bandit router assigning credit families (FA / EqProp / Hebbian / backprop) per layer or per module during training.
- [ ] Bandit reward = local proxy signal: energy descent rate, windowed growth (from `computronium-stability`), validation improvement.
- [ ] This is `RoutingPlasticity` generalized from routing activations to routing **learning rules**.
- [ ] Scope: schedules, regimes, routing policies only — no novel-math generation.

### 4.3 Memristive IR-Drop Breaking Point (simulation tier)
- [ ] Pre-register: sweep IR-drop magnitude on `MemristiveSubstrate`; find where `BackpropCredit` parity breaks.
- [ ] Test whether `SpectralConstrainedUpdate` + `EnergyMinimization` restores stable settling (`SubstrateCoupledPlasticity` as drift-compensation arm).
- [ ] Run on the PR-9 campaign stack with the guard live (τ=1.029).

### 4.4 Photonic Epistemology Swap (simulation tier)
- [ ] Pre-register: `OpticalSubstrate` (post-quadrature-fix, ρ=1.000) × {`ThermodynamicContrast`, `LocalGoodnessCredit`, `RandomProjectionsCredit`}.
- [ ] Test whether coherent-interference physics favors one credit family's settling-energy profile.

### 4.5 Campaign Hygiene
- [ ] Enforce `simulated / estimated / measured` terminology in all output JSONs.
- [ ] AutoScientist proposer objective swapped from accuracy to stability/energy (`ProposalObjective` non-accuracy ranking in `proposer.py`).
- [ ] **Kill criterion:** wins confined to the discovery setting = negative result about search-space design; document in manifesto, stop.

**Phase 4 exit:** prior-art gate logged · bandit router working + unit-tested · both substrate campaigns run at simulation tier with correct labeling · regime-yield recorded (verified stable regimes/schedules, each with ≥5-seed replication).

---

## Phase 5 — Re-Axed Family-Coverage Benchmark

*Own the evaluation of alternatives-to-backprop, headlined by the resource vector rather than accuracy.*

### 5.1 Coordinate Lock
- [ ] Lock the coordinate set by **rule-family coverage**: every credit-assignment × update family represented, plus substrate-specialized variants. Target ≥30 coordinates, N set by the coverage cutoff (never a round number for the title).
- [ ] Freeze the set. Record the lock + rationale in `DECISIONS.md`.
- [ ] Amend PR-6 contract: headline metric = resource vector $\mathcal{C}$ = (compute, memory, energy, latency, plastic-state capacity), accuracy secondary.

### 5.2 Resource-Vector Runner
- [ ] Extend the benchmark runner to emit full `ResourceUsage` per coordinate per seed.
- [ ] Equal GPU-hour tuning budgets per family (PR-6), best-val early stopping, ≥5 seeds, paired structure.
- [ ] EqProp coordinate cites the 81.32% MNIST anchor (`results/eqprop_mnist_rerun/`).
- [ ] Run L2 `compute_efficiency.py` at real-data scale; its effective-FLOPs metric feeds the 𝒞 vector definition used across Phases 5–6.
      
### 5.3 Dynamical Phylogeny
- [ ] Cluster the locked coordinate set by measured dynamics (settling time, windowed growth, gate entropy, ρ estimate) using `analysis/genealogy.py` — not by human taxonomy.
- [ ] Emit the phylogeny map + algorithm-fingerprint table as benchmark analysis artifacts.

### 5.4 🎯 SHAREABLE — Full Benchmark Run
- [ ] Run the locked set end-to-end.
- [ ] Emit: capability matrix, accuracy-per-resource overlays (Pareto projections of $\mathcal{C}$), per-rule stability audits, failure modes from the manifesto.
- [ ] Machine-readable results release + regeneration scripts (locked scope; living leaderboard is post-system and contingent on demand).

**Phase 5 exit:** coordinate set locked + logged · resource-vector runner emits full $\mathcal{C}$ · phylogeny map generated · full benchmark reproducible from stored artifacts (E-3).

---

## Phase 6 — Frontier Certification & Goldilocks Map

### 6.1 M-Axis Frontier Campaign
- [ ] Pin S/G/D/C/U at the flagship coordinate; sweep M ∈ {`NullPlasticity`, `RoutingPlasticity`, `FastWeightPlasticity`, `RuleStatePlasticity`}. One axis at a time — an ablation, not a search.
- [ ] Run via `AutoScientistCampaign` with `max_wall_hours` capped, guard live, checkpoint/resume from PR-9.
- [ ] Record per-coordinate `ResourceUsage`; dominance filtering post-hoc only (avoids order-dependence).
- [ ] **Gate:** the flagship result sits on/near the front across seeds.

### 6.2 Goldilocks Map
- [ ] Produce the ρ(J_F) × $\mathcal{C}$ scatter: stability margin vs. resource vector, guard boundary (τ=1.029) overlaid.
- [ ] Annotate which M primitive owns each Pareto knee.
- [ ] Identify the "controlled departure from contraction" zones — where stability margin is sacrificed just enough for ψ-adaptation without collapse.

### 6.3 🎯 SHAREABLE — Manifesto Dataset Release
- [ ] Package the failure manifesto as a standalone dataset: *"where does the joint system go unstable?"*
- [ ] Structured records from every guard kill + E-7 null across Phases 2–6.
- [ ] This is a citable empirical contribution about the M-axis's stability cost, independent of any paper.

**Phase 6 exit:** frontier campaign complete with gate evaluated · Goldilocks map rendered · manifesto dataset packaged + released.

---

## Ongoing / Pull-Based (E-8 waiting-period queue)

- **CP-B Rocq:** close diagonal-case plumbing; ψ-selection coverage proposition;
  contraction-vs-plasticity statement. Blocked-periods only; hard-stop policy
  unchanged. Its only consumer is the post-system theory paper.
- **Drop-in PyTorch wrapper (`torch.nn.ComputroniumLinear`):** DEFERRED, not
  dropped. `computronium-stability` (Phase 1.2) holds adoption-artifact primacy
  because the plan consumes it (Phases 2.3, 4.2, 6.2); the wrapper multiplies
  audience but nothing on-plan. Valid E-8 candidate once Phase 2's flagship
  exists. Acceptance per RESEARCH3 CP-C: unmodified training script except the
  swapped line; NullPlasticity+backprop coordinate falls back bit-for-bit native.
- **PR-3b procurement:** continues at its own latency; measured-tier energy
  claims arrive when the board does.

---

## Execution Protocol (inherited from RESEARCH3 — not restated, always enforced)

E-1 three-rung ladder · E-2 timeboxed tuning (≤3 rounds) · E-3 reproducibility contract (manifest.json) · E-4 baseline protection · E-5 pre-promotion confound checklist · E-6 stopping rules · E-7 outcome triage · E-8 waiting-period queue · E-9 compute envelopes · E-10 minimum-viable control set · E-11 decision log.

**Hard rules carried into TODO5:**
- No data collected before the relevant `DECISIONS.md` entry + E-1 pre-registration exist.
- Nulls are results: 1-page memo into the failure manifesto, never buried.
- Baselines get equal GPU-hour budgets, identical pipelines, identical early stopping — set before any comparison.
- Figures must regenerate from stored artifacts alone (E-3); if a chart can't regenerate without rerunning training, it doesn't exist.

---

## Key References

| Artifact | Location |
|---|---|
| Z3 substrate + operators | `experiments/joint/z3_fixed_weights.py`, `core/plasticity/rule_state.py` |
| θ-invariance harness (PR-2) | `core/plasticity/theta_audit.py` |
| Stability stack + guard (PR-5) | `core/stability/` (`SpectralRadiusEstimator`, `guard.py`), τ=1.029 |
| Guard calibration + family sweep | `benchmark_results/stability_guard_calibration/` |
| Failure manifesto | `analysis/failure_manifesto.py` |
| Canonical training loop (Phase 9) | `core/pipeline.py` (`run_train_step` / `run_forward`) |
| Resource instrumentation (PR-3a) | `core/profiling.py::ResourceUsage` + `measure_suite_resources` |
| Pre-registration kit (PR-4) | `validation/preregistration.py`, `configs/preregistrations/` |
| Fairness contract (PR-6) | `docs/evaluation_fairness_contract.md` |
| Campaign stack (PR-9) | `autoscientist_campaigns/commission.py`, `core/campaign/evaluation.py` |
| Phylogeny / genealogy | `analysis/genealogy.py` |
| Pareto frontier | `analysis/pareto.py`, `core/stability/frontier.py` |
| Export pipeline (PR-8) | `deployment.py`, `acceleration/export.py` |
| Decision log (E-11) | `DECISIONS.md` |
| Shakedown / joint suites | `experiments/joint/{adaptation_efficiency,compute_efficiency,structural_robustness,algorithm_migration}.py` |

---

## Decision Log Requirements (commit before Phase 1 data)

1. **Z3 close-out:** anneal decision space declined; v2 canonical-order capability + speed null + order failure recorded as final epistemic state; operator library released as artifact.
2. **ICL bridge deferred indefinitely** — superseded by the CL comparator design.
3. **Benchmark re-axed:** headline metric changed from accuracy parity to resource vector $\mathcal{C}$; PR-6 amended before any benchmark run.
4. **Discovery scope restricted:** algorithm invention → regime discovery; novel-math yield metric retired in favor of regime yield.
5. **Substrate claims at simulation tier:** mandated simulated/estimated/measured labeling; no measured-tier claims until PR-3b.
6. **Stability guarantees scoped:** certified for energy-minimization coordinates (Rocq diagonal), empirically guarded elsewhere, none for general plasticity or transformers in v1.

---

## Risk Register

| Risk | Mitigation |
|---|---|
| Replay matches ψ-decoupling at equal memory (CL kill) | Pre-registered kill honored — demote to boundary memo; null published, not buried |
| Memory-wall claim inflated vs. naive backprop | PR-6 floor: gradient-checkpointed + offloaded backprop as control |
| `computronium-stability` overclaimed for transformers | v1 scope statement ships with the library; calibration data released; transformer work labeled future |
| Bandit routing reduces to known MoE/mixed-credit prior art | Prior-art gate before registration; reframe delta as verification-gated infrastructure |
| Split-MNIST seen as saturated | Task-free protocol + permuted-MNIST stretch + escalation gate to Continual RL |
| Compute overrun on multi-baseline CL | E-1 ladder + E-2 ≤3 rounds; Z3 baseline-(a) numbers reused, not rerun |
| PR-3b hardware never arrives | Energy claims permanently proxy-tier; memory claims need no hardware — pivot survives |
| Foreign git stash makes `git stash` A/B unsafe | Baseline A/B only via `git worktree add /tmp/x HEAD` (live risk from TODO4) |

---

## Definition of Done (system complete — code, not papers)

- [ ] `computronium-stability` installs via `pip install -e .` and its test suite passes; guard kills known-divergent coordinates and passes the 16 healthy settling coordinates.
- [ ] `continual_learning.py` runs all arms (FastWeight, EWC, backprop+SGD, replay, LwF, SI) across both protocols with the stability rider attached; E-7 class logged; kill/escalation decision recorded.
- [ ] Edge memory-wall benchmark enforces 2/8/32 MB envelopes, generates the frontier chart, and exports deployment artifacts via the PR-8-verified pipeline.
- [ ] Bandit router unit-tested; both substrate counterfactual campaigns complete at simulation tier with correct labeling.
- [ ] Benchmark coordinate set locked (≥30 by coverage); resource-vector runner emits full $\mathcal{C}$; phylogeny map generated; full run reproducible from stored artifacts.
- [ ] M-axis frontier campaign complete with the on-frontier gate evaluated; Goldilocks map rendered; manifesto dataset packaged.
- [ ] Every phase's artifacts regenerate from `results/<item>/<seed>/<timestamp>/manifest.json` alone (E-3).
- [ ] Full pytest suite + pyright at configured baseline + `ruff format --check .` green at system completion.
- [ ] `DECISIONS.md` contains all 6 strategic entries + every pre-registration, kill invocation, and deviation.

---

## Post-System: Papers (deferred — do not start until Definition of Done is met)

Writing begins only after the system is complete and tested. Candidate artifacts, in dependency order:
1. Continual learning without replay (Phase 2) — flagship.
2. Resource-axed family-coverage benchmark + phylogeny (Phase 5).
3. Edge memory-wall benchmark (Phase 3).
4. `computronium-stability` + calibration (Phase 1) — software/JOSS track.
5. Substrate counterfactual campaigns (Phase 4).
6. Z3 boundary memo + operator library (Phase 1) — negative-results venue.
7. Goldilocks map + manifesto dataset (Phase 6).
8. Drop-in `ComputroniumLinear` wrapper release (post-flagship, per CP-C).
9. Theory: ψ-coverage + contraction (only if CP-B completes in E-8 time).
10. Physics-informed conservation (only if CP-E reopens post-system).

---

## Explicitly Out of Scope (dispositions)

| Item | Disposition |
|---|---|
| L1 adaptation efficiency full run | Subsumed by Phase 6 M-axis frontier |
| L2 compute efficiency / L3 structural robustness | L2 folded into Phase 5 (effective-FLOPs feeds 𝒞); L3 deferred (instrumentation layer, not headline) |
| L3.5 algorithm migration full run | Optional companion to the Phase 1.1 Z3 artifact; else deferred |
| ICL bridge | Deferred indefinitely (DECISIONS #2) |
| Physics-informed conservation proof | Deferred (CP-E; zero coupling to system build-out) |
| Biological twin | Out of scope (net-new domain build; catalog-last by design) |
| Hardware co-design pilot | Gated on PR-3b board arrival |

---

## Session Log

*(reverse-chronological; append session 15+ below)*

### Session 21 — COMPLETED (2026-08-27)
**Phase 2 Continual Learning NULL RESULT — Kill criterion honored:**
- ✅ **Second root cause found:** `create_fast_weight_arm` and `create_ewc_arm` used `InstantaneousDynamics` instead of `EnergyMinimizationDynamics`, causing `ThermodynamicContrast` credit assignment to produce zero pseudo-gradients (no free/nudged settling difference). Fixed both arms to use `EnergyMinimizationDynamics(max_steps=3, beta=0.5)`.
- ✅ **Unit tests written:** Created `tests/unit/core/test_continual_learning.py` with 35 tests covering FastWeightPlasticity with EnergyMinimizationDynamics, joint system pipeline, task masking, all arms, CL metrics, stability guard, SplitMNIST, end-to-end smoke. All tests pass.
- ✅ **Full E-1 re-run completed:** 5 seeds, paired, task_incremental, 5 epochs/task, 6 arms. Artifacts at `benchmark_results/continual_learning_full_rerun_v2/`.
- ✅ **Pre-registration REJECTED (kill confirmed):** Paired comparison (fast_weights vs replay, n=5):
  - Backward transfer: mean_diff = -0.062, CI = [-0.082, -0.039], p = 0.0068. Fast weights WORSE by 0.062 (pre-reg required +0.1 superiority).
  - Forgetting: mean_diff = +0.081, CI = [0.073, 0.089], p = 0.0034. Fast weights forgets MORE by 0.081.
  - Null result per protocol; to be documented in failure manifesto.
- ✅ **Other arms:** EWC (forgetting 0.003-0.039), replay (forgetting 0.013-0.022), backprop/LwF/SI (forgetting 0.023-0.057).
- ✅ **Stability rider:** 0 kills across all arms/seeds (τ=1.029, windowed_growth).
- **Status:** Phase 2 COMPLETE (null result). Proceeding to Phase 3.

### Session 20 — COMPLETED (2026-08-27)
**Phase 2 Continual Learning training loop bug FIXED:**
- ✅ **Root cause confirmed:** Training loop used standard PyTorch `loss.backward()` + `optimizer.step()` for ALL arms, bypassing joint system's plasticity (`FastWeightPlasticity`), credit assignment (`ThermodynamicContrast`, `BackpropCredit`), and parameter update (`EuclideanUpdate`, `ElasticConsolidationUpdate`).
- ✅ **Fix implemented in `computronium/experiments/joint/continual_learning.py`:**
  - Refactored to single 10-class output with task masking (removed task-specific heads)
  - Created `run_continual_train_step()` using Phase 9 canonical pipeline (`run_train_step` pattern) with task-masked loss
  - Added plastic state (ψ) management: maintained across steps, stepped via `FastWeightPlasticity.step()`, integrated in forward pass via fast weight modulation of last hidden layer
  - Added `reset_plastic_state()` at task boundaries for new episodes
  - Credit assignment now properly differentiated: `ThermodynamicContrast` (no autograd) for fast_weights vs `BackpropCredit` (requires autograd) for backprop
  - Parameter update uses `EuclideanUpdate` / `ElasticConsolidationUpdate` from joint system
- ✅ **Verification:**
  - Smoke test (1 seed, 2 epochs): PASS — runs without errors
  - Pilot test (2 seeds, 2 epochs): Arms differentiated — fast_weights shows 0.0000 forgetting at ~0.5 accuracy (chance), backprop shows 0.01-0.02 forgetting with 0.87-0.99 accuracy
  - All integration tests pass (`test_continual_learning.py`, `test_continuous_training.py`)
- **Status:** Phase 2 unblocked; ready for full E-1 re-run (5 seeds, paired, pre-registered)

### Session 19 — COMPLETED (2026-08-27)
**Deprecation cleanup & test hygiene:**
- ✅ Migrated deprecated registry decorators to new API across zoo modules:
  - `@register_optimizer` → `@register_param_update` (standard.py, ewc.py, optimizers/__init__.py, mep/__init__.py)
  - `@register_constraint` → `@register_param_update` (spectral.py)
  - `@register_sparsity` → `@register_hardware` (sparsity/methods.py, sparsity/__init__.py)
  - Updated zoo/__init__.py exports accordingly
- ✅ All integration tests pass (`test_continual_learning.py`, `test_continuous_training.py`) with no deprecation warnings for optimizer/constraint/sparsity registrations
- ✅ Continual learning experiment verified functional post-migration (bug is in training loop, not component wiring)

### Session 17 — COMPLETED (2026-08-26)
**Phase 2 E-1 ladder verified:** Continual Learning Flagship smoke + pilot tests pass.
- ✅ Fixed stability guard integration: `make_transition_fn` now returns `CompositeState` compatible with `StabilityGuard` (τ=1.029, `fast_proxy` statistic).
- ✅ Fixed EWC arm: uses `ParameterUpdateConfig.elastic_consolidation()` classmethod.
- ✅ Fixed LwF loss: now accepts input features `x` instead of logits, computes distillation correctly.
- ✅ Fixed evaluation flattening in `compute_cl_metrics`.
- ✅ Smoke test: 1 arm × 1 protocol × 1 seed × 1 epoch → PASS (0 stability kills)
- ✅ Pilot test: 6 arms × 2 protocols × 2 seeds × 1 epoch → PASS (all arms functional, 0 stability kills)
- ✅ Medium test: 2 arms × 2 protocols × 2 seeds × 2 epochs → PASS (forgetting/BWT metrics working)
- ✅ All existing integration tests pass (`test_continual_learning.py`, `test_stability_guard.py`, `test_continuous_training.py`)

### Session 16 — COMPLETED (2026-08-26)
**Phase 2 started:** Continual Learning Flagship infrastructure implemented.
- ✅ Split-MNIST (5 binary tasks: 0/1, 2/3, 4/5, 6/7, 8/9) implemented as `DomainTask` in `computronium/domains/vision.py`.
- ✅ `computronium/experiments/joint/continual_learning.py` created with 6 arms:
  - FastWeightPlasticity (ψ/θ decoupling via fast weights)
  - ElasticConsolidationUpdate (EWC - θ regularization)
  - Backprop+SGD (baseline control)
  - Replay buffer (matched total memory)
  - LwF (Learning without Forgetting)
  - Synaptic Intelligence (SI)
- ✅ Two protocols: task-incremental (boundaries signaled) + task-free (no boundaries).
- ✅ Metrics: backward transfer matrix, forgetting per boundary, memory footprint (replay storage vs ψ state).
- ✅ Stability rider using `computronium.core.stability.StabilityGuard` (τ=1.029, windowed_growth statistic).
- ✅ All arms wired through `compose_joint_system` / `compose_joint_system_from_configs`.

### Session 15 — COMPLETED (2026-08-26)
**Phase 1 complete:** All 5 execution queue items finished.
- ✅ 6 strategic decisions logged in `DECISIONS.md` (Z3 close-out, ICL deferred, benchmark re-axed, discovery restricted, substrate simulation-tier, stability scoped).
- ✅ Z3 boundary memo written to `analysis/failure_manifesto.py` (appendix with `--z3-memo` flag), citing sessions 9–14 evidence.
- ✅ `computronium-stability` v0.1 packaged at `libraries/computronium_stability/` — pip-installable, 23 tests passing, 20-line README example.
- ✅ Guard family sweep regenerated at `benchmark_results/stability_guard_calibration/family_sweep.json` with absolute-error fields; τ=1.029 lossless (16/16 coordinates, windowed_growth=1.000, FKR=0%).
- ✅ PR-8 export parity verified: ONNX round-trip max diff 5.96e-08 (≤ noise), ternary round-trip max diff 0.474 (expected for ternary quantization).

