# TODO12.md — Active Plan: The Credit-Channel Repair Program

> **Opened 2026-09-06.** Successor to [TODO11.md](TODO11.md) (R11 core
> complete; D1–D16 + F1–F3 demonstrated; 2026-09-06 fundamental-research
> sprint closed P1a/P3/P1b positive, P2 open-negative, P4/P5 resolved).
> Research catalog: [RESEARCH4.md](RESEARCH4.md).
>
> **Identity (unchanged):** Computronium is an ML library whose every claim
> is a live demonstration. Tests are the evidence system. A claim stands
> only while the current code re-demonstrates it, on demand, in under two
> minutes. Verification is continuous, not archival.
>
> **Prime directive:** *Nothing is claimed that the suite does not re-show at
> HEAD. The demo suite is the proof; the README quotes it; everything else
> is history or hypothesis.*
>
> **State:** The 2026-09-06 sprint's fundamental-research queue is EMPTY of
> one-probe items. The two library pulls it surfaced are the entry points:
> **(A) learned PEPITA feedback projections** (RESEARCH4 Lever 2) and
> **(B) P2's lower-priority cells** (RESEARCH4 Lever 5 — objective
> consistency via contrastive credit + normalized propagation). RESEARCH4's
> Phase 1 (Kill the Optimizer Crutch: `LocalAdamUpdate` on PEPITA/ePC) is
> the recommended first probe — it is a U-axis-only change, cheap, and
> directly tests the unifying hypothesis: *the credit direction is
> approximately right; only the magnitude is broken.*

---

## 🎯 The Unifying Diagnosis (from RESEARCH4)

Every local-algorithm defect measured in the knowledge base is a
manifestation of **one problem: the credit signal loses fidelity as it
propagates through depth, and the loss compounds.** Backprop's "cheat" is
the exact transpose Jacobian, which preserves credit fidelity. Local rules
approximate it, and each approximation leaks information differently:

| Measured defect | Evidence | Credit-channel failure mode |
|---|---|---|
| PEPITA collapse at depth/width | P4/P5: fixed B is directionally random | **Misaligned channel** — credit projected through random basis uncorrelated with feature space |
| ePC geometric decay | F1 audit: ~4×/layer attenuation, exact 0.0 at layer 1 by depth 20 | **Attenuating channel** — credit magnitude shrinks per layer |
| Width fragility | P4: PEPITA/ePC explode/collapse across widths; ff_hybrid robust | **Unnormalized gain** — credit/activity scale compounds ∝ width |
| FF error-blindness | LM audit: pure FF flat at chance; readout_error fixes it | **Disconnected channel** — credit never sees the task loss |
| sPC nudge trapping | F1/D12: hidden credit norms exactly 0.00 | **Blocked channel** — settle geometry traps credit at output |
| P2 frozen-error LM failure | 13 regimes: corrected forward fits, free settle at chance | **Train/inference objective gap** — ε-corrected objective diverges from free-forward CE |
| Naive STDP/Hebbian | F2/R11.3.14: subspace collapse, no task signal | **Absent channel** — correlation only, no task credit |
| Optimizer crutch | P3: ePC gradient 400× too small for Euclid; Muon load-bearing | **Low-rank credit** — optimizer compensates for poor credit quality |

**The fix surface — six levers (RESEARCH4 §2):**

1. **Credit-space normalization** (highest novelty) — orthogonalize/spherically-normalize the credit signal as it propagates down ("Muon applied to the backward signal"). Directly attacks the ePC ~4×/layer geometric decay.
2. **Learned/adaptive feedback projections** — fix PEPITA's core: make B learnable (adaptive FA alignment or PEPITA-as-inference-network).
3. **Structural gain homeostasis** — promote scattered renorms (spectral, unit-RMS, homeostatic, μPC) into an always-on pipeline primitive.
4. **Task-coupled local signals** — generalize `readout_error=True`: give every layer a local signal provably correlated with task loss.
5. **Objective consistency** — fix P2: stay contrastive (epc_thermo×Muon works on LM) but repair signal propagation (levers 1+2), rather than forcing frozen-error.
6. **Optimizer–credit co-design** — design credit to emit well-conditioned pseudo-gradients so the optimizer requirement relaxes toward plain SGD/Adam.

**Research program order (RESEARCH4 §6):**

1. **Phase 1: Kill the Optimizer Crutch** — `LocalAdamUpdate` (per-layer Adam) on PEPITA/ePC width sweeps. If fragility disappears, hypothesis confirmed: direction ≈ right, magnitude broken.
2. **Phase 2: Learn the Feedback** — learned B with autoencoder-style local objective; re-run P4/P5 width sweeps.
3. **Phase 3: Self-Normalizing Credit** — relative error (εᵢ = (nudgedᵢ − freeᵢ) / ‖freeᵢ‖) in ePC; does the stable-width window widen?
4. **Phase 4: Local Target Delivery** — predictive targets for FF: each layer predicts next layer's activity; prediction error = credit signal.
5. **Phase 5: Architecture Co-Design** — residual (landed) + error buses (dedicated channels carrying error alongside forward pass).
6. **Phase 6: Plasticity-Native Learning** — true per-pathway routing (P-axis), fast-weight adaptation, metaplasticity.

---

## 🔬 The Attack Plan — Three Workstreams

The plan executes RESEARCH4's program through three workstreams that can
run in parallel, each producing demo-grade evidence (D-table or F-table
entries) as it goes. Workstreams are **pull-based**: they land only when
a demo, campaign, or research paragraph needs them.

### Workstream A — Credit Normalization & Gain Control (Levers 1, 3, 6)

**Goal:** Make the credit signal non-attenuating and gain-normalized so
local rules work without Muon/OrthoAdam crutches.

| Step | Description | Artifact | Validation |
|---|---|---|---|
| **A1** | `LocalAdamUpdate` (U-axis): per-layer Adam state (m, v) + bias correction; config `ParameterUpdateConfig.local_adam(step_size=, beta2=, eps=)`; dispatch/factory/validate/exports | `computronium/ontology/update.py` + unit lock `test_local_adam_update.py` | Parity with global Adam on single-layer; state-reuse fail-loud |
| **A2** | Probe: PEPITA width sweep (w32/64/128/256) with LocalAdam vs Muon vs Euclid | `scripts/probes/phase1_local_adam_pepita.py` | Does w128/256 collapse disappear? Does w32 explosion disappear? |
| **A3** | Probe: ePC width sweep with LocalAdam | `scripts/probes/phase1_local_adam_epc.py` | Does ePC at w32 stop exploding? Does depth 8+ train without Muon? |
| **A4** | Credit-space normalization (C-axis): add `credit_norm` option to settle/credit path — orthogonalize/spherically-normalize credit per layer as it propagates down (spectral radius → 1) | `LocalGoodnessCredit`, `ThermodynamicContrast`, `RandomProjectionsCredit` modifications | F1 ePC depth audit re-run: does ~4×/layer decay flatten to ~1×? Does credit reach layer 1 at depth 20? |
| **A5** | Structural gain homeostasis (pipeline primitive): per-layer activity-scale normalization as first-class `SettleConfig.gain_control: Literal["none", "unit_rms", "spectral", "homeostatic"]` wired into `StateDynamics.settle` and `CreditAssignment.compute_pseudo_gradient` | `StateDynamicsConfig`, `CreditAssignmentConfig`, demo test | Re-run P4 width-fragility sweep expecting razor-thin window to widen |
| **A6** | Optimizer–credit co-design pass: once credit is well-conditioned, sweep optimizer axis again to see how far we can relax Muon→Adam→SGD | Campaign YAML + demo test | D16 re-run with normalized credit: does SGD work? |

### Workstream B — Learned Feedback & Target Coupling (Levers 2, 4)

**Goal:** Fix PEPITA's directional collapse and FF's error-blindness by making feedback learnable and targets local.

| Step | Description | Artifact | Validation |
|---|---|---|---|
| **B1** | `LearnedFeedbackCredit` (C-axis): `LocalGoodnessCredit` subclass where B is a learnable parameter updated by a secondary local rule. Two variants: (a) **Adaptive FA** — B updated to minimize `cos(B·e, ΔW_estimate)` using local activity; (b) **PEPITA-as-inference-network** — B trained through autograd alongside θ (reconstruction loss: B reconstructs pre-synaptic activity from post-synaptic error). Config: `CreditAssignmentConfig.local_goodness(learned_feedback: Literal["adaptive_fa", "inference_net"], feedback_lr=, feedback_update_every=)` | `computronium/ontology/credit.py` + unit lock | P5/P4 width sweeps re-run: does learned B eliminate bidirectional fragility? |
| **B2** | Autoencoder-style B-training objective: B learns pseudo-inverse of forward weights using only local information (no weight transport). Probe at depths 4/8/16. | `scripts/probes/phase2_learned_feedback.py` | Compare fixed vs learned B at depth 16; does depth-attenuation problem dissolve? |
| **B3** | Predictive targets for FF (C-axis): each layer predicts next layer's activity; prediction error IS the credit signal. New credit type `local_predictive` in `CreditAssignmentConfig`. | `computronium/ontology/credit.py` + demo test | Compare against `ff_hybrid` on MNIST + LM; does it match without global CE? |
| **B4** | Contrastive targets per layer: each layer sees activity under (a) correct class, (b) random/corrupted class; difference = learning signal. FF's original idea applied per-layer. | `computronium/ontology/credit.py` | Does it beat ff_hybrid on LM? |

### Workstream C — Objective Consistency & Architecture (Levers 5, + P2 Resolution)

**Goal:** Resolve P2's open-negative by staying contrastive (the working PC-family LM instrument) and repairing signal propagation, plus architecture co-design.

| Step | Description | Artifact | Validation |
|---|---|---|---|
| **C1** | Re-test P2 with contrastive credit (epc_thermo) + credit normalization (A4) + learned feedback (B1) on LM — the objective-consistency fix. | `scripts/probes/phase5_contrastive_lm.py` | Does ePC×thermo×normalized-credit train LM at depth? |
| **C2** | Error buses (G-axis): dedicated channels carrying error signals alongside forward pass. Each layer reads from error bus, writes to it. Error bus updated by local accumulation of prediction errors. | `GeometryConfig.error_bus: bool`, `ErrorBusGeometry` wrapper | Does depth wall dissolve at 20+ for all credit rules? |
| **C3** | Normalized architectures (G-axis): LayerNorm/RMSNorm/weight normalization built into geometry. | `GeometryConfig.normalization: Literal["none", "layernorm", "rmsnorm", "weight_norm"]` | Does width fragility disappear at architecture level? |
| **C4** | Sparse/dynamic architectures (G-axis + P-axis): only subset of neurons active; credit assignment easier. Connects to routing plasticity. | `GeometryConfig.sparsity`, `RoutingPlasticity` integration | Fault tolerance + energy measurement |

---

## 📋 Immediate Next Actions (Ordered by Risk & Leverage)

### 1. Phase 1 Probe: `LocalAdamUpdate` + PEPITA/ePC Width Sweeps (Week 1)

**Why first:** U-axis only, no new credit rules, no new architectures. Cheap,
grounded in our two biggest wins (μPC normalizes init space, OrthoAdam
normalizes momentum space), directly tests the unifying hypothesis.

**Implementation:**
- Add `LocalAdamUpdate` to `computronium/ontology/update.py` (per-layer
  Adam state; config `ParameterUpdateConfig.local_adam()`).
- Wire dispatch in `factory.py`/`spec.py`/`joint.py`; `SystemConfig.validate()`
  whitelist; root + ontology exports; CLI listings.
- Unit lock: `tests/unit/core/test_local_adam_update.py` (per-layer state
  isolation, bias correction, distinct from global Adam, state-reuse
  fail-loud).
- Probe: `scripts/probes/phase1_local_adam.py` — PEPITA/ePC width sweeps
  (w32/64/128/256, depth 4, seeds 0–2, mnist quick) with LocalAdam vs
  Muon vs Euclid.

**Success criteria:** PEPITA at w128 trains; ePC at w32 does not explode.
If both hold → optimizer crutch is a scale problem, not direction problem.

### 2. Phase 2 Pull: Learned Feedback Projections for PEPITA (Week 2)

**Why second:** Single biggest bottleneck for PEPITA. If B tracks forward
weights, depth-attenuation problem solves for entire FA/PEPITA family.

**Implementation:**
- `LearnedFeedbackCredit` in `credit.py` — B as learnable parameter with
  secondary local update rule (adaptive FA alignment or inference-net
  reconstruction).
- Config gating: `local_objective="pepita_learned"` or new
  `learned_feedback` field.
- Probe: `scripts/probes/phase2_learned_feedback.py` — fixed vs learned B
  at depths 4/8/16, capacity-matched.

**Success criteria:** Learned B eliminates PEPITA's bidirectional width
fragility; depth 16 trains without Muon crutch.

### 3. Phase 3 Probe: Credit-Space Normalization on ePC (Week 2–3)

**Why third:** Highest novelty lever. "Muon applied to the backward
signal." Directly attacks ePC's ~4×/layer geometric decay measured in F1
audit.

**Implementation:**
- Add `credit_norm: Literal["none", "orthogonal", "spherical", "rms"]` to
  `CreditAssignmentConfig` and `StateDynamicsConfig`.
- In settle/credit path: as credit propagates from layer ℓ to ℓ−1,
  orthogonalize/spherically-normalize it (spectral radius → 1).
- Probe: `scripts/probes/phase3_credit_norm_epc.py` — F1 ePC depth audit
  re-run with credit_norm on.

**Success criteria:** Credit reaches layer 1 at depth 20 with non-vanishing
norm (~1×/layer instead of ~4×/layer). If validated, whole agenda
confirmed.

### 4. P2 Resolution via Contrastive Path (Parallel, Week 2+)

**Why:** P2's open-negative is a high-value finding if resolved. The
working PC-family LM instrument is epc_thermo×Muon (contrastive), NOT the
jpc frozen-error gradient. The fix: stay contrastive + repair signal
propagation (levers 1+2).

**Implementation:**
- Re-run P2 probe (`scripts/probes/p2_jpc_lm.py`) with epc_thermo×Muon +
  credit_norm (A4) + learned_feedback (B1) at registered width (w816×7).
- If contrastive path trains LM at depth, P2's "frozen-error fails" is
  resolved as "wrong objective for this task family" and the contrastive
  path becomes the PC-family LM demo (D17 candidate).

---

## 🎯 Demo/Capability Targets (D-table / F-table Extensions)

Each workstream produces live demonstrations. Target capabilities:

| Target | Description | Workstream | Demo Test |
|---|---|---|---|
| **D17** | PC-family LM: epc_thermo×Muon (contrastive) trains transformer at 1h scale | C | `test_demo_pc_lm_contrastive.py` |
| **D18** | LocalAdam eliminates optimizer crutch: PEPITA/ePC train at depth/width without Muon | A | `test_demo_local_adam_crutch_kill.py` |
| **D19** | Learned feedback: PEPITA with learned B matches BP at depth 16, width-matched | B | `test_demo_learned_feedback_pepita.py` |
| **D20** | Credit normalization: ePC credit reaches layer 1 at depth 20 (norm > 0) | A | `test_demo_credit_norm_epc_depth.py` |
| **D21** | Predictive targets: FF with local predictive credit trains LM without global CE | B | `test_demo_predictive_targets_ff.py` |
| **F4** | Credit-channel failure map: all 8 failure modes + their fixes in one figure | A/B/C | `test_demo_credit_channel_fixes.py` |

---

## 🧪 Probe-First Discipline (Standing)

1. **Throwaway probes first** (`scripts/probes/phase*.py`) — measure
   levers before touching tests or library code.
2. **Promote to demo test only if** the finding holds at probe scale
   (variance-aware asserts, multi-seed where claimed).
3. **Gallery lock + RESULTS.md paragraph** on promotion.
4. **Drift immunity:** re-pin manifest → green lock ×2 consecutive runs
   before declaring landing.
5. **Walltime discipline:** demo suite stays on CPU; GPU only for
   registered-scale campaigns (measured, not assumed).

---

## 🔧 Library Wiring Checklist (Per Primitive Pull)

When adding a new ontology primitive (credit, update, dynamics, geometry,
plasticity, substrate), the **single-source checklist** (from AGENTS.md):

1. **Registry row** — add class to layer's registry (e.g.,
   `DYNAMICS_REGISTRY` in `dynamics/__init__.py` for StateDynamics).
2. **Config classmethod** — `ConfigClass.primitive()` returns config whose
   `*_type` matches registry key.
3. **Wiring lockstep lock** — `tests/property/test_dynamics_wiring_lock.py`
   (or equivalent per axis) proves registry ↔ config classmethods ↔ root
   `__all__`/`_LAZY` ↔ root `TYPE_CHECKING` imports stay in sync.
4. **Export surfaces** — root `__all__` + `_LAZY` + `TYPE_CHECKING` import
   block; `ontology/__init__.py` imports + `__all__`.
5. **Contract invariants** — read the Protocol docstring (activation
   layout, settle mutation/autograd contract, free/nudged semantics).
6. **Validation** — if `SystemConfig.validate()` needs new compatibility
   branch, whitelist consistently in *all* credit/substrate branches.
7. **Demo (if shipping)** — static `_ARMS` table pattern; `_train_arm`/
   `_probe_arm` extracted; walltime printed never recorded; one `DEMOS`
   registry row in `visualization/gallery.py`; re-pin
   `docs/figures/manifest.json` via gallery lock.
8. **Probe conventions** — throwaway probes in `scripts/probes/` with
   measured-regime numbers and docstring citing the demo they informed.

---

## 📦 Standing Directives (Carried from TODO11, Binding)

- **`benchmark_results/` stays untracked and gitignored — never re-add it.**
- **README: never edit it.** Evidence lives in `docs/RESULTS.md` and the gallery.
- **Test-execution discipline:** never run tests without showing output and walltime.
- **Lint/type debt is deprioritized:** ruff clean passively; pyright only on genuinely new modules.
- **Device policy:** demo suite on CPU (kernel-launch-bound); GPU for registered-scale campaigns.
- **DataLoader workers:** `num_workers=2` faster at demo scale; `0` is flake mitigation.
- **GitHub CI not in use:** local gates are the acceptance criteria.

---

## 🎯 The Next-Session Plan (Ordered)

1. **Phase 1 probe** — `LocalAdamUpdate` + PEPITA/ePC width sweeps (throwaway probe → promote if holds).
2. **Phase 2 pull** — `LearnedFeedbackCredit` for PEPITA (library change + probe).
3. **Phase 3 probe** — credit-space normalization on ePC (F1 depth audit re-run).
4. **P2 contrastive re-test** — epc_thermo×Muon + credit_norm + learned_feedback on LM.
5. **Workstream synthesis** — combine A4 + B1 + C1 into the "normalized contrastive local rule" demo (D17/D18/D19/D20).

**Gate after each step:** probe output + walltime visible → ruff format/check on changed files → pyright on new modules → targeted tests for touched modules → gallery lock green if demo promoted.

---

## 🧭 Open Questions & Adaptive Branches

The plan is open to discovery. Key decision points:

| Question | If YES | If NO |
|---|---|---|
| Does LocalAdam kill the optimizer crutch (PEPITA w128 trains, ePC w32 bounded)? | Accelerate A5/A6 (gain homeostasis + co-design); Phase 2/3 become amplification, not rescue | Phase 2 (learned feedback) becomes primary; Phase 3 (credit norm) becomes essential |
| Does credit-space normalization flatten ePC's ~4×/layer decay? | The unifying hypothesis validated; all levers compose toward parity | Re-examine: is attenuation in settle geometry, not credit propagation? |
| Does learned B eliminate PEPITA's width fragility? | PEPITA becomes competitive local rule; D13 upgraded | PEPITA stays slow arm; focus shifts to ff_hybrid + ePC as local-credit carriers |
| Does contrastive + normalized credit train LM at depth (P2 resolution)? | PC-family LM demo lands (D17); jpc frozen-error retired as wrong objective for LM | P2 stays open-negative; contrastive path documented as the working PC-family LM instrument |

---

## 📜 Register C Hygiene Pass (Carried, Deferred)

The Register C hygiene pass (R11.2) remains **deferred** — it lands only
when a demo, campaign, or research paragraph needs it. Current clean
state: ruff clean, property suite 679 passed, pyright clean on new
modules. The hygiene pass is a separate workstream, not a blocker for
research.

---

## 🏁 Completion Criteria for TODO12

TODO12 closes when:

1. **Phase 1 validated or falsified** — LocalAdam probe result recorded
   (positive or negative) with demo-grade evidence.
2. **At least two of {Phase 2, Phase 3, P2 contrastive} produce demo-grade
   findings** — promoted to D-table/F-table with gallery locks.
3. **The credit-channel failure map (F4) is live** — all 8 failure modes
   + their fixes demonstrated in one figure, mechanism-audit ratchets locked.
4. **The optimizer crutch is either killed or its boundary mapped** —
   clear statement of which local rules need which optimizer, and why.

The plan is **adaptive**: each probe result re-orders the remaining work.
The unifying hypothesis (credit-channel fidelity) is the compass; the
demo suite is the proof.