# TODO12.md — Active Plan: The Credit-Channel Repair Program

> **Opened 2026-09-06 (rev 2, code-alignment review).** Successor to
> [TODO11.md](TODO11.md) (R11 core complete; D1–D16 + F1–F3 demonstrated;
> 2026-09-06 sprint closed P1a/P3/P1b positive, P2 open-negative, P4/P5
> resolved). Research catalog: [RESEARCH4.md](RESEARCH4.md).
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
> **State:** The sprint's fundamental-research queue is EMPTY of one-probe
> items. The two library pulls it surfaced — learned PEPITA feedback
> (RESEARCH4 Lever 2) and P2's untried cells (Lever 5) — are entry points.
> RESEARCH4's Phase 1 (Kill the Optimizer Crutch) is the recommended first
> probe: cheap, U-axis-only, and a direct test of the unifying hypothesis —
> *the credit direction is approximately right; only the magnitude is
> broken.* Rev 2 aligns the plan to verified code seams (snapshot capture,
> credit dispatch surfaces, the legacy `AdaptiveFA` rule, the P4/F1 probe
> instruments) and folds in every carried TODO11 obligation.

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
| FF error-blindness | LM audit: pure FF flat at chance; `readout_error` fixes it | **Disconnected channel** — credit never sees the task loss |
| sPC nudge trapping | F1/D12: hidden credit norms exactly 0.00 | **Blocked channel** — settle geometry traps credit at output |
| P2 frozen-error LM failure | 13 regimes: corrected forward fits, free settle at chance | **Train/inference objective gap** — ε-corrected objective diverges from free-forward CE |
| Naive STDP/Hebbian | F2/R11.3.14: subspace collapse, no task signal | **Absent channel** — correlation only, no task credit |
| Optimizer crutch | P3: ePC gradient 400× too small for Euclid; Muon load-bearing | **Low-rank credit** — optimizer compensates for poor credit quality |

**The fix surface — six levers (RESEARCH4 §2):**

1. **Credit-space normalization** (highest novelty) — orthogonalize/spherically-normalize the credit signal as it propagates down ("Muon applied to the backward signal"). Directly attacks ePC's ~4×/layer decay.
2. **Learned/adaptive feedback projections** — fix PEPITA's core: make B learnable (transport-free objectives only).
3. **Structural gain homeostasis** — promote scattered renorms (spectral, unit-RMS, homeostatic, μPC) into an always-on pipeline primitive.
4. **Task-coupled local signals** — generalize `readout_error=True`: give every layer a local signal correlated with task loss (incl. reward-modulated STDP).
5. **Objective consistency** — fix P2: stay contrastive (epc_thermo×Muon works on LM) and repair propagation (levers 1+2), rather than forcing frozen-error.
6. **Optimizer–credit co-design** — design credit to emit well-conditioned pseudo-gradients so the optimizer requirement relaxes toward plain SGD/Adam.

**Research program order (RESEARCH4 §6):** Phase 1 kill the optimizer
crutch → Phase 2 learn the feedback → Phase 3 self-normalizing credit →
Phase 4 local target delivery → Phase 5 architecture co-design → Phase 6
plasticity-native learning. **Honest bar: near-parity per regime first**
(depth, width, task family), with the energy/plasticity/distributed
benefits as the differentiator.

### Coverage Map — every RESEARCH4 idea has a plan slot

| RESEARCH4 item | Plan slot |
|---|---|
| Lever 1 credit-space normalization | **A4** |
| Lever 2 learned/adaptive feedback | **B0–B2** |
| Lever 3 structural gain homeostasis | **A5** |
| Lever 4 task-coupled local signals | **B3–B5** (`readout_error` landed) |
| Lever 5 objective consistency | **C0–C1** |
| Lever 6 optimizer–credit co-design | **A0/A1/A6** |
| Fix 1 options: reconstruction / alignment / slow co-adaptation | **B1** (all three objectives) |
| Fix 2 options: relative error / per-layer β / spectral norm | **A4** (modes) + **A5** |
| Fix 3 options: predictive / contrastive / temporal targets | **B3 / B4 / B6** |
| Fix 4 options: sign / per-layer Adam / spectral step | **A1** ladder (+`SpectralConstrainedUpdate` exists) |
| Fix 5: residual (landed) / error buses / normalized archs / sparse | **C2–C4** |
| Fix 6: fast-weights / routing (realized, F3) / metaplasticity | **D1–D3** |
| Unifying-hypothesis decisive test | **A0/A1** (magnitude-vs-direction ladder) |
| Energy/plasticity/distributed payoff (the "why") | **Capstone** (resource-vector accounting) |

---

## 📜 Carried Queue from TODO11 (binding — none of these may silently drop)

| Item | Status | Lands as |
|---|---|---|
| **LM ladder runs, 10–20 min/arm** (user directive: gradual budget; candidate subset {transformer/bp/adam, transformer/bp/muon, transformer/ff_hybrid/muon, mlp/ff_hybrid/muon, mlp/epc_thermo/muon, mlp/pepita/muon}) | QUEUED, user-gated | **D17** (the D-number TODO11 reserved) |
| **P2 untried cells**: exact jpc inference-network ε (free-vs-nudged difference), γ (activity step) grid, width 512, inference steps > H | OPEN | C1 |
| **TransformerGeometry × ePC settle path** (block-structured settle with error variables; TransformerGeometry is bias-free — the ePC settle kernel needs a block extension) | PREREQ for transformer PC-family cells | C1 prereq |
| **Registered-scale P-axis campaign** with the matched-effective-lr protocol pinned in the manifest (Path B full) | QUEUED | D3 |
| **Reward-modulated STDP** (F2's OPEN verdict: "the STDP fixed point destroys class structure" — the supervised-error-term audit is the remaining gap) | OPEN | **B5** |
| **NaturalGradientUpdate**: rename or implement diag-Fisher before any natural-gradient mechanism claim (as-touch) | OPEN | A0/A6 touch |
| **Snapshot state generalization** (found during this review): `TrainerSnapshot` (`core/system_trainer/trainer.py:182`) captures only `update._momentum_buffers` — state capture works for the momentum-buffer family (Euclidean, Riemannian/Muon) but **`AdamUpdate`/`OrthoAdamUpdate` `_m`/`_v`/`_t` and any credit-internal state are silently dropped on resume**; the R11.2.24 bitwise-resume lock passed because its fixtures used buffer-family updates | DISCOVERED | A1/B1 prereq |
| **Persistent-ψ across batches** (`train_step` contract change; ψ currently re-initializes per episode — the F3 scope-honest boundary) | CONDITIONAL — only if a registered fast-weight memory claim (D1/D3) needs multi-batch episodes | D1/D3 enabler |

---

## 🔬 The Attack Plan — Four Workstreams

Each produces demo-grade evidence (D/F-table entries) as it goes and is
**pull-based**: it lands when a demo, campaign, or research paragraph
needs it. Probe IDs are workstream-step (`a0_…`, `b1_…`); sessions, not
calendar weeks.

### Workstream A — Magnitude & Credit-Channel Normalization (Levers 1, 3, 6; Phases 1 & 3)

**Goal:** Make the credit signal non-attenuating and gain-normalized so
local rules work without the Muon/OrthoAdam crutch.

| Step | Description | Artifact | Validation |
|---|---|---|---|
| **A0** | **Zero-new-primitive probe.** `NaturalGradientUpdate` is already a per-tensor mean-\|grad\| normalizer (effective step = step_size — the D16 lesson). Re-run the fragile cells (pepita w32/w128, epc w32) from `scripts/probes/p4_width_fragility.py` with a natural-gradient arm + per-cell lr micro-sweep (its lr IS its step size). Also rename-or-diag-Fisher follow-up rides here (as-touch) | Extend `p4_width_fragility.py` with an `--update` axis (harness already carries per-credit LR table + per-layer activity-std instrumentation) | If magnitude normalization alone widens the stable band, the hypothesis is already half-confirmed with zero library code |
| **A1** | **The ablation ladder primitive** — separate magnitude from direction. Ladder rungs, all momentum-EMA based: (i) `UnitRMS`-normalized momentum (m̂ → unit-RMS per tensor, **no orthogonalization**) — the decisive rung; (ii) `LocalAdam` per-tensor scalar second moment (LAMB-style: u = m̂/√(mean v̂)) — RESEARCH4's literal ask; (iii) optional sign-momentum (1-bit hardware rung). Config classmethods on `ParameterUpdateConfig`; dispatch in factory/spec/joint; `SystemConfig.validate()` whitelist; root+ontology exports; CLI listings | `computronium/ontology/update.py` + `tests/unit/core/test_update_ladder.py` (per-tensor normalization identity, distinct-from-Muon, distinct-from-Adam, state-reuse fail-loud) | **Pre-registered predictions (RESEARCH4):** UnitRMS ≈ Muon on PEPITA w128 / ePC w32 ⇒ magnitude is the whole story; UnitRMS < Muon ⇒ orthogonalization carries direction signal beyond scale. Either terminus is a finding |
| **A2** | PEPITA width sweep (w32/64/128/256, depth 4, seeds 0–2) × {Euclid, UnitRMS, LocalAdam, Muon} on both instruments: the P4 LM harness and the D13 MNIST regime | `scripts/probes/a2_ladder_pepita.py` | Does w128/256 collapse disappear? Does w32 explosion disappear? |
| **A3** | ePC width + depth sweep with the ladder | `scripts/probes/a3_ladder_epc.py` | Does ePC at w32 stop exploding? Does depth 8+ train without Muon? |
| **A4** | **Credit-space normalization (C-axis).** Seam: the layer error tensor at pseudo-gradient formation. Modes on `CreditAssignmentConfig`: `credit_norm: Literal["none","relative","rms","beta_adaptive","spectral"]` — `relative`: εᵢ/(‖freeᵢ‖) (RESEARCH4 Fix-2 option 1); `rms`: per-layer unit-RMS ε; `beta_adaptive`: per-layer βᵢ tuned to hold the error signal at unit scale (Fix-2 option 2 — the ePC-native version, since ÷β is the cap in question); `spectral`: spectral-radius→1 rescale of the propagated error (option 3). Applies to `ThermodynamicContrast` (ePC's settled εᵢ before εᵢᵀaᵢ₋₁) and the FA/PEPITA per-hop propagated error. **Never normalizes toward fabricated signal** (zeros stay zeros) | `computronium/ontology/credit.py` + unit lock; **gates the full property suite** (credit-semantics change, TODO11 precedent) | Re-run the F1 instrument (`scripts/probes/f1_epc_depth.py` — already measures per-layer credit norms): does ~4×/layer flatten to ~1×, and does ePC **learn at depth 8–20 under the simple F1 regime** (D14's faithful regime already trains depth 20 — the sharper test is the simple one)? |
| **A5** | **Structural gain homeostasis as a pipeline primitive.** `StateDynamicsConfig.gain_control: Literal["none","unit_rms","spectral"]` wired into the settle path — start on **instantaneous + ePC only**: energy-family settles carry Lyapunov/energy locks (L4) whose landscapes a settle-time renorm would change; an energy_minimization compatibility branch is its own audited pull | `StateDynamicsConfig`, settle kernel, demo test; property suite gate | P4 sweep re-run expecting the razor-thin stable-width window to widen; DeepHebbianChain (R11.3.14) is the existence proof the recipe works |
| **A6** | **Co-design pass:** once credit is well-conditioned (A4) + magnitude-controlled (A1), sweep the optimizer axis again — how far does Muon relax toward Adam/SGD? Re-pin D16 with normalized-credit columns; fold in the natural-gradient rename/diag-Fisher resolution | Campaign YAML + D16 extension | The crutch map: which local rules still need which optimizer, with matched-step controls (P3 protocol) |

### Workstream B — Learned Feedback & Task Coupling (Levers 2, 4; Phases 2 & 4)

**Goal:** Fix PEPITA's directional collapse and error-blindness by making
feedback learnable and targets local — **transport-free objectives only**
(the L3 weight-transport freeness lock is the guard: ‖B − Wᵀ‖ stays > 1e-3,
separate storage).

| Step | Description | Artifact | Validation |
|---|---|---|---|
| **B0** | **Audit the legacy seam first.** `computronium/core/local_learning/rules/fa.py:148` has `AdaptiveFA` (Akrout et al. 2019) — but its `_update_feedback_weights` pulls `fb` toward `param.data` (or `param.data.T`): **it reads forward weights, i.e. soft weight transport**; its bio-alignment property test sits xfail'd (`tests/property/biology/test_biology_axioms.py:365`, "feedback LR too small to show alignment in 50 steps"). Extract what's reusable (slow feedback timescale, alignment metric) and record the transport verdict | Probe note + `b0` docstring citing this file | The ontology port must NOT inherit the transport; the xfail stays xfail until a transport-free rule passes it |
| **B1** | **`LearnedFeedbackCredit` (C-axis).** B becomes credit-internal state updated by a secondary local rule, three transport-free objectives (RESEARCH4 Fix 1): (a) **reconstruction** — B trained to map post-synaptic error back to pre-synaptic activity (autoencoder-style local objective); (b) **update-direction alignment** — minimize angle between B·e and the realized ΔW direction (self-supervised from the update itself, no W read); (c) **slow co-adaptation** — B on a slower timescale tracking the changing Jacobian through local signals only. Config: `CreditAssignmentConfig.local_goodness(learned_feedback=…, feedback_lr=, feedback_update_every=)` | `computronium/ontology/credit.py` + unit lock. Integration requirements: system-scoped state (AdamUpdate precedent) + fail-loud reuse; **TrainerSnapshot must capture credit state** (see carried queue — generalize snapshot to a per-axis state protocol); deterministic under seed (R11.2.24 bitwise resume); campaign `_CREDIT_FACTORIES`/`_CREDIT_ALIASES` dispatch (`core/campaign/evaluation.py:290`); full property-suite gate (credit semantics) | P4/P5 sweeps re-run: does learned B eliminate bidirectional width fragility? Does the D13 `ff − pepita > 0.2` ratchet gap close? |
| **B2** | Fixed-vs-learned B at depths 4/8/16, capacity-matched, MNIST + LM cells | `scripts/probes/b2_learned_feedback_depth.py` | Does the depth-attenuation problem dissolve for the FA/PEPITA family? |
| **B3** | **Predictive targets for FF (C-axis):** each layer predicts the next layer's activity; the prediction error IS the credit signal (predictive coding's error, FF's architecture). New credit type `local_predictive` — full registry wiring per the checklist below. Design note: `TargetInversionCredit` currently propagates targets through `Wᵀ` (**weight transport by construction**) — the honest variant propagates targets through learned B (compose with B1) | `computronium/ontology/credit.py` + demo test | vs `ff_hybrid` on MNIST + LM: does it match without global CE? |
| **B4** | **Per-layer contrastive targets:** each layer sees its activity under (a) correct-class input, (b) corrupted-class input; the activity difference is the layer-local signal (FF's original idea, made per-layer) | `computronium/ontology/credit.py` | Does it beat ff_hybrid on LM (where pure FF's global-goodness fails)? |
| **B5** | **Reward-modulated STDP** — the supervised error term `TemporalTraceCredit` lacks by construction (F2: "declares `phases=(FREE,)` and never consumes `loss`"). Add a config-gated reward/error term on the timing-STDP path; closes F2's OPEN verdict either way | `credit.py` + F2 test re-audit | Collapse stops + readout ≥ random-init ⇒ F2 was a missing term; persists ⇒ a verified constraint of timing-STDP, honestly closed |
| **B6** | *(optional, last)* **Temporal targets:** use the settling trajectory (not just the final equilibrium) as credit — EqProp-adjacent; only if B3–B5 leave the target-delivery question open | probe only | — |

### Workstream C — Objective Consistency & Architecture (Lever 5; Phase 5)

**Goal:** Resolve P2 the honest way (contrastive + repaired propagation),
and co-design the architecture for local learning.

| Step | Description | Artifact | Validation |
|---|---|---|---|
| **C0** | **Bank the LM baselines first** (carried queue): the 10–20-min ladder arms at verified regimes with matched-step protocol riding along; curves + ppl table → **D17** per the static-arms convention. The repair program then measures lift-over-*this* baseline, not over 2.5-min smoke | `scripts/probes/lm_comparison.py --minutes 15 --arms …` → D17 demo | User-gated budget; findings stabilize before any 60-min run |
| **C1** | **P2 resolution via the contrastive path:** epc_thermo×Muon already trains LM at registered width (train 2.81, val ~21 vs bp 9.9) — the open question is closing the gap to parity. Cells: (a) contrastive credit + A4 credit_norm + B1 learned feedback at w816×7; (b) P2's untried cells verbatim (inference-network ε, γ grid, width 512, steps > H); (c) prereq: the **TransformerGeometry × ePC settle extension** (block-structured error routing) before any transformer PC-family cell | `scripts/probes/c1_contrastive_lm.py` | Contrastive+repairs reach bp-parity or a mapped boundary; frozen-error retired as wrong-objective-for-LM if the untried cells also fail |
| **C2** | **Error buses (G-axis):** dedicated channels carrying error alongside the forward pass; each layer reads/writes the bus; bus updated by local accumulation of prediction errors. `GeometryConfig.error_bus` + geometry support; residual (`GeometryConfig.residual`, landed) composes | geometry + settle/credit support + demo | Does the depth wall dissolve at depth 20+ for ALL credit rules (RESEARCH4 Phase-5 experiment)? |
| **C3** | **Normalized architectures (G-axis):** LayerNorm/RMSNorm/weight-norm as geometry config — the scale problem fixed at the architecture level rather than the optimizer level | `GeometryConfig.normalization` | Width fragility disappears architecturally? (Cross-check A5: pipeline-level vs architecture-level gain control) |
| **C4** | **Sparse/dynamic architectures (G×P):** activity sparsity makes credit assignment easier (fewer active weights); connects to routing plasticity and the compute-efficiency benchmark | `GeometryConfig` + `RoutingPlasticity` integration | Fault tolerance + compute measurement per the benchmark suite |

### Workstream D — Plasticity-Native Learning (Phase 6)

**Goal:** The timescale-separation payoff. Note: RESEARCH4's prerequisite
("realize true per-pathway routing") is **already DONE** — F3's realization
landed per-gate drive + per-unit masks (2026-09-05). What remains:

| Step | Description | Artifact | Validation |
|---|---|---|---|
| **D1** | **ψ-only adaptation:** freeze θ entirely; can routing/fast-weight ψ solve the A→B distribution switch with zero θ updates? Ties to the `adaptation_efficiency` benchmark suite; the Z3 frozen-θ machinery (D5) is the precedent | probe → benchmark run | Adaptation time + energy vs θ-updating controls; parameter invariance exact (‖θ_after − θ_before‖ = 0) |
| **D2** | **Metaplasticity:** the learning rate itself is a learned local quantity (confident layers update slowly, uncertain quickly) — local gain control on the update timescale; per-layer adaptive lr as a U-axis/P-axis hybrid | probe | Does it beat fixed-lr on the fragile cells without global lr tuning? |
| **D3** | **Registered-scale P-axis campaign** (carried deliverable): the manifest pins the matched-effective-lr protocol per arm (the F3 lesson — retention at nominal lr is confounded by construction); GPU per doctrine | campaign YAML + run | The first finding-grade P-axis claim |

---

## 🏆 Capstone — The Payoff, Made Measurable

RESEARCH4's payoff table (energy, plasticity, dynamic networks, distributed
operation, substrate compatibility) becomes a **resource-vector accounting**
demo once A/B/C land:

- **Memory:** no stored-activation backward sweep — the D4 memory profiler
  (`test_demo_memory_budget.py`) already measures saved bytes; extend to the
  repaired local rules.
- **Energy:** simulated-energy methodology (the substrate models' stated
  terms) for local-update-only vs full backprop at matched task accuracy.
- **Compute/latency:** per-episode cost at matched accuracy (the F3
  discipline: walltime never enters records; measured proxies only).
- **Substrate compatibility:** the repaired channel under
  Memristive/Neuromorphic substrates (D6's five arms) — local
  self-consistency is what runs without global clocks.

Deliverable: one campaign FrontierRecord table over
𝒞 = (compute, memory, energy, latency, plastic-state capacity), repaired
local rules vs backprop at matched accuracy — the "why it matters"
paragraph with numbers.

---

## 🎯 Demo/Capability Targets

| Target | Description | Workstream | Demo Test |
|---|---|---|---|
| **D17** | LM ladder result at 10–20-min arms (carried TODO11 numbering — first claim on the number) | C0 | per static-arms convention |
| **D18** | The optimizer crutch killed or mapped: PEPITA/ePC train at fragile cells under ladder updates; magnitude-vs-direction verdict locked | A1–A3 | `test_demo_update_ladder.py` |
| **D19** | Credit-space normalization: ePC learns at depth 8–20 under the simple regime; per-layer decay ~1× | A4 | `test_demo_credit_norm_epc_depth.py` |
| **D20** | Learned feedback: PEPITA with learned B competitive at depth 16, width-matched | B1–B2 | `test_demo_learned_feedback_pepita.py` |
| **D21** | Predictive/contrastive local targets train without global CE | B3/B4 | `test_demo_local_targets.py` |
| **F2-close** | Reward-modulated STDP verdict (either terminus) | B5 | re-audit inside `test_demo_spiking_plateau.py` |
| **D22** | ψ-only adaptation (θ frozen) solves the switch | D1 | `test_demo_psi_only_adaptation.py` |
| **F4** | The credit-channel failure map: all 8 failure modes + their repairs in one figure, mechanism ratchets locked | A/B/C | `test_demo_credit_channel_map.py` |

---

## 🧪 Probe-First Discipline (Standing)

1. **Throwaway probes first** (`scripts/probes/<workstream><step>_<topic>.py`)
   — measure levers before touching tests or library code; **reuse existing
   instruments before writing new ones** (P4 harness for width sweeps, F1
   harness for depth/credit-norm audits, D13 regime for MNIST iteration,
   `lm_comparison.py` for LM arms).
2. **Pre-register the prediction** in the probe docstring before running
   (RESEARCH4's predictions are falsifiable — record them verbatim).
3. **Promote to demo only if** the finding holds at probe scale
   (variance-aware asserts, multi-seed where claimed).
4. **Gallery lock + RESULTS.md paragraph** on promotion; drift immunity =
   re-pin → green lock ×2 consecutive runs.
5. **Audit-of-the-audit (R11.5.5a) applies to our own landings** — F3's
   premature attribution is the standing caution: lr-matched controls
   (P3 protocol) before any mechanism claim; walltime never enters records.
6. **Gates per credit/dynamics-semantics change:** full property suite +
   wiring lockstep locks, not just targeted tests.

---

## 🔧 Library Wiring Checklist (Per Primitive Pull — verified surfaces)

1. **Registry/dispatch row** — dynamics: `DYNAMICS_REGISTRY`
   (`ontology/dynamics/__init__.py`); updates: factory/spec/joint dispatch
   (`_UPDATE_CLASSES` map); **credits: `_CREDIT_FACTORIES` +
   `_CREDIT_ALIASES` + (if contrastive) `_CONTRASTIVE_CREDITS` in
   `core/campaign/evaluation.py:290`, plus the axis-probe copy
   (`tests/unit/core/test_axis_probe.py:82`) and
   `_CREDIT_TYPE_ALIASES` (`cli/commands/train.py:24`)**.
2. **Config classmethod** — `ConfigClass.<primitive>()` whose `*_type`
   matches the dispatch key.
3. **Wiring lockstep lock** — the per-axis property lock proves registry ↔
   config classmethods ↔ root `__all__`/`_LAZY` ↔ `TYPE_CHECKING` imports
   stay in sync; fix what it flags, never bypass.
4. **Export surfaces** — root `__all__` + `_LAZY` + `TYPE_CHECKING` block;
   `ontology/__init__.py`.
5. **Contract invariants** — read the Protocol docstrings first (activation
   layout, settle mutation/autograd contract, free/nudged semantics,
   phases/requires_autograd declarations).
6. **Validation** — `SystemConfig.validate()` whitelists the new type in
   *all* relevant credit/substrate branches consistently (R5.2 retro).
7. **Snapshot/resume** — if the primitive holds state (optimizer moments,
   feedback matrices), the TrainerSnapshot capture must include it;
   bitwise-resume lock extended to cover the new state (see carried queue).
8. **Demo (if shipping)** — static-arms table, `_train_arm`/`_probe_arm`
   extracted, walltime printed never recorded, one `DEMOS` row in
   `visualization/gallery.py`, `docs/figures/manifest.json` re-pinned via
   the gallery lock.
9. **Probe conventions** — throwaway probes in `scripts/probes/` with
   measured-regime numbers and a docstring citing the demo they informed.

---

## 📦 Standing Directives (Carried from TODO11, Binding)

- **`benchmark_results/` stays untracked and gitignored — never re-add it.**
- **README: never edit it.** Evidence lives in `docs/RESULTS.md` and the gallery.
- **Test-execution discipline:** never run tests without output + walltime visible; measure levers in throwaway scripts before touching tests.
- **Lint/type debt deprioritized:** ruff clean passively; pyright on genuinely new modules only.
- **Device policy:** demo suite on CPU (measured: kernel-launch-bound); GPU where FLOP-bound (registered-scale campaigns, large widths).
- **DataLoader workers:** `num_workers=2` at demo scale; `0` is flake mitigation.
- **GitHub CI not in use:** local gates are the acceptance criteria.
- **Runs budget:** gradual scaling — next trial budget ~10–20 min/arm; 60-min runs only as findings stabilize (user directive 2026-09-06).
- **Demo API publishability roadmap deferred — research first** (user directive 2026-09-05). This plan is research-scoped accordingly; the roadmap resumes after the repair program.

---

## 🎯 The Next-Session Plan (Ordered)

1. **A0 probe** — natural-gradient magnitude-normalization arm on the P4 fragile cells (zero new primitives; extends `p4_width_fragility.py`).
2. **A1** — the ablation ladder primitive (`UnitRMS` + `LocalAdam`) per the wiring checklist; snapshot-state generalization lands with it.
3. **A2/A3 probes** — PEPITA/ePC sweeps across the ladder on both instruments; verdict against the pre-registered predictions.
4. **B0 audit → B1 pull** — legacy-AdaptiveFA audit, then `LearnedFeedbackCredit` (reconstruction objective first — cheapest local objective).
5. **C0** — the carried LM ladder runs (user-gated) bank D17 baselines while A/B probes iterate.
6. **A4 probe** — credit_norm on the F1 depth instrument (the decisive novelty lever).

**Gate after each step:** probe output + walltime visible → ruff
format/check on changed files → pyright on new modules → targeted tests →
full property suite when credit/dynamics semantics change → gallery lock
green if a demo was promoted.

---

## 🧭 Open Questions & Adaptive Branches

| Question | If YES | If NO |
|---|---|---|
| Does magnitude normalization alone rescue the fragile cells (A0/A1: UnitRMS ≈ Muon)? | The crutch is scale; A4/A5 become the depth levers; parity-with-cheap-optimizer is in reach | Direction (orthogonalization) carries signal; the crutch map (A6) documents *why* per rule — RESEARCH4's prediction falsified is still a finding |
| Does credit_norm flatten ePC's ~4×/layer decay (A4)? | The unifying hypothesis validated at the credit channel; levers compose toward parity | The attenuation lives in settle geometry, not credit propagation → C2 (error buses) is promoted |
| Does learned B eliminate PEPITA's width fragility (B1)? | PEPITA becomes a competitive local rule; D13 upgraded; B3 composes on top | PEPITA stays the slow arm; ff_hybrid + ePC carry the local-credit story (they already do on LM) |
| Do the P2 untried cells + contrastive repairs close LM (C1)? | PC-family LM demo lands; the objective-consistency lever is validated | The boundary is mapped honestly: contrastive+repairs is the working instrument, frozen-error retired with evidence |
| Does ψ-only adaptation solve the switch (D1)? | The plasticity payoff is demonstrated, not claimed | The ψ-timescale boundary is mapped; metaplasticity (D2) is the next lever |

---

## 🏁 Completion Criteria for TODO12

1. **Phase 1 verdict locked** — the magnitude-vs-direction ladder measured on
   PEPITA/ePC fragile cells; pre-registered predictions confirmed or
   falsified with demo-grade evidence (D18).
2. **At least two of {A4 credit-norm, B1 learned feedback, C1 contrastive-LM}
   produce demo-grade findings** — D-table entries with gallery locks.
3. **The credit-channel failure map (F4) is live** — all 8 failure modes +
   their repairs demonstrated in one figure, mechanism-audit ratchets locked.
4. **The optimizer crutch killed or its boundary mapped** — the A6 map states
   which local rules need which optimizer at matched effective step, and why.
5. **The carried TODO11 queue is either landed or explicitly re-dated** —
   D17, P2 cells, transformer-ePC prereq, P-axis campaign, F2 verdict.
6. **The capstone resource-vector table exists** — energy/memory/compute for
   the repaired local channel vs backprop at matched accuracy.

The plan is **adaptive**: each probe result re-orders the remaining work.
The unifying hypothesis (credit-channel fidelity) is the compass; the demo
suite is the proof.