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
> **State (2026-09-06, rev 13 — D22 honest-miss; F5 landed as pinned
> miss; ⚠️ CLAIM A SCOPE AUDIT: ff_hybrid's autograd chain IS the
> learning signal (output-pseudo-loss backprop) — canonical locality
> wording holds verbatim only for requires_autograd=False credits +
> PEPITA-style routing; per-layer contrastive targets (B3/B4) promoted
> to the top of the lever queue; D17 PAUSED after seed 0):** F5 (`test_demo_resource_vector.py`,
> gallery-locked ×2) executed the pre-registered schema with measured
> bytes/FLOPs: both physical-advantage targets FALSIFIED at HEAD
> (ff_hybrid autograd realization stores ≥ backprop, costs ~1.3× FLOPs;
> thermo's exactly-0 proves the O(1) class is real) — Claim A stands,
> Claim B awaits a non-autograd local-rule realization. Rev-13 items: (1) Highlight-Reel Step 3
> (D22 ψ-only adaptation) CLOSED as a mapped boundary — both
> pre-registered predictions falsified in `scripts/probes/d22_psi_only.py`:
> no landed ψ law consumes a task-loss signal, so ψ-only adaptation cannot
> acquire Task B with θ bitwise frozen; demo withheld per probe-first
> rule 3; revival condition = a supervised ψ-law primitive,
> pre-registered. (2) F5's resource-accounting schema WRITTEN before any
> measurement (`scripts/probes/f5_resource_vector.py` docstring — memory
> leads, energy footnoted, OrthoAdam SVD charged honestly, both variants
> reported). (3) D17 (Step 1) run started and **PAUSED at seed 0** (user
> budget directive 2026-09-06: no multi-hour runs — time/electricity).
> Seed-0 salvaged arms (`benchmark_results/d17_seed0.json`): 
> **transformer/ff_hybrid/muon val_ppl 5.05 (2,200 steps) vs
> transformer/bp/adam 27.55 (15,217 steps)** — ff_hybrid far AHEAD at
> 15 min, single-seed, protocol INCOMPLETE (seeds 1–2 not run; verdict
> band NOT adjudicable). Also fixed en route: `_val_sets` mixed-ctx
> defect (both families' windows were cut at max(ctx) — fine at smoke
> scale, crashed the registered mlp arm's eval; per-family ctx now).
> **Standing lesson: smoke-verify every registered arm end-to-end
> (1-min arms) before launching minute-scale arms.**
>
> **State (2026-09-06, rev 12 — pre-registration tightening + suite green):**
> external review applied. (1) D17's target is now a fixed **verdict band**,
> not a point (<15% headline parity / 15–25% competitive-with-caveat /
> \>25% mechanical MISS → C1 fallback); mean-over-seeds-0–2 is the reported
> number, full curve is the artifact. (2) The canonical locality claim is
> fixed verbatim ("forward-local credit with a single readout supervision
> term — no backward sweep through the hidden layers"); credit-locality and
> physical-advantage claims ride separate lines; F5 reports BOTH the
> plain-update and OrthoAdam variants. (3) F5's resource-accounting schema
> must be written before any measurement; memory leads (depth-scaling,
> near-definitionally true), energy is the carefully-footnoted secondary.
> (4) D22 gains lr/effective-step-matched controls + a θ-fine-tuning
> comparison on the same switch. (5) **The suite is green by construction:
> 1772 passed / 0 failed** — all legacy red disposed (see rev-12 record
> below). (6) The two-minute identity clause is scoped to the fast demo
> tier; registered-scale claims re-show via the `slow` tier. D17's
> pre-registration is IN the probe docstring (`lm_comparison.py`) — the
> run is turnkey.
>
> **State (2026-09-06, rev 11 — HIGHLIGHT-REEL PIVOT, user directive):**
> the repair program has its winners; show them off. Three active steps,
> in order: **(1) D17** — the LM ladder, now UN-GATED: Transformer ×
> ff_hybrid (star local rule) vs Transformer × backprop at 10–20-min
> arms; **(2) the capstone resource-vector accounting demo** on the
> Step-1 winner (memory/energy Pareto vs backprop at matched accuracy);
> **(3) ψ-only adaptation** — θ frozen, Task B learned via routing ψ
> only (D22). **PARKED:** PEPITA (five-cause audit chain complete,
> realization retired for the local-credit story), Z3-terminology
> defense (the demo stands on its own: frozen θ is asserted exactly),
> deep-Hebbian / naive-STDP rescue work. Pre-registered targets below
> are *targets, not claims* — measured numbers only (prime directive
> unchanged). The parked items satisfy the carried-queue rule by
> explicit re-dating, not silent drop.
>
> **State (2026-09-06, rev 10):** F4 LANDED (`test_demo_credit_channel_map.py`,
> gallery lock green ×2, full property suite 1685 passed / 3 pre-existing
> baseline fails). PEPITA readout rung FALSIFIED — bounding hidden gain AND
> unit-normalizing the output-weight step still diverges (output act_std up
> to 7e10): the runaway lives in the weight trajectory itself, not per-step
> direction shape; five causes now ruled out (feedback_scale, centered-e1,
> row space, hidden gain, output step shape). `NaturalGradientUpdate`
> RENAMED `MeanNormUpdate`/`mean_norm` (it is a mean-|grad| normalizer, not
> Fisher; carried-queue item closed; fake `"fisher"` alias dropped). D16
> RE-PINNED with a matched-step unit_rms column — **unit_rms is
> regime-shaped**: crutch-killer on LM width (D18), chance-level on MNIST
> quick at every lr (RMS normalization holds step magnitude fixed near the
> loss floor → convergence noise floor). Remaining: C0 (user-gated D17),
> optional sign-momentum rung, B2–B6, C1, capstone.
> **State (2026-09-06, rev 9):** A5-depth rung FALSIFIED (gain_control
> gives no depth lift; credit norms still explode — the depth wall is
> credit-side, confirmed from the activity side). **A6 co-design map
> ASSEMBLED** (see Workstream A table): crutch dead for ePC-width and
> outright in the faithful regime; Muon's residual value is depth-only;
> PEPITA diverges under every landed lever (readout-path suspect).
> Remaining: C0 (user-gated D17), D16 re-pin, F4 figure; optional
> PEPITA readout rung, sign-momentum rung.
> **State (2026-09-06, rev 8):** A5 LANDED — `gain_control` on
> `StateDynamicsConfig` (hidden-layer renorm at settle emit, instantaneous
> + ePC). Probe verdict: gain_control bounds hidden acts by construction
> but PEPITA still diverges — **the runaway reroutes through the
> unnormalized readout** (output act_std up to 7e10, val_ppl saturated
> 4.85e8 at every lr incl. 1e-5). With A2+B1+A5 combined, ALL of
> {feedback_scale, centered-e1, fixed-B row space, hidden gain} are
> ruled out as the PEPITA driver; the remaining suspect is readout-path
> divergence (persistent saturated-e1 update direction on the output
> weights). PEPITA stays the slow arm — ff_hybrid + ePC carry the
> local-credit story. B1 landed rev 7 (learned feedback, transport-free,
> snapshot-captured credit state; row-space prediction falsified).
> Remaining: C0 (user-gated D17), A6 map, F4 figure; optional
> ePC-depth×gain_control rung.
> **State (2026-09-06, rev 7):** B1 LANDED — `learned_feedback` on
> `local_goodness` (transport-free reconstruction B, closed-form ridge,
> EMA, snapshot-captured credit state). Probe verdict: learned B does
> NOT stop the PEPITA runaway (act_std still ~20×/layer under unit_rms)
> — the fixed-B row space is exonerated; the unbounded activity loop is
> the driver and **A5 (settle-path gain homeostasis) is the indicated
> PEPITA repair**. A0–A4 + composition closed (rev 5/6). Remaining:
> A5, C0 (user-gated D17), A6 map, F4 figure.
> **State (2026-09-06, rev 6):** B0 DONE — legacy AdaptiveFA's
> alignment is proven soft weight transport (frozen-W smoking gun);
> B1's spec is written (reconstruction objective, L3-guarded, snapshot-
> protocol state). A0–A4 + composition all closed (rev 5 below). Next:
> implement B1, then C0 (user-gated D17), A6 map, F4 figure.
> **State (2026-09-06, rev 5):** A0–A4 landed AND the A4×D14
> composition test is CLOSED (faithful regime self-sufficient — SGD
> 0.528 at depth 20 — credit_norm harmful there; ε is dynamics in the
> reparameterized channel). The credit-channel picture is now
> three-regime: simple regime needs credit_norm (depth) + unit_rms
> (width); PEPITA needs B1 (row space); faithful regime needs nothing
> new. Remaining: B1, C0 (user-gated D17), A6 map assembly, F4 figure.
> **State (2026-09-06, rev 4):** A0–A4 all landed. D18 demo pinned
> (crutch dead for ePC at w32–64); A3 mapped the depth wall as
> credit-side; A4 landed credit_norm (5 modes) with the mechanism
> verified (spectral: norms ~1.0 flat through depth 16; depth-8 lift
> 0.195 vs 0.113 matched) — the D14 faithful-regime composition is the
> one decisive test left on the unifying hypothesis. B1 (learned
> feedback — PEPITA's repair) and C0 (D17 LM baselines, user-gated)
> follow. A0 DONE, A1 LANDED, A2 COMPLETE with
> defect audit, **D18 DEMO LANDED** (`test_demo_update_ladder.py`,
> gallery lock green ×2, full property suite 679 passed). The pinned
> record IS the verdict: ePC w64 unit_rms 32.5 vs muon 101.2; ePC w32
> 42.5 vs muon 191.7 (multi-seed, 600 fixed steps, deterministic);
> PEPITA control explodes as audited. **The optimizer crutch is dead
> for ePC at w32–64.** Next: A3 (depth under unit_rms) → A4 (credit_norm)
> → B1 (learned feedback — PEPITA's indicated repair).

---

## 🎬 The Highlight Reel Program (rev 11 — ACTIVE)

> User directive 2026-09-06: stop writing the mechanic's logbook; win the
> race. Ignore the broken algorithms and showcase the high performers —
> **ff_hybrid**, **OrthoAdam**, and **ePC** — as end-to-end, slide-grade
> results. Everything below is probe-first (pre-register the target in
> the probe docstring, measure, then promote), and every "juicy" number
> in this section is a **pre-registered target to hit or honestly miss**,
> never a claim until a demo pins it.

| Step | Goal | Action | Pre-registered target (honest fallback: record what measures) |
|---|---|---|---|
| **1 — D17 headline** | Local, biologically-plausible learning trains a Transformer on LM | Un-gate and run the C0 ladder: `transformer/ff_hybrid/muon` vs `transformer/bp/adam` (+ `mlp/ff_hybrid/muon` local reference), 10–20-min arms, matched-step protocol, seeds 0–2; promote to a fixed-step D17 demo (walltime-budgeted arms can never be gallery-pinned — walltime printed, never recorded). **Pre-registration lives in the probe docstring (`lm_comparison.py`) — written before the run** | Reported number = **mean val_ppl over seeds 0–2** (per-seed values recorded; the full curve is the artifact, not the endpoint). Gap = (ff_hybrid − bp/adam)/bp/adam, verdict band fixed in advance: **<15% = headline parity** · **15–25% = competitive-with-caveat** (table + gap pinned, no parity claim, F5 proceeds on measured accuracy) · **>25% = MISS → C1 fallback triggers mechanically** (no post-hoc argument) |
| **2 — Capstone figure (F5)** | ✅ **CLOSED rev 13 — honest miss pinned** (see the F5 demo row): the physical advantage is NOT realized at HEAD; the non-autograd local-rule realization is the lever | High performers are physically superior for next-gen hardware | Resource-vector accounting on the Step-1 winner: extend the D4 memory profiler to ff_hybrid (no stored-activation sweep), simulated-energy per the substrate models' stated terms (local updates vs global matmuls), per-episode compute at matched accuracy; one FrontierRecord table + Pareto frontier figure. **The accounting schema (which substrate terms count, what a local update costs vs a backward matmul, whether optimizer bookkeeping — momentum buffers, SVD — is charged) is WRITTEN before any measurement** — no computing the ratio after seeing it. Report **both** the plain-update variant (clean locality) and the OrthoAdam variant (best performance) | Order-of-magnitude memory-bandwidth saving (**~10×** target) LEADS — backprop must store all layer activations; ff_hybrid doesn't; the ratio scales with depth and is near-definitionally true. **~5×** simulated-energy saving is the carefully-footnoted SECONDARY: charge OrthoAdam's SVD honestly; if it eats the win, that is a real finding ("the physical advantage survives on the forward pass; the optimizer is the cost") — report, don't hide |
| **3 — D22 instant adaptation** | ✅ **CLOSED rev 13 — honest miss.** Probe verdict (`scripts/probes/d22_psi_only.py`): ψ-only adaptation cannot acquire Task B with θ bitwise frozen — the ψ contract carries no task-loss signal (controls designed in before the run caught it: lr-matched fine-tune acquires B 0.984 at real forgetting cost, ψ-only sits at the frozen-null floor). Lever: supervised ψ term (B5-generalization) or D2 metaplasticity — *revival condition: a ψ-law primitive with a local supervised term, pre-registered* | Single-episode adaptation with **‖θ_after − θ_before‖ = 0 asserted exactly** was demonstrated (bitwise) but the mechanism is empty without acquisition — demo withheld per probe-first rule 3 |

**Ruthlessly parked (rev 11) — each with its explicit revival CONDITION
(no parked item may read as silently dropped):**
- **PEPITA (anything further):** revive only via a new probe with a
  pre-registered prediction about the *weight-trajectory* growth channel
  (the one unexplained observation left: output-weight magnitude grows
  ~1e4 in ~600 steps even with unit-normalized steps — a ‖W_out‖-vs-‖B‖
  per-step tracker would close it) or about the
  faithful-forward-modulation realization. The five-cause audit chain is
  otherwise complete and closed.
- **Z3-terminology defense:** revive never as prose; the D22 exact-θ
  assert supersedes the naming debate entirely.
- **Deep-Hebbian / naive-STDP rescue work (B2, B5, B6):** revive only
  when a neuromorphic-paper-shaped deliverable exists that needs them.
- **C1 / C2–C4:** below the reel; C1 re-opens **mechanically iff the D17
  pre-registered verdict lands in the >25% band** (same threshold, same
  run — the branch is a lookup, not a judgment call).

**Canonical claim wording (binding everywhere — README-grade artifacts,
RESULTS.md, demos, prose):** "forward-local credit with a single readout
supervision term — no backward sweep through the hidden layers." Never
"fully local"; never "local learning beats backprop". Muon/OrthoAdam are
global per-matrix operations (SVD polar over the whole weight tensor) —
they are NOT part of the locality claim, only of the performance
configuration. **Claim A (credit locality)** and **Claim B (physical
advantage: no stored activations, no backward sweep — true regardless of
optimizer)** are always reported on separate lines so a skeptic's "the
SVD isn't local" cannot sink the memory/energy result.

⚠️ **CLAIM A SCOPE AUDIT (rev 13, `scripts/probes/f5b_closed_form_ff.py`
— pre-registered P4 CONFIRMED):** under `InstantaneousDynamics`, the
nudged pass differs from free ONLY at the output act (β·onehot nudge);
hidden nudged acts == free hidden acts, so every hidden-layer goodness
term in the ff objective is EXACTLY ZERO. The autograd ff realization's
sole nonzero term is the output pseudo-loss ‖out_f‖²−‖out_n‖², and its
gradient reaches the hidden weights ONLY via the autograd chain
(closed-form detached variant: exactly 0 hidden gradient → chance
0.104 vs autograd 0.778 on mnist quick d4/w128/300 batches, seeds 0–2;
and 0 bytes saved vs 1108 KiB). **The chain IS the learning signal —
ff_hybrid at HEAD performs a backward sweep through the hidden layers**
(output-pseudo-loss backprop). Consequences: (1) the canonical wording
above is FALSE for ff_hybrid specifically — it holds verbatim only for
the requires_autograd=False credit family (thermo, measured 0 bytes)
and PEPITA-style closed-form inverse routing; every artifact quoting
Claim A must scope it to those credits until per-layer nudged passes
exist. (2) F5's memory miss is now fully EXPLAINED (the chain is
load-bearing — 1108 KiB at d4/w128, 48.5→74.5 M FLOPs with it).
(3) The honest path to a true layer-local ff with the physical
advantage is per-layer contrastive/predictive targets (B3/B4), which
give each hidden layer its own nudged pass — promoted to the top of
the lever queue as the Claim A repair.

**Identity-clause scope (rev 12):** the two-minute re-demonstration
identity applies to the **fast demo tier**; registered-scale claims
(D17/D18-style, `slow`-marked) re-show via the slow tier at their
registered budget. The demo suite remains the proof at both tiers.

**Binding constraint retained:** the demo suite is still the proof. A
parked claim can be resurrected only through a new probe with a
pre-registered prediction.

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
| **Snapshot state generalization** (found during this review): `TrainerSnapshot` (`core/system_trainer/trainer.py:182`) captures only `update._momentum_buffers` — state capture works for the momentum-buffer family (Euclidean, Riemannian/Muon) but **`AdamUpdate`/`OrthoAdamUpdate` `_m`/`_v`/`_t` and any credit-internal state are silently dropped on resume**; the R11.2.24 bitwise-resume lock passed because its fixtures used buffer-family updates | ✅ **LANDED 2026-09-06 with A1** — `get_state()`/`load_state()` named-group protocol; Adam `_m/_v/_t` captured; bitwise resume green | A1 |
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
| **A0** | ✅ **DONE 2026-09-06** — zero-new-primitive probe run with matched-step Muon controls. Verdict (recorded in `scripts/probes/p4_width_fragility.py` docstring): **split terminus per rule** — ePC's direction is right (magnitude normalization alone rescues it), PEPITA's failure is directional (every magnitude rung explodes; RESEARCH4 Fix-4 falsified for PEPITA). Natural-gradient rename-or-diag-Fisher still rides the next update-axis touch | `p4_width_fragility.py` `--update` axis (muon / natural_gradient / unit_rms / local_adam) + per-cell lr micro-sweep | A0 probe output + walltime in probe docstring |
| **A1** | ✅ **LANDED 2026-09-06** — `UnitRMSUpdate` (momentum-EMA → unit-RMS per tensor, no orthogonalization) + `LocalAdamUpdate` (scalar per-tensor second moment, LAMB-style `u = m̂/√(mean v̂)`). Full wiring checklist done: config classmethods (`ParameterUpdateConfig.unit_rms()` / `.local_adam()`), dispatch in spec `_UPDATE_CLASSES` + factory + joint, root `_LAZY`/`__all__`/`TYPE_CHECKING` + `ontology/__init__` exports. **Snapshot state generalization landed with it**: `get_state()`/`load_state()` named-group protocol on Euclidean/Riemannian/Adam/UnitRMS/LocalAdam; `TrainerSnapshot.opt_state` is now `dict[str, dict[str, Tensor]]`; `AdamUpdate._m/_v/_t` captured (carried-queue item closed); bitwise-resume test updated + passing | `computronium/ontology/update.py` + `tests/unit/core/test_update_ladder.py` (8 locks: unit-RMS identity, rank-1 direction preserved vs Muon, LocalAdam direction-preserving vs Adam, fail-loud reuse ×2, bitwise state round-trip ×2) | All gates green: ruff clean, pyright 0 errors, 14 targeted tests pass. **A2 partial verdict (probe docstring): ePC w32 trains without Muon on every rung — unit_rms best (34.9 @ 3e-4); ePC w64 (Muon-exploded cell) trains under unit_rms to ppl 28.3 — CRUTCH DEAD for ePC at w32–64; PEPITA explodes on all rungs (directional failure confirmed)** |
| **A2** | ✅ **COMPLETE 2026-09-06** — ePC width sweep done under unit_rms: **w32 34.9, w64 28.1–28.5 (seeds 0/1/2), w128 25.3–27.4, w256 24.8–29.6 — ePC is WIDTH-ROBUST without Muon** (Muon required w≥256). PEPITA explodes identically across seeds. **Defect audit executed per standing caution** (see probe docstring): feedback_scale inert under normalized rungs; step-0 stds healthy; centered-e1 ruled out — the runaway is structural to the DFA-style realization (fixed B row-space + unbounded activity loop), repairs are B1 or A5. Remaining for D18 demo promotion: D13 MNIST instrument cross-check | `scripts/probes/p4_width_fragility.py` docstring (full record) | Multi-seed confirmed on headline cells; full grid seed-0 |
| **A3** | ✅ **DONE 2026-09-06** — the simple-regime (F1) depth wall PERSISTS under the ladder: unit_rms walls at depth 8+ (best 0.206 @ depth 2, lr 0.02, non-monotonic lr response); Muon better at depth 8 (0.276) but walls at 16/20; Euclid walls from depth 8. Consistent with F1's own record — the wall is a credit-channel property, NOT an optimizer property. Split with A2 locked: **magnitude fixed the WIDTH axis (LM); the DEPTH axis needs A4 or the D14 faithful composition** | `scripts/probes/a3_ladder_epc.py` (verdict in docstring) | Probe output + walltime in docstring; credit norms reproduce the F1 attenuation signature |
| **A4** | ✅ **LANDED 2026-09-06** — `credit_norm: Literal["none","relative","rms","beta_adaptive","spectral"]` on `CreditAssignmentConfig` (+ classmethods), `_apply_credit_norm` helper applied at pseudo-gradient formation: ThermodynamicContrast (both block & plain paths, ε refs per transition), PEPITA per-hop `err`, FA/DFA per-hop propagated error. Zeros stay zeros; non-finite credit passes through untouched (Riemannian diverged-step precedent). **Verdict:** mechanism CONFIRMED — spectral flattens per-layer credit norms to exactly ~1.0 through depth 16 (vs 4×/layer decay / exact-0.0 trapping at HEAD); audit showed the hidden-layer signal exists but is budget-independent attenuated (3e-6→4e-3, settle budgets 5/15/30 equivalent — NOT a budget artifact, NOT topology trapping). Learning: depth 8 acc 0.195 (spectral+euclid@0.2) vs 0.113 (none, matched) — real lift, boundary honest: not yet Muon-level (0.276) in the 60-batch simple regime; the decisive test is the D14 faithful-regime composition (next-session item 1). Gates: property suite 679 passed (credit-semantics gate), 6 unit locks (`test_credit_norm.py`), ruff net-improved vs baseline, pyright clean on new code | `computronium/ontology/credit.py` + `tests/unit/core/test_credit_norm.py` | Probe outputs + walltime in session record; **D19 (demo) deferred until the faithful-regime composition lands** — a simple-regime demo would pin a weak claim |
| **A4×D14** | ✅ **COMPOSITION CLOSED 2026-09-06** — see next-session item 1 and the probe docstring: faithful regime self-sufficient (SGD 0.528, Adam 0.828 @ depth 20); credit_norm harmful there (ε = dynamics). Unifying-hypothesis composition test resolved | `scripts/probes/a4_faithful_composition.py` | Predictions falsified honestly; single-seed arms, D14 multi-seed baseline |
| **A4-legacy-row** | *(superseded description)* **Credit-space normalization (C-axis).** Seam: the layer error tensor at pseudo-gradient formation. Modes on `CreditAssignmentConfig`: `credit_norm: Literal["none","relative","rms","beta_adaptive","spectral"]` — `relative`: εᵢ/(‖freeᵢ‖) (RESEARCH4 Fix-2 option 1); `rms`: per-layer unit-RMS ε; `beta_adaptive`: per-layer βᵢ tuned to hold the error signal at unit scale (Fix-2 option 2 — the ePC-native version, since ÷β is the cap in question); `spectral`: spectral-radius→1 rescale of the propagated error (option 3). Applies to `ThermodynamicContrast` (ePC's settled εᵢ before εᵢᵀaᵢ₋₁) and the FA/PEPITA per-hop propagated error. **Never normalizes toward fabricated signal** (zeros stay zeros) | `computronium/ontology/credit.py` + unit lock; **gates the full property suite** (credit-semantics change, TODO11 precedent) | Re-run the F1 instrument (`scripts/probes/f1_epc_depth.py` — already measures per-layer credit norms): does ~4×/layer flatten to ~1×, and does ePC **learn at depth 8–20 under the simple F1 regime** (D14's faithful regime already trains depth 20 — the sharper test is the simple one)? |
| **A5** | ✅ **LANDED 2026-09-06** — `StateDynamicsConfig.gain_control: Literal["none","unit_rms","spectral"]` (field + `instantaneous()`/`error_predictive_coding()`/`energy_minimization()` classmethod params; other dynamics ignore it until their own audited pull). `_apply_gain_control` renormalizes **hidden layers only** at settle emit — unit_rms = μPC per-sample unit RMS (a·√d/‖a‖), spectral = unit spectral norm of the batch matrix; input/output pass through (output carries the readout logits); zero/non-finite layers untouched. **Probe verdict (prediction FALSIFIED, `--gain-control unit_rms`):** pepita w32/w128 under unit_rms still diverge — hidden acts bounded (~0.8) by construction but the explosion **reroutes through the unnormalized readout** (output act_std 5.8e4–7.3e10; val_ppl saturated 4.85e8 at lrs 1e-4/3e-4 AND 1e-5). A5 does not repair PEPITA; audit chain now: feedback_scale, centered-e1, row space (B1), hidden gain — all ruled out. Remaining suspect: readout-path divergence (saturated softmax ⇒ e1 ≈ ±onehot ⇒ persistent class-constant update direction on the output weights) — testable via output-side gain control or e1 saturation handling, but per the standing caution this is an observed behavior of the current realization, not a PEPITA-in-principle claim | `computronium/ontology/dynamics/_dynamics.py` + `tests/unit/core/test_gain_control.py` (7 locks: passthrough identity, per-sample unit RMS hidden-only, spectral σ=1 hidden-only, zero/non-finite untouched, short-acts untouched, instantaneous settle bounded ×2 modes) | Property suite 679 passed (dynamics-semantics gate) + wiring lockstep green; ruff/pyright clean on new code (baseline legacy errors unchanged) |
| **A6** | **Co-design pass:** once credit is well-conditioned (A4) + magnitude-controlled (A1), sweep the optimizer axis again — how far does Muon relax toward Adam/SGD? Re-pin D16 with normalized-credit columns; fold in the natural-gradient rename/diag-Fisher resolution | Campaign YAML + D16 extension | The crutch map: which local rules still need which optimizer, with matched-step controls (P3 protocol) |

#### A6 Co-Design Map (assembled 2026-09-06 from A0–A5 + B1 evidence; matched-step protocol per P3)

| Rule × axis | Landed lever(s) | Measured state | Optimizer requirement |
|---|---|---|---|
| ePC — width (LM) | unit_rms (A1/A2) | Width-robust w32–256 without Muon (D18 pinned: w64 32.5, w32 42.5) | **Crutch dead** — unit_rms (momentum-EMA normalize-the-momentum) is the canonical rung; Muon strictly dominated at small widths |
| ePC — depth (simple regime) | credit_norm spectral (A4), gain_control (A5) | Partial lift only (0.195 vs 0.113 @ depth 8); no lift from activity-side renorm; credit norms still explode (A5 rung: 1.1e7) | Muon best-in-class at depth 8 (0.276); orthogonalization's residual value is **depth-only** |
| ePC — depth (faithful regime) | none needed (A4×D14) | Self-sufficient: plain SGD 0.528 @ depth 20; credit_norm HARMS (ε is dynamics) | SGD suffices — the strongest co-design result in the program |
| ff_hybrid | readout_error (landed pre-TODO12) | Width-robust on LM; carries the local-credit story | Trains under plain rungs |
| PEPITA — width (LM) | A2/B1/A5 audit chain | 🅿️ **PARKED rev 11** — diverges under every landed lever; five causes ruled out; realization retired | Explodes on every non-Muon rung (Muon small-lr = stable-flat) — parked, not a high performer |
| NaturalGradientUpdate | — | Mean-\|grad\| magnitude normalizer, not Fisher (A0) | ✅ **RESOLVED 2026-09-06 — renamed `MeanNormUpdate`/`mean_norm`** (touching `_UPDATE_CLASSES`, factory, joint, CLI listings, evaluation map, probes/tests; fake `"fisher"` alias dropped). No diag-Fisher was implemented — the A6 map shows the momentum-EMA family (unit_rms) dominates it, so the honest resolution is the rename |

**Map-level conclusions:** (1) the optimizer crutch is dead for ePC on
the width axis and dead outright in the faithful regime; (2) Muon's
irreplaceable signal, where it exists, is depth-side — but A4/A5 show
credit/activity-side normalization does not substitute for it yet;
(3) the momentum-EMA normalizer (unit_rms) beats instantaneous
normalizers (natgrad) and per-coordinate Adam (local_adam) — the
magnitude family to standardize on; (4) D16 re-pin should add the
unit_rms-vs-Muon matched-step column and the faithful-regime SGD row.

### Workstream B — Learned Feedback & Task Coupling (Levers 2, 4; Phases 2 & 4)

**Goal:** Fix PEPITA's directional collapse and error-blindness by making
feedback learnable and targets local — **transport-free objectives only**
(the L3 weight-transport freeness lock is the guard: ‖B − Wᵀ‖ stays > 1e-3,
separate storage).

| Step | Description | Artifact | Validation |
|---|---|---|---|
| **B0** | ✅ **DONE 2026-09-06** — smoking-gun probe (`scripts/probes/b0_adaptive_fa_audit.py`): with W FROZEN, no activity, and zero gradients (only enabling the branch), legacy `AdaptiveFA._update_feedback_weights` drives cos(B, Wᵀ) monotonically up (0.116→0.160, 0.288→0.447 over 5000 updates) — **the legacy alignment is pure soft weight transport** (it reads `param.data`; Akrout's non-transport activity term was dropped in the port). The xfail's reason ("feedback LR too small to show alignment in 50 steps") = the transport is present but slow; the xfail stays xfail forever under the L3 lock, by construction. Reusable for B1: slow feedback timescale (feedback_lr 1e-4 ≪ lr), cos(B, Wᵀ) metric, feedback_scale | `scripts/probes/b0_adaptive_fa_audit.py` | Probe output + walltime in docstring; transport verdict recorded |
| **B0-legacy-row** | *(original description)* **Audit the legacy seam first.** `computronium/core/local_learning/rules/fa.py:148` has `AdaptiveFA` (Akrout et al. 2019) — but its `_update_feedback_weights` pulls `fb` toward `param.data` (or `param.data.T`): **it reads forward weights, i.e. soft weight transport**; its bio-alignment property test sits xfail'd (`tests/property/biology/test_biology_axioms.py:365`, "feedback LR too small to show alignment in 50 steps"). Extract what's reusable (slow feedback timescale, alignment metric) and record the transport verdict | Probe note + `b0` docstring citing this file | The ontology port must NOT inherit the transport; the xfail stays xfail until a transport-free rule passes it |
| **B1** | ✅ **LANDED 2026-09-06** — `CreditAssignmentConfig.local_goodness(learned_feedback=, feedback_lr=0.5, feedback_update_every=1)`; extended `local_goodness`, NOT a new credit_type (registry surfaces untouched). Learned B is credit-internal state (`LocalGoodnessCredit._learned`, same deterministic CRC-seeded init as fixed B): per weight, closed-form ridge regression `post @ C ≈ e1` with `B = Cᵀ·feedback_scale` (autoencoder-style, autograd-free, reads only settled activations + the e₁ broadcast — never `param.data`, L3 honored), EMA-blended at `feedback_lr` every `feedback_update_every` steps; non-finite settles skip the update (diverged-step precedent). `get_state()`/`load_state()` per the A1 protocol (learned-B matrices + step counter, string keys, fail-loud shape-mismatch reuse); **TrainerSnapshot now captures credit state** (`credit_state` named-group axis, restore fails loud — carried-queue lesson closed for credits too). **PROBE VERDICT (pre-registered prediction FALSIFIED):** `p4_width_fragility.py --learned-feedback --update unit_rms` — pepita w32/w128 STILL explode (val_ppl 4.8e8; act_std ~20×/layer at lrs 1e-4/3e-4, ~5 min on RTX 3080). Learned B changes the update's row space and the runaway persists ⇒ the fixed-B row space is exonerated; the driver is the unbounded settle-activity loop. **A5 (settle-path gain homeostasis) is the indicated PEPITA repair**, per the pre-registration's else-branch. Library-side B1 stands as honest infrastructure (unit locks + bitwise resume green) | `computronium/ontology/credit.py` + `tests/unit/core/test_learned_feedback.py` (5 locks: B moves + reconstruction strictly improves, transport-free trajectory vs perturbed W, bitwise state round-trip, fail-loud reuse, fixed path untouched) + `tests/integration/test_learned_feedback_resume.py` (snapshot carries credit state; resume bitwise) | Full property suite 679 passed (credit-semantics gate); ruff clean on new code; pyright clean on new code (legacy findings unchanged) |
| **B2** | 🅿️ **PARKED (rev 11 — PEPITA family).** Fixed-vs-learned B at depths 4/8/16, capacity-matched, MNIST + LM cells | `scripts/probes/b2_learned_feedback_depth.py` | Does the depth-attenuation problem dissolve for the FA/PEPITA family? |
| **B3** | ▶️ **PROMOTED rev 13 — the Claim A repair** (see the scope audit above): per-layer targets give each hidden layer its own nudged pass, the only honest route to a true layer-local ff with the physical advantage. **Predictive targets for FF (C-axis):** each layer predicts the next layer's activity; the prediction error IS the credit signal (predictive coding's error, FF's architecture). New credit type `local_predictive` — full registry wiring per the checklist below. Design note: `TargetInversionCredit` currently propagates targets through `Wᵀ` (**weight transport by construction**) — the honest variant propagates targets through learned B (compose with B1) | `computronium/ontology/credit.py` + demo test | vs `ff_hybrid` on MNIST + LM: does it match without global CE? |
| **B4** | **Per-layer contrastive targets:** each layer sees its activity under (a) correct-class input, (b) corrupted-class input; the activity difference is the layer-local signal (FF's original idea, made per-layer) | `computronium/ontology/credit.py` | Does it beat ff_hybrid on LM (where pure FF's global-goodness fails)? |
| **B5** | 🅿️ **PARKED (rev 11 — naive-STDP rescue).** **Reward-modulated STDP** — the supervised error term `TemporalTraceCredit` lacks by construction (F2: "declares `phases=(FREE,)` and never consumes `loss`"). Add a config-gated reward/error term on the timing-STDP path; closes F2's OPEN verdict either way | `credit.py` + F2 test re-audit | Collapse stops + readout ≥ random-init ⇒ F2 was a missing term; persists ⇒ a verified constraint of timing-STDP, honestly closed |
| **B6** | *(optional, last)* **Temporal targets:** use the settling trajectory (not just the final equilibrium) as credit — EqProp-adjacent; only if B3–B5 leave the target-delivery question open | probe only | — |

### Workstream C — Objective Consistency & Architecture (Lever 5; Phase 5)

**Goal:** Resolve P2 the honest way (contrastive + repaired propagation),
and co-design the architecture for local learning.

| Step | Description | Artifact | Validation |
|---|---|---|---|
| **C0** | ▶️ **UN-GATED rev 11 — this is Highlight-Reel Step 1** (the D17 ladder). Bank the LM baselines: the 10–20-min ladder arms at verified regimes with matched-step protocol riding along; curves + ppl table → **D17** per the static-arms convention. The repair program then measures lift-over-*this* baseline, not over 2.5-min smoke | `scripts/probes/lm_comparison.py --minutes 15 --arms …` → D17 demo | User-gated budget — **gate lifted by the 2026-09-06 highlight-reel directive**; findings stabilize before any 60-min run |
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

## 🏆 Capstone — The Payoff, Made Measurable (▶️ ACTIVE — Highlight-Reel Step 2)

RESEARCH4's payoff table (energy, plasticity, dynamic networks, distributed
operation, substrate compatibility) becomes a **resource-vector accounting**
demo — now the ACTIVE Step 2 of the highlight reel, run on the Step-1
winner (ff_hybrid presumed; the D17 result decides):

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
paragraph with numbers. Pre-registered targets: ~10× memory bandwidth,
~5× energy (see the Highlight Reel table). Lands as demo **F5**
(`test_demo_resource_vector.py`) + Pareto frontier figure.

---

## 🎯 Demo/Capability Targets

| Target | Description | Workstream | Demo Test |
|---|---|---|---|
| **D17** | ▶️ **PAUSED at seed 0 (rev 13 — user budget directive).** Seed-0 arms salvaged (`benchmark_results/d17_seed0.json`): transformer/ff_hybrid/muon **5.05** vs transformer/bp/adam **27.55** vs mlp/ff_hybrid/muon 27.94 — ff_hybrid far ahead at 15 min, but SINGLE-SEED; the pre-registered verdict band (<15% / 15–25% / >25% gap, seeds 0–2 mean) is NOT adjudicable until seeds 1–2 run. Resume = seeds 1–2 only (~90 min GPU; user-gated). If resumed and the gap holds, D17 lands as headline parity | C0 | probe JSONs + D17 demo promotion after verdict |
| **D18** | ✅ **LANDED 2026-09-06** — crutch killed for ePC w32–64: ePC trains at both fragile widths under unit_rms (w64 32.5, w32 42.5 multi-seed means) while Muon at its registered lr explodes (101.2 / 191.7); PEPITA control explodes (audit-backed structural). Demo: fixed-step arms (deterministic — walltime budgets CANNOT be gallery-pinned), LM task, seeds 0–2, CPU | A0–A2 | `test_demo_update_ladder.py` + `docs/figures/run_records/d18_update_ladder.json` |
| **D19** | Credit-space normalization: ePC learns at depth 8–20 under the simple regime; per-layer decay ~1× | A4 | `test_demo_credit_norm_epc_depth.py` |
| **D20** | Learned feedback: PEPITA with learned B competitive at depth 16, width-matched | B1–B2 | `test_demo_learned_feedback_pepita.py` |
| **D21** | Predictive/contrastive local targets train without global CE | B3/B4 | `test_demo_local_targets.py` |
| **F2-close** | Reward-modulated STDP verdict (either terminus) | B5 | re-audit inside `test_demo_spiking_plateau.py` |
| **D22** | ❌ **HONEST MISS (rev 13, probe verdict — NOT promoted).** `scripts/probes/d22_psi_only.py`: θ bitwise frozen (SHA identical) but routing-ψ does NOT acquire Task B — b 0.656 → 0.637–0.668 across psi_lr {0.05,0.2,0.5} × ep {200,600} (noise); FastWeight-ψ same (0.637–0.648; lr 0.5 degrades A to 0.566). lr-matched θ-fine-tune acquires B 0.984 at real forgetting (A 0.660 vs ψ 0.969≈null 0.973). ROOT CAUSE: the ψ-step contract feeds ψ only the FREE (target-free) settled activity — no landed plasticity law consumes a loss/target term (the B5 missing-supervised-term disease, confirmed on the P-axis). Mechanism assert (‖Δθ‖=0) trivially true; value claim evaporated exactly as the pre-registration's else-branch allowed. ψ-timescale boundary mapped; lever = supervised ψ term (B5-generalization) or D2 metaplasticity | D1 | probe only — demo withheld per probe-first rule 3 |
| **F5** | ✅ **LANDED 2026-09-06 (rev 13) — the capstone miss pinned.** `test_demo_resource_vector.py` (~5 s CPU, gallery lock green ×2, manifest re-pinned additive-only, RESULTS.md paragraph): measured saved-bytes + measured FLOPs at depths 4/16. **~10× memory target FALSIFIED at HEAD** — ff_hybrid's autograd realization saves MORE than backprop (177.5 vs 143.5 KiB @d16, peak==total); **~5× energy target FALSIFIED** — ff_hybrid ~1.3× bp FLOPs (1.349 vs 0.939 M), ortho premium real (muon 1.349 vs euclid 1.218 M = "the optimizer is the cost"), thermo ties bp. The O(1)-memory class is real (thermo saves EXACTLY 0 at every depth) — Claim A (algorithmic locality) stands, Claim B NOT demonstrated at HEAD; asserts are the ratchet a non-autograd ff_hybrid must flip (implementation artifact per standing caution) | Capstone | `test_demo_resource_vector.py` + `docs/figures/run_records/f5_resource_vector.json` |
| **F4** | ✅ **LANDED 2026-09-06** — `test_demo_credit_channel_map.py`: all 8 failure modes in one figure; two mechanisms LIVE (A4 spectral repair: ePC depth-8 0.108→0.170 with per-layer credit norms flattened to ~1.0; blocked channel: sPC hidden norms exactly 0.0 vs ePC > 0) + record ratchets locked against D18/D16/F1/F2/D14 pinned data. **D19 supersedes the deferred simple-regime demo** — F4's live cell IS the A4 demonstration | A/B/C | `test_demo_credit_channel_map.py` + `docs/figures/run_records/f4_credit_channel_map.json` |

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
- **Runs budget (TIGHTENED 2026-09-06):** no multi-hour or unattended
  long runs — they waste time and electricity (user directive). Per-arm
  budgets stay ~10–20 min ONLY with explicit per-batch approval;
  smoke-verify every registered arm end-to-end at ~1-min arms first
  (the D17 seed-0 ctx defect would have surfaced in minutes);
  registered-scale claims may re-show via the `slow` tier at their
  budget, but launching them is a per-session user decision, never a
  background default.
- **Demo API publishability roadmap deferred — research first** (user directive 2026-09-05). This plan is research-scoped accordingly; the roadmap resumes after the repair program.

---

## 🎯 The Next-Session Plan (Ordered, rev 11 — Highlight Reel)

1. **D17 ladder (Step 1) — PAUSED at seed 0.** Seed-0 salvaged:
   ff_hybrid 5.05 vs bp/adam 27.55 (single-seed). Resume = seeds 1–2
   only, after explicit user approval of the ~90-min GPU budget:
   `uv run python scripts/probes/lm_comparison.py --minutes 15 --arms
   transformer/ff_hybrid/muon,transformer/bp/adam,mlp/ff_hybrid/muon
   --seed 1` (then `--seed 2`); copy `benchmark_results/lm_comparison.json`
   → `d17_seed<N>.json` after each. Compute the seed-mean, apply the
   pre-registered band mechanically, then promote to the fixed-step D17
   demo (scaled pinned regime — walltime-budgeted arms can never be
   gallery-pinned) + gallery lock.
2. **F5 resource-vector demo (Step 2)** — winner of Step 1 through the
   capstone accounting: extend the D4 memory profiler to ff_hybrid,
   simulated-energy per substrate terms, per-episode compute; one
   FrontierRecord table + Pareto figure; static-arms convention.
3. ✅ **D22 probe verdict (Step 3) — HONEST MISS, 2026-09-06.** Both
   pre-registered predictions falsified (`scripts/probes/d22_psi_only.py`,
   ~2 s CPU): ψ-only adaptation cannot acquire Task B with θ bitwise
   frozen — no landed ψ law consumes a task-loss signal (root cause,
   same disease as B5's STDP error term). Demo NOT promoted (probe-first
   rule 3); fine-tune-vs-ψ value comparison recorded honestly in the
   probe docstring. Step 3 is closed as a mapped boundary, not a pin.
4. Only if Step 1 misses its target: re-open C1 (P2 contrastive-LM
   parity + propagation repair) — the reel's fallback path.

**Gate after each step:** probe output + walltime visible → ruff
format/check on changed files → pyright on new modules → targeted tests
→ gallery lock green if a demo was promoted.

---

## 🎯 The Next-Session Plan (historical, rev 3–10 — repair program, closed at rev 10)

1. ✅ **A4×D14 composition — DONE 2026-09-06, clean informative
   negative** (`scripts/probes/a4_faithful_composition.py`): faithful
   regime is SELF-SUFFICIENT (mupc+adam 0.828 at depth 20 replicates
   D14; **plain SGD reaches 0.528** — the dynamics do the heavy
   lifting); credit_norm actively HARMS it (spectral/adam 0.545,
   spectral/euclid 0.258, rms ≈ chance) because ε in the
   reparameterized regime is injected into the FORWARD — it is
   dynamics, not just credit, and rescaling breaks the μPC/β scale
   structure. Both pre-registered predictions falsified → the A6
   co-design map's key entry: **levers do not naively compose;
   credit_norm is a simple-regime tool, the faithful regime needs
   none.** The unifying hypothesis's composition test is CLOSED.
2. ✅ **B1 pull — LANDED 2026-09-06 (see B1 row).** Reconstruction
   objective shipped transport-free with snapshot-captured credit state;
   the probe falsified the row-space prediction: learned B does not stop
   the PEPITA runaway. **Consequence: A5 is now the next pull** (the
   pre-registration's else-branch — settle-path gain homeostasis bounds
   the unbounded activity loop). B1's remaining objectives (b)
   update-direction alignment, (c) slow co-adaptation stay available if
   A5 also fails to stabilize PEPITA.
3. **C0** — the carried LM ladder runs (user-gated) bank D17 baselines.
4. **A6 map assembly** — fold A0–A5 verdicts into the co-design table:
   ePC needs magnitude (unit_rms) + nothing else; ff_hybrid needs
   readout_error; PEPITA diverges under every landed lever (A2/B1/A5
   audit chain) — its row is "current realization diverges; readout-path
   suspect"; Muon's residual value is depth-only (A3). Re-pin D16 with
   normalized-credit columns; natural-gradient rename/diag-Fisher rides
   this touch.
5. **F4 figure** — assemble the credit-channel failure map from the
   landed audit chains (each failure mode now has measured evidence and
   ruled-out causes).
6. **Optional cheap rungs** — ✅ **(a) DONE 2026-09-06: ePC depth ×
   gain_control — prediction FALSIFIED** (`a3_ladder_epc.py
   --gain-control unit_rms`, 48 s CPU): no depth lift (unit_rms
   0.245@2 → walls at 8+ as before; credit norms still explode —
   1.1e7 at depth 8). The depth wall is credit-side, now confirmed from
   the activity side too. (b) output-side gain control / e1 saturation
   handling for the PEPITA readout suspect remains open (probe-only,
   standing caution applies).
7. **Sign-momentum rung** (A1 iii, optional) — D18 pinned unit_rms
   clearly ahead of Muon on ePC at w32–64; orthogonalization's residual
   value is now a DEPTH question (A3 decides the hardware story).

**Landed this session (rev 10, 2026-09-06):**
- **PEPITA readout rung** (`p4_width_fragility.py --readout-norm
  unit_rms`, new `_ReadoutNormProxy`): both pre-registered predictions
  FALSIFIED (~10 min RTX 3080). Readout step shape ruled out — the
  divergence survives hidden-gain bounding + output-step normalization;
  output act_std 5e4–7e10 means the output-weight MAGNITUDE grows
  through the weight trajectory, not the normalized step. PEPITA's
  five-cause audit chain is complete; realization stays retired.
- **`NaturalGradientUpdate` → `MeanNormUpdate`** (carried queue closed):
  honest rename, `"fisher"` alias dropped; wiring locks + full suite
  green. If a Fisher mechanism is ever wanted, it is a NEW primitive
  with a diag-Fisher implementation, not this class.
- **D16 re-pin** (matched-step unit_rms column): **unit_rms is
  regime-shaped** — vision-quick chance at every lr 0.002–0.1 (lr grid
  probed; loss oscillates min 0.80/last 2.17 — RMS normalization
  random-walks near the loss floor) while D18 holds the LM-side win.
  Asserted in the demo as a boundary lock; D14 keeps the faithful-regime
  row (no duplication).
- **F4 landed** (see demo table row): the completion criterion "credit-
  channel failure map is live" is met.

**Demo-infrastructure lessons (for future D-table landings):**
- Walltime-budgeted arms can never be gallery-pinned (record drifts) —
  fixed-step arms are the pattern (D18: 600 steps, ~25 s/arm CPU).
- Walltime is printed, never recorded — putting it in the record dict
  silently breaks the lock's byte-stability.
- MNIST quick does NOT reproduce the LM width-fragility regime (ePC
  trains at w32 under both updates there) — D18-style LM demos must run
  the tiny-Shakespeare harness; MNIST demos can't carry LM findings.
- Manifest re-pin: render_gallery → merge per-capability updates →
  never reorder/shuffle existing sha pins (verified additive-only).

**Session-context notes for future work:**
- A5 probe command (reuse verbatim):
  `uv run python scripts/probes/p4_width_fragility.py --update unit_rms
  --cells pepita:32 pepita:128 --lrs 1e-4 3e-4 --gain-control unit_rms`
  (+ a 1e-5 rung). CAVEAT: under gain_control, post-settle act_std is
  bounded by construction on hidden layers — stability reads are
  val_ppl/finite loss, not act_std. The saturated val_ppl sentinel
  485165195.41 (= exp(23.0)) appears in every diverged arm — treat it
  as "diverged", not a measured perplexity.
- B1 probe command (reuse verbatim, do not re-derive):
  `uv run python scripts/probes/p4_width_fragility.py --update unit_rms
  --cells pepita:32 pepita:128 --lrs 1e-4 3e-4 --learned-feedback`
  (~5 min, RTX 3080). Learned-B knobs: `feedback_lr=0.5`,
  `feedback_update_every=1` were used — a slower cadence is untested.
- The P4 harness now takes `--update {muon,natural_gradient,unit_rms,
  local_adam}`, `--cells credit:width …`, `--lrs …` — reuse it; do not
  write a new width-sweep instrument.
- ePC arm lrs: unit_rms best at 3e-4 (w32) / 1e-4–3e-4 (w64); natgrad
  best at 1e-4. PEPITA explodes at ≥3e-5 on every non-Muon rung — don't
  re-litigate with higher lrs.
- Probe arms are 75 s walltime on CUDA (`lmc.DEVICE`); a full
  A2 sweep is ~25 arms ≈ 35 min. Budget accordingly.
- `NaturalGradientUpdate` rename-or-diag-Fisher (A0 as-touch) still
  open — it is a mean-|grad| magnitude normalizer, not Fisher; rename
  touches `_UPDATE_CLASSES`, factory, joint, CLI listings.
- Gate discipline: A1 landed without running the full property suite
  (update-axis change, not credit/dynamics semantics); run the full
  suite once before the D18 demo promotion since demo locks pin across
  axes.

**Gate after each step:** probe output + walltime visible → ruff
format/check on changed files → pyright on new modules → targeted tests →
full property suite when credit/dynamics semantics change → gallery lock
green if a demo was promoted.

---

## 🧭 Open Questions & Adaptive Branches

| Question | If YES | If NO |
|---|---|---|
| Does magnitude normalization alone rescue the fragile cells (A0/A1: UnitRMS ≈ Muon)? | **ANSWERED (split terminus, 2026-09-06):** YES for ePC (w32/w64 train without Muon under unit_rms); NO for PEPITA (directional failure — B1 is the repair). The unifying hypothesis holds per-rule | — |
| Does credit_norm flatten ePC's ~4×/layer decay (A4)? | **ANSWERED (2026-09-06, split):** the decay flattens (spectral: norms exactly ~1.0 through depth 16) and depth-8 learning lifts at matched lr — but simple-regime parity with Muon is not reached; the faithful-regime composition is the decisive remaining test. NOT a settle-geometry/topology problem (audit: hidden ε exists, budget-independent) — C2 not promoted yet |

| A3/A4 sharpened branch (2026-09-06): width = optimizer-side (D18
| pinned); depth = credit-side (A4 landed, composition pending). The
| honest claim so far: "the crutch is magnitude on width, direction on
| PEPITA, and per-layer attenuation on depth" — each axis has its
| lever landed and measured.
| Does learned B eliminate PEPITA's width fragility (B1)? | PEPITA becomes a competitive local rule; D13 upgraded; B3 composes on top | **ANSWERED (2026-09-06, NO — for the reconstruction objective):** learned B (row-space changed, transport-free) does not stop the runaway; the fixed-B row space is exonerated, the unbounded activity loop is the driver → **A5 is the indicated repair**. Objectives (b)/(c) remain untried |
| Do the P2 untried cells + contrastive repairs close LM (C1)? | PC-family LM demo lands; the objective-consistency lever is validated | The boundary is mapped honestly: contrastive+repairs is the working instrument, frozen-error retired with evidence |
| Does ψ-only adaptation solve the switch (D1)? | The plasticity payoff is demonstrated, not claimed | **ANSWERED (2026-09-06, rev 13, NO — `d22_psi_only.py`):** routing-ψ and FastWeight-ψ both fail to acquire Task B with θ bitwise frozen (the ψ contract carries no task-loss signal); fine-tune acquires B at real forgetting cost. The ψ-timescale boundary is mapped; the next lever is a supervised ψ term (B5-generalization to the P-axis) or metaplasticity (D2) |

---

## 💡 New Improvement Opportunities (surfaced rev 13 — F5, 2026-09-06)

- **Non-autograd ff_hybrid realization is the memory/energy lever** —
  F5 pinned that LocalGoodnessCredit's autograd realization stores
  ≥ backprop and costs ~1.3× FLOPs. The O(1)-memory class is real
  (thermo: exactly 0 bytes). A hand-written local-goodness update
  (per-layer closed-form/elementwise ops, no autograd graph — the B1
  ridge-solver pattern is precedent) would flip F5's ratchets and turn
  the capstone miss into the ~10×/~5× headline. This is a dynamics/
  credit-realization pull, pre-register before building.
- **Peak-vs-total saved bytes are identical for every landed credit** —
  even "local" losses hold all layers' graphs simultaneously at HEAD.
  A time-resolved live-bytes probe (pack/unpack hooks with a running
  set) is the instrument that would distinguish O(1)-peak from
  O(depth)-peak if the non-autograd realization lands.
- **Muon's SVD premium is scale-dependent** (1.349 vs 1.218 M FLOPs at
  16-wide demo scale = +11%) — at D17 scale (320-wide) the premium
  grows as d³; the F5 schema's per-matrix SVD charge is the right
  accounting when the registered-scale F5 rerun happens.

## 💡 New Improvement Opportunities (surfaced rev 13, 2026-09-06)

- **Supervised ψ term is the missing P-axis primitive** — D22's root
  cause generalizes F2's B5 finding: neither RoutingPlasticity nor
  FastWeightPlasticity consumes a loss/target term, so no ψ-only
  adaptation claim is possible at HEAD. The indicated lever: a
  `plasticity` law whose `step` receives the NUDGED-phase settled
  activity (target-conditioned Hebbian outer — ψ becomes a local
  associative readout learned with one supervision term, composing
  with the credit-locality story). Pre-register before building;
  the D22 probe is the reusable instrument.
- **Pipeline ψ-step phase choice** — `run_train_step` steps ψ on
  `credit.phases[0]` (always FREE for landed credits). A
  `psi_phase: Literal["free","nudged"]` knob (or NUDGED-first credit)
  would let supervised ψ laws exist without a new settle path. Design
  caution: modulation of the nudged settle interacts with contrastive
  credit semantics — needs its own audit.
- **D22 probe instrument is reusable** for any future ψ claim:
  `_probe`/`_probe_modulated`/`_theta_sha256` + the A→B switching-task
  batch helper — do not re-derive (flattened `create_switching_task`,
  settle-path probing — note `forward_pass` returns per-sample outputs
  for flat MLPs; probes must read through `dynamics.settle`).

## 💡 New Improvement Opportunities (surfaced rev 10, 2026-09-06)

- **unit_rms's convergence noise floor** — RMS normalization holds step
  magnitude fixed as the loss approaches the floor (measured: MNIST
  quick loss oscillates min 0.80 / last 2.17 over 150 batches at every
  lr). A decayed step_size schedule (or floor-relative normalization)
  would make the momentum-EMA family viable in easy regimes — candidate
  `unit_rms_decay` mode if a demo needs it.
- **PEPITA weight-trajectory channel** — the last ruled-out-free
  observation: output-weight magnitude grows even with unit-normalized
  steps and bounded hidden acts. Candidate mechanism: the momentum EMA
  on the output weight accumulates a persistent direction while
  normalization removes only its scale... yet per-step displacement is
  lr-bounded, so the growth rate (~1e4 in ~600 steps) implies the
  explosion enters through a non-step path (feedback-weight B growth?
  settle-internal state?). A probe tracking ‖W_out‖ per step vs ‖B‖
  per step would close it — cheap, probe-only.
- **F4's ratchet pattern is the template** for future consolidated
  claims: live arms for the mechanisms not otherwise demoed, record-
  derived asserts for the pinned ones — no duplicate harnesses.
- **`MeanNormUpdate` is now honestly named but load-bearing** in D16
  (natural column) — if it is ever removed, D16's column and the
  boundary story must be re-pinned, not silently dropped.
- **Diverged-arm sentinel hygiene** (carried from rev 9): still worth
  fixing — `lm_comparison._eval` reports exp(23.0) for every diverged
  arm; an explicit NaN/flag would make F4-style ratchets cleaner.

---

## 💡 New Improvement Opportunities (surfaced by A5, 2026-09-06)

- **The diverged-arm val_ppl sentinel** (exp(23.0) exactly, all arms):
  `lm_comparison._eval` likely clamps/saturates on non-finite logits —
  worth a look so diverged arms report NaN or an explicit flag instead
  of a plausible-looking number (evidence-hygiene defect).
- **Output-layer gain control is the untested half of A5** — the A5
  verdict is strictly "hidden-layer renorm is insufficient"; a
  readout-side bound (or e1 saturation handling) is the direct follow-up
  for the PEPITA readout suspect. Design caution: normalizing logits
  changes the CE landscape — needs its own pre-registration.
- **ePC × gain_control depth rung is free to try** — the flag is already
  on `error_predictive_coding()`; adding `--gain-control` pass-through
  to `a3_ladder_epc.py` is a one-line change and answers whether
  settle-path gain control beats credit_norm's partial depth-8 lift.
- **μPC-unit-RMS as the canonical A5 mode**: hidden acts settle to
  ~0.8 std under unit_rms (slightly below 1.0 — the √d/‖a‖ scale
  interacts with the batch matrix norm); if a demo pins A5, assert on
  the per-row RMS identity (the unit lock), not raw std.
- **F4 map input**: the PEPITA misaligned-channel row is now the
  deepest audit chain in the program — four ruled-out causes
  (feedback_scale, centered-e1, row space, hidden gain) with the
  readout-path suspect explicitly named.

---

## 💡 New Improvement Opportunities (surfaced by B1, 2026-09-06)

- **B1 library infra is reusable beyond PEPITA**: the reconstruction
  learned-B machinery + `TrainerSnapshot.credit_state` axis work for any
  credit that holds matrices (B3's predictive targets compose with it
  directly; RandomProjectionsCredit could adopt learned B via the same
  ridge update — its per-hop chained structure differs, don't assume).
- **PEPITA diagnosis is now three-for-three on ruled-out causes**:
  feedback_scale (inert, A2), centered-e1 (inert, A2), fixed-B row
  space (exonerated, B1). The unbounded settle-activity loop is the
  only remaining candidate — A5's validation has a sharp target: if
  gain_control alone stabilizes pepita w32/w128, the causal chain is
  closed (Muon's orthogonalization was masking the loop via step shape).
- **Ridge regression as a local-learning primitive**: the closed-form
  `post @ C ≈ e1` solver (float32, trace-scaled λ, non-finite guard) is
  a general transport-free local rule — candidate for a future
  `LocalRegressionUpdate`/credit primitive if B3 needs it.
- **F4 map input**: PEPITA's misaligned-channel row now carries a
  mechanism-audit chain (3 ruled-out causes, loop identified) — the
  strongest-audited row of the figure.
- **Standing caution still binds (user directive)**: the runaway is
  "structural to the current DFA-style realization at HEAD", not
  "PEPITA-in-principle"; A5 verdict language should keep the same
  honesty (a faithful forward-modulation PEPITA remains untested).

---

## 💡 New Improvement Opportunities (surfaced by A0–A2, 2026-09-06)

- **unit_rms as the new default local-update rung** — it beat both
  natural_gradient and local_adam on ePC and needs no SVD. If the A2
  sweep holds multi-seed, D16's optimizer map should add a
  unit_rms-vs-Muon matched-step column (Muon may be strictly dominated
  at small widths; orthogonalization's value would then be depth-only).
- **Momentum-EMA matters for normalized rungs** (unit_rms 34.9 vs
  natgrad 41.2 on ePC w32): the A6 co-design map should treat
  "normalize-the-momentum" (not "normalize-the-gradient") as the
  canonical magnitude family.
- **B1 credit state must implement `get_state()`/`load_state()`** — the
  new snapshot protocol is the pattern; learned-B matrices are
  credit-internal state (the carried-queue failure mode, now fixed for
  updates, must not reappear for credits).
- **`LocalAdamUpdate` direction-preservation property** (scalar
  denominator keeps pseudo-gradient direction) may be reusable as a
  credit-side normalizer for A4's `rms` mode — same math, C-axis.
- **PEPITA directional diagnosis is now quantitative**: non-Muon rungs
  explode at act_std growth ~20×/layer regardless of lr — a clean
  input for the F4 credit-channel failure map (misaligned-channel row).
- **⚠️ Standing caution (user directive 2026-09-06): do not prematurely
  condemn possibilities that may be implementation defects.** The PEPITA
  "directional failure" verdict is an observed behavior at HEAD, not a
  settled mechanism claim. Before B1's design locks it in, audit: (a) is
  `feedback_scale=0.01` Muon-specific — retune per update rung (its
  effective step shape differs from orthogonalized steps)? (b) does the
  PEPITA error path lack the per-hop normalization the ladder rungs
  implicitly removed, i.e. is the explosion in the credit channel rather
  than the update? (c) is the act-std explosion a cause or a symptom
  (compare free-settle stds at step 0 vs after training)? Muon's
  orthogonalization may have been silently acting as the missing gain
  control — in which case A5 (settle-path gain homeostasis) + unit_rms,
  not learned-B, is the honest PEPITA repair. Verdict language in
  RESULTS/docs should say "PEPITA fails under the current fixed-B
  configuration at these step shapes", not "PEPITA's direction is
  fundamentally wrong" until (a)–(c) are measured.

---

## 🏁 Completion Criteria for TODO12

**Rev 11 criteria (highlight reel — the active bar):**

0. **Suite green by construction (rev 12 gate, MET):** zero unexplained
   red — see the disposition record below.
1. **D17 pinned** — Transformer × ff_hybrid vs Transformer × backprop at
   10–20-min equivalent arms, seeds 0–2, gallery-locked; the headline
   table exists whatever the gap measures; the verdict is the
   pre-registered band (parity / competitive-with-caveat / MISS),
   never a post-hoc reading.

**Rev-12 suite disposition record (2026-09-06 — every red test fixed,
quarantined with mechanism, or retired):**
- `test_lightning_integration::test_pl_trial_runs_backprop` — **FIXED**:
  `run_pl_trial`/`run_pl_trial_with_wandb` emitted the bare `accuracy`
  key that imp-46 abolished; now return `nudged_fit_accuracy`; the
  engine's two readers prefer the parity key with legacy fallback.
- `test_readme_snippet_lock` — **FIXED (lock semantics)**: README is
  immutable by standing directive while the demo harness legitimately
  evolves (loader caps, seeds); the lock now exempts the two harness
  line shapes (`def _flatten(`, `for x, y in `) and keeps full
  verbatim teeth on every API line — any composed-system drift still
  fails.
- `test_energy_invariants::test_control_lyapunov_free_energy_decreases`
  — **QUARANTINED with mechanism** (strict xfail): the tracked history
  is the Hopfield activation energy, not the docstring's error-based
  Control-Lyapunov V; monotonicity is init-dependent and the init
  flipped when `bf0d218f` (bisected) changed import-time RNG
  consumption. Register C: rides the legacy energy-semantics pass.
- `tests/unit/core/test_stability_standalone.py` — **RETIRED**: a
  55-test verbatim mirror of `tests/unit/stability/test_stability_api.py`
  importing the `computronium_stability` distribution name that
  setuptools never actually mapped (dead pyproject override removed;
  no production consumers). Coverage retained via the api tests.
- `tests/slow/test_continual_learning.py::test_train_step_uses_joint_system_pipeline`
  — **FIXED**: asserted the bare `accuracy` key imp-46 abolished;
  updated to `nudged_fit_accuracy`.
- Verification: full suite **1772 passed / 0 failed** (60 skipped,
  32 xfailed, 1 pre-existing non-strict xpass) — 11m14s walltime.
2. ✅ **F5 pinned (rev 13) — landed as an HONEST MISS.** The
   FrontierRecord table + figure exist (`test_demo_resource_vector.py`,
   gallery-locked ×2); the accounting schema was written first; both the
   plain-update and ortho variants reported. The pre-registered ~10×
   memory / ~5× energy targets are falsified at HEAD — ff_hybrid's
   autograd realization stores more than backprop and costs ~1.3×
   FLOPs; the O(1)-memory class is real (thermo saves exactly 0).
   The miss is the finding: Claim B needs a non-autograd local-rule
   realization — that is the next capstone lever, not a silent drop.
3. **D22 pinned** — ψ-only adaptation demo with exact ‖Δθ‖=0 assert,
   lr-matched controls, the θ-fine-tuning comparison, and Task-A
   retention on the matched footing.
4. **RESULTS.md carries the three-step story** — one paragraph per step,
   numbers quoted from the pinned records only.

**Rev ≤10 criteria (repair program — status at rev 10, kept for record):**

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

**Status at rev 10 (2026-09-06):** criteria 1–4 MET (D18 pinned; A4+B1
demo-grade via F4's live cell + D18; F4 live with ratchets; A6 map
re-pinned into D16). Criterion 5: D17 user-gated (C0), P2 cells /
transformer-ePC prereq slotted C1, P-axis campaign D3, F2 verdict B5 —
all explicitly queued, none silently dropped. Criterion 6 (capstone
resource-vector table) is the remaining open deliverable.