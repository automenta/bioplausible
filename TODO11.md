# TODO11.md — Active Plan: The Library, Completed and Connected

> **Opened 2026-09-02 (draft).** Successor to [TODO10.md](TODO10.md)
> (R10 closed: D1–D7 demonstrate all six ontology axes; three consecutive
> green gate runs banked; the acceptance session — *read, run, change one
> thing, see it matter* — is available to any stranger). This plan contains
> **all remaining TODO10 work**: Register B capability pulls, the Register C
> hygiene pass, the carried registers, and the research-track spine's
> open prerequisites. Research catalog: [RESEARCH3.md](RESEARCH3.md).
> Landing-cost / wiring-hygiene work moved to [TODO.md](TODO.md) (2026-09-04).
>
> **Identity (reaffirmed from R10 v4):** Computronium is an ML library whose
> every claim is a live demonstration. Tests are the evidence system. A claim
> stands only while the current code re-demonstrates it, on demand, in under
> two minutes. Verification is continuous, not archival.
>
> **Prime directive:** *Nothing is claimed that the suite does not re-show at
> HEAD. The demo suite is the proof; the README quotes it; everything else is
> history or hypothesis.*
>
> **State: CORE COMPLETE (2026-09-04).** All planned R11 capability pulls and
> hygiene items landed at HEAD; R11.3.12 (ePC, D12) pulled 2026-09-04. R11.2.24
> (resumable trainer, fold_in RNG) pulled 2026-09-04. User-directed general-
> improvements session (2026-09-04): R11.2.23 (sample-weighted metrics +
> `val_ppl`) and R11.4.1 v1 (`SystemModule` facade) landed; `GradientCredit`
> fail-loud resolved (Watch). CP-6 opened 2026-09-04: R11.3.13 (depth
> metrics) + R11.3.11 μPC init landed; depth-frontier E-1 pilot run —
> boundary ≈ depth 8 confirmed, μPC lift unconfirmed pending multi-seed
> pilot (see Remaining Items). The
> library demonstrates all 12 capabilities (D1–D12) at demo scale;
> `comp repro` 8/8; property suite 670 passed; demo gate 13/13; gallery lock
> green; `comp gallery` renders all figures. Session 2026-09-04 (continued):
> R11.3.14 (deep Hebbian chain) pulled with a live subspace-collapse
> boundary; R11.1.10 (LazyStateDynamics) pulled with a measured
> settle-count refutation. R11.3.11 audit-driven revision: μPC verdict
> downgraded from "refuted" to OPEN after an instrument audit found two
> regime mismatches (plain-MLP vs paper's residual architecture; Euclidean
> SGD vs paper's Adam/β-grid); residual geometry landed as a capability
> and the in-regime re-test still shows no lift under our trainer —
> the jpc-faithful port is the remaining gap. Remaining items are
> explicitly **pull-based** or **deprioritized** — they land only when a
> demo, campaign, or research paragraph needs them. 2026-09-04 (continued):
> **F1 — the failure-manifesto figure (first *finding* figure, CP-6)
> landed**: the four measured depth-failure faces consolidated in one demo
> test + gallery figure + RESULTS.md paragraph (demo gate now 14/14, ~158 s).
> Later 2026-09-04: **F2** (spiking plateau, audited) and **D13** (local
> credit × Muon, single-seed) landed — demo gate 18/18, ~163 s.
> **Next-session plan queued 2026-09-05** (see "The Next-Session Plan").
> **2026-09-05 session: plan items 1–4 landed.** Item 1 — D13 multi-seed> probe (`scripts/probes/d13_multiseed.py`): the Muon lift HOLDS across
> seeds 0–4 (FF×Muon 0.838±0.009 vs FF×Euclid 0.568±0.041, per-seed
> min +0.241); multi-seed grid promoted into the D13 demo test with
> variance-aware asserts — the lift claim is multi-seed verified and
> quotable. Item 4 — **FF/PEPITA realized as distinct algorithms**
> (`local_objective` wired: "ff" = goodness contrast, legacy path
> byte-identical; "pepita" = softmax output error × fixed random inverse
> projections, closed form); realized PEPITA learns slowly at demo
> budget (0.23 Muon / 0.11 Euclid, 1 epoch), asserted as
> lift-over-own-baseline only; unit ratchet locks ff ≠ pepita gradients.
> Item 2 — **F1 ePC depth audit: the wall survives fix #1.** ePC's
> credit reaches every hidden layer at depth 8 (norms > 0 — the sPC
> last-layer-only pathology is gone) yet still walls (0.108 ≈ chance);
> the signal decays geometrically through depth
> (`scripts/probes/f1_epc_depth.py`). Candidate REAL wall; verdict OPEN
> pending the jpc-faithful trainer regime. Item 3 — **F2 homeostatic
> audit: the collapse survives fix #1.** Synaptic scaling
> (`homeostatic_scaling=True`) holds row norms at target yet the readout
> still collapses (0.36 → 0.18) — the STDP fixed point destroys class
> structure, not norm growth. OPEN pending reward-modulated STDP.
> Demo gate green after deliberate manifest re-pins; property suite
> 679 passed. **2026-09-05 (continued): Path A EXECUTED — the jpc-faithful
> regime landed as D14, and BOTH open verdicts resolved positive.** ePC
> now expresses residual geometry (free equilibrium = residual feedforward
> bitwise — the D12 invariant holds through the skip path); the PC-native
> weight gradient (Δθᵢ ∝ (∂sᵢ/∂θᵢ)ᵀ εᵢ, settled errors frozen, one
> reverse-mode sweep) + Adam + β grid + inference steps = H compose the
> paper regime. At depth 20 / width 128 residual: **μPC+β=10 generalizes
> (test 0.69–0.83 over seeds 0–2; D14 asserts live) while default init
> memorizes (train 1.00 / test ≤ 0.24)** — the μPC lift is REAL, the
> earlier "no lift" verdicts were trainer-regime artifacts — and the F1
> depth wall DISSOLVES (0.8 test at depth 20 where sPC/thermo/Euclid
> walled at chance from depth 8). The "candidate real wall" upgrade from
> earlier this session was premature in the other direction: the wall was
> regime-bound all along. F1 figure re-framed (regime boundary, not
> physics); D14 is the first composed-regime capability (D14 row).
> β=1e3 lands in the memorization corner at this scale — β is a working
> knob, not a monotone dial.
> **2026-09-05 (continued): CP-6 Path B E-1 smoke EXECUTED — the P-axis
> Pareto axes discriminate, promoted to F3.** With Path A closed, the
> session turned to the doctrine's queued Path B deliverable (the
> settling-time vs basin-stability trade-off per plasticity primitive).
> E-1 smoke first (`scripts/probes/p_axis_pareto.py`): at the A20/B20 toy
> scale the axes did NOT discriminate (compute proxy arm-invariant by
> construction, walltime launch-bound noise, basin ≈ 0.88 everywhere,
> mastery precondition unmet) — per RESEARCH3's stated risk. At the
> registered A40/B40 regime (5 seeds, CPU) the axes separate cleanly:
> routing retains what null forgets (0.272±0.027 vs 0.194±0.076, ≥ null
> 4/5 seeds) with the widest decision basins (perturbation agreement 0.50
> vs 0.42 at radius 4.0) at +22% episode latency; fast_weights pays most
> latency (+45%) and ψ-capacity (512) with NO retention benefit on this
> path — its ψ re-initializes per episode under the `train_step` contract
> (scope-honest boundary, not a verdict on fast weights). The compute
> proxy is arm-invariant (asserted): ψ-update cost is visible ONLY in
> walltime — a measured proxy gap. Promoted: `test_demo_paxis_pareto.py`
> (F3, ~11 s), gallery row + `f3_paxis_pareto.png`, RESULTS.md row.
> Walltime drift discipline: absolute latency NEVER enters `record["data"]`.
> **⚠ The attribution story in THIS paragraph is superseded by the
> mechanism audit below — read both before quoting anything.** User
> directive: Demo API publishability roadmap deferred — research first.
> **2026-09-05 (continued): F3 MECHANISM AUDIT — the attribution story was
> premature; the audit ratchets are now the claim.** User-directed
> R11.5.5a discipline applied to our own fresh landing. Findings: (1)
> RoutingPlasticity's `modulate` applies `mean(sigmoid(gate_logits))` as
> a per-sample SCALAR GAIN, and the gate-strength trace is FLAT at
> 0.502 ± 0.0006 across the whole walk — the "routing" primitive
> implements constant gain control, not routing; (2) the retention
> "advantage" is an effective-lr effect: null at lr 0.015 (≈ 0.5 gain ×
> 0.03) retains 0.294 vs routing's 0.288 — statistically identical (this
> confound also touches D3's registered mechanism reading; its
> comparative claims stand); (3) FastWeightPlasticity's `modulate`
> injects `proj(outer(x, y_target))` — target-correlated bias from the
> RAW target — into every layer; gradient-live but modest (θ gap
> ~0.9%/episode from identical inits, monotone), retention below null.
> **Audit-of-the-audit:** the first draft's "flat 1.42 fw θ divergence"
> was the per-coordinate INIT DRAW, not the modulation — caught by
> re-running with identical inits; and even the quantized latency-ratio
> bins flipped run-to-run (routing's ψ cost sits at the noise floor), so
> record["data"] now carries NO walltime at all (figure x-axis =
> ψ-capacity; latency stays a live assert + stdout). F3's record carries
> a `mechanism_audit` block and the test ratchets the audit facts (gate
> ≈ 0.5 flat, fw θ-gap monotone growth, lr-brake control) — if a
> primitive's behavior changes, the ratchets fire and force re-audit.
> **Registered gap: realize the primitives** — real per-gate routing and
> real fast-weight modulation are unpulled work; no P-axis mechanism
> claim is quotable until then. The comparative orderings and the
> Pareto-instrument discrimination (E-1 gate) stand.
> **2026-09-05 (continued): the realization gap CLOSED — the P-axis
> primitives are now their advertised mechanisms, and the audit ratchets
> were re-derived for the realized behavior.** RoutingPlasticity: per-gate
> input-projection drive (gates differentiate) + per-unit sigmoid masks in
> `modulate` (mask std 0.081 — flat scalar gain gone; flat-MLP re-spec of
> pathway gating). FastWeightPlasticity: `pipeline.py` now steps ψ on the
> first phase's SETTLED activity (settled output as post) — the raw-target
> bias is gone; modulation stays gradient-live (θ gap 3.4%/episode,
> monotone). Outcome: routing's retention advantage SURVIVES realization
> (0.273 vs null 0.194, 4/5 seeds) but the effective-lr confound survives
> too (null@0.015 retains 0.294 > routing) — ordering recorded, never a
> mechanism claim; the retention-ordering OPEN item is now lr-matched
> controls per arm. F3 ratchets re-derived (mask-std ratchet replaces the
> flatness ratchet); D3 green unchanged at fixed lr; property 679 passed;
> demo gate 19/19 + gallery lock green after a deliberate two-record
> re-pin (swap_plasticity + paxis_pareto), drift-immunity proven ×2
> consecutive runs. Probe: `scripts/probes/p_axis_realize.py`.
> **2026-09-05 (continued): fast/slow demo-gate split landed** (the
> R11.5.7 re-baseline): D14 is the slow tier (`pytest -m slow -k demo`,
> ~120 s); the fast gate (D1–D13 + F1–F3 + lock) is back to ~190 s.
> Remaining open work is the registered-scale P-axis campaign with
> lr-matched controls per arm — the first finding-grade P-axis claim.
> **2026-09-05 (continued): the lr-matched control pilot EXECUTED and
> promoted — the retention confound is CLOSED, quantitatively.** Routing's
> effective step is exactly half of null's (Δθ 0.0014 vs 0.0028/episode;
> matched null lr 0.0154 = the mask-mean-0.5 prediction) and null@matched
> retains 0.294 vs routing's 0.273 — the retention ordering is effective
> learning rate ALONE; no routing mechanism claim is quotable. Fast-
> weights' step matches null's (0.032 ≈ 0.03): its retention deficit is
> REAL. Promoted into F3 (`record["mechanism_audit"]["lr_matched_audit"]`
> + ratchets: routing matched-advantage ≤ 0.01, fw ≤ 0). Campaign-design
> deliverable extracted: retention must be measured at matched effective
> lr per arm. Probe: `scripts/probes/p_axis_lr_matched.py`. Demo gate fast
> tier + gallery lock green after deliberate F3 re-pin.
> **2026-09-05 (continued): D15 LANDED — the performance hunt's headline.**
> User directive applied: hunt for high performers across the ontology
> (probes `scripts/probes/performance_hunt*.py`, digits for cheap
> iteration — measured ~28% faster than mnist; CPU beats CUDA for MLPs,
> consistent with the standing device policy). Finding: **the U-axis
> moves the depth wall, capacity-matched** — at identical geometry and
> parameter count, BP×Euclid is at chance at depth 16 (0.114) while
> BP×Muon generalizes (0.834 ± 0.036); FF×Muon 0.930 ± 0.001 ≥ BP×Muon
> 0.911 ± 0.007 at depth 4 / width 256 — local learning beats backprop
> at matched capacity. Both Muon arms set repo records for held-out
> accuracy at this budget (previous: D13's 0.84 at width 32). Promoted
> as D15 (`test_demo_uaxis_depth_frontier.py`, slow tier ~240 s, seeds
> 0–2 on the headline pairs, single-seed depth curves per the F1
> convention, parameter counts in the record). Slow tier now = D14+D15;
> gallery lock green after deliberate re-pin, drift-immunity ×2.
> **2026-09-05 (continued): D16 LANDED — the U-axis coverage map.** Ran
> queued item (0): `performance_hunt4.py`, fixed to be **capacity-matched
> across geometries** (user correction — the first sweep's 26.5k–416k
> param spread was unfair; retuned to 47.7k–57.5k, max/min < 1.25
> asserted) and fixed to the pipeline contracts (GraphGeometry: node dim
> == batch dim per D9 — each 32-sample batch is one graph over an 8×4
> grid of batch positions; partial test batches dropped; mnist batches
> flattened once at materialization). Promoted as
> `test_demo_uaxis_coverage.py` (slow tier, ~188 s, timeout 600): Muon ≥
> Euclid on every geometry; spectral matches Muon only on attention;
> natural gradient at chance on mlp/graph/lattice, unstable on attention
> — asserted as upper bound, OPEN regime question; local ff×Muon trails
> bp×Muon on all four at this budget. Slow tier now = D14+D15+D16
> (~8 min); gallery lock green after re-pin (d16 figure rendered).
> **2026-09-05 (continued): the optimizer-family axis landed — SGD vs
> Adam was the missing coordinate.** User correction: the hunt had
> swapped orthogonalization rules but never the optimizer family — the
> ontology had NO Adam (the `EuclideanUpdate` "SGD/Adam" docstring was a
> dead claim). `AdamUpdate` implemented (Kingma & Ba, per-coordinate
> moments + bias correction, system-scoped state that fails loud on
> cross-geometry reuse — the D13 lesson) and wired per the primitive
> checklist: `ParameterUpdateConfig.adam(step_size=1e-3, beta2=, eps=)`,
> dispatch in factory/spec/joint, `SystemConfig.validate()` whitelist,
> root + ontology exports, CLI listings. Locks:
> `tests/unit/core/test_adam_update.py` (torch.optim.Adam parity,
> bias-corrected first step, state-reuse fail-loud, SGD/Adam
> distinctness). **The hunt's headline moved: Adam BEATS Muon on
> attention (0.900±0.001 vs 0.874) while Muon keeps mlp/graph/lattice —
> NO update rule dominates the map**, and Adam lifts local credit to
> parity with Muon on mlp (ff×Adam 0.899 vs ff×Muon 0.896). The
> "Muon is the robust default" reading was an artifact of sweeping only
> the SGD family. D16 extended with the adam + ff/adam columns (~265 s,
> slow tier); gallery re-pinned; property 679 passed; fast gate 19/19.
> The learning-algorithm hunt is now OPEN as a directive: credit ×
> optimizer-family cells are the new map (pepita×Adam, spectral×Adam,
> and Adam for the D14 jpc regime are unexplored).
> **2026-09-05 (continued): the hunt's first two cells resolved — the
> natural-gradient "boundary" was a step-size artifact, and the
> pepita×Adam cell is regime-dependent.** Probe
> `scripts/probes/hunt_cells.py`: (1) natural gradient is flat at chance
> across lr 0.01–1.0, but lr 1e-3 trains on ALL FOUR geometries (mlp
> 0.875 / attention 0.875 / lattice 0.888 / graph 0.342) — the update is
> mean-|grad|-normalized so its effective step IS its step_size, and the
> original D16 sweep's lr 0.1 was a ~10× overshoot that collapsed every
> geometry. D16's OPEN regime question is RESOLVED (instrument defect,
> not a learning boundary); D16 re-run with natural lr 1e-3 and the map
> re-pinned — still no rule dominates (natural joins the competitive
> set). (2) pepita×Adam: Adam partially rescues realized PEPITA at
> width 32 (0.11 → 0.17, ≈ Muon's 0.20; promoted into D13 as a
> lift-over-own-baseline ratchet) but Muon clearly dominates at width 64
> (0.31 vs 0.14) — the cell is regime-dependent, no optimizer-family
> mechanism claim. Remaining hunt cells: spectral×Adam, Muon+Adam
> hybrid (orthogonalized Adam), Adam for the D14 jpc trainer regime.
> **2026-09-05 (continued): the Muon+Adam hybrid cell LANDED as a
> library primitive — OrthoAdam dominates the coverage map's headline
> cells.** Probe `scripts/probes/hunt_hybrid.py`: Adam moments with
> Muon's SVD-polar orthogonalization applied to matrix-shaped
> first-moment directions (rescaled to Adam's step magnitude; vector
> params keep plain Adam), ortho_lr calibrated on mlp across
> {1e-3, 3e-3, 1e-2} → 3e-3. Measured (D16 regime, seeds 0–2): mlp
> 0.930 / attention 0.911 / lattice 0.924 — beats BOTH parents on all
> three — and beats Adam on graph (0.411 vs 0.332, where Muon keeps its
> win 0.433; the one geometry the hybrid does not take). Landed per the
> primitive checklist: `OrthoAdamUpdate` + `ParameterUpdateConfig.ortho_adam`
> (new `ortho_lr` field), dispatch in factory/spec/joint, validate
> whitelist, campaign evaluation thunks, CLI listings, root + ontology
> exports; locks `tests/unit/core/test_ortho_adam_update.py` (5 locks:
> matrix direction orthogonal, vector params = plain Adam, state-reuse
> fail-loud, distinct-from-Adam, spec round-trip). D16 extended with the
> ortho_adam column (~310 s slow tier); D16's headline re-framed from
> "no update rule dominates" to "OrthoAdam dominates the headline
> cells"; RESULTS.md re-pinned; gallery lock green after deliberate
> re-pin. **2026-09-05 (continued): the registered-scale OrthoAdam probe
> EXECUTED and promoted into D15 — the hybrid moves the depth frontier
> beyond Muon and rescues local credit at scale.**
> `scripts/probes/ortho_adam_scale.py` (depth/width grid, seeds 0–2,
> mnist quick 300 batches, TEST): at depth 16 / width 128 BP×OrthoAdam
> 0.878 ± 0.035 beats BP×Muon 0.834 ± 0.036 (D15's registered number
> reproduces exactly) while plain Adam partially collapses (0.303 ±
> 0.079) — Adam's second-moment normalization is itself depth-fragile
> and orthogonalizing its momentum repairs it; at depth 8 ortho 0.929 >
> muon 0.899 > adam 0.852; and FF×OrthoAdam 0.947 ± 0.002 at depth 4 /
> width 128 (~119k params) is the repo's best held-out accuracy per
> parameter at this budget class (D15's previous local record 0.930
> needed 400.9k params). D15 extended with the adam/ortho_adam headline
> arms + the three width-128 local cells (~380 s slow tier, ratchets:
> ortho > muon at d16, adam < 0.5 at d16, ff/ortho tops ff/muon at
> w128); RESULTS.md re-pinned; gallery lock green after deliberate
> re-pin.

---

## 🎯 The Next-Session Plan (queued 2026-09-05, user-directed — ordered by risk and leverage; revised same day by the R11.5.5a course correction)

The 2026-09-04 session landed F1/F2 (initially framed as live *findings*)
and D13 (a large positive claim). The 2026-09-05 course correction
(R11.5.5a above) **demotes F1 and F2 from "findings" to OPEN engineering-
defect audits**: a passing test that asserts a failure mode proves the
current regime, not physics — the "failure manifesto" posture is retired
until a wall survives its known fixes. Four workstreams gate CP-6, in
this order:

### 1. Critical vulnerability first: secure the D13 Muon claim (multi-seed audit) — ✅ LANDED 2026-09-05, outcome (a)

**Risk:** D13's 3.2× lift (≈ 0.85 vs ≈ 0.26) is the largest positive claim
in the repository and rests on ONE seed — exactly the failure mode the
μPC audit exposed (a single-seed 2× effect that evaporated under audit).

**Action:** multi-seed probe of the full D13 grid — seeds 0–4 (≥ 5, per
PR-4 / E-10 Minimum-Viable Control Set) × {bp, ff, pepita} × {euclidean
lr 0.2, muon lr 0.02}, demo regime (width 32, 150 batches, MNIST quick).
Throwaway probe first (`scripts/probes/d13_multiseed.py`); promote into
the demo test (per-seed record fields + variance-aware asserts) only if
the lift holds.

**Outcome: (a) — the lift holds.** FF×Muon 0.838 ± 0.009 vs FF×Euclid
0.568 ± 0.041 across seeds 0–4; per-seed lift min +0.241 (never below
the demo assert margin). BP×Muon 0.843 ± 0.013. The multi-seed grid for
the local arms is promoted into `test_demo_uaxis_muon_swap.py`
(`record["multi_seed"]`, asserts: min per-seed FF lift > 0.15, mean
PEPITA lift > 0.05). **Standing constraint lifted: the lift number is
multi-seed verified and quotable.**

### 2. F1 audit-fix: does ePC close the sPC depth wall? — ✅ LANDED 2026-09-05, outcome (b): the wall survives fix #1

**Reframe:** F1's "depth wall" is sPC-specific — our layered settle traps
the nudge (hidden contrast exactly 0.00), which is precisely the pathology
ePC (Goemaere et al., D12) was written to solve. The damning-conclusion
reading ("local learning dies at depth") was premature: F1 currently
demonstrates that *our sPC solver is inadequate at depth*.

**Action:** run the F1 harness with `ErrorPredictiveCodingDynamics`
swapped in — depths 2/8/20, same budget and the D12 regime (ePC trains at
1/3 settle budget; ÷β contrastive credit caps deep-stack signal, see D12
notes). Measure: (a) does the credit signal reach layer 1 (per-layer
norms, the F1 mechanism probe); (b) does ePC *learn* at depth 8–20 where
sPC walls?

**Outcomes:** ePC learns at depth 20 → F1 was a baseline defect; the
"wall" moves or dissolves, and the manifesto figure is re-framed as an
sPC-vs-ePC solver contrast (a capability story, D12-adjacent). ePC also
walls → the first candidate *real* wall — but only after the jpc-faithful
trainer regime (plan item 4A) is also applied. **F1's RESULTS.md/figure
framing is re-pinned after this audit, not before.**

**Measured (probe `scripts/probes/f1_epc_depth.py`, promoted into the F1
test as `record["arms"]["epc"]`):** ePC depth 2 learns (0.443) with
credit at every layer; depth 8 walls (0.108 ≈ sPC's 0.098) DESPITE credit
reaching every hidden layer (min hidden norm 2.96e-05 > 0 — the
last-layer-only pathology is gone); at depth 20 the signal decays
geometrically to exact 0.0 at layer 1 (~4×/layer compounding). The wall
is upgraded from "solver inadequacy" to **candidate real wall**, with
fix #2 (jpc-faithful trainer: Adam, β grid, steps=H) the remaining
instrument gap. Asserts: ePC hidden norms > 0 at depth 8 (mechanism
live), ePC depth-2 > 0.3 (positive control), ePC depth-8 ≤ sPC + 0.05
(wall persists, OPEN message). RESULTS.md F1 paragraph re-pinned.

### 3. F2 audit-fix: does homeostatic scaling stop the STDP collapse? — ✅ LANDED 2026-09-05, outcome (b): the collapse survives fix #1

**Reframe:** F2's silent-network confound was an implementation defect
(fixed); the remaining "collapse" is the known pathology of *naive,
unconstrained* STDP — runaway potentiation without gain control. It is a
missing term in our local rule, not a neuromorphic limit.

**Action:** add a homeostatic mechanism to the timing-STDP path
(`TemporalTraceCredit` + spike update) — synaptic scaling
(normalize incoming weight rows to a target norm) and/or an Oja-style
decay term `− η·E[y²]·w` (the DeepHebbianChain recipe, R11.3.14), config-
gated so the naive rule stays available for contrast. Re-run the F2
feature probe: centroid readout random-init vs homeostatic-STDP-trained.

**Outcomes:** collapse stops and readout ≥ random-init → F2 was a missing
term; the honest spiking story becomes "STDP + homeostasis is trainable"
(a capability path) and the refutation slot re-opens for a properly
controlled rule. Collapse persists under homeostasis → a real candidate
finding, but only alongside the supervised-error-term audit (reward-
modulated STDP) — no boundary verdict before then.

**Measured (promoted into the F2 test as `record["homeostatic_audit"]`):**
synaptic scaling implemented on the timing-STDP path
(`CreditAssignmentConfig.temporal_trace(homeostatic_scaling=True,
homeostatic_target=...)` — descent zero exactly at ||row|| = target).
Scaling is LIVE (row norms held at target: 5.94 vs target 5.7, 7.15 vs
8.0) yet the centroid readout still collapses 0.36 → 0.18 at every
target tried (0.5/1.0/2.0/init-norm/8.0). Gain control is necessary but
not sufficient: the STDP fixed point itself destroys class structure.
Verdict stays OPEN pending the reward-modulated STDP audit. RESULTS.md
F2 paragraph re-pinned.

### 4. Ontology debt: realize FF vs PEPITA (byte-identity gap) — ✅ LANDED 2026-09-05

**Problem:** D13's audit found `local_objective` is dead config — no
credit reads it, so FF and PEPITA run byte-identical pseudo-gradients.
"You cannot claim local credit × Muon if the two primary local credit
algorithms in the factory table are indistinguishable in the codebase."

**Action:** implement the actual distinction in the credit layer (branch
`LocalGoodnessCredit` or split classes):
- **FF:** per-layer goodness threshold — layer-local loss, no inverse pass;
- **PEPITA:** forward differential + inverse propagation modulation.
Keep the D13 record's identical-numbers evidence in the history (the gap
was real); wire `local_objective` (or a new field) so the configs mean
something; property suite gates the change (credit semantics change).

**Then:** re-run the D13 multi-seed grid over the *realized* rules — does
Muon's lift apply to both local mechanisms, or is it specific to one?
That answer is itself D-table material.

**Landed 2026-09-05:** `local_objective` is now
`Literal["ff","pepita"]` and is READ by `LocalGoodnessCredit`:
- **"ff"** = the legacy autograd goodness-contrast path (byte-identical
  to the old behavior; default everywhere, so existing records hold);
- **"pepita"** = closed-form error-modulated update: e = onehot(y) −
  softmax(free_out); per-layer err = e @ B (B: fixed random orthogonal
  (out,width), crc32(name)-seeded, scaled by `feedback_scale`); grad =
  −errᵀ @ nudged_pre / batch. PEPITA's e = y − ŷ in probability space is
  load-bearing — the raw nudged differential β·(onehot − logits) is
  dominated by the constant one-hot term and learns nothing.
- Surfaces: `create_pepita_mlp` + `create_native_pepita_mlp` now pass
  "pepita" (parity test green); the stale
  `test_native_pepita_mlp` xfail ("returns empty pseudo-gradients")
  removed — it XPASSes live. D13 demo `_CREDITS` uses the realized
  rules; new ratchet asserts `ff/euclid − pepita/euclid > 0.2` (fires if
  the two ever re-collapse to identical numbers).
- Re-run D13 multi-seed over realized rules: the Muon lift is FF-specific
  in magnitude — FF 0.838 vs realized-PEPITA 0.226 (Muon, 1 epoch; 0.30
  at 3 epochs). Realized PEPITA learns (beats its Euclid baseline by
  +0.12 mean) but is far slower at demo budget — asserted only as
  lift-over-own-baseline, never FF-parity. D-table material recorded.

### 5. CP-6 path choice: close the jpc loop (A) or commission the P-axis campaign (B) — RECOMMENDATION RECORDED 2026-09-05: Path A

With F1 landed, CP-6's doctrine offers two doors — **decide explicitly,
don't drift**:

- **Path A — close the jpc loop (μPC verdict).** Architecture is in the
  library (`GeometryConfig.residual`); the missing piece is the trainer
  regime: Adam on weights + activity GD with a β grid (paper 1e3→1e-2) +
  inference steps = H (a `JPCFaithfulTrainer` or `SystemTrainer` config
  support). Then the depth-8/width-128 residual sweep. Lift under the
  paper's exact regime → massive positive finding for deep local
  learning; still no lift → the depth wall is a verified constraint of
  the algorithm, not an optimizer artifact. Either terminus closes
  R11.3.11 permanently.
- **Path B — commission the P-axis Pareto campaign (the core CP-6
  deliverable).** Draft the campaign YAML: pin S/G/D/C/U at the flagship
  coordinate, sweep P ∈ {Null, Routing, FastWeight}; wire the PR-5
  stability guard (`probe_interval_for_overhead`) into the loop; GPU per
  the doctrine. Goal: the first *finding* figure of the P-axis — the
  settling-time (latency) vs basin-stability trade-off curve per
  plasticity primitive, the figure that justifies the 6-D ontology to
  hardware researchers.

**Sequencing note:** item 1 blocks quoting D13 anywhere; item 2 changes
what "local credit" means before any registered-scale claim uses it; the
A/B decision is orthogonal to both and can be made while 1–2 run. Check
`RESEARCH3.md` protocol (E-1 smoke → pilot → full) governs whichever
campaign path is chosen.

**2026-09-05 session recommendation (recorded, user ratifies): Path A —
close the jpc loop.** Items 2 and 3 both ended at "the wall/collapse
survives known fix #1; the remaining gap is the jpc-faithful trainer
regime." One landing now resolves TWO OPEN verdicts (μPC lift AND the
depth wall) with the same instrument. Path B (P-axis Pareto) stays the
CP-6 deliverable and follows immediately — with F1/F2 verdicts final,
the failure figure is grounded before the campaign commissions.
Concretely next: `JPCFaithfulTrainer` or `SystemTrainer` config support
(Adam on weights + activity GD, β grid 1e3→1e-2, inference steps = H,
width 512 residual), then the depth-8 sweep; check the output
premultiplier a_L = N^{-1} CE-softmax subtlety first (see Notes).

---

## 📜 Standing Directives (carried, binding)

These are session-established user directives and measured facts. They bind
every workstream below.

- **`benchmark_results/` stays untracked and gitignored — never re-add it**
  (user directive 2026-09-02, superseding earlier TODO10 language).
- **README: never edit it** (user directive 2026-09-03). The README/snippet
  drift-lock machinery is retired: `test_readme_snippet_lock` stays red at
  HEAD by directive and is not a gate. Evidence lives in `docs/RESULTS.md`
  and the gallery.
- **Test-execution discipline (2026-09-02):** never run tests without showing
  output and walltime (`--durations` in addopts; pipe through `tail`/`grep`,
  never silent `head`-truncation). Minimize redundant test executions:
  measure levers in throwaway scripts before touching tests.
- **Lint/type debt is deprioritized (2026-09-03):** ruff sits clean and stays
  clean passively (per-line markers self-flag on touch); pyright runs only
  on genuinely new modules when it adds signal. R11.2.2 and remaining
  lint-adjacent items are as-touch work, never a workstream. Real
  development progress is the priority.
- **Device policy (measured 2026-09-02, RTX 3080):** the demo suite stays on
  **CPU** — tiny Digital builds (784→32→10, batch 64, Python settle loop)
  are kernel-launch-bound, and CUDA ran *slower* (D2 hit 60 s timeout).
  GPU-first applies where work is FLOP-bound: registered-scale studies,
  campaign fleets, large hidden dims, long horizons. Rule: *prefer GPU where
  appropriate — measured, not assumed* (AGENTS.md), with the demo-suite CPU
  verdict as the standing counter-example.
- **DataLoader workers:** `num_workers=2` measured faster at demo scale
  (13.2 s vs 20.7 s per epoch). `num_workers=0` is the *flake* mitigation
  (D7 precedent), not a speed rule.
- **GitHub CI is not yet in use** (2026-09-02): the gates that matter are the
  locally runnable invocations recorded in this plan; workflow edits are
  bookkeeping, not acceptance criteria.

---

## 🎯 The Demonstration Table (D1–D12)

| #  | Capability                                                          | Demo test                                      |
|----|---------------------------------------------------------------------|------------------------------------------------|
| D1 | Six-axis composition is real                                        | `test_demo_compose_6axis.py`                   |
| D2 | One trainer, every credit rule                                      | `test_demo_swap_credit.py`                     |
| D3 | The P-axis swap matters                                             | `test_demo_swap_plasticity.py`                 |
| D4 | The memory profiler is honest                                       | `test_demo_memory_budget.py`                   |
| D5 | Frozen θ is a guarantee, bitwise                                    | `test_demo_z3_frozen_theta.py`                 |
| D6 | The substrate axis is physical (memristive IR-drop + neuromorphic spike dropout, five arms) | `test_demo_substrate_swap.py` |
| D7 | The D-axis settles in time                                          | `test_demo_spike_settle.py`                    |
| D8 | The G-axis is a swap (capacity-matched conv vs flat)                | `test_demo_geometry_swap.py`                   |
| D9 | The G-axis is a swap (capacity-matched graph vs flat, structural generalization) | `test_demo_graph_geometry_swap.py` |
| D10| The G-axis is a swap (capacity-matched attention vs flat, permutation sensitivity) | `test_demo_attention_geometry_swap.py` |
| D11| The G-axis is a swap (capacity-matched 3D lattice vs flat — 206,090 vs 203,530 params, a 1.25% gap after reducing the larger lattice arm — spatial noise robustness) | `test_demo_spatial_lattice_geometry_swap.py` |
| D12| The D-axis settles without signal decay (ePC: free equilibrium = feedforward bitwise, nudged signal reaches every layer, 1/3 settle budget) | `test_demo_epc_fast_settle.py` |
| D13| The U-axis is a swap (local credit × Muon: FF×Muon 0.838±0.009 over 5 seeds vs 0.568±0.041 on Euclidean; FF and realized PEPITA are distinct algorithms; SVD polar factor + momentum-orthogonalization locked) | `test_demo_uaxis_muon_swap.py` |
| D14| Depth 20 trains under the jpc-faithful regime (ePC + PC-native weight gradient + Adam + β grid + steps=H): μPC generalizes (test ≈ 0.83) where default init memorizes (train 1.00 / test ≤ 0.24); the F1 depth wall is regime-bound — **slow tier** (`pytest -m slow -k demo`) | `test_demo_jpc_faithful_depth.py` |
| D15| The U-axis moves the depth wall, capacity-matched (identical geometry, swapped update/credit, mnist quick 300 batches, TEST acc): depth 16/width 128 (349,450 params each) BP×Euclid 0.114±0.000 (chance) / BP×Adam 0.303±0.079 / BP×Muon 0.834±0.036 / **BP×OrthoAdam 0.878±0.035 — the hybrid moves the frontier beyond Muon**; depth 4/width 256 (400,906 params both) FF×Muon **0.930±0.001** ≥ BP×Muon 0.911±0.007 — local learning beats backprop at matched capacity; width-128 local cells — **FF×OrthoAdam 0.947±0.002** (~119k params, repo-best acc/param at this budget) > FF×Adam 0.939 > FF×Muon 0.920; BP degrades gracefully under Muon through depth 16 where Euclid cliffs — **slow tier** | `test_demo_uaxis_depth_frontier.py` |
| D16| The U-axis coverage map, capacity-matched across geometry (4 geometries 47.7k–57.5k params × 6 update rules × seeds 0–2): **OrthoAdam (the Muon+Adam hybrid, a library primitive) dominates the headline cells — mlp 0.930 / attention 0.911 / lattice 0.924, beating BOTH parents, and beats Adam everywhere** (graph 0.411, where Muon keeps its win 0.433); Muon ≥ Euclid on EVERY geometry; Euclidean/Adam/Muon/Natural are distinct optimizer families (natural's early "chance" cell resolved 2026-09-05 as a step-size artifact — at its working lr 1e-3 it learns on all four geometries); spectral geometry-conditioned (attention only); local ff×Muon trails bp×Muon on all four at this budget — **slow tier** | `test_demo_uaxis_coverage.py` |
| F3 | The P-axis Pareto, REALIZED (CP-6 E-1 → finding instrument → mechanism audit → realization): per-gate input-projection drive + per-unit sigmoid masks (mask std 0.081; flat 0.5 scalar gain gone) and settled-activity fast weights (raw-target bias gone, θ gap 3.4%/episode monotone); retention ordering survives realization but the effective-lr confound survives too (null@0.015 > routing) — orderings recorded, lr-matched controls the OPEN item; realized-mechanism ratchets locked live | `test_demo_paxis_pareto.py` |

---

## ✅ Completed This Session (2026-09-05 — learning-algorithm hunt, first cells)

| Item | Description | Key Evidence |
|------|-------------|--------------|
| **Natural-gradient regime question RESOLVED** | The D16 "natural gradient at chance" cell was a step-size artifact, not a learning boundary (R11.5.5a applied before verdicting): `NaturalGradientUpdate.step` divides each tensor's gradient by its mean |grad| — effective step ≈ step_size — so the sweep's lr 0.1 was a ~10× overshoot that collapsed every geometry. At lr 1e-3 it learns on ALL four geometries | Probe `scripts/probes/hunt_cells.py` (chance across lr 0.01–1.0) + micro-sweep (1e-4 → 0.805, 1e-3 → 0.875 on mlp). D16 re-run with natural lr 1e-3: mlp 0.875 / attention 0.875 / lattice 0.888 / graph 0.342; map re-pinned, slow tier green (~259 s), fast gate 19/19 + lock green after deliberate D16 re-pin. Follow-up (as-touch): the update is a normalized-gradient placeholder, not a Fisher update — rename or implement diag-Fisher before any natural-gradient mechanism claim |
| **Muon+Adam hybrid LANDED: `OrthoAdamUpdate` dominates the map** | Probe `scripts/probes/hunt_hybrid.py` first (ortho_lr calibrated on mlp across {1e-3, 3e-3, 1e-2} → 3e-3), then the primitive landed per the update-primitive checklist. OrthoAdam = Adam moments + Muon's SVD-polar orthogonalization of matrix-shaped first-moment directions (rescaled to Adam's step magnitude; vector params plain Adam). Measured (D16 regime, seeds 0–2): mlp 0.930±0.002 / attention 0.911±0.003 / lattice 0.924±0.003 — beats BOTH parents — and beats Adam on graph (0.411 vs 0.332; Muon keeps graph 0.433) | `computronium/ontology/update.py` (`OrthoAdamUpdate`, `ParameterUpdateConfig.ortho_adam` + `ortho_lr` field); dispatch factory/spec/joint (spec chain deduplicated into a `_UPDATE_CLASSES` dispatch map); `SystemConfig.validate()` whitelist; campaign evaluation thunk; CLI listings; root + ontology exports; locks `tests/unit/core/test_ortho_adam_update.py` (5 locks). D16 extended with the ortho_adam column (~310 s slow tier, asserts: ortho > adam everywhere, ortho ≥ muon − 0.03); headline re-framed "OrthoAdam dominates the headline cells"; RESULTS.md re-pinned; gallery lock green after deliberate re-pin |
| **Registered-scale OrthoAdam probe → promoted into D15** | The hybrid's dominance survives depth: at depth 16 / width 128, BP×OrthoAdam 0.878±0.035 > BP×Muon 0.834±0.036 (D15's number reproduces exactly) while BP×Adam collapses to 0.303±0.079 — **Adam's second-moment normalization is itself depth-fragile; orthogonalizing its momentum repairs it**. FF×OrthoAdam 0.947±0.002 at depth 4 / width 128 (~119k params) is the repo's best held-out accuracy per parameter at this budget class | Probe `scripts/probes/ortho_adam_scale.py` (depth/width grid, seeds 0–2); D15 extended with the bp/adam + bp/ortho_adam d16 headline arms and the three width-128 ff local cells (~380 s slow tier, timeout raised to 900); ratchets: ortho > muon at d16, adam < 0.5 at d16, ff/ortho > ff/muon at w128 with per-seed floor 0.93; RESULTS.md D15 row re-pinned; gallery lock green after deliberate re-pin |
| **pepita×Adam cell** | Regime-dependent: Adam partially rescues realized PEPITA at width 32 (0.110 → 0.173, ≈ Muon's 0.202) but Muon clearly dominates at width 64 (0.31 vs 0.14) — no optimizer-family mechanism claim; asserted in D13 as lift-over-own-baseline only | `tests/integration/test_demo_uaxis_muon_swap.py` new hunt arm `pepita/adam` (ratchet: adam > euclid + 0.03) + w64 probe numbers in `scripts/probes/hunt_cells.py`; D13 record re-pinned, fast gate 19/19 + lock green, D13 drift-immunity re-proven |

## ✅ Completed This Session (2026-09-05 — plan items 1–4)

| Item | Description | Key Evidence |
|------|-------------|--------------|
| **Item 1** | D13 multi-seed audit — **outcome (a): the Muon lift holds** | `scripts/probes/d13_multiseed.py` (seeds 0–4 × 6 arms): FF×Muon 0.838±0.009 vs FF×Euclid 0.568±0.041, per-seed lift min +0.241; BP×Muon 0.843±0.013. Multi-seed grid promoted into the D13 demo test (`record["multi_seed"]`, variance-aware asserts) |
| **Item 2** | F1 ePC depth audit — **the wall survives fix #1** | `scripts/probes/f1_epc_depth.py` + `record["arms"]["epc"]` in the F1 test: ePC credit reaches every hidden layer at depth 8 (norms > 0) yet walls (0.108 ≈ chance); signal decays ~4×/layer. Candidate real wall; OPEN pending jpc-faithful regime |
| **Item 3** | F2 homeostatic audit — **the collapse survives fix #1** | `CreditAssignmentConfig.temporal_trace(homeostatic_scaling=True, homeostatic_target=…)` implements synaptic scaling on the timing-STDP path; `record["homeostatic_audit"]`: norms held at target (5.94 vs 5.7), readout still collapses 0.36→0.18 at every target tried. OPEN pending reward-modulated STDP |
| **Item 4** | FF/PEPITA realized as distinct algorithms | `local_objective: Literal["ff","pepita"]` READ by `LocalGoodnessCredit` — "ff" = legacy goodness contrast (byte-identical); "pepita" = softmax output error × fixed random inverse projections (crc32(name)-seeded, `feedback_scale`-scaled), closed form. `create_pepita_mlp`/`create_native_pepita_mlp` wired; stale pepita xfail removed (XPASSes live); unit ratchet `TestLocalGoodnessRealization` (ff ≠ pepita, deterministic feedback); D13 ratchet `ff/euclid − pepita/euclid > 0.2`. Realized-PEPITA D13 multi-seed: 0.226 Muon / 0.106 Euclid (1 epoch) |
| **Item 5** | CP-6 Path A EXECUTED — **jpc-faithful regime landed as D14; both open verdicts resolved positive** | ePC residual support (`_build_forward_with_errors` now carries the skip; lock: residual free-eq == feedforward bitwise + nudged reaches all hidden layers, `test_residual_geometry.py` 6 tests). Manual jpc loop: ePC settles (steps = H, β grid) → PC-native weight gradient with frozen ε → Adam. Smoke (depth 8): 1.000 train at β ≥ 100 both inits. Pilot (depth 20, seeds 0–2): μPC β=10 test 0.686/0.828/0.831 (train 0.91–0.92) vs default β=10 test 0.142/0.237/0.234 (train 0.997 — memorization) and β=1e3 degenerate (test ≈ 0.09–0.36). Probe: `scripts/probes/jpc_faithful.py`. Demo: `test_demo_jpc_faithful_depth.py` (~107 s, timeout 600) with gallery `DEMOS["jpc_faithful_depth"]` + `_fig_jpc_faithful_depth`. μPC-lift verdict: **REAL at depth 20 under the faithful regime**; F1 depth wall: **regime-bound, dissolved**. Property 679 passed; demo gate + gallery lock green after re-pin |

Property suite 679 passed; demo gate 17/18 + gallery lock green after
deliberate manifest re-pins (D13/F1/F2 record changes are intended);
credit-semantics change gated by the full property suite.

## ✅ Completed This Session (2026-09-05 — CP-6 Path B, research)

| Item | Description | Key Evidence |
|------|-------------|--------------|
| **F3** | The P-axis Pareto, E-1 smoke → promoted → **mechanism-audited the same session** (see header). Final state: comparative orderings + audit ratchets locked live; no mechanism claim | Probe `scripts/probes/p_axis_pareto.py` (E-1), audit `scripts/probes/p_axis_mechanism_audit.py` (gate trace, lr control, θ divergence — incl. the init-draw artifact correction), demo `tests/integration/test_demo_paxis_pareto.py` (~11 s: 4 walks + gate trace + θ divergence). Gallery `DEMOS["paxis_pareto"]` → `_fig_declared` (ψ-capacity vs retention scatter + basin curves), RESULTS.md audited row. Record drift discipline evolved twice: absolute walltime out, then quantized ratios out too — record["data"] carries zero walltime; drift-immunity proven ×2 consecutive runs |

## ✅ Completed This Session (2026-09-05 — P-axis realization)

| Item | Description | Key Evidence |
|------|-------------|--------------|
| **Realize the P-axis primitives** | The F3 audit's registered gap CLOSED. `RoutingPlasticity`: fixed per-gate input→gate projection drive (gates differentiate across units/samples) + `modulate` applies per-unit sigmoid masks via per-layer fixed gate→unit projections (seeded by layer index; `to(device)` moves them). `FastWeightPlasticity` + `pipeline.py`: ψ steps ONCE per episode on the first phase's SETTLED activity (settled output passed as post) instead of the raw pre-settle target | Probe `scripts/probes/p_axis_realize.py` (registered A40/B40, 5 seeds): routing mask std 0.081 (real per-unit gating), retention 0.273±0.028 vs null 0.194±0.076 (4/5 seeds), mastery 0.398 (masters A slower); fw θ gap 0.034 monotone, retention 0.155 (below null); null@0.015 retained 0.294 — the effective-lr confound survives realization. F3 test re-audited: mask-std ratchet (> 0.05) replaces the flatness ratchet; record `mechanism_audit` gains `per_unit_mask_std`; D3 green unchanged. Property 679 passed; `test_psi_engagement` routing control now asserts exact inequality (toy-scale per-episode ψ effect ~1e-7 is below approx tolerance — documented at the assert); demo gate 19/19 + gallery lock green after deliberate re-pin of swap_plasticity + paxis_pareto, drift-immunity ×2 |
| **Fast/slow demo-gate split** (R11.5.7 re-baseline) | D14 marked `pytest.mark.slow` — the suite's first slow-tier resident. Fast gate (default invocation) = D1–D13 + F1–F3 + gallery lock, **19 passed in ~188 s** (was ~300 s); slow tier = D14 + D15 (~6 min total). Gate Commands + R11.5.7 re-baselined; gallery lock unaffected (hashes on-disk records regardless of markers) | `tests/integration/test_demo_jpc_faithful_depth.py`; both tiers verified green same-day |
| **D15 — performance hunt: Muon moves the depth wall** (user-directed: hunt high performers, capacity-matched) | Probes (`scripts/probes/performance_hunt.py` mnist grid, `performance_hunt2.py` digits sweeps — digits ~28% faster per step, CPU beats CUDA for MLPs) found the unexplored cell: Muon × depth. Promoted as `test_demo_uaxis_depth_frontier.py` (slow tier, ~240 s): capacity-matched headline pairs (seeds 0–2, variance-aware asserts) + single-seed depth curves; record carries per-arm parameter counts — no capacity confound by construction | See D-table D15 row. Repo-record held-out accuracy: FF×Muon d4 w256 0.930±0.001, BP×Muon d16 w128 0.834±0.036 at chance-matched Euclid 0.114 |
| **MEP Newton–Schulz wired as opt-in; whitening mechanism found** (user prompt: "are we using the MEP kernels?") | `ortho_steps` was dead config — the pipeline's Muon always ran the full SVD. Now: `ortho_steps=0` (default) = exact SVD polar factor; `>0` = canonical Muon NS (`newton_schulz5`, quintic coefficients, replaces the naive under-converging iteration in `core/optimization/strategies/update.py`; fp32 — bf16 is catastrophically slow on CPU). **Finding: the FF×Muon lift is whitening-driven** — NS at 5 steps preserves BP×Muon (0.868) but collapses FF×Muon to 0.29 (width 32, 5 seeds); SVD default restored, D13/D15 claims intact. Conv×Muon also probed: conv/ff/muon wide 0.971±0.002 on digits, CUDA 5.3× faster than CPU for conv (data must be pre-moved to device) | `computronium/ontology/update.py`, `core/optimization/strategies/update.py`; probes `performance_hunt3.py`; D13/D15 re-pinned under SVD default, slow tier + lock green ×2 |
| **Multi-geometry × multi-update coverage map QUEUED** (user directives: don't fixate on conv, don't put all eggs in the Muon basket) | Written, NOT yet run: `scripts/probes/performance_hunt4.py` — the capacity-identical U-axis sweep the ontology claims but never measured: 4 geometries {ff, attention, graph (real 28×28 pixel-grid edges), spatial-lattice 3D} × 4 update rules {euclidean, muon, spectral, natural} with bp credit, plus ff/muon as the local reference; mnist quick 150 batches, seeds 0–2, test acc. Run it FIRST next session; promote discriminating cells per the D8–D12 capacity-matching convention; keep the sweep breadth-first (more geometries × updates) before giving any single cell budget-scaling | `scripts/probes/performance_hunt4.py` |
| **lr-matched control pilot** (P-axis E-1, closes the retention confound) | Per-arm effective step measured from identical inits (θ displacement per episode); null lr matched by log-interp on a displacement grid; full 5-seed walk at the matched lr. **Verdict: routing's retention advantage is effective-lr ALONE** — matched null lr 0.0154 (exactly the mask-mean-0.5 prediction), null@matched retains 0.294 vs routing 0.273 (advantage −0.020). Fast-weights' step matches null's (matched lr 0.032): its deficit (0.155 vs 0.184) is REAL. Promoted into the F3 test: `record["mechanism_audit"]["lr_matched_audit"]` + `_assert_lr_matched` ratchets (routing ≤ 0.01, fw ≤ 0). **Campaign-design deliverable: the registered P-axis campaign must measure retention at matched effective lr per arm** — the retention axis at nominal lr is confounded by construction | `scripts/probes/p_axis_lr_matched.py` → `tests/integration/test_demo_paxis_pareto.py`; fast gate + gallery lock green after deliberate F3 re-pin |
| **D16 — the U-axis coverage map** (queued item 0: `performance_hunt4.py`) | Capacity-matched (user correction: first sweep was 26.5k–416k params — unfair; retuned mlp 55.1k / attention 50.0k / graph 47.7k / lattice 57.5k, spread < 1.25× asserted) 4 geometries × updates × seeds 0–2, bp credit + ff/{muon,adam} reference. **Muon ≥ Euclid on every geometry** (lifts +0.068/+0.038/+0.102/+0.050); spectral competitive only on attention (0.879 vs muon 0.874); natural gradient at chance on mlp/graph/lattice, unstable on attention 0.377±0.114 (OPEN regime question, asserted < 0.6); local ff×Muon trails bp×Muon on all four at this budget. Pipeline-contract fixes: GraphGeometry node-dim==batch-dim (D9 semantics — 8×4 grid over batch positions), partial test batches dropped, mnist flattened once at materialization. Slow tier now D14+D15+D16 (~9 min) | `scripts/probes/performance_hunt4.py` → `tests/integration/test_demo_uaxis_coverage.py` (~265 s, timeout 600) + gallery `DEMOS["uaxis_coverage"]` (`_fig_declared` heatmap) + RESULTS.md row; gallery lock green after re-pin |
| **AdamUpdate — the SGD-vs-Adam optimizer-family axis** (user correction: "are we forgetting there's a difference between SGD and Adam?") | The ontology had NO Adam — the `EuclideanUpdate` "SGD/Adam" docstring was a dead claim; the hunt had swapped orthogonalization rules but never the optimizer family. Landed: `AdamUpdate` (per-coordinate m/v moments, bias correction, global-norm clip shared with Euclidean, system-scoped state that fails loud on cross-geometry reuse); `ParameterUpdateConfig.adam(step_size=1e-3, momentum=β1, beta2=, eps=, grad_clip=)`; dispatch in `factory.py`/`spec.py`/`joint.py`; `SystemConfig.validate()` whitelist; root `_LAZY`+`__all__`+TYPE_CHECKING + ontology exports; CLI listings. **Findings: Adam BEATS Muon on attention (0.900±0.001 vs 0.874) — no update rule dominates the map**; Adam mid-pack on mlp (0.892) and lattice (0.895), ≈ Euclid on graph (0.332); **Adam lifts local credit to parity on mlp (ff×Adam 0.899 ≈ ff×Muon 0.896)** — the "Muon is the robust default" reading was an SGD-family-only artifact. D16 extended (adam + ff/adam columns, ~265 s); gallery re-pinned; property 679 passed; fast gate 19/19 | `computronium/ontology/update.py` (`AdamUpdate`, `ParameterUpdateConfig.adam`); `tests/unit/core/test_adam_update.py` (4 locks: torch.optim.Adam parity, bias-corrected first step, state-reuse fail-loud, SGD/Adam distinctness); probe `scripts/probes/performance_hunt4.py` |

**Unblocked next (pull order):** (0) ~~run `performance_hunt4.py`~~
**DONE 2026-09-05 — promoted as D16** (see header). Follow-ups opened by
the map: budget-scale the discriminating cells (Muon is the robust
default everywhere; spectral's attention corner is the only competitor);
(0b) **the learning-algorithm hunt (user directive 2026-09-05) — first
cells RESOLVED 2026-09-05** (see header):
- ~~pepita×Adam~~ **DONE** — regime-dependent: Adam partially rescues
  realized PEPITA at width 32 (0.11→0.17 ≈ Muon's 0.20, promoted into
  D13 as a lift-over-own-baseline ratchet); Muon dominates at width 64
  (0.31 vs 0.14, `hunt_cells.py`). No mechanism claim.
- ~~natural-gradient regime question~~ **DONE** — step-size artifact.
  The update is mean-|grad|-normalized (effective step ≈ step_size);
  lr 0.1 was a ~10× overshoot. At lr 1e-3 it learns on all four
  geometries (D16 re-run, map re-pinned). Probe: `hunt_cells.py` +
  the lr micro-sweep (1e-4 → 0.805, 1e-3 → 0.875, ≥0.01 → chance).
  Follow-up (as-touch): the "natural gradient" is a normalized-gradient
  placeholder, not a Fisher update — rename or implement diag-Fisher
  before any natural-gradient mechanism claim.
- **Still OPEN:** ~~Muon+Adam hybrid~~ **DONE — landed as `OrthoAdamUpdate`,
  dominates the headline cells (see session table)**; remaining:
  spectral×Adam (ill-posed as a credit×family cell — spectral is an
  update rule; the honest remaining cells are hybrid variants, e.g.
  NS-orthogonalized Adam instead of SVD), Adam for the D14 jpc regime
  (`ParameterUpdateConfig.adam` native trainer config — the manual loop
  already uses Adam); the graph arm is semantically weak (D9 node==batch
  contract — needs an R11.2.9-class latent-graph path before it can join
  the map honestly).
- Budget-scaling note (hunt): ~~OrthoAdam at demo budget is the new
  record-holder on mlp~~ **DONE 2026-09-05 — promoted into D15** (see
  session table): the hybrid beats Muon at depth 16, plain Adam
  collapses there, and FF×OrthoAdam 0.947 at ~119k params is the
  repo-best acc/param. Natural next pulls: NS-orthogonalized-Adam
  variant (SVD per step is the deep-sweep cost center — NS is Muon's
  cheaper recipe; measure whether it preserves the OrthoAdam lift, the
  D13 whitening question for the hybrid); the OrthoAdam×D14 jpc regime
  (the manual loop's torch.optim.Adam could become OrthoAdam — does
  orthogonalized-momentum Adam lift μPC's depth-20 regime further?);
  GPU registered-scale confirmation when a conv/large-width study
  commissions.
- (1) ~~lr-
matched controls per arm~~ **DONE 2026-09-05** — the pilot quantified the
confound completely
(routing's advantage = effective-lr alone; fw's deficit real) and the
campaign-design requirement is extracted: measure retention at matched
effective lr per arm. The **registered-scale P-axis campaign (Path B
full)** is the remaining deliverable — its manifest must pin the
matched-lr protocol; (2) ~~the fast/slow demo-gate split~~ **DONE
2026-09-05** — D14 is slow-tier marked (`pytest.mark.slow`), the fast gate
(D1–D13 + F1–F3 + lock) is ~190 s, the slow tier (`-m slow -k demo`) runs
D14 in ~120 s; R11.5.7 + Gate Commands re-baselined; (3) persistent-ψ
across batches (`train_step` contract change) only if a registered
fast-weight memory claim needs multi-batch episodes.

---

## ✅ Completed This Session (2026-09-04)

### R11.1 — Capability Pulls (Register B)

| Item | Description | Key Evidence |
|------|-------------|--------------|
| **R11.1.1** | Neuromorphic spike dropout (D6 five-arm) | `SubstrateConfig.neuromorphic(sparsity=...)`, `create_neuromorphic_mlp` |
| **R11.1.2a** | ConvGeometry (D8) | im2col via substrate forward; capacity-matched (3,940 vs 3,818 params) |
| **R11.1.2b** | GraphGeometry (D9) | Adjacency message passing; capacity-matched (1.3k vs 1.7k params) |
| **R11.1.2c** | AttentionGeometry (D10) | Multi-head self-attention + FFN; capacity-matched (~100k params) |
| **R11.1.2d** | SpatialLattice3DGeometry (D11) | 3D neural cube; capacity-matched (~200k params) |
| **R11.1.3** | Tile × dynamics matrix documented as strict xfails | 7 tile pairings with mechanism-level reasons; `comp repro` 8/8 |
| **R11.1.4** | Tile-mesh settle kernel | Flips 7 tile xfails → xpass; `test_tile_settle_kernel.py` lock |
| **R11.1.5** | Adapter shape-probing, fail-loud | `_probe_linear_dims` walks `nn.Linear` chain; raises `TypeError` on failure |
| **R11.1.6** | _TaskTrainer scheduler/tracker/safety | Cosine/step/linear/cosine_warmup; `SafetyConfig`/`SafetyWrapper`; GPU-verified |
| **R11.1.7** | Diffusion target term (nudged-Langevin) | `compute_energy_from_state(target, beta)`; fidelity probe passes |
| **R11.1.8** | Ontology facade merge | `_dynamics.py`→`dynamics/_dynamics.py`, `_substrate.py`→`substrate/_substrate.py` |
| **R11.1.9** | Timing-asymmetric STDP wired to 5-D pipeline | Spike rasters, eligibility traces, configurable threshold |
| **R11.3.11b** | Residual feedforward geometry + in-regime μPC re-test (pulled 2026-09-04, audit-driven) | **Capability landed:** `GeometryConfig.residual` (skip between equal-width hidden layers, `a_ℓ = a_{ℓ−1} + φ(W_ℓ a_{ℓ−1} + b_ℓ)`; input/output projections unscaled) — forward, `route`, settle kernel (`SubstrateSettleKernel` + compiled `_eqprop_settle_loop`), and spec round-trip all carry it. Lock: `tests/integration/test_residual_geometry.py` (5 tests): manual-trace bitwise match, eager≡compiled parity (bitwise; the initial parity "failure" was a test-side RNG-order bug — built both systems from one seed), spec round-trip, fail-loud on non-feedforward. **In-regime re-test** (`scripts/probes/mupc_residual_regime.py`): residual depth-8/width-128 MNIST, seeds 0–2, μPC 0.137 vs default 0.139 — **still no lift under our trainer**. Verdict downgraded from "refuted" to OPEN: architecture family now matches the paper, but the trainer regime still does not (paper: Adam weights, activity step β ∈ {1e3..1e-2} tuned per run, inference steps = H, width 512; ours: Euclidean SGD, β=0.5, 60 fixed settle steps). Next pull for a clean answer: jpc-faithful port (Adam on weights, large-β activity GD, steps=H) |
| **R11.1.10** | LazyStateDynamics (pulled 2026-09-04) | Rewritten as a real sequential (Gauss–Seidel) EqProp settle: per-layer in-place updates reading freshest neighbors, substrate forward-operator bottom-up, per-sweep activation cache, fail-loud on non-layered/recurrent. Wired per the primitive checklist: registry `"lazy"`, `StateDynamicsConfig.lazy()`, root `__all__`+`_LAZY`+TYPE_CHECKING, thermo-contrast validate whitelist. Lock: `tests/integration/test_lazy_dynamics.py` (5 tests, ~5 s): monotone per-sweep Hopfield energy, nudge pulls output toward target, MNIST quick-mode 150-batch training > 2.5× chance, fail-loud non-layered/recurrent, sweep-count observable. **Measured refutation:** the plan's "settle-count contrast" expected Gauss–Seidel to win in sweeps — it does NOT at demo scale (34 sweeps vs Jacobi 21 at 256→64×6→10, τ=1e-2, step 0.05); no dominance claimed |

### R11.2 — Hygiene Pass (Register C)

| Item | Description | Key Evidence |
|------|-------------|--------------|
| **R11.2.1** | Ruff baseline clean | `ruff check .` clean at HEAD; E501 disabled forever |
| **R11.2.3** | Root `PlasticityConfig` twin-class resolution | Single source in `computronium.state.transitions`; ~140 lines deleted |
| **R11.2.4** | Joint `to_spec`→`from_spec` round-trip | `TestJointSystemSpecRoundTrip` locked (recurrent + FF bitwise) |
| **R11.2.5** | `init_scale` functional | Shared `_linear_stack` helper; default≡0.1 bitwise; 0.2≡0.1×2 |
| **R11.2.7** | Energy dedup: `_state_energy_vector` | PredictiveSettling / SpikeIntegration / Diffusion share it |
| **R11.2.8** | `FrontierRecord.seed` required | Campaign record `seed: int` no default; clean break (no compat) |
| **R11.2.10** | Params-moved locks (10/10 factories) | `test_params_moved.py`; fidelity expanded 48→60 coords |
| **R11.2.11** | imp-27 resolved-by-contract | `settle` protocol docstring + AST census lock (`test_settle_caller_census.py`) |
| **R11.2.12** | Tile family fold: `equitile`→`tile` | 7 deployments, CLI `FAMILY_MAP`, metamodel, tolerances all canonical `tile` |
| **R11.2.18** | `test_scaling_invariants` xpass removed | Marker removed; now asserts `acc > 0.3` live |
| **R11.2.20** | Timebox closed | All scoped items landed; no finding class stretched past its box |
| **R11.2.21** | Zoo Registry deleted | 6 files + ~30 consumers stripped; all surfaces resolve native 5-D factories |
| **R11.2.22** | Fidelity-gate determinism | `check_coordinate_fidelity(seed, fork_rng)`; verdicts deterministic |
| **R11.2.24** | Resumable trainer (`fold_in` RNG) | `TrainerSnapshot` + `from_snapshot`; interrupted == uninterrupted **bitwise** (`tests/integration/test_trainer_resume.py`); pure `fold_in` locked by hypothesis (`tests/property/test_fold_in_rng.py`) |
| **R11.2.25** | `torch.compile` settle fast paths | `compiled=True` now covers **both** energy families: sPC layered settle (2.0× train_step, bitwise parity) and `EnergyMinimizationDynamics`/`SubstrateSettleKernel` loop (1.75× settle, parity 9.5e-7, autograd-graph parity for thermo credit locked). Eager path byte-identical when off. Locks in `test_compiled_settle.py`; probes `torch_compile_settle.py` / `torch_compile_eqprop_kernel.py` |
| **R11.2.23** | Metric aggregation contract (pulled 2026-09-04, user-directed "general improvements" session) | Trainer epoch metrics are **sample-weighted** sums (ragged final batch no longer over-weights); `validate()` reports `val_ppl = exp(mean CE)` from the same per-sample normalization. Lock: `tests/unit/core/test_trainer_metric_aggregation.py` (weighted-mean identity via delegating spy, ragged batches, ppl identity) |

### R11.3 — Research Track (RESEARCH3 Spines)

| Item | Description | Key Evidence |
|------|-------------|--------------|
| **R11.3.1** | PR-9 Campaign commissioning | Smoke kill→resume cycle at HEAD; unbuffered pre-kill trail; `records/episodes.json` |
| **R11.3.2** | PR-2 θ-audit harness | `theta_audit()` context manager; SHA-256 over name+device+dtype+bytes |
| **R11.3.3** | PR-5 Calibrated stability guard | Demo-harvest ROC within 0.005% of deployed τ; `fast_proxy` calibration-only; artifact + live lock |
| **R11.3.12** | ePC fast-settling solver (D12) | `ErrorPredictiveCodingDynamics` — error reparameterization per Goemaere et al. (arXiv:2505.20137, ICML 2026); free equilibrium = feedforward bitwise, nudged signal reaches every hidden layer (sPC's reaches none), trains at 1/3 budget; demo + gallery figure + round-trip |
| **R11.3.11** | Multi-seed depth-frontier pilot (E-1, pulled 2026-09-04; **superseded by the in-regime re-test, see below**) | `scripts/probes/mupc_multiseed_frontier.py`: depths 4/8 × seeds 0–3 × spc/default vs spc/mupc, compiled settle, 477 s. μPC lift at depth 8 absent (0.135 vs 0.133). **Instrument audit finding:** the pilot applied μPC init to a plain MLP — outside the paper's tested domain (arXiv:2505.13124 Table 1 is specified and tested on residual networks; skip connections are load-bearing for the (N·L)^{-1/2} hidden scale) |
| **R11.3.14** | Deep Hebbian chain with per-layer activity normalization (pulled 2026-09-04, user-directed plan-and-fix) | `computronium/models/native/deep_hebbian_native.py`: `DeepHebbianChain` — spectral renorm (unit gain at init) + tanh + batch Oja decay + unit-RMS activity renorm per layer; plain-torch local learning (no backprop, no nudging). Per-layer pre-renorm signal norms O(1) at depth 10/50/100 (the tile-chain runaway-gain/NaN pathology is structurally fixed); unnormalized control decays to ~1e-14. Dominant-direction 2-class readout 1.000 at every depth. **Honest boundary (R11.5.5):** 10 direction-coded classes → L1 1.00 / L10 0.52 / L100 0.20 (> 0.1 chance): activity covariance effective rank collapses 5.1→1.5 through the chain under compounding tanh distortion + renorm + Oja spectral sharpening — Sanger, gain scaling, and per-step spectral renorm do NOT rescue it. Lock: `tests/integration/test_deep_hebbian_chain.py` (8 tests, ~7 s) |
| **F1 (R11.5.5 manifesto)** | The depth-boundary figure — **upgraded to candidate REAL wall 2026-09-05** (ePC depth audit, plan item 2: the wall survives known fix #1 — ePC's credit reaches every hidden layer at depth 8 yet still walls; remaining gap = jpc-faithful trainer). Demoted from *finding* to OPEN audit 2026-09-05 (R11.5.5a); landed 2026-09-04 as a *finding*, CP-6 | `tests/integration/test_demo_failure_manifesto.py` (D-axes table gains no row — this is a finding, not a capability): four measured depth-failure faces, same pipeline, same terms, all at demo scale (~18 s incl. the ePC arm): (1) **depth wall** — backprop 0.72→0.50→0.11 across depths 2/4/8, MNIST quick 60 batches, width 32 (flat across lr 0.02–0.2 — not an lr artifact); sPC walls at chance at this budget, **but the audit found the mechanism is last-layer-only training**: per-layer credit norms exactly 0.00 for every hidden weight matrix (asserted live in the record), budget softens the wall (0.21 at 60 settle steps); **ePC arm (2026-09-05)**: depth 2 learns 0.443, depth 8 walls 0.108 despite hidden credit norms > 0 — signal decays ~4×/layer, exact 0.0 at layer 1 by depth 20; asserts: ePC hidden norms > 0 at depth 8, ePC depth-2 > 0.3, ePC depth-8 ≤ sPC + 0.05 (OPEN pending jpc-faithful regime); (2) **μPC no lift** — `spc_mupc ≤ spc + 0.05` asserted at every depth (0.124 vs 0.105 at depth 8; OPEN, not refuted); (3) **runaway gain** — tile-hebbian init norm ratio 1.4→7.2e2→3.2e5 at depths 10/50/100, monotone growth asserted; (4) **subspace collapse** — Oja 10-class readout 0.99→0.23 toward chance, first layer ≈1.0 asserted. Record `f1_failure_manifesto` (+`arms.epc`), gallery `DEMOS` row + `_fig_failure_manifesto` (1×3 panels), RESULTS.md paragraph re-pinned 2026-09-05. Probes: `scripts/probes/failure_manifesto.py`, `scripts/probes/failure_manifesto_audit.py`, `scripts/probes/f1_epc_depth.py` |
| **F2 (R11.5.5 spiking)** | The spiking plateau figure — **collapse verified robust to known fix #1, 2026-09-05** (homeostatic audit, plan item 3: synaptic scaling holds norms yet the readout still collapses — the STDP fixed point destroys class structure; OPEN pending reward-modulated STDP). Demoted from *finding* to OPEN audit 2026-09-05 (R11.5.5a); landed 2026-09-04 | `tests/integration/test_demo_spiking_plateau.py` (~10 s): (1) **the confound, instrument defect** — default init leaves hidden LIF layers silent (spike fraction < 1e-4 at depth 4, width 32, threshold 1.0) → exactly zero STDP gradient on every hidden weight matrix → frozen readout: historic "spiking at chance" measured a silent network; `init_scale=1.0` restores spiking (0.15–0.45/layer) and gradient reach (all norms > 0, asserted); (2) **the plateau, mechanism** — supervised accuracy 0.048 (chance) even with healthy spiking: `TemporalTraceCredit` declares `phases=(FREE,)` and never consumes `loss` — no supervision path by construction (category fact; a supervised spiking claim needs an error term, e.g. reward-modulated STDP — OPEN); (3) **runaway gain, spiking edition** — unsupervised STDP training collapses class structure: centroid readout on hidden membranes 0.36 (random init) → 0.18 (STDP-trained), asserted; **homeostatic audit (2026-09-05)** — `homeostatic_scaling=True` holds hidden row norms at target (live, asserted) yet readout still collapses (record `homeostatic_audit`). Probes: `scripts/probes/spiking_learning.py`, `scripts/probes/spiking_gain_audit.py`. Gallery `DEMOS` row + `_fig_spiking_plateau` (1×3). RESULTS.md re-pinned 2026-09-05. |
| **D13 (U-axis × local credit)** | The U-axis is a swap — and **local credit × Muon is real, multi-seed verified** (pulled 2026-09-04, user hypothesis; multi-seed promotion + FF/PEPITA realization 2026-09-05, plan items 1+4) | `tests/integration/test_demo_uaxis_muon_swap.py` (~15 s): one coordinate, three credits {bp, ff, pepita} × one swapped update {euclidean lr 0.2, muon lr 0.02}: FF×Muon **0.838 ± 0.009 over seeds 0–4** vs FF×Euclid 0.568 ± 0.041 (per-seed lift min +0.241, asserted per seed); BP×Muon 0.843. **FF and realized PEPITA are distinct algorithms** (ratchet: ff/euclid − pepita/euclid > 0.2 fires if the byte-identity defect ever returns); realized PEPITA (softmax error × fixed random inverse projections) learns slowly at demo budget — 0.226 Muon vs 0.106 Euclid, asserted as mean lift > 0.05 only. **Instrument history:** (1) `RiemannianOrthogonalUpdate` now orthogonalizes the momentum buffer (Muon's recipe); (2) the polar factor is the SVD `U @ Vh` (reduced QR is sign-arbitrary, cos ≈ 0), locked by `test_muon_polar_factor_is_descent_aligned`; (3) `EuclideanUpdate` momentum buffers fail loud on cross-geometry reuse. Multi-seed probe: `scripts/probes/d13_multiseed.py`. Property suite 679 passed; demo gate green (manifest deliberately re-pinned). |

### R11.4 — Adoption Surface

| Item | Description | Key Evidence |
|------|-------------|--------------|
| **R11.4.2** | PR-6 Fairness contract draft | `docs/FAIRNESS_CONTRACT.md` v0.1 (F-1..F-6, consumers table) |
| **R11.4.1 (v1)** | `SystemModule` drop-in nn.Module facade (pulled 2026-09-04, user-directed "general improvements" session) | `computronium/nn/system_module.py`, root export. Plain-PyTorch inference (`forward` under `no_grad`/`eval`), `fit_step` for internal credit assignment (no optimizer), `parameters()`/`train()` delegate to geometry, `to(device/dtype)` moves geometry (mirrors SystemTrainer's `geometry.to(device)` convention; pyright-strict clean). Lock: `tests/unit/nn/test_system_module.py`. Scope-honest: this is the wrapper *surface*, not pip packaging |

---

## 📋 Remaining Items (Pull-Based — No Schedule)

These land **only when a demo, campaign, or research paragraph needs them**.

| Item | Trigger | Category |
|------|---------|----------|
| **R11.3.14** Deep Hebbian fix: per-layer activity normalization | **LANDED 2026-09-04** (see Completed below) | ~~Capability~~ ✅ |
| **R11.3.11** Multi-seed depth-frontier pilot | **Pulled 2026-09-04; verdict downgraded** (see R11.3.11b in Completed). No μPC lift under our trainer even in-regime (residual); the clean answer needs a jpc-faithful trainer port — **pull-based** | Research (architecture ✅, trainer regime OPEN) |
| **R11.1.10** LazyStateDynamics | **Landed 2026-09-04** (see Completed) | ~~Capability~~ ✅ |
| **R11.1.11** Domain extensions | Benchmark/demo/research needs: `wikitext2`/`penn_treebank` (LM), `mountain_car`/`lunar_lander` (RL), `diabetes`/`california_housing` (tabular), `ett_h1` (time series), PDE suite (Heat/Wave/Burgers/Navier-Stokes) | Capability |
| **R11.2.14** Latency proxy | **Landed 2026-09-04** — `estimate_train_step_flops` (`core/profiling.py`): deterministic structure-derived FLOPs per train_step (matmul rounds per weight matrix × settle structure from `dynamics_type`, incl. the spike-substrate one-matmul-per-layer subtlety); intended as a *relative* comparator — absolute latency stays with the repeated-timing path in `analyze_joint_system`. Lock: `tests/unit/core/test_latency_proxy.py` (determinism, depth/settle-step scaling, **proxy ordering matches measured walltime**, non-layered rejection) | ~~Hygiene~~ ✅ |
| **R11.2.9** `substrate_coupled` plasticity engagement | Campaign manifest needs it; probe fixed-dim `step` assumptions; now also the home of any future latent-graph ternary learning path (see Notes) | Hygiene |
| **R11.2.13** Campaign stability proxy | Cheap per-episode proxy for stability axis | Hygiene |
| **Compute proxy: count ψ-update work** | `_episode_resources` (campaign/evaluation.py) is arm-invariant across P-axis primitives — measured in F3 (asserted live): the ψ-update cost is visible in walltime (+22%/+45%) but absent from the compute/energy MACs. Add the primitive's per-step ψ arithmetic to the proxy when the registered P-axis campaign needs the work axis to discriminate | Hygiene |
| **P-axis primitives realized (closed the F3 mechanism-audit gap, 2026-09-05)** | **LANDED 2026-09-05** (see header + session table). Remaining P-axis OPEN item: lr-matched controls per arm before any retention mechanism claim; an episode that persists ψ across batches (the `train_step` contract re-initializes ψ per episode) if a registered fast-weight claim needs multi-batch memory | ~~Capability (P-axis)~~ ✅ core |
| **R11.2.15** `demo/tests/` 28 stale failures | Rebuild with R11.4 UI, or before if path touched | Hygiene |
| **R11.2.16** TF-IDF weighting / `V_nudged` | Research track wants strengthened PC Lyapunov xfail | Hygiene |
| **R11.3.4** AutoScientist P-axis frontier | Tangible Checkpoint 5 — first *finding* figure (Pareto over 𝒞) | Research |
| **R11.3.11** μPC depth scaling | **RESOLVED 2026-09-05 (D14): the μPC lift is REAL** at depth 20 under the jpc-faithful regime (test 0.69–0.83 vs default 0.14–0.24, seeds 0–2) — the earlier "no lift" verdicts were trainer-regime artifacts (Euclidean SGD, β=0.5, fixed 60 steps). Init landed 2026-09-04 (`GeometryConfig.init_scheme="mupc"`); regime landed 2026-09-05 (D14). Probe history: `scripts/probes/jpc_faithful.py` | ~~Research~~ ✅ (verdict: lift real, depth-dependent — saturated/invisible at depth 8, decisive at depth 20) |
| **Path A: jpc-faithful trainer** | **LANDED 2026-09-05 as D14** (see Completed). Remaining tail (pull-based): promote the manual jpc loop into a `JPCFaithfulTrainer`/`SystemTrainer` regime config if a registered-scale sweep needs it; wider width 512 replication; the a_L = N^{-1} CE-softmax subtlety never bit us at width 128 (CE on 1/N-scaled logits trained fine) — revisit only at width 512 | ~~Research~~ ✅ core |
| **R11.3.13** Depth-metric classes | **Landed 2026-09-04** | `computronium/ontology/depth.py`: `DepthMetric` Protocol, `FixedDepth`, `ShortestPathDepth` (BFS from sources, edge direction row←col matching `GraphGeometry._aggregate`), `LongestPathDepth` (DAG Kahn; fail-loud on cycles), `max_depth`. `GraphGeometry.num_nodes` + `node_depths(metric)`. Root + ontology exports. Lock: `tests/unit/core/test_depth_mupc.py` (12 tests, incl. default-init bitwise lock) |
| **R11.2.23** Energy-framed metric contract | **Pulled 2026-09-04** (see R11.2.23 in Completed) — live trainer sample-weighted metrics + `val_ppl`. FabricPC's legacy `EvalMetric` design informed the contract; FabricPC itself is archived | ~~Hygiene~~ ✅ |
| **Path B: P-axis Pareto campaign** | **E-1 smoke DONE (F3, 2026-09-05): the axes discriminate at demo scale, mechanisms realized, and the lr-matched pilot fixed the protocol: retention must be measured at MATCHED effective lr per arm (nominal-lr retention is confounded by construction — routing's advantage is effective-lr alone).** Remaining: registered-scale pilot → full (AutoScientist commissioning, S/G/D/C/U pinned at the flagship coordinate, sweep P ∈ {Null, Routing, FastWeight, RuleState}, wider dims where walltime differentiates reliably, PR-5 guard wired via `probe_interval_for_overhead`, GPU-first). RESEARCH3 protocol governs | Research |
| **R11.3.5** Z3 flagship registered commission | Tangible Checkpoint 6 — ≥95% on 3 tasks, exact Δθ=0, ≤20% fine-tuning steps, ≥5 seeds | Research |
| **R11.3.6–3.10** Boundary mapping, CL, task-family, provenance, companions | Pull when research paragraph needs them | Research |
| **R11.4.1** Drop-in PyTorch wrapper | **v1 pulled 2026-09-04** (see R11.4.1 in Completed) — remaining: pip packaging + acceptance test per RESEARCH3 PR sequence | Adoption |
| **R11.4.3** Live demo UI | API stable — ships only when library is stable; rebuilds `demo/tests/` | Adoption |
| **R11.4.4** Hygiene sweep | Only when blocks a figure, test, or fresh checkout | Adoption |

### Deprioritized (As-Touch Only)

| Item | Reason |
|------|--------|
| **R11.2.2** Pyright baseline | User directive: as-touch on legacy modules; new modules stay strict. Not a workstream. |

---

## 📋 Register D — Carried Deferred (unchanged from TODO10)

| Item | Reason |
|------|--------|
| Coverage floor (~16.8%) | opt-in `--cov`; raise after API stabilizes |
| Rocq general-case formalization | CP-B pull-based; diagonal case done with paper proof; ψ-coverage proposition is the next statement |
| `test_ontology_parity.py` decomposition | Slow-marked; split fast/slow only if gate iteration speed demands |
| Physical hardware (PR-3b / CP-D) | Latency-gated procurement per RESEARCH3; proxy tier (R11.3.2b) decouples all software-side claims |

---

## 🔒 R11.5 — The Standing Rules (R10.3 verbatim, renumbered)

- **R11.5.1 No test, no feature.** Every feature ships with an integration
  test that demonstrates it working end-to-end.
- **R11.5.2 No claim without a live demonstration.** When a test is removed,
  flaky, or failing, its claim disappears from the front page automatically —
  the system degrades to silence, never to stale assertions.
- **R11.5.3 Corroboration never carries.** Registered numbers are history:
  labeled, scoped, provenance-annotated, confined to RESULTS.md's back
  section and the research track.
- **R11.5.4 Scope honesty.** Demo-scale demonstrations speak for demo scale;
  registered claims live in the research track. Neither borrows the other's
  clothes.
- **R11.5.5 Refutations ship with the same pipeline** — same figure factory,
  same docs, same terms. Standing candidate: the **spiking family's learning
  claim**. Status at HEAD (2026-09-04): R11.1.9 wired timing-asymmetric STDP
  (rasters, eligibility traces, threshold) into the 5-D pipeline, but no demo
  measures whether it *learns* — the pre-wiring Hebbian-plateau result
  (TODO10: spiking at chance on MNIST) is history, not a live refutation.
  First spiking pull must show one or the other: plateau (refutation figure,
  same pipeline) or learning (capability claim). Both refutation slots are now
  filled: F1 (2026-09-04) is the live depth-boundary manifesto figure, and
  F2 (2026-09-04) shows the spiking plateau — audited, with the historic
  chance-level numbers exposed as a silent-network confound. The new OPEN
  follow-up: a supervised spiking claim needs an error term (reward-
  modulated STDP or similar) — pull-based.
  **R11.5.5a (2026-09-05, user course correction — binding corollary):** a
  refutation is a *boundary* only after the algorithm's known theoretical
  fixes have been applied and the failure persists (for the depth wall:
  ePC's error reparameterization, μPC residual scaling, a faithful trainer
  regime; for STDP collapse: homeostatic synaptic scaling / Oja decay /
  intrinsic plasticity; for optimizer effects: Adam or the tuned regime).
  **A test only proves the code behaves as written — not that the algorithm
  is mathematically doomed.** F1 and F2 are therefore DEMOTED from
  "findings" to OPEN engineering-defect audits (see The Next-Session Plan):
  the "failure manifesto" posture is retired until a wall survives its
  known fixes. A passing test that asserts a failure mode is evidence of
  the current regime, never of physics.
- **R11.5.6 Pull rule.** A backlog item is pulled only if it ends in a live
  demonstration, a gallery figure, or a RESULTS.md capability paragraph.
  Infrastructure is justified by the capability it lets the suite show,
  never by itself.
- **R11.5.7 Gates (tiered, per AGENTS.md test-execution tiers).** Per-commit
  duties are **scoped to changed files** (format + lint + pyright + targeted
  tests). The standing fast gates — property suite, demo gate
   (`pytest tests/integration/ -k "demo or gallery_lock"`, now SPLIT:
   fast tier ~190 s excluding the `slow` marker, slow tier
   `pytest -m slow -k demo` for D14+D15+D16 (~13.5 min; D14 ~120 s,
   D15 ~380 s, D16 ~310 s) — re-baselined 2026-09-05 after the
   OrthoAdam arms landed; run
   the slow tier on round close or D14/D15/D16-adjacent changes),
  drift locks, positive control — run on their triggers (demo/gallery/
  lock-adjacent changes), never per-edit. The full CI order and repo-wide
  ruff/pyright are R11.2's deliverable and a round-close event, not a
  habit. No new verification rounds are commissioned in R11; R11 spends
  R6–R10's trust.

---

## 🔒 Gate Commands (Quick Reference)

```bash
# Property locks (fast CI gate) — 670 passed
uv run pytest tests/property/ -q

# Demo gate, FAST tier (D1–D13 + F1–F3 + gallery lock) — 19 passed, ~190 s
# (default addopts exclude the `slow` marker; D14 lives in the slow tier)
# NOTE: invoke as `python -m pytest` — see Watch (user-site pytest drift)
uv run python -m pytest tests/integration/ -k "demo or gallery_lock" -q

# Demo gate, SLOW tier (D14 jpc-faithful depth-20 + D15 depth frontier
# with OrthoAdam arms + D16 coverage map with ortho_adam) — 3 passed, ~13.5 min
uv run python -m pytest tests/integration/ -m slow -k "demo or gallery_lock" -q

# Gallery lock — figure data checksums match manifest
uv run python -m pytest tests/integration/test_gallery_lock.py -q

# Reproducibility — 8/8 native families bitwise identical
uv run comp repro --seed 42 --device cpu

# Gallery re-render from on-disk records (deliberate re-pin; `--run` re-runs
# the demo suite first and needs >2min; NOTE: `--run` inherits the default
# `-m 'not slow'` addopts, so D14's on-disk record is reused — run the slow
# tier first when a deliberate D14 re-pin is wanted)
uv run comp gallery

# Root exports
uv run python -m pytest tests/unit/core/test_root_exports.py -q
```

---

## 👁️ Watch (Live Items Only)

- **Settle-loop cost (measured 2026-09-04):** both energy-family settle loops
  now have `torch.compile` fast paths behind `StateDynamicsConfig.compiled=True`:
  sPC layered settle (2.0× train_step, bitwise) and EqProp kernel loop
  (1.75× settle; autograd-through-compiled verified for thermo credit).
  Guards keep the compiled path on the common case (digital, no recurrent,
  momentum=0, no tracking/checkpointing); the compiled EqProp path runs a
  fixed step budget (skips the eager convergence early-exit). Remaining
  headroom: extend to SpikeIntegration (D7) with the same recipe when the
  spiking demo is pulled; batch-per-step 4–8× stacks for free. The R11.3.11
  depth frontier is affordable — run it with `compiled=True`.
- **`GradientCredit` fail-loud — RESOLVED (2026-09-04):** `allow_unused=True`
  zero-fill replaced with a `RuntimeError` naming the detached weights
  (`credit.py`, `GradientCredit.compute_pseudo_gradient`; `BackpropCredit`
  is the same class). A future dynamics that detaches activations now fails
  loudly instead of silently degrading to last-layer-only learning.
  `LocalGoodnessCredit` keeps its zero-fill (surplus recurrent self-connection
  weights legitimately receive `None`).
- **`uv run pytest` resolves the USER-site pytest (2026-09-04):** the launcher
  at `~/.local/bin/pytest` (shebang `/usr/bin/python`) shadows the venv's,
  importing protobuf 6.33.6 from user site against gencode 7.35.1 stubs →
  `VersionError` at gRPC test collection. Invoke tests as
  **`uv run python -m pytest`** (guaranteed venv python, protobuf 7.36.1).
  Gate commands above updated accordingly.
- **Plain `uv sync` strips dev extras (2026-09-04):** a bare `uv sync
  --upgrade` re-syncs the venv to main-deps-only, silently removing
  optuna/scipy/torchvision (they live in the `dev` extra group, not main
  deps). Symptom: sudden ModuleNotFoundError at import. Durable fix:
  `uv sync --dev --all-extras`. pyproject was briefly double-listing
  optuna/scipy in main deps during triage — reverted; groups are canonical.
- **D1/D8 record drift absorbed 2026-09-04:** `comp gallery` (render-only
  re-pin) flagged compose_6axis/geometry_swap data changed vs the old
  manifest — same class as the 2026-09-03 sweep-regime note. Manifest
  re-pinned from current on-disk records; demo gate + gallery lock green
  after. If the lock fires again: check test asserts first, then re-render.
- **Demo-gate budget drifted past R11.5.7's ≤90 s (RESOLVED 2026-09-05 by
  the fast/slow split):** D8–D12 additions plus F1 grew the gate, then D14
  pushed it to ~300 s. The gate is now split: fast tier (D1–D13 + F1–F3 +
  lock, ~190 s, default invocation) and slow tier (`-m slow -k demo`,
  D14, ~120 s) — future slow demos join the `-m slow` resident set
  (candidates: any test needing >60 s or a timeout marker).

- **axis_probe `[2-0]` flake** — no recurrence since 2026-08-31.
- **D3 mechanism confound (found by the F3 audit, 2026-09-05; updated
  after the primitive realization same day):** the retention advantage
  SURVIVES realized routing (real per-unit gates now, not scalar gain) —
  but null@0.015 still retains more (0.294 vs routing 0.273), so the
  effective-lr explanation remains live there too. D3's mechanism
  wording ("pathway gating") should still be softened when D3 is next
  touched; its assert messages should carry the audit pointer, and any
  future mechanism claim needs lr-matched controls per arm.
- **Walltime never enters `record["data"]` (2026-09-05, F3 lesson — twice
  refined):** the gallery lock hashes data at 1e-6 and walltime drifts
  every run. First fix (quantized latency RATIO vs null) still tripped
  the lock: routing's ψ cost sits at the noise floor (~+20% ≈ run-to-run
  drift), so its ratio flipped 0.25-bins between runs. Final rule:
  record["data"] carries ZERO walltime — Pareto cost axes use
  deterministic quantities (ψ-capacity); latency-ordering claims are
  live asserts with margins; absolute values stay on stdout/probes.
  Drift-immunity for any record is proven by two consecutive demo+lock
  green runs, not one.
- **Muon orthogonalizers are TWO algorithms (2026-09-05, user prompt on
  the MEP kernels):** the pipeline's `RiemannianOrthogonalUpdate` had been
  using the exact SVD polar factor while `ortho_steps` sat as dead config.
  Now wired: `ortho_steps=0` (default) = exact SVD polar factor —
  full-spectrum whitening, the configuration of record for D13/D15;
  `ortho_steps>0` = canonical Muon Newton–Schulz (quintic coefficients,
  `newton_schulz5` in `core/optimization/strategies/update.py`; the old
  naive `0.5·X(3I−XᵀX)` variant under-converged from Frobenius
  normalization, orthonormality err ~0.85). **Measured finding: the
  FF×Muon lift is WHITENING-driven** — NS preserves BP×Muon (0.868) but
  collapses FF×Muon to 0.29 at width 32 (SVD maps even tiny singular
  values to 1, dense exploration for low-rank FF buffers; NS keeps the
  spectrum). The MEP Triton/CUDA NS kernels implement the naive
  iteration — upgrade them to the quintic coefficients before any GPU
  registered sweep that opts into NS (cross-device numerics).
- **Suppression-comment verdict REVERSED back (2026-09-05):** `#
  ruff: ignore[x]` fires RUF103 (unknown directive) under the current
  ruff — the 2026-09-05 note claiming it is preferred/applied is wrong
  for this version; the 2026-09-04 note was right. `# noqa: <code>`
  works and did NOT trigger the repo-local `noqa-comments` rule on the
  F3 test. Best: avoid suppressions (e.g. `torch.nn.functional.normalize`
  instead of `x = x / x.norm()`). Register C sweep still owns the legacy
  conversions.
- **CUDA tolerance boundaries** shift xfail edges — CPU/GPU tests kept separate.
- **R11 sweep regime note (2026-09-03):** repo-wide ruff autofix shifted import/init order, moving D2/D7 record data. Tests pass asserts; manifest re-rendered. If figure lock fires again, check test asserts first, then re-render.
- **D8 record determinism:** seed must precede loader draw (`torch.manual_seed(42)` before `_materialize`); DEVICE must be `"cpu"`.
- **Conv = GPU pointer (measured 2026-09-03):** conv-family is first FLOP-bound path (15× CUDA speedup); CUDA nondeterministic run-to-run. Registered-scale conv studies go GPU-first; committed demo records stay CPU.
- **`benchmark_results/` stays untracked** (standing directive).
- **Fidelity probes RNG-order sensitivity — RESOLVED (R11.2.22):** `check_coordinate_fidelity` is seeded + fork-rng'd.
- **Stale eager-default metric lookups:** `d.get("free_accuracy", d["accuracy"])` evaluates default eagerly — safe idiom is nested `get`.
- **Registry-era removals:** transfer-weight loading, proposer objective ranking deleted with zoo — re-home onto native factories.
- **`equitile` deprecated:** family registrations, CLI maps, tolerances, metamodel branches key on `"tile"`. Residual mentions cosmetic — rename on next touch.

---

## 🎯 Tangible-Result Checkpoints (R11 Edition)

| # | Checkpoint | Condition |
|---|------------|-----------|
| 1 | Capability pulls demonstrated (R11.1) | Demo suite green, walltime ≤ 90s ✅ |
| 2 | Truthful gates (R11.2) | ruff/pyright green at HEAD or explicitly scoped; CI order enforceable ✅ |
| 3 | Commissioned campaign stack (R11.3.1) | Iterate → interrupt → checkpoint → resume cycle recorded ✅ |
| 4 | Calibrated stability guard (R11.3.3) | ROC-calibrated kill thresholds (<5% false-kill, >95% kill, <10% overhead) ✅ |
| 5 | Adoption surface (R11.4) | Wrapper v1 (pip-installable, smoke suite) and/or live demo UI — **pull-based** |
| 6 | First research-shaped result (R11.3.4) | P-axis Pareto frontier over 𝒞, annotated per knee — **pull-based** |
| 7 | Discovery bet (R11.3.5) | Z3 flagship at registered scale; either outcome tangible per pre-registered fallback — **pull-based** |

Sequencing: 1–4 complete; 5 after API stabilizes (done); 6–7 are RESEARCH3 CP-A's tail. No checkpoint blocks on a later one.

---

- **Graph/tree panels + ergonomic builders landed (2026-09-05, second
  slice):** ``graph_panel``/``tree_panel`` add structural charts —
  layouts ``layered`` (y = longest-path depth, the R11.3.13 notion),
  ``tree`` (root-hierarchical, x by leaf order), ``circle``, ``spring``
  (deterministic, seeded; numpy-only, no networkx dep). ``node_values``
  colors nodes by a metric (per-node depth, activation norm) with a
  colorbar; ``edge_weights`` scale line widths; arrows when
  ``directed``. Author specs with the builder family (``figure_spec``,
  ``bars_panel``, ``lines_panel``, ``scatter_panel``, ``heatmap_panel``,
  ``graph_panel``, ``tree_panel``) — keyword-completable, exported from
  ``computronium.visualization``. Decision table in ``_demo_api.py``'s
  docstring: comparisons → bars; curves → lines; frontiers → scatter;
  matrices → heatmap; structure → graph. 15 unit locks in
  ``tests/unit/visualization/test_demo_api.py`` (incl. layered-depth
  ordering, tree hierarchy, fail-loud layouts). Demo-side convention:
  ``record["figure"] = figure_spec("D# — claim", panel(...), ...)``.

---

## 📈 Demo API — Publishability Roadmap (planned 2026-09-05, pull-based)

Honest verdict from the final check: the declared-in-record API delivers
*consistency* and *reproducibility* (one renderer, spec checksummed with
the data, drift-immune lock) but NOT yet *publication polish*. The gaps,
planned in dependency order — each lands when a figure needs it (R11.5.6),
except item 1 which is a tracked migration:

1. **Migrate the 14 legacy figure factories to declared specs**
   (D1–D12, F1, F2). Migration ratchet: a lock that walks `DEMOS` and
   asserts every record declares `data["figure"]` — write it as a tally
   test now (reports 14 undeclared), flip to strict when the count hits
   zero. After this, bespoke factories are dead code and delete.
2. **Seed variance on the page.** Multi-seed claims (D13's ±0.009, the
   D14 probe's per-seed spreads) live in record fields but the figures
   can't show them: add `yerr` to `bars` (from per-seed arrays: mean ±
   range) and a mean±band mode to `lines`. Error bars are the single
   biggest publishability gap — a claim without visible variance invites
   the single-seed skepticism the μPC audit earned.
3. **Vector export.** `save()` is PNG@150 only; publication needs PDF/SVG.
   `render_gallery(..., formats=("png", "pdf"))` + `comp gallery` flag;
   manifest lists per-format files. Cheap, unblocks docs/paper embeds.
4. **Captions on the figure.** Record-level `caption` (regime summary,
   one-line reading guide, chance note) rendered as a figure footnote and
   reused verbatim in RESULTS.md's figure table — one source of truth for
   what the figure claims, provenance visible without opening the record.
5. **Label polish.** Smart bar-label offsets (labels currently clip at
   ylim=1.0), per-series `fmt`, a diverging-cmap helper for signed
   heatmaps (credit/contrast matrices), `legend_loc` defaults per panel
   type tuned once.
6. **Graph polish** (when a structural demo is pulled): curved/multi-edge
   routing, community/group shading, a labeled colorbar title for
   `node_values` ("node depth" vs unlabeled color).
7. **Cookbook docs.** `docs/DEMO_API.md`: the decision table, one minimal
   worked example per panel (lift from the unit locks), and the record →
   gallery → lock data flow. The stranger's path from "I measured
   something" to "it's in the gallery, locked" in one page.
8. **D9/D11 migration exemplars** (first consumers of `graph_panel`):
   migrate the graph-geometry and lattice demos to declared specs with
   node_values = per-node depth (R11.3.13's `node_depths` metric — the
   API and the metric finally meet on the same figure).

Sequencing note: 1 is the ratchet and unblocks deleting ~300 lines of
hand-styling; 2–4 are the publishability core; 5–8 ride on touch. The
demo gate (~300 s) argues for the fast/slow split BEFORE adding more
demos — D14's three depth-20 arms are the natural first slow-tier
resident (carried from the session notes).

---

## 📝 Notes for the Next Editor (2026-09-05 session, OrthoAdam at scale)

- **Adam is depth-fragile under this pipeline: the new finding of this
  slice.** BP×Adam is fine at demo scale (D16: 0.892 on mlp d2/w64) but
  collapses partially at depth 16 / width 128 (0.303 ± 0.079, high seed
  variance) — the second-moment normalization amputates the gradient
  signal where it is smallest, compounding through depth. OrthoAdam's
  SVD-polar replacement of the matrix direction restores graceful
  degradation (0.878, matching Muon's regime behavior). If quoting
  Adam anywhere in a depth claim, state the depth dependence.
- **Repo-best acc/param claim, stated carefully:** FF×OrthoAdam 0.947 ±
  0.002 at depth 4 / width 128 (~119k params) vs D15's FF×Muon 0.930 at
  depth 4 / width 256 (400.9k params) — different geometries, so the
  honest frame is "beats the previous local-credit record's ACCURACY
  with ~3.4× fewer parameters", not a capacity-matched comparison (the
  capacity-matched claim inside the w128 cells is ff/ortho > ff/adam >
  ff/muon at identical geometry).
- **D15 runtime grew to ~380 s (timeout raised to 900).** The slow tier
  is now D14 (~120 s) + D15 (~380 s) + D16 (~310 s) ≈ 13.5 min. If it
  grows further, consider per-demo sharding (`-k`) in the slow-tier gate
  command rather than trimming arms.
- **Determinism note:** the promoted D15 numbers match
  `scripts/probes/ortho_adam_scale.py` exactly — the probe and the demo
  agree bitwise at the same seeds/loader draw, so the probe docstring's
  numbers are quotable as of this landing.

---

## 📝 Notes for the Next Editor (2026-09-05 session, OrthoAdam landing)

- **The OrthoAdam recipe, precisely:** per step — update Adam's m/v;
  for ndim==2 params, replace the update direction with the SVD polar
  factor of the bias-corrected first moment m̂, rescaled to ‖adam_step‖_F
  so `ortho_lr` is magnitude-comparable to Adam's per-tensor step; ndim<2
  params take plain Adam at `step_size`. The rescale is load-bearing:
  without it a raw polar factor has norm √min(m,n) and the lr scale is
  geometry-dependent. ortho_lr was calibrated on mlp (1e-3 → 0.912,
  3e-3 → 0.931, 1e-2 → 0.901, seed 0) — 3e-3 is the config of record.
  Implementation: `OrthoAdamUpdate(AdamUpdate)` in
  `computronium/ontology/update.py` — inherits `_clip`, `_state`
  (fail-loud reuse), and the config plumbing.
- **`ParameterUpdateConfig` gained an `ortho_lr: float = 0.003` field**
  (default chosen so legacy configs are unaffected; asdict round-trip
  carries it like beta2/eps). `_update_from_config` in spec.py was
  refactored from an elif chain to a `_UPDATE_CLASSES` alias map while
  adding the dispatch branch — the map (not a chain) is the pattern for
  the next update primitive.
- **D16 slow-tier re-run lessons:** the ortho-vs-adam assert margin must
  come from the measured per-geometry gaps (mlp +0.038 / attention
  +0.010 / graph +0.079 / lattice +0.029) — the first draft's flat 0.03
  margin failed on attention; final ratchets: ortho > adam + 0.005 and
  ortho > muon − 0.03 (graph is the one geometry where the hybrid trails
  Muon, −0.022). Also: grep-filtered pytest output hid a failure line —
  always `tail` the summary, never grep-only, when a run's outcome is
  unknown.
- **Gate evidence this session:** slow tier D16 single run green (~259 s,
  faulthandler dump at 120 s is cosmetic — `faulthandler_timeout`
  prints stacks without killing); fast gate 19/19 (~205 s); gallery
  lock green after deliberate D13+D16 re-pin; D13 drift-immunity
  proven ×2 (its re-run is in the fast gate). D16's re-pin has one
  consecutive green (the emitting run) — prove ×2 on the next slow-tier
  run before quoting its absolute numbers.
- **Stray module (pre-existing):** `tests/unit/core/test_stability_standalone.py`
  imports `computronium_stability` from a foreign path
  (`/home/me/bioplausible`) and breaks `pytest tests/unit/` collection.
  Not from this session — quarantine or delete in the hygiene pass.

---

## 📝 Notes for the Next Editor (2026-09-05 session, hunt cells)

- **`NaturalGradientUpdate` is NOT natural gradient.** `step` is
  `param − lr · g / (mean|g| + 1e-8)` per tensor (update.py, "Simplified"
  comment) — a per-tensor-normalized SGD whose effective step equals
  lr. Any lr ≥ 0.01 destabilizes MNIST MLPs into output collapse
  (trainacc exactly chance, monotone loss growth — measured at
  0.01/0.1/1.0); working lr is 1e-3. Before any natural-gradient
  mechanism claim: implement diag-Fisher (EMA of g² per parameter, damp,
  multiply) or rename the primitive. D16's map currently quotes it as
  "natural (lr 1e-3)" — a family label, not a mechanism claim.
- **pepita×Adam is regime-dependent, and that is the finding.** w32/D13
  scale: Adam 0.173 ≈ Muon 0.202 (both » euclid 0.110); w64
  (`hunt_cells.py`): Muon 0.306 » Adam 0.143. The rescue-vs-Muon
  ordering flips with width — the D13 ratchet only asserts
  lift-over-own-Euclid-baseline. If this cell is ever quoted, state the
  regime.
- **Probe `hunt_cells.py` cell-label caveat:** its "spectral" credit
  rows fall through `_credit` to pepita — there is no spectral credit
  rule; the useful outputs are the pepita×optimizer and bp×natural
  rows. Fix the labels if the probe is rerun.
- **Gate evidence this session:** slow tier D16 single run green (~259 s,
  faulthandler dump at 120 s is cosmetic — `faulthandler_timeout`
  prints stacks without killing); fast gate 19/19 (~205 s); gallery
  lock green after deliberate D13+D16 re-pin; D13 drift-immunity
  proven ×2 (its re-run is in the fast gate). D16's re-pin has one
  consecutive green (the emitting run) — prove ×2 on the next slow-tier
  run before quoting its absolute numbers.

---

## 📝 Notes for the Next Editor (2026-09-05 session, demo API + re-pin fix)

- **The common demo API landed (`computronium/visualization/_demo_api.py`):**
  figures are DECLARED IN THE RECORD — a demo test puts a ``figure`` spec
  under ``data["figure"]`` next to the measured data, and ONE generic
  renderer (``gallery._fig_declared``, registered for D13/D14) produces
  the gallery figure. Consistency (labels, chance lines, palettes, value
  labels) is by construction. Panel vocabulary:
  ``bars`` (grouped, ``horizontal``, ``series_colors``/``series_labels``,
  ``fmt``), ``lines`` (``x``, ``xticklabels``, ``log_y``/``symlog_thresh``,
  ``annotate``, ``vline`` for depth boundaries), ``scatter`` (the
  Pareto/frontier panel: ``connect``, ``point_labels`` — built for the
  CP-6 P-axis trade-off figure), ``heatmap`` (annotated grids — tile ×
  dynamics matrices). Common keys: ``title`` per panel, ``chance``,
  ``legend_loc``; figure keys: ``title``, ``figsize``, ``layout`` grid.
  D13's record shows the color-coding pattern: shape the declaration as
  groups = credit rule × series = update rule and the renderer colors +
  legends automatically. Unit locks: ``tests/unit/visualization/test_demo_api.py``
  (10 tests — value labels, chance lines, per-series colors, fail-loud on
  unknown types). Legacy hand factories keep working; migrate on touch.
- **The gallery re-pin hassle is FIXED: the lock is now drift-immune.**
  Root cause: the lock hashed raw record floats, so documented 1e-7
  multithreaded reduction churn tripped it spuriously.
  ``gallery.canonicalize_floats`` (round to 1e-6, normalize −0.0) is now
  the single hashing convention for the lock's data sha, the emitter's
  provenance ``config_sha256`` (conftest), and the lock's
  emitter-consistency check (which previously DUPLICATED the hashing
  inline — the duplication is what broke first). Proven: demo re-run →
  record re-emitted → lock stays green WITHOUT re-pin. The lock now
  fires only when a demo changes what it demonstrates — exactly its job.
  One-time cost: all records re-emitted once under the new convention
  (full demo suite run 2026-09-05, 19 passed).

---

## 📝 Notes for the Next Editor (2026-09-05 session, Path A)

- **D14 / jpc-faithful regime (Path A landing):** the manual loop is the
  reference implementation until a trainer config is pulled:
  1. ePC settle free (steps = H, `convergence_threshold=0.0` forces the
     fixed budget — no early exit) and nudged (β from the grid);
  2. **PC-native weight gradient**: rebuild the forward with the settled
     errors FROZEN (`eps = [e.detach() for e in dynamics._last_errors]`),
     take d(β·CE)/dθ — this equals the paper's Δθᵢ ∝ (∂sᵢ/∂θᵢ)ᵀ εᵢ (the
     reverse-mode sweep carries the propagated error to each weight);
  3. `torch.optim.Adam(weights, lr=1e-3)` on the transition weights.
  `ErrorPredictiveCodingDynamics._build_forward_with_errors` now takes
  `residual=` and applies the `_apply_stack` skip semantics
  (h = h_in + φ(W h_in + b) when widths match); ε rides ON TOP of the
  skip (skip is part of the prediction). Lock:
  `test_residual_geometry.py::test_residual_epc_free_equilibrium_is_feedforward_bitwise`.
- **β is a working-regime knob, not a monotone dial.** At depth 20 /
  width 128 / 150 batches: β=10 generalizes (mupc test 0.78 mean),
  β=1e3 lands in the memorization corner (train 1.00, test 0.09–0.36 —
  including one arm BELOW chance). When a result looks impossible
  (test < chance), get the train/test breakdown before suspecting the
  instrument: train 1.000 + test 0.089 = memorization, not a broken
  evaluator. Audit lesson applied in both directions this session.
- **The μPC lift is depth-dependent:** invisible at depth 8 (both inits
  saturate ≈ 1.0 train under the faithful regime), decisive at depth 20.
  Depth-of-the-lift-question must match depth-of-the-claim — the depth-8
  "no lift" verdicts were asking the question where no answer exists.
- **Demo gate is now ~300 s** (D14 ≈ 107 s added). Still serial. The
  fast/slow split (R11.5.7 re-baseline) is now clearly due — D14's
  three depth-20 arms are the first demo that could live in a slow tier.
- **Probes carry the multi-seed evidence:** `scripts/probes/jpc_faithful.py`
  docstring + run history (smoke grid + depth-20 pilot); the D14 demo
  single-seed asserts use comparative margins inside the measured
  per-seed gaps (≥ 0.54).

---

## 📝 Notes for the Next Editor (2026-09-05 session)

- **2026-09-05 landings (plan items 1–4):** see the session table above.
  Carry these forward:
  - **PEPITA's error must be in probability space.** The raw nudged
    differential under `InstantaneousDynamics` is β·(onehot − logits) —
    dominated by the constant one-hot term, so every sample yields
    nearly the same update and nothing is learned (measured: 0.10 at
    every lr/scale). e = onehot(y) − softmax(free_out) is the signal.
  - **PEPITA feedback determinism:** `LocalGoodnessCredit._inverse_projection`
    caches B per (name, shape, device, dtype), seeded `zlib.crc32(name)`
    — deterministic across runs and process order; orthogonal init when
    `orthogonal_init=True`, scaled by `feedback_scale` (0.01 default;
    scale is irrelevant to Muon arms, which normalize the update).
  - **The φ′ (tanh derivative) mask HURTS realized PEPITA** (0.165 vs
    0.226 with Muon) — do not add it.
  - **`homeostatic_scaling` is a separate gate** from
    `homeostatic_target` — the latter is shared with `HomeostaticCredit`
    (default 1.0) and could not be repurposed without changing that
    default. Scaling descent is zero exactly at ||row|| = target.
  - **Demo gate is now ~180 s** (D13 multi-seed promotion + F1 ePC arm +
    F2 homeostatic arm). Still serial (record drift under xdist).
    Re-baseline R11.5.7's ≤90 s rule when the fast/slow split happens.
  - **Manifest re-pinned 2026-09-05** for D13/F1/F2 — all three diagnosed
    as intended record changes (new fields/arms), per retro (e).
  - **ruff's repo config PREFERS `# ruff: ignore[...]` over `# noqa`**
    (a repo-local `noqa-comments` rule fires on `# noqa` and the
    suppression DOES apply — contradicting the 2026-09-04 note below;
    that note's RUF103 concern applies to *invalid rule names*, not the
    mechanism). New suppressions in touched code used
    `# ruff: ignore[<name>]`. Verify suppression applies when adding one.
  - **`test_ontology_parity.py` is `pytestmark = [slow]`** — invisible to
    the default addopts `-m 'not slow...'`; run with `-m slow -k pepita`
    to exercise the FF/PEPITA preset-parity locks.

---

## 📝 Notes for the Next Editor (2026-09-04)

- **All core R11 items complete; R11.3.12 (ePC/D12) pulled 2026-09-04** —
  D1–D12 at demo scale; `comp repro` 8/8; property 670 passed; demo gate
  13/13; gallery lock green.
- **Registry is gone — never re-add it** (2026-09-03). Ontology is the composition surface; models resolve through native factories and `compose_*`. `KernelRegistry` (acceleration/) is unrelated and stays.
- **README is never edited** (2026-09-03). No sunset condition.
- **Ontology package layout (R11.1.8):** implementations in `_`-prefixed modules (`dynamics/_dynamics.py`, `substrate/_substrate.py`); `__init__.py` = docstrings + re-exports only.
- **PlasticityConfig single source:** `computronium.state.transitions` owns it; `core/joint/transition.py` re-exports. Never redefine — import.
- **Geometry dispatch single source (R11.1.2a):** `computronium.ontology.geometry_from_config` is the one topology_type→implementation dispatcher. Never re-inline — add a branch. New `GeometryConfig` tuple fields must be added to `_geometry_spec_parts`'s JSON tuple-restore list.
- **Tile × dynamics matrix (R11.1.3 + R11.1.4):** 7 tile strict xfails flipped xpass, promoted to live locks in `test_native_smoke.py` and `test_validation_all.py`. Single unlock: target-responsive TileMesh settle kernel. `native_tile_ep` re-added to REPRO_MODELS. `comp repro` 8/8. New lock `test_tile_settle_kernel.py`. **User directive: Tile geometry potential realized later.**
- **Diffusion target term (R11.1.7):** `DiffusionDynamics.compute_energy_from_state` accepts optional `target`, `beta` (nudged-Langevin). Fidelity probe passes. PredictiveSettlingDynamics fallback remains target-unwired (no geometry uses it).
- **PR-5 instrument (R11.3.3):** `calibrate_demo_harvest` (stability/calibration.py) single calibration surface; artifact `docs/figures/registered/stability_guard_pr5.json`. Known-bad = manufactured explosive family — re-calibrate against real diverged runs when failure manifesto accumulates. Deploying kill switch = wiring `probe_interval_for_overhead` (102 episodes) into AutoScientist loop.
- **Demo-test record determinism (D8):** seed *before* materializing loader batches; workers spawn per loader *iteration*, so materialize once and share. Match parameter counts and assert parity for fairness.
- **`equitile` deprecated (2026-09-03):** canonical key is `"tile"`. Residual mentions cosmetic (test names, model names, benchmark variables, docstrings) — rename on next touch, never as sweep.
- **ePC single source (R11.3.12 / D12, 2026-09-04):** `ErrorPredictiveCodingDynamics`
  lives in `dynamics/_dynamics.py` next to `PredictiveSettlingDynamics`.
  Attribution: Goemaere et al., "ePC: Fast and Deep Predictive Coding in
  Digital Simulation", arXiv:2505.20137 (ICML 2026) — class + config
  docstrings carry it. Surfaces wired: `dynamics/__init__`, ontology
  `__init__`, root `_LAZY` + `__all__`, `StateDynamicsConfig.error_predictive_coding()`,
  `from_spec` branch (factory.py), `SystemConfig.validate()` predictive-settling
  credit branch (accepts both PC dynamics). Free-phase equilibrium =
  feedforward pass bitwise (zero-init errors are the fixed point); nudged
  phase = β·CE driven through full-graph reverse-mode AD (requires
  `torch.enable_grad()` inside settle — pipeline runs no_grad for
  ThermodynamicContrast). Out-of-place adds only in the error forward —
  in-place adds pin the autograd graph (same CUDA-leak rule as geometry.py).
  Demo claims are structural (equilibrium, propagation, budget) — accuracy
  parity is NOT claimed: ePC ≈ 0.44 vs sPC ≈ 0.55 at (32,32)/150 batches;
  the ÷β contrastive credit caps ePC's learning signal on deeper stacks
  (candidates if revisited: PC-native weight gradient (∂ŝ/∂θ)ᵀε or a
  contrast-β decoupled from the loss weight).
- **μPC init + depth metrics (R11.3.11 init + R11.3.13, landed 2026-09-04):**
  `GeometryConfig.init_scheme` (`Literal["default","mupc"]`) is the single
  init lever — "default" is byte-identical to legacy (locked bitwise in
  `test_depth_mupc.py`); "mupc" replaces fan-in init with N(0,1) × depth
  scaling and supersedes `init_scale`. Wired in `_linear_stack` (feedforward
  + recurrent) and GraphGeometry (layers + head). `GraphGeometry.node_depths(metric)`
  is the R11.3.13 seam: per-node effective depth replaces layer-counting on
  graph topologies. `asdict` round-trip carries the new field (no
  `_geometry_spec_parts` change needed — str, not tuple).
  **Trap learned the hard way:** when rescaling weights in place
  (`Parameter.data.mul_`), initialize from `torch.randn`, never `torch.empty`
  — garbage × scale is still garbage, and the D9 graph demo silently learned
  garbage for 5 epochs before recovering (caught by the demo gate, exactly
  its job). Frontier probes: `scripts/probes/mupc_depth_frontier.py`
  (boundary + μPC-unconfirmed) and `scripts/probes/mupc_compiled_device.py`
  (device verdict: compiled CUDA 80 vs CPU 55 ms/step at width 32 — CPU
  still wins; compile 2.6× CPU). Next research step: multi-seed pilot.
- **R11.3.14 deep Hebbian chain (landed 2026-09-04):** implementation home
  is `computronium/models/native/deep_hebbian_native.py` — a plain-torch
  chain, deliberately NOT the tile graph (its per-edge σ caps interact badly
  with full connectivity). Recipe: spectral renorm at init + tanh + batch
  Oja (`w += lr·(yᵀa/n − E[y²]·w)`) + unit-RMS activity renorm per layer.
  Key measured findings (module docstring carries them):
  (1) the primary pathology was runaway per-layer gain (1.2–1.5×/layer
  compounding → inf/NaN); activity renorm fixes it structurally;
  (2) the trained chain transmits its *dominant direction* indefinitely
  (2-class readout 1.000 at depth 100);
  (3) a rank-10 class subspace decays (L1 1.00 → L100 0.20 > 0.1 chance):
  activity covariance effective rank collapses ~0.5/layer. Sanger (GHA),
  gain scaling, and per-step spectral renorm do not rescue it — renorm
  amplifies whatever the spectrum favors each layer. This is the third
  failure mode of the depth-boundary triad (error rules: telescoping
  decay; unnormalized local: runaway gain; normalized Oja: subspace
  collapse) — a candidate CP-6 finding figure.
  **Determinism trap (D8-class):** class means must be drawn ONCE per
  trial (seeded generator passed to both train and eval draws) — the
  first draft regenerated means per call, silently mismatching
  train/eval geometries. Same lesson as seed-before-loader.
  Readout convention: nearest-centroid (linear scores), NOT one-hot ridge
  — ridge without a bias term cannot represent ordered/interval class
  structure along a 1-D code and silently reports chance.
- **ruff 0.16 selector migration (2026-09-04, env drift fixed):** the venv
  ruff upgraded to 0.16.6, which dropped long-form rule names — pyproject's
  `ignore`/`per-file-ignores` no longer parsed and EVERY ruff invocation
  failed (`line-too-long`→E501, `magic-value-comparison`→PLR2004,
  `no-self-use`→PLR6301, `invalid-argument-name`→N803,
  `non-lowercase-variable-in-function`→N806, `raise-vanilla-args`→RSE102,
  `unused-function-argument`→ARG001, `unused-method-argument`→ARG002,
  ambiguous-unicode→RUF001/2/3, `float-equality-comparison`→PLR0133,
  `undefined-export`→F822, `non-empty-init-module`→INP001,
  `non-augmented-assignment`→PLR6104, subprocess/random S-codes, `assert`→
  S101). Selector strings are canonical codes only from here on. Repo-wide
  `ruff check` now reports ~580 findings under the (renamed but wider)
  effective set — Register C scope, not per-commit blockers.
- **`DeepHebbianChain` is local-only by design:** weights are
  `requires_grad=False` nn.Parameters (in-place Oja under `no_grad`);
  do not wire it into SystemTrainer/autograd credit — it is the local
  feature-learning arm, evaluated via readout.
- **LazyStateDynamics landing (2026-09-04):** the pre-existing class was a
  stub (single-tensor routing, `(acts**2).mean()` energy, unregistered);
  rewritten in place as a sequential Gauss–Seidel settle. Key measured
  facts: (1) per-sweep Hopfield energy is monotone non-increasing;
  (2) the nudged phase works like the Jacobi kernel (output nudge each
  sweep); (3) Gauss–Seidel does NOT converge in fewer sweeps than Jacobi
  at demo scale — measured 34 vs 21 at (256→64×6→10, τ=1e-2, step 0.05).
  The sequential sweep's value is the on-demand/memory strategy, not
  speed — scope-honest claim only. ReLU nets have multiple fixed points:
  Jacobi and Gauss–Seidel legitimately land on different ones (both are
  fixed points of the same map) — never assert fixed-point equality
  between the two settles.
- **Pyright ignore-comment convention (2026-09-04):** `# type: ignore`
  comments do NOT suppress pyright errors in this repo's config
  (pyrightconfig.json, basic mode); use `# pyright: ignore[<rule>]` placed
  on the exact line pyright reports (for multiline calls, on the offending
  argument's line, not the call opener). Ruff's PGH003 forbids bare
  `# pyright: ignore` — always name the rule.
- **Repo's `# ruff: ignore[...]` comments are invalid (2026-09-04):** ruff
  only honors `# noqa`; the `# ruff: ignore[x]` idiom is flagged RUF103
  under the current rule set and its suppressions are NOT applied. This
  is why repo-wide ruff reports ~580 findings despite "per-line markers
  self-flag on touch" — the markers never worked. Register C scope: a
  one-time sweep converting `# ruff: ignore[` → `# noqa: ` (codes must be
  translated to ruff names) would restore the intended suppression
  system; do it in the hygiene pass, never per-commit.
- **Multi-seed pilot verdict (R11.3.11, 2026-09-04 — DOWNGRADED after the
  audit):** the initial "μPC lift refuted" was premature on two counts.
  (1) Domain mismatch: the pilot ran plain MLPs, but the paper's Table 1
  parameterization is specified and tested on residual networks — the
  (N·L)^{-1/2} hidden scale assumes a skip path; without one the scaled
  branch has nothing to correct. `GeometryConfig.residual` now makes the
  paper's architecture family expressible (locked in
  `test_residual_geometry.py`). (2) Trainer mismatch: the paper uses Adam
  on weights, activity GD with β up to 100 (grid-searched), and inference
  steps = H (not convergence); ours uses Euclidean SGD, β=0.5, fixed 60
  settle steps. In-regime re-test (residual, width 128, seeds 0–2): μPC
  0.137 vs default 0.139 — no lift under our trainer, but the paper's
  optimizer/β regime is still untested. Status: OPEN, not refuted. Do not
  quote "μPC refuted" anywhere; the honest statement is "no lift under the
  computronium trainer; jpc-faithful port (Adam, β grid, steps=H) is the
  remaining instrument gap."
- **Audit lesson (2026-09-04, user-prompted):** before publishing a
  negative verdict, check the instrument against the source paper's
  stated regime. Two systematic mismatches (architecture family, optimizer
  regime) hid inside a plausible "refuted" conclusion. Refutations ship
  with the same pipeline — and that pipeline must demonstrably implement
  the claim's own terms (R11.5.5 applied to refutations themselves).
- **Ternary × gradient credit = strict-mechanism xfail (2026-09-04):** the
  property certification `test_substrate_with_backprop_credit[ternary]` was a
  silent casualty of the GradientCredit fail-loud landing — never gated after
  it. Mechanism (recon, not regression): `TernarySubstrate.quantize_weights`
  STE-quantizes *substrate-owned latent* weights (`detach().clone()`), so the
  forward graph is severed from the geometry parameters by design — no
  autograd gradient can reach them, and the pairing only ever produced
  silent zeros (no learning) under the old zero-fill. Ternary learning
  routes through the substrate update operator (`ternary_update` writes
  latent + re-quantizes); pairing ternary with gradient credit needs a
  latent-graph path — fold into R11.2.9 (`substrate_coupled` engagement) if
  a research paragraph ever wants learned ternary through the 5-D pipeline.
  Marked dynamic `pytest.xfail` with the mechanism string in
  `tests/property/test_axis_certifications.py` (R11.1.3 precedent). Lesson:
  **fail-loud changes gate the property suite before landing** — this one
  shipped and the first property run caught it a session later.
- **Test acceleration (2026-09-04, user directive "tests take too damn long"):**
  property suite now runs `pytest -n auto` (pytest-xdist, already a dev dep):
  **105 s → 55 s**, verified stable ×3. Demo gate stays **serial** — parallel
  demo runs re-emit records with float drift and trip the gallery lock
  (mechanism below). Record the canonical gates:
  `uv run python -m pytest tests/property/ -q -n auto` and the serial demo
  gate. Two RNG-order-fragile property tests were the only parallel
  failures: `test_deep_network_accuracy[100]` and
  `test_eqprop_vs_backprop_accuracy` built models from unseeded global RNG
  (pass/fail depended on which tests ran earlier in the process) — seeded;
  the former's claim was then refuted (see next bullet). Known intermittent
  `XPASS`: `test_backprop_memory_grows_with_depth[25]` (non-strict, CUDA
  memory measurement noise — pre-existing, harmless).
- **Deep-settle EqProp refutation (R11.5.5 slot filled, 2026-09-04):**
  `test_deep_network_accuracy[100]`'s claim ("100-settle-step EqProp >
  30% acc after 3 steps") is false at every seed — and MORE training decays
  accuracy further (3→10→30 steps: 0.22→0.03→0.0). It previously passed
  only via an unseeded draw. Converted to a strict-mechanism xfail citing
  the R11.3.11 boundary; this is the EqProp instance of the depth/settle
  signal-loss boundary. Candidate for a live failure figure (same pipeline)
  if the multi-seed pilot's levers rescue it.
- **Float-reduction record drift (D2/D7 absorbed, 2026-09-04):** demo
  records can drift at the 1e-7 level run-to-run — multithreaded CPU
  reduction order in some kernels varies with scheduling, especially under
  parallel test workers. Asserts are tolerance-based (green); the gallery
  lock's sha check is not. Manifest re-pinned after mechanism review via
  `render_gallery` directly. `comp gallery` itself was broken at HEAD —
  its CLI imported `_FACTORIES`, which the gallery refactor renamed to the
  `DEMOS` registry; fixed (`computronium/cli/gallery.py`). Per retro (e):
  when the lock fires, diagnose (isolation re-runs, byte diffs) before
  re-pinning — both drift classes this session were diagnosed, not pinned
  blind.
- **P-axis standardization (user directive, 2026-09-04):** "M-axis" is
  retired — the plasticity axis is the **P-axis** everywhere (code
  docstrings/comments, demo D3 wording, gallery figure titles, TODO/docs;
  archives untouched). README's uncommitted M→P edits are the user's own
  and stay. No `M`-prefixed identifiers exist in code, so this was a
  documentation-level sweep (19 files).
- **#2–#5 session (2026-09-04):**
  - **Deep Hebbian probe** (`scripts/probes/deep_hebbian_chain.py` — read
    its docstring): the hebbian tile chain *explodes* at depth, not
    decays — per-layer gain ~1.2–1.5× at init compounds to inf by depth
    500, and one Hebbian local_update NaNs the activities (positive
    feedback, no gain control). Track 54's "maintains signal through 50
    layers" is unverifiable at HEAD — its
    `measure_signal_propagation` method no longer exists. Sharpened CP-6
    thesis: the depth bottleneck is **structural gain control** —
    error-based rules die by telescoping decay, unnormalized local chains
    by runaway gain, μPC's parameterization IS the normalization. Next
    lever: unit-layer-gain init or homeostatic scaling on the tile chain.
  - **Compiled LIF settle (R11.2.25 extension):** `spike_integration(..., compiled=True)`
    runs the per-layer LIF loop as one graph — bitwise parity on
    membranes, spike counts, and rasters (`test_compiled_settle.py`, now
    7 locks). **Demos stay eager**: flipping D7 to compiled busts its
    60 s timeout (compile warmup ~60 s vs 0.3 ms/step saved at demo
    scale) — compiled settle paths are registered-scale levers only.
  - **Wheel acceptance (R11.4.1/CP-5):** pyproject's flat `packages`
    list shipped only top-level modules — fixed with
    `packages.find` (subpackages verified in the wheel). New
    `tests/integration/test_wheel_acceptance.py`: builds the wheel,
    installs into a fresh venv (torch via `--system-site-packages`),
    runs the stranger's first minute (import → compose → `SystemModule`
    forward → `fit_step`). CP-5's pip-packaging door is now demonstrably
    open.
- **Deep Hebbian lead — SUPERSEDED by the probe (2026-09-04):** the
  "hundreds of layers" recollection is not realized by the current
  `DeepHebbianChain` implementation (see probe findings above). The lead
  survives as the gain-control thesis; track 54's evidence string is
  orphaned history — retire or re-home it on next touch of
  `nebc_tracks.py`.
- **Metric aggregation contract (R11.2.23, pulled 2026-09-04):**
  `SystemTrainer.train_epoch`/`validate` now accumulate **sample-weighted**
  sums (`trainer.py`) — a ragged final batch no longer counts as a full
  batch-weight — and `validate()` adds `val_ppl = exp(mean CE)`. Epoch
  numbers shift microscopically vs old records (only the ragged batch
  differs); demo gate + gallery lock re-verified green at the landing. The
  lock (`tests/unit/core/test_trainer_metric_aggregation.py`) checks the
  weighted-mean identity through a delegating spy (`_SpySystem`) because
  `_ComposedSystem` attributes are read-only — reuse that pattern for
  trainer instrumentation instead of mocks.
- **GradientCredit fail-loud (2026-09-04):** detached-weight zero-fill is
  gone; `BackpropCredit is GradientCredit` (alias). Anything that relied on
  silent zeros now raises. `LocalGoodnessCredit` intentionally keeps
  `allow_unused` (surplus recurrent self-connections).
- **SystemModule (R11.4.1 v1, 2026-09-04):** `computronium/nn/system_module.py`,
  exported root + `computronium.nn`. Training stays credit-internal —
  `fit_step`, never `loss.backward()`. Remaining for CP-5: pip packaging,
  acceptance test (RESEARCH3 PR sequence), `to(device)` passthrough if a
  consumer needs it.
- **Remaining pull-based items:** R11.1.10, R11.1.11, R11.2.9/13/14/16, R11.3.4–3.11, R11.3.13, R11.4.1/4.3/4.4. Land only when demo/campaign/research needs them.
- **F1 landing (2026-09-04):** the first *finding* figure follows the demo
  pattern exactly (static `_ARMS`/depth tables at module scope, helpers
  extracted to satisfy PLR0915, one `DEMOS` row `_fig_failure_manifesto`,
  gallery re-pin, RESULTS.md front paragraph). Conventions worth carrying:
  (1) capability id **F-series** ("F1") marks findings vs capabilities —
  future refutation figures continue it; (2) the D8 trap recurs per
  arm-loop: seed **before** the loader draw, then per-arm reseed for model
  init — single-seed `spc_mupc` vs `spc` diffs are order-dependent
  otherwise (first run measured a 0.06 "lift" at depth 4 that vanished
  once the shuffle draw was pinned); (3) the μPC arm's assert encodes
  "no lift" as `mupc ≤ spc + 0.05` with the OPEN verdict in the message —
  keep that phrasing if the multi-seed pilot lands; (4) `TileAlgorithm`
  (core/local_learning) builds a depth-100 chain in <1 s — the cheap
  runaway-gain instrument. The four arms run in ~12 s total; keep F1
  lean if arms are added.
- **F1 audit revision (2026-09-04, user skepticism — instrument before
  verdict):** a follow-up probe (`scripts/probes/failure_manifesto_audit.py`)
  checked the negative arms for implementation defects before letting them
  stand. Findings: (1) BP's depth-8 collapse is NOT an lr artifact (flat
  0.108–0.112 across lr 0.02–0.2); (2) the sPC wall IS budget-sensitive
  (0.098 at 15 settle steps → 0.212 at 60) and its mechanism under our
  layered settle is **last-layer-only training** — per-layer thermo-contrast
  norms are exactly 0.00 for every hidden weight matrix (now asserted live
  in the F1 record), so the "depth wall" for sPC is the random-feature
  readout boundary of last-layer-only training, not hidden-credit decay.
  This is consistent with D12 (hidden nudged deviations exactly zero) but
  reframes the R11.3.11 frontier's sPC arm: depth enters through
  random-feature quality only. Whether a hidden-layer contrast is
  achievable in a layered settle at all (e.g., settle-time error
  propagation or a PC-native weight gradient (∂ŝ/∂θ)ᵀε) is OPEN — a
  candidate instrument improvement, not a physics verdict. The failure
  arms are now framed as "under this instrument regime" everywhere
  (test docstring, RESULTS.md, figure title).
- **F2 landing (2026-09-04):** the gain-control thesis now spans THREE
  families: error-based credit (last-layer-only under the layered settle),
  tile-Hebbian chains (init runaway 1.4→3.2e5), and spiking STDP (readout
  0.36→0.18 under unsupervised potentiation). All three are instrument-
  audited, not assumed: the spiking audit (spiking_gain_audit.py) swept
  threshold/step/init_scale and located the silence. Convention: the
  confound (silent layers) and the mechanism (no error path, runaway gain)
  are separate record fields with separate asserts — never let a fix to the
  instrument silently upgrade a mechanism claim. Next lever, pull-based:
  reward-modulated or error-gated STDP (a supervised TemporalTrace variant)
  — the C-axis slot for a spiking capability claim.
- **D13 landing (2026-09-04):** the audit trail matters more than the
  number. Measured facts to carry: (1) reduced QR is NOT the polar factor
  — its R-diagonal is sign-arbitrary, cos(ortho, grad) measured ≈ 0, and
  every "Muon" lr trained at chance; the SVD polar factor (U @ Vh) gives
  cos 0.55–0.80 and instant learning — the property lock
  `test_muon_polar_factor_is_descent_aligned` (cos > 0.4) is the ratchet;
  (2) orthogonalization MUST follow momentum accumulation (Muon's recipe);
  raw single-batch orthogonalization amplifies the noise floor — BP×Muon
  at chance was the symptom; (3) `EuclideanUpdate._momentum_buffers` were
  keyed by bare parameter name and silently corrupted on cross-geometry
  reuse (campaign/HPO/probe hazard) — shape-mismatch now fails loud;
  (4) **`local_objective` is dead config**: it appears only in
  `CreditAssignmentConfig`, no credit reads it — FF and PEPITA currently
  run byte-identical pseudo-gradients (the D13 record shows identical
  numbers). The factory-table FF/PEPITA distinction is NOT realized in
  code — a real capability gap (new improvement opportunity below).
  Single-seed: the 0.85-vs-0.26 effect is far beyond the seed noise that
  faked the μPC 2× claim, but multi-seed verification is still pending —
  do not quote outside RESULTS until then. As-touch lever: the
  create_ff_mlp/create_pepita_mlp factories hardwire EuclideanUpdate —
  expose update choice once the local_objective gap closes.
- **Single-seed audit scope of F1:** every number in the F1 record is
  single-seed but *live* (re-demonstrated each demo-gate run), and the
  assertions are comparative margins (decay deltas, ratio monotonicity,
  ≤ spc + 0.05) rather than headline values — the seed-noise failure mode
  that faked the 2× μPC claim cannot flip an assertion silently. The
  registered-scale multi-seed pilot (R11.3.11 tail) remains the
  research-grade instrument.
- **Resumable trainer (R11.2.24, landed 2026-09-04):** `fold_in(base, epoch, batch, *, domain)`
  (SplitMix64, `computronium/core/system_trainer/_resume.py`) + `TrainerSnapshot`
  (epoch, global_step, history, theta, opt_state). `SystemTrainer(resumable=True)`
  reseeds the global torch RNG per epoch (domain `DOMAIN_EPOCH`, fixes the
  DataLoader shuffle draw) and per batch, so *every* downstream draw — shuffle,
  substrate noise, projection masks — is a pure function of coordinates.
  Resume: `snap = trainer.snapshot()` → `SystemTrainer.from_snapshot(system=…,
  config=…, train_data=…, snapshot=snap)`; `max_epochs` counts **total** epochs.
  Opt-in flag: `resumable=False` (default) leaves legacy trajectories byte-for-byte
  unchanged — do not flip the default without re-pinning all demo records.
  Restores into `EuclideanUpdate._momentum_buffers`; updates without optimizer
  state restore an empty opt_state (fail-loud `TypeError` if snapshot has state
  but the update object has nowhere to put it).
- **Follow-ups unlocked by R11.2.24 (as-touch / pull-based):**
  campaign episodes (R11.3.1) can set `resumable=True` to make kill→resume
  bitwise rather than statistically equal; `CheckpointManager`'s global
  RNG-state capture (`checkpoint.py`) becomes redundant per-episode once
  trainers run resumable — retire it only when no consumer needs stream-position
  resume; `fold_in` is the canonical seed derivation for any future per-batch
  keyed randomness (probes, campaign shard seeds).
- **Lint/type debt deprioritized:** ruff clean passively; pyright on new modules only. Legacy findings carry per-line noqa markers that self-flag on touch.

### New Improvement Opportunities (opened 2026-09-04, pull-based)

- **FF / PEPITA as distinct algorithms (capability gap, opened by the D13
  audit):** **LANDED 2026-09-05** — `local_objective` is wired as
  `Literal["ff","pepita"]` and read by `LocalGoodnessCredit`; see plan
  item 4 for the measured outcome and ratchets.

- **R11.5.5 failure-manifesto paragraph (CP-6 candidate):** **LANDED
  2026-09-04, audited 2026-09-05 — CLOSED as a physics claim.** F1's
  depth wall dissolved under the jpc-faithful regime (D14: depth 20
  trains to ≈ 0.8 test); F2's STDP collapse survives homeostatic scaling
  and stays OPEN pending reward-modulated STDP (a real candidate finding,
  properly conditioned). The F1 figure remains at HEAD as the measured
  boundary of the sPC/thermo/Euclid instrument — regime-scoped, never
  physics. The three-session arc (finding → audit → resolution) is the
  R11.5.5a doctrine working as designed in BOTH directions: the μPC audit
  stopped a false positive, the D14 audit stopped a false negative.
- **Suppression-system repair (Register C):** convert the repo's invalid
  `# ruff: ignore[x]` comments to working `# noqa` codes (they currently
  suppress nothing). One-time sweep; unlocks the "self-flag on touch"
  mechanism the lint directive depends on.
- **jpc-faithful μPC trainer port (R11.3.11 tail):** the remaining gap
  for a clean μPC verdict — Adam (or tuned-η) weight optimizer, activity
  GD with a β grid (paper grid 1e3→1e-2), inference steps = H, width 512.
  Reference: github.com/thebuckleylab/jpc. Also verify how jpc applies the
  output premultiplier a_L = N^{-1} before a CE-softmax readout (the
  paper clamps z_L to y with MSE; naive 1/N logits may underflow CE —
  check before porting the scale to our output layer).
- **Single-seed audit: every accuracy number quoted in probe docstrings
  and RESULTS.md back-section is single-seed. The μPC refutation shows
  seed noise can fake 2× effects. On next touch of any registered-scale
  claim, add a second seed or mark it explicitly unverified.
- **`settle`/`compute_energy` type skew:** the StateDynamics Protocol
  declares `CompositeState` but the pipeline (and every demo) passes
  `SystemState` — every new consumer needs pyright-ignore noise. Fix the
  Protocol annotation to `SystemState` (or a union) on next touch of
  `_dynamics.py`.
- **`DeepHebbianChain` readout helper:** the deleted one-hot ridge helper
  was the wrong evaluation for 1-D-coded classes; if a future demo needs
  ridge readouts, add a bias term (this is why nearest-centroid is the
  convention now).

### Sprint Retro (2026-09-03, binding for future sessions)

- (a) Tests run **once at close** — mid-session gates are ruff + pyright on
  changed files only (seconds); behavioral questions get throwaway probe
  scripts; file moves/renames get grep + pyright, no test runs at all.
- (b) Any signature/config break (required fields, renamed identifiers):
  AST-walk the *entire* repo including `tests/` for call sites before
  finishing the item — eyeballing three test files missed
  `test_power_preregistration.py` this sprint.
- (c) Behavior inherited from deleted code is not automatically correct —
  when the recon shows the old implementation was itself the debt, land the
  fail-loud upgrade, don't preserve a silent fallback.
- (d) When a plan item is phrased as either/or ("fold into X or drop"),
  cross-check the repo's current naming conventions before picking a
  direction; plan phrasing can lag the codebase (the equitile→tile case).
- (e) **Diagnose nondeterminism at the data layer first** — a figure-lock
  drift that survives re-pins is an unseeded draw (D8's loader shuffle),
  not a manifest problem; three re-pin loops were spent before the seed
  order was checked. When the same sha keeps changing: seed the stream,
  run the demo twice, compare bytes — never re-pin between.
- (f) Fairness/capacity requirements arrive mid-item and reshape the demo
  (conv ≈1/10 params) — apply them by re-balancing the *weaker* arm, not by
  asserting superiority of the over-provisioned one.
- Work lean: one Register item per landing, each with a test that
  demonstrates it. Don't pull infrastructure "just in case".
- RESEARCH3 protocol (E-1 smoke → pilot → full; E-11 DECISIONS.md) governs
  every R11.3 pull. Infra-failures don't consume tuning rounds.

---

## 🚪 The R12 Fork (2026-09-04 — decision point, not work)

R11 is **core complete**. D1–D12 demonstrate every axis; `comp repro` 8/8;
property suite, demo gate, gallery lock green at HEAD. What remains is not
cleanup — it is choosing which future the completed library serves. The
remaining checkpoints are three different futures:

| Future | Checkpoint | What it produces |
|--------|-----------|------------------|
| Make it visible | CP-5 Adoption | Wrapper / UI — someone who isn't you can use it |
| Make it say something | CP-6 First finding | P-axis Pareto frontier over 𝒞 — first figure that is a *finding*, not a demonstration |
| Big swing | CP-7 Discovery bet | Z3 flagship, pre-registered with fallback |

**Standing recommendation: CP-6 before CP-5.** The instrument was made
honest at real cost; the payoff of honesty is a finding, and adoption
follows what the instrument shows, not its surface. The backlog already
points there — μPC depth scaling (R11.3.11), depth-metric classes
(R11.3.13), ePC's deep-stack credit limitation (Notes: contrastive ÷β)
are all deep-EqProp-boundary territory, and the PR-5 stability guard is
calibrated and idle. Demo-scale machinery + registered-scale GPU (conv
speedup measured) is exactly the sweep CP-6 needs. CP-7 rides CP-6's
findings; CP-5 stays pull-based until a finding gives people a reason to
adopt.

**Decision (2026-09-04, user): proceed with CP-6 first — and all three
options will eventually be built.** Sequencing commitment: CP-6 → CP-5 →
CP-7, with each door pulled when its predecessor lands a reason. First
concrete step: R11.3.11 + R11.3.13 as one landing (μPC init + depth
metrics), probe-first per RESEARCH3 E-1.

---

## 🔬 CP-6 Execution Doctrine (2026-09-04 — external strategy review, applied)

R11 core-complete means the posture shifts: not building a library —
operating a completed, verified, honest instrument. Three priorities,
everything else ruthlessly ignored while CP-6 runs.

### 1. Prime objective — interrogate the deep-EqProp boundary (R11.3.11 + R11.3.13)

The library treats locality, energy, and physical constraints as
first-class; plain BP and plain PC both decay through deep local-learning
regimes (~10 layers). The question CP-6 answers: do **μPC** (depth-scaled
init) and **ePC** (error reparameterization) actually solve this on
non-trivial topologies, or merely shift the failure mode?

- E-1 smoke probe **done** (`scripts/probes/mupc_depth_init.py` — read its
  docstring before re-deriving: boundary at depth ≥ 8 is not PC-specific;
  μPC ≈ 2× PC learning at depth 8 under a real budget).
- Implement `ShortestPathDepth`/`LongestPathDepth` (R11.3.13) so "depth" is
  measurable per-node on `GraphGeometry`/`TileMesh`, where layer-counting
  fails; then sweep μPC × ePC × substrate-noise across depth.
- Deliverable: *the exact depth and substrate-noise constraint at which
  local credit physically breaks down vs global backprop.*

### 2. Leave the CPU sandbox — registered-scale GPU campaigns

The demo suite (D1–D12) is CPU/kernel-launch-bound by measured verdict; it
proves the *ontology*, not a *finding*. Registered-scale work runs GPU with
the `torch.compile` settle fast paths (R11.2.25) already landed; conv-family
is the measured 15× FLOP-bound path.

- Commission AutoScientist on a registered-scale sweep of the
  **stability–plasticity frontier**: map the 𝒞-vector Pareto frontier
  (compute, memory, energy, latency, plastic-state capacity) across
  plasticity primitives (Routing vs FastWeight vs Null).
- Deliverable: one figure showing the trade-off between settling time
  (dynamical latency) and basin stability per plasticity primitive — the
  first *finding* (checkpoint 6), not a demonstration.
- PR-5 stability guard is calibrated and idle — wire
  `probe_interval_for_overhead` into the campaign loop when it starts.

### 3. Hunt for refutations (standing rule R11.5.5)

The goal is the *physics of learning*, not benchmark wins over PyTorch —
accuracy horse-races are solved and boring. A negative result from the deep
μPC sweep **is the finding**: publish it in the failure manifesto with the
same pipeline, same figures, same terms (*"local energy-based learning
collapses at depth L > 12 due to X; μPC scaling fails to rescue it because
of Y"*). A rigorously documented failure boundary for local, asynchronous,
energy-minimizing systems is a real contribution to neuromorphic and
biologically-plausible ML. The instrument stays honest even when the
hypothesis dies — and the spiking-family learning slot (R11.5.5) remains
open for exactly this kind of live refutation.

---

## ⚡ Performance Proposals — Evaluated (2026-09-04 external review)

Assessed against measured regime facts (demo suite is CPU and
kernel-launch-bound, not FLOP-bound; conv-family is the first GPU-bound
path at 15×; `KernelRegistry` already hosts Triton-family kernels in
`acceleration/`). All are **pull-based per R11.5.6** — perf work lands
when a registered-scale study or campaign needs it, never speculatively.

| Proposal | Verdict | Note |
|----------|---------|------|
| `torch.compile` on settle/credit paths | **Viable, pull with a registered-scale GPU study** |compile helps FLOP-bound paths; the demo suite's Python settle loop is launch-bound and CPU-pinned — compile would not move the gate. Caution: settle loops have data-dependent convergence breaks and `no_grad`/`enable_grad` context switches; wrap whole-settle, not per-step. |
| `torch.vmap` over settle steps | **Rejected (category error)** | Settling is sequentially dependent; `vmap` maps over batch axes, not time. Batch parallelism already exists via the loader. |
| Triton kernels for substrate ops | **Viable, as-touch** | `acceleration/` kernel infrastructure exists (snn/contrastive kernels); extend when a commissioned study's profile shows the operator is the bottleneck. Not TODO.md R4.3 (that is `UV_LINK_MODE`). |
| `system.compile()` static-graph export | **Pull with CP-6/CP-7** | Only worth building when campaign fleets hit abstraction overhead at registered scale; measure first (profiling infra exists). |

---

## Termination Criterion

R11 closes when a stranger can, in one sitting: compose a system from *any*
geometry, substrate, and dynamics the ontology declares (not just the
Feedforward/Recurrent/Digital/Memristive/EnergyMinimization set R10
demonstrated), watch it train in the demo suite (the live UI is a separate
adoption round, R11.4.3 — presentation, not a library-completeness gate),
and find
the repo's own gates — ruff, pyright, property locks, demo gate, figure
lock — green at HEAD without caveats. The library is then *complete relative
to its own ontology*: every axis declares only primitives that exist and
demonstrate themselves. Research claims remain where they belong — the
corroboration appendix and RESEARCH3, pull-based, never the front page.