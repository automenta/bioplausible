# TODO9.md — Active Plan

> **Opened 2026-09-01.** Successor to [TODO8.md](TODO8.md) (closed 2026-09-01 — all phases complete).
> Research catalog: [RESEARCH3.md](RESEARCH3.md). This doc owns the R6 handoff, the pull-based
> backlog, and the deferred register.
>
> **State:** **R6 complete 2026-09-01** — PR-1 ✅ · PR-2 ✅ (pre-existing, verified) ·
> PR-3a ✅ (pre-existing, verified) · PR-4 ✅ (pre-existing, verified) · PR-7 shakedown ✅
> (L3.5 → L1 → L2/L3 all run end-to-end, metrics + PR-3a resources populate). Next per
> RESEARCH3 critical path: **Z3 flagship** (CP-A), with PR-7's harvested configs feeding
> PR-5 guard calibration. Verification: touched-file tests 3 passed / 0 failed,
> shakedown suites green, ruff/pyright net-zero on touched files.
>
> **R7 opened 2026-09-01 — instrument-honesty sweep.** The registered null is **reclassified:
> a symptom, not a result**. Ten suspect-instrument areas, one probe each (R7 below); no
> interpretation of any null while a suspect remains unprobed. Termination criterion:
> **if it works it will be obvious.**
>
> **R7 first pass 2026-09-01 (probes 1/2/6/7):** 2 PASS (rotation parity, construction seeding) ·
> 1 DEFECT FOUND + FIXED (12 silent dispatch fallbacks in the 5-D compose path — imp-25 class
> eradicated) · 1 DEFECT FOUND (`backward_flops` ≡ 0; energy recorded state energy) ·
> data-confirms imp-36 saturation. The sweep produces signal.
>
> **R7 second pass 2026-09-01 (first-pass closures):** all four closed — ① rotation invariant
> **pinned** (`TestTaskRotation`) · ② compute/energy semantics **fixed** (total train-step MACs
> incl. documented 2× backward estimate; state vs consumed energy split; drift-proof derived
> alias) · ③ **campaign exposure audit CLEAN** — all six commissioned campaigns (1200 artifact
> records + 62 SQLite coords): zero silent-substitution exposure, defect was latent · ④ full
> gate **1258 passed / 0 failed**, ruff/pyright net-zero. Remaining probes: 4 → 5 → 3 → 9 → 8
> + positive control (imp-52).
>
> **R7 third pass 2026-09-01 — sweep complete: all registered probes run.** #4 metric
> honesty census → provenance table (`docs/metric_provenance.md`), pipeline schema closed
> (`METRIC_SCHEMA`), `accuracy` → `nudged_fit_accuracy` everywhere, **1 defect found+fixed**
> (distributed trainer's `train_acc` silently constant 0.0) · #5 settle caller census →
> 26/26 sites bind the returned state, AST lint lock pinned · #3 ψ engagement locks →
> pipeline-level chain green (ψ moves → modulate reaches activations → frozen-ψ control
> changes metrics) for routing/fast_weights · **imp-52 positive control → instrument sees
> the planted effect** (lr=0 stays chance, lr=0.1 hits ceiling on a stationary task, stable
> across seeds; task-stream non-stationarity property discovered + pinned) · resource
> revalidation → compute/energy vary, nonnegative, state split holds · #9 power audit →
> **UNDERPOWERED** (MDE@80% = 0.314 vs top observed d = 0.205) · #8 probe-the-probe → 18
> engineered broken variants fail their probes, correct ones pass. Gate **1300 passed /
> 0 failed** (42 new lock tests), ruff/pyright net-zero on touched files.
>
> **R7 closes 2026-09-01 with a meaning change: the null is now explained, not merely
> uninterpreted.** The instrument sees a planted effect end-to-end through the campaign claim
> chain (imp-52); the commissioned campaign was structurally incapable of producing a
> learnable signal (imp-54: teacher redraw every episode) and underpowered even for the
> per-episode effects it did show (imp-55: MDE@80% 0.314 vs top observed d 0.205). The known
> code-level instrument lies are fixed and probe-the-probe locked. **The defect hunt moves
> from code defects to experimental-design defects** — successor plan: **R8** below. The old
> null stays uninterpreted; it is a property of the old design, not a finding.
>
> **R8 Track 1 complete 2026-09-01 — Z3 flagship gate landed (R8.1 + R8.2).** Every
> `evaluate_z3` run now embeds its own engagement gate + planted-ψ control: exact θ
> invariance (bitwise; was 1e-6-tolerance; `theta_sha256` emitted for artifact-level
> checks), ψ non-constancy + task-conditioning (per-task ψ trajectories + final ψ
> vectors recorded), a mechanistic ψ→gate wiring probe (isolated RNG, stream-clean), an
> RNG-aligned frozen-ψ control arm (identical meta-state/task/seed/budget — only ψ
> stepping differs), and per-task above-chance probe accuracy (chance 0.5 + margin 0.1).
> Gate PASSES at registered scale (50 meta/20 eval) across 3 seeds; the planted effect is
> large — engaged vs ψ-frozen gaps: parity +0.32..0.42, last_symbol +0.20..0.34,
> threshold +0.04..0.14. The ψ-disabled arm (recipe `feedback=False`) is the
> engineered-broken variant: the gate flags exactly the ψ items and its embedded control
> is bit-identical to treatment (probe-the-probe). θ sha256 identical across
> engaged/disabled arms — the plant is non-confounded. `claims_scope: psi_engaged` is
> emitted per run and self-downgrades to `plumbing_only` on gate failure. Locks:
> `tests/property/test_z3_engagement.py` (12). Quick-scale read (10 meta epochs): only
> last_symbol above chance — a capability/scale fact (the quick budget cannot acquire the
> decoder), NOT an instrument defect; the gate is a registered-scale instrument.
> **The Z3 flagship registered run (RESEARCH3 CP-A) is unblocked.** Gate **1317 passed /
> 0 failed** (was 1300; +17 R8 locks), ruff/pyright net-zero on touched files.
>
> **R8 Track 2 started 2026-09-01 — R8.3 Option A foundation.** `episode_batch` gained
> `teacher_key`: the synthetic teacher derives from (campaign_id, coordinate, seed)
> alone — stationary across episodes (accumulated learning becomes representable) while
> inputs keep varying per episode; `evaluate_episode(stationary_teacher=True)` threads
> it through. The legacy per-episode redraw is preserved byte-for-byte (commissioned
> artifacts stay reproducible) and the imp-54 non-stationarity stays pinned for
> per-episode-adaptation scope. Behavioral lock: a linear probe trained on early episodes
> generalizes >0.9 to late episodes under the stationary teacher and stays at chance
> (<0.35) on the legacy stream (`tests/property/test_stationary_teacher.py`, 5 locks).
> **Completed same day — R8.3 pilot + calibration, R8.4 label gate, R8.5 controls: see
> the R8 Track 2 completion record above; R8.6 re-scoped → R9.**
>
> **R8 Track 2 complete 2026-09-01 — R8.3/R8.4/R8.5 landed; the instrument is
> qualified.** R8.3: `stationary_teacher`/`teacher_noise` threaded through
> `CampaignStack` (config-recorded, provenance-stamped per record
> `teacher_stationary`/`teacher_noise`); difficulty calibrated —
> `CALIBRATED_TEACHER_NOISE = 0.5` puts the oracle at ≈0.86 (band 0.125..0.86,
> no saturation headroom problem); **stationary pilot run** (persistent-θ arms,
> 40 episodes × 3 seeds, registered shape): controls PASSED both variants,
> per-arm variance measured (`benchmark_results/stationary_pilot_{noiseless,
> calibrated}.json`). R8.4: `validation/power_preregistration.py` —
> `PowerPreregistration` (effect/variance/n/α/stratification/scope/stream),
> derived `mde_cohens_d`/`mde_metric` + `n_for_target_power`, label gate
> (`claim_grade`/`pilot`/`plumbing`/`instrument_check`; a declared rung CAPS
> the label even when gates pass; accumulated_learning requires the
> stationary stream per imp-54), JSON round-trip; `CampaignStack.run_campaign`
> records label + prereg in the campaign config. R8.5: embedded positive
> controls enforced — claim-grade requires a declared control arm; the
> `frozen` (lr=0) update value composes as a planted control coordinate;
> post-run `verify_embedded_control` over the campaign's records; failed or
> missing control → `CampaignRunResult.quarantined`. Gate **1339 passed /
> 0 failed** (was 1317; +22 locks: `test_power_preregistration.py` 13,
> stationary threading/noise/calibration 9), ruff/pyright net-zero on touched
> files.
>
> **R8.6 re-scoped → R9.** The pilot's power numbers (routing deficit d≈1.8 →
> n≈6/group; fast_weights vs null d≈0.17..0.47 → n≈72..529 at 40 episodes)
> make one thing plain: effect size is a design property, not a scaling
> property. A registered re-commission of the same M-axis contrast on a
> stationary stream would buy a smaller-n version of a boring question. Per
> the 2026-09-01 strategic review: **R8 is instrument qualification; the
> discovery phase is R9 — surgical stress tests in regimes where Backprop's
> superpowers are liabilities and the ontology's axes are strictly
> necessary** (R9 below). R8's gates (prereg label, embedded controls,
> fidelity re-evaluation) are R9's commissioning machinery.
>
> **Numbering:** improvement items continue TODO8's append-only ledger from **imp-42**;
> imp-1..41 remain canonical in TODO8.md.

## Policy (carried from TODO8; extended by R7/R8)

- Zero backwards compatibility · GPU-first for all training paths · serial pytest only (xdist hangs in this env)
- No new tests for broken capability — xfail with precise reasons
- **The System's own ParameterUpdate owns Δθ — external torch optimizers must not drive composed systems** (custom-loss harnesses route through `core.pipeline.apply_autograd_update`)
- **No scientific conclusion from any campaign delta until both arms pass an implementation-fidelity check.** A failed fidelity check is *inconclusive*, never a refutation. Deltas on known-defective axes are quarantined from attribution, not interpreted.
- Observed-but-unregistered deltas are never interpreted (pre-registration precedes comparison)
- **A null is a symptom, not a result.** No null is read as evidence of no effect while an R7 instrument suspect remains unprobed; each probe either finds a defect (fix it, re-run the affected campaign tier) or passes (pin it where cheap, move on)
- **Construct validity gates claims, not just metric honesty (R8).** Metric honesty (imp-46) and construct validity (imp-54) are different: honest metrics on a non-stationary stream still cannot support accumulated-learning claims. Every campaign declares its claim scope (per-episode adaptation / accumulated learning / resource-efficiency / stability / M-axis plasticity) up front
- **Power preregistration (imp-55).** A commission that does not state expected effect size, variance estimate, n/group, and MDE@80% — or that sits below the power floor — is labeled `pilot` / `plumbing` / `instrument-check`, never claim-grade
- **Embedded positive controls (imp-52 extension).** Claim-grade campaigns carry a planted-effect control arm (e.g. lr=0 coordinate, ψ-frozen arm) inside the commission itself; if the embedded control fails, the campaign is quarantined — every campaign self-validates
- **Termination criterion — "if it works it will be obvious":** the evidence of a working microscope is a robust, replicable, stratified-stable effect that survives fidelity filtering, a Pareto frontier with real tradeoffs, and ψ engagement with measurable behavioral differences — keep probing until the result is obvious

## ✅ Carried Complete (detail: TODO8.md Completed Record + git log)

P0–P5 (registry, trainers, smoke, quarantine, substrate-native settle, campaign schema + pyright policy) ·
R1 (device threading, auto-device runners, placement guard, construction seeding; EqProp MNIST ≈ 5.6 s CUDA) ·
U-bypass sweep · R2.1/2.3–2.6 (zoo retirement, registry unification, xpass split, skip census) ·
R3.1–3.3 + R3.9 (diffusion autograd, credit bodies, FA repair, D×C fencing; fidelity manifest 48/72) ·
R5.1a/b/c (smoke/quick/replication campaigns; kill→resume lifecycles; golden manifests) · R5.3–5.5 ·
R5b-0/A–F (fidelity gate, retrospective audit, instrument repairs imp-17..24, pre-registration,
locked campaign, defect-filtered evidence chain, discovery locks, discovery report + live demo tab).
Registered scientific state: **null claim** (top pooled effect stratified-unstable at the 0.05 floor).

## 🎯 R6 — RESEARCH3 Handoff — ✅ COMPLETE (2026-09-01)

**DoD (RESEARCH3): PR-1 + PR-2 merged · PR-7 shakedown green — met.**

Audit-before-building findings (2026-09-01): most of the startup sequence already existed from
R5b work — PR-2/PR-3a/PR-4 were built during the campaign/fidelity push under different names,
and PR-1's Z3 site was fixed in the v4 redesign (2026-08-26). Remaining gaps were the two
non-Z3 shakedown suites.

| # | Item | Status |
|---|------|--------|
| PR-1 | Optimizer-phase hygiene: no Adam state crosses a phase boundary | ✅ Z3 site pre-fixed (v4 per-task rebuild: `_adapt_all_tasks`, fine-tune baseline, meta-train warmup). **This session:** `adaptation_efficiency.py` (fresh Adam at A→B boundary — was one optimizer across both phases), `algorithm_migration.py` (fresh Adam at A0→A1 task boundary), `structural_robustness.py` (fresh optimizer per damage scenario — was shared across all three). `compute_efficiency.py` N/A (single phase, no boundary) |
| PR-2 | θ-invariance audit harness | ✅ pre-existing — `core/plasticity/theta_audit.py` (`ThetaInvarianceAudit` ctx mgr, `ThetaAuditReport`, `require_frozen`) + `tests/unit/core/test_theta_audit.py` + consumers (`z3_fixed_weights.py` PR-2 audit, `_claims.py`, `failure_manifesto.py`, `scripts/z3_reverification_audit.py`) |
| PR-3a | Software resource instrumentation in every suite runner | ✅ pre-existing — `resources.py::ResourceUsage` (canonical 5-axis record) + `core/profiling.py::measure_suite_resources` wired into all four shakedown suites + `benchmarks/algorithm_migration.py`; JSON artifacts verified to carry `compute/forward_flops/wall_time_ms` (+ full detail fields) |
| PR-4 | Statistics kit | ✅ pre-existing — `validation/statistics.py` (bootstrap percentile + BCa CI, Cohen's d/dz, Cliff's δ, BH-FDR, power analysis, permutation test, Fisher one-sided) + `docs/preregistration_template.md` + `configs/preregistrations/` (R5b-A JSON is a live consumer) |
| PR-7 | Switching-machinery shakedown (L3.5 → L1 → L2/L3, smoke scale) | ✅ green 2026-09-01 — all four suites complete end-to-end in `--quick` mode, every metric populates, `resources` present in each emitted JSON. L3.5: migration + θ-change reported. L1: adaptation times + accuracy reported. L2: gate-entropy/effective-FLOPs discriminate routing (1.0 active routes, 87.5% FLOPs reduction vs 8.0 dense). L3: pre-damage/recovery/final accuracy reported. Configs harvested into `benchmark_results/*/` for PR-5 calibration (pull-based — PR-5 guard exists; recalibrate only if a campaign needs it) |

**PR-7 shakedown notes (instrumentation-quality, claims-scoped — not defects for plumbing runs):**
all M-arms report identical means in L1/L3 (`PlasticityModulatedModel`/`SimpleMLP` harnesses build ψ
but recovery/adaptation timing is θ-optimizer-driven); the ontology-pipeline path (`run_train_step`
with imp-22 ψ wiring) is where M-axis contrasts become interpretable — see imp-43 before any L1/L3
*claim*. L3.5's nonzero θ-change is by design there (both phases train θ; the Δθ=0 claim lives in Z3).

## 🎯 R7 — Instrument-Honesty Sweep (the null is a symptom) — CLOSED 2026-09-01 (superseded by R8)

The registered null (top pooled effect stratified-unstable at the 0.05 floor) is **not interpreted**.
If the microscope worked, the evidence would be obvious: a robust, replicable, stratified-stable
effect surviving fidelity filtering; a Pareto frontier with real tradeoffs; ψ engagement producing
measurable behavioral differences. Everything dissolving into "inconclusive"/"unstable" is evidence
the instrument still lies somewhere. The job is not to interpret the null — it is to find the next
defect.

**Rule of engagement:** take the next probe → if it finds a defect, fix it and re-run the affected
campaign tier; if it passes, pin it as a lock test where cheap and move on. Discovery locks and the
fidelity manifest re-evaluate on any re-commission — never silently preserved (R5b-D rule). Z3
flagship (CP-A) proceeds in parallel only on validated instruments.

| # | Suspect | Probe | imp | Status |
|---|---------|-------|-----|--------|
| 1 | Task rotation structurally broken (R5b-B 0/48 ancestor) | Every coordinate in a multi-family grid campaign appears in **both** task families across its scheduled visits — assert over commissioned artifacts, pin as lock test (rev d fixed the rotation; the invariant itself is unpinned) | imp-44 | ✅ PASS + LOCKED 2026-09-01 — 48/48 coords / 240/240 strata both families; engine-level lock pinned (`TestTaskRotation`: visit-count alternation covers all families, repeat visit flips family) |
| 2 | Resource accounting fake (imp-17 ancestor) | `compute`/`memory`/`energy`/`latency`/`plastic_state_capacity` must **vary** across coordinates in the heterogeneous r5b_b grid — per-axis span over `records/episodes.json`; a constant axis is a fiction (imp-17/imp-35 collapse signature), not a finding | imp-45 | ✅ FIXED 2026-09-01 — episode path records total train-step MACs (forward settle/phase work + documented 2× backward estimate) and splits energy: consumption axis = work-derived estimate (monotone via `MAC_ENERGY_J`), state free energy → `state_energy_j`; suite path wired likewise; serializer/loader made drift-proof (`consumed_energy_estimate_j` is a derived alias, not stored state). Pre-fix r5b_b/r51c **resource records quarantined for resource claims** (task_loss attribution unaffected — discovery locks re-verified green). ψ-capacity was already a real discriminator (0/128/512 by primitive) |
| 3 | ψ engagement unproven in benchmark harnesses | In each suite: does ψ actually change, and does that change alter the measured behavior? (engagement lock per harness) | imp-43 | ✅ LOCKED (pipeline level) 2026-09-01 — `tests/property/test_psi_engagement.py`: ψ moves under task input, `modulate` reaches activations, and a frozen-ψ DI control changes train-step metrics (routing + fast_weights). Suite verdicts + Z3 gate checklist pinned in `_claims.py` (`psi_engaged` scope added). L1/L3 harnesses stay `plumbing_only`; **Z3 flagship remains gated** on a suite-level engagement lock + positive-control run |
| 4 | Metric honesty still leaky beyond `free_accuracy` (imp-20 ancestor) | For every metric emitted by campaign/benchmark paths: trace the state it is computed on — target-free, or supervision-contaminated? Any leaky metric quarantines the evidence chain that consumes it | imp-46 | ✅ DEFECTS FOUND + FIXED, LOCKED 2026-09-01 — census in `docs/metric_provenance.md` (emitter × state × target-free × claim-consumer × verdict). Pipeline schema closed (`METRIC_SCHEMA`); bare `accuracy` → `nudged_fit_accuracy` (quarantined); `evaluate_episode` claim reads made strict (`free_loss`/`free_accuracy`/`free_energy` only — leaky emitter now KeyErrors, proven by meta-test); trainer `train_acc` switched free-first; **distributed trainer's `train_acc` was silently constant 0.0** (read a metrics key nothing writes) — now computed from free-state activations. Locks: `test_metric_provenance.py`. Task-loss attribution upgraded provisional → claim-grade for the campaign chain |
| 5 | Settle mutation contracts ambiguous (imp-27 ancestor) | Every `settle()` caller uses the **returned** state — a caller reading the input state silently compares pre-settle activations (manufactures "non-descent"/"no-effect" results; the fidelity probe already ate this once) | imp-47 | ✅ PASS + LOCKED 2026-09-01 — census: all 26 `.settle(` call sites bind or return the settled state, zero violations. Canonical contract pinned on the `StateDynamics` protocol docstring. Enforcement: `tests/property/test_settle_caller_census.py` AST lock (rejects bare-expression `.settle(...)` statements; self-checks scan-not-blind ≥20 sites + flags a planted violation) |
| 6 | Silent dispatch fallbacks remain (imp-24/imp-25 ancestor) | Grep **all** dispatch tables for bare `else:`; each must raise on unknown values — any silent fallback manufactures fake ablation arms | imp-48 | ✅ DEFECT FOUND + FIXED 2026-09-01 — the imp-24 fix covered only the joint compose path; `spec.py` (geometry/dynamics/update/plasticity), `factory.py` (both round-trip sites), and `joint.py` geometry still silently substituted defaults (a `diffusion` config ran Instantaneous; a typo'd update ran SGD; a typo'd plasticity ran Null — fake M-arms). 12 raise-conversions. **Exposure audit: CLEAN** — all six commissioned campaigns (r51c/r5b_b artifacts: 1200 records; quick_gpu×3/smoke_cpu SQLite: 62 coords) contain zero silent-substitution values; the defect was latent, no historical records quarantined. Legitimate else-sites audited and left: `_credit_from_config` (already raises), MEP binary mode, activations explicit-default, pareto metadata fallback (documented, imp-35-mitigated) |
| 7 | Construction seeding not universal (R1.4 ancestor) | Every factory call in every campaign/benchmark/harness is seeded, or θ init rides ambient RNG? Unseeded arms contaminate M-axis comparisons where ψ is supposed to be the only difference | imp-49 | ✅ PASS 2026-09-01 — every joint suite seeds θ-init (`torch.manual_seed(seed)` at each `evaluate_*` entry; structural_robustness also seeds `random`), campaign path is `episode_seed`-seeded (imp-11 lock) |
| 8 | Fidelity probes themselves wrong | Probe-the-probe: for each fidelity probe, engineered ground truth — a deliberately broken implementation must fail it, a correct one must pass; a probe that passes a broken instrument wrong-foots the entire defect-filtering pipeline | imp-50 | ✅ PASS + LOCKED 2026-09-01 — `tests/property/test_fidelity_meta_validation.py`: 18 cases, each probe gets a correct (passes) and engineered-broken (fails) variant — identity/noisy settle, zero pseudo-gradient, step_size=0 update, inert ψ, ψ-insensitive modulate, θ mutation vs audit, even-cycle sampler, constant resource stub, leaky metric emitter. All detectors work; the manifest means something |
| 9 | Statistical test mis-specified | Is the 0.05 claimable floor correct? Is stratification/direction-merge implemented correctly? Is the test powered for the effect sizes/variance at this scale? A mis-specified test converts "no evidence of effect" into "evidence of no effect" — the exact error the fidelity policy forbids | imp-51 | ✅ VERDICT: UNDERPOWERED 2026-09-01 — `scripts/power_audit.py` over r5b_b task_loss (claim-grade per imp-46): MDE@80% = **0.314** (n=160/group, α=0.05) vs top observed d = **0.205** (null vs fast_weights, power 0.45); pooled d = 0.138 (power 0.23). The 0.05 floor is correctly specified but unreachable at this scale — the registered null is **not** evidence of no effect. Detecting the top observed effect needs ≈2.3× group size (n≈376/group) or a design that enlarges the effect |
| 10 | The "obvious" result is missing | Not a probe — the sweep's termination criterion (Policy): keep probing until the result is obvious | — | — |
| 11 | The microscope has never seen a known effect end-to-end | **Positive control (imp-52):** plant an obvious synthetic effect (lr=0 vs lr>0; trained vs untrained; correct credit vs deliberately inert credit; ψ engaged vs ψ frozen where ψ modulates output) and require the instrument to detect it with high confidence. An instrument self-test, not a scientific claim. **Policy: no campaign is interpreted unless the positive-control probe detects the planted effect.** | imp-52 | ✅ PASS + LOCKED 2026-09-01 — `tests/property/test_positive_control.py`: through the real `evaluate_episode` claim chain, lr=0 arm stays at chance while lr=0.1 arm reaches ceiling (acc 1.0) on a stationary task, stable across 3 seeds; resource revalidation rides along (compute/energy vary, nonnegative, state split, ψ-capacity discriminates). **Discovery:** the per-episode `synthetic` stream redraws the teacher every episode — no fixed θ can accumulate learning across the smoke-scale stream (pinned as `test_per_episode_stream_is_non_stationary_by_design`; see imp-54). Control D (ψ) locked at pipeline level by `test_psi_engagement.py` |

**Third-pass execution record** (2026-09-01 — order 4 → 5 → 3 → imp-52 → resource
revalidation → 9 → 8, all closed; 42 new lock tests, gate 1300/0):

1. **#4 metric honesty (imp-46):** census table `docs/metric_provenance.md`; pipeline
   schema closed (`METRIC_SCHEMA` in `core/pipeline.py`); `accuracy` → `nudged_fit_accuracy`
   (quarantined) with strict `free_*` claim reads in `evaluate_episode`; trainer/lightning/
   tradeoff-track consumers switched free-first; distributed trainer silent-zero `train_acc`
   defect fixed. *Task-loss attribution upgraded provisional → claim-grade for the campaign
   chain; suite-level training-loss-threshold metrics (adaptation/migration time) remain
   diagnostics — any adaptation *claim* must re-trigger on a target-free probe metric.*
2. **#5 settle callers (imp-47):** 26/26 clean; contract on the Protocol; AST lint lock with
   self-checks (planted violation flagged, scan floor enforced).
3. **#3 ψ engagement (imp-43):** pipeline-level chain green for routing/fast_weights;
   suite verdict table + Z3 gate checklist in `_claims.py`. L1/L3 harness M-arms stay
   `plumbing_only` — a harness-level engagement rewire is the upgrade path, not a fix.
4. **imp-52 positive control:** instrument detects the planted lr effect end-to-end through
   `evaluate_episode` claim fields; stationary-task control vs per-episode stream
   distinction pinned (imp-54).
5. **Resource revalidation:** guardrail folded into the positive-control suite (variance,
   nonnegativity, state/consumption split, ψ-capacity discriminator).
6. **#9 power audit (imp-51):** UNDERPOWERED — see table; scale-up or redesign required
   before the null can mean anything.
7. **#8 probe-the-probe (imp-50):** 18 engineered variants — every probe detects its
   broken case and passes its correct case.

**What remains before any campaign interpretation** (superseded by the R8 gates below —
both gates must pass; neither is bypassable):
- **Gate A — Z3 flagship gate:** suite-level ψ engagement lock (θ exact invariance + ψ
  non-constancy + ψ-dependent operator selection/behavior + above-chance probe accuracy +
  frozen-ψ control) and a planted-ψ-effect positive control run through the Z3 suite itself.
  Pipeline-level locks exist; suite-level wiring does not — Z3 output stays unread until both
  land. Definition of done: Z3 quick mode either passes the gate or fails with a precise,
  named instrument defect (else Z3 is classified `plumbing_only` and fixed).
- **Gate B — powered, stationary campaign redesign:** a claim-scope decision, a stationary
  task design (or an explicit per-episode scope), power preregistration, and embedded
  positive controls — re-commission only when the design passes the power gate, then
  re-verify the discovery locks on the new commission (R5b-D rule). Definition of done: a
  future campaign can pass the power gate without relying on the non-stationary synthetic
  stream.
- Guardrails unchanged: do not interpret the registered null; do not treat pre-fix
  r5b_b/r51c resource records as valid; do not treat the saturated stability axis as
  evidence of no stability tradeoff; do not close R7 because probes pass — the termination
  criterion is **the result becomes obvious**, and a planted effect detected at ceiling is
  the first honest datapoint, not the finish line.

## 🎯 R8 — Powered, Stationary Evidence (successor to R7) — CLOSED 2026-09-01 (R8.6 re-scoped → R9)

R7 established that the instrument can see a planted effect, but the commissioned campaign
was underpowered and structurally non-stationary. R8 closes the gap between **instrument
validity** (what R7 delivered) and **claim validity** (what R8 must deliver). The next defect
hunt is not primarily in the code — it is in the experimental design.

**Two gates, run as parallel tracks:**

| # | Item | Done when | Status |
|---|------|-----------|--------|
| R8.1 | Z3 suite-level engagement lock | θ exact invariance (norm/hash equality, not tolerance-only), ψ non-constancy, ψ-dependent operator selection/behavior, above-chance probe accuracy, frozen-ψ control changes metrics — the suite fails unless all hold | ✅ 2026-09-01 — embedded in every `evaluate_z3` run (`psi_gate` verdict, fails loudly by name): exact θ invariance (`ThetaAuditReport.invariant`, `theta_sha256`), ψ non-constancy + task-conditioning (`psi_history`/`final_psi` per task), ψ→gate wiring probe, RNG-aligned frozen-ψ control, per-task probe > chance+0.1. PASSES ×3 seeds at registered scale; quick scale fails capability items only (named, expected). Locks: `test_z3_engagement.py` |
| R8.2 | Z3 positive control | a planted ψ-mediated effect (ψ disabled/frozen vs ψ engaged) is detected through the Z3 suite itself; if not, Z3 is classified `plumbing_only` and fixed before interpretation | ✅ 2026-09-01 — engaged vs ψ-disabled arms (same seed → identical θ by construction: forced warm-up is ψ-independent, phase 2 freezes θ; sha256-verified) differ across all task metrics; best gap parity >0.29 every seed, direction favors ψ on all tasks. Disabled arm's gate fails with exactly the ψ items (probe-the-probe). Z3 verdict upgraded `plumbing_only` → `psi_engaged` in `_claims.py` (run-conditional emission) |
| R8.3 | Stationary task design | the campaign task family supports accumulation, or the campaign explicitly declares a per-episode-adaptation scope (imp-54). Options: (A) stationary synthetic family — teacher seed = f(campaign_id, coordinate, seed); (B) held-out probe design separating per-episode adaptation from accumulated learning; (C) task-switch phases A→B matching the L1 hypothesis | ✅ 2026-09-01 — Option A implemented + **pilot run**: `stationary_teacher`/`teacher_noise` threaded through `CampaignStack` (config-recorded; per-record provenance metadata); difficulty calibrated (`CALIBRATED_TEACHER_NOISE = 0.5` → oracle ≈0.86, no ceiling saturation — pinned by lock); legacy stream byte-reproduced; accumulation behavioral lock green. **Pilot numbers** (persistent-θ arms, 40 ep × 3 seeds, lr=0.01): noiseless — null 0.354±0.118, fast_weights 0.394±0.011, routing 0.288±0.025, control 0.146 **PASS**; calibrated σ=0.5 — null 0.371±0.085, fw 0.360±0.010, routing 0.263±0.006, control 0.150 **PASS**. d from 3 seeds is order-of-magnitude only. *Semantics note: the CampaignStack path rebuilds θ per episode — stationary teachers make per-episode-adaptation claims stable, but accumulated-learning claims run the persistent-θ chain (the pilot harness), which is what R9.1's retention design must thread deliberately.* |
| R8.4 | Power preregistration | every commission declares expected effect size, variance estimate, n/group, MDE@80%, α, stratification structure; below-floor commissions are labeled `pilot`/`plumbing`/`instrument-check` (imp-55) | ✅ 2026-09-01 — `validation/power_preregistration.py`: `PowerPreregistration` (claim/metric/scope/stream/effect/variance/n/α/stratification), derived `mde_cohens_d`/`mde_metric`, `n_for_target_power` planner; label gate `claim_grade`/`pilot`/`plumbing`/`instrument_check` — claim-grade derived (never declared), a declared rung caps even when gates pass, accumulated_learning demands the stationary stream (imp-54); `scripts/power_audit.py` now imports the shared MDE helper. Enforced at commission time: `CampaignStack.run_campaign(preregistration=..., require_claim_grade=...)` records label + prereg in the campaign config and fails loudly by name |
| R8.5 | Embedded positive controls | every claim-grade campaign contains a planted-effect control arm (lr=0 coordinate; ψ frozen vs engaged); control failure quarantines the campaign | ✅ 2026-09-01 — claim-grade requires `embedded_control` in the preregistration; the `frozen` (lr=0) update value composes as an explicit planted-control coordinate (dispatch raises on unknown values as ever); post-run `verify_embedded_control` checks the control arm's mean target-free accuracy against chance ± tolerance over the campaign's records; failed/missing control → `CampaignRunResult.quarantined` + event log. Verified live in both pilot runs (control passed) |
| R8.6 | Re-commission powered campaign | n/design passes the power gate without the non-stationary synthetic stream; discovery locks + fidelity manifest re-evaluated on the new commission (R5b-D rule) | **re-scoped → R9** 2026-09-01 — the machinery (label gate, embedded controls, stationary stream, pilot variance) is in place; the registered campaign itself moved into R9.1's retention design, where the effect is enlarged by the task structure rather than bought with n. Pilot planning numbers: routing-deficit d≈1.8 → n≈6/group; fw-vs-null d≈0.17–0.47 → n≈72–529/group at 40 episodes — episode budget and task structure are the levers, not raw n |

**Claim-scope rule (pairs with R8.3/R8.4):** each campaign states up front which effect type
its design can support — per-episode adaptation, accumulated learning, resource-efficiency,
stability, M-axis plasticity. The old smoke campaign cannot support accumulated-learning
claims because of imp-54; that constraint is written into any retrospective of it.

**Execution order (superseded):** both tracks completed 2026-09-01 — Track 1
(R8.1 → R8.2, the Z3 gate) and Track 2 (R8.3 pilot/calibration → R8.4 label
gate → R8.5 controls); R8.6 re-scoped into R9 (see below). Track 1's Z3 gate
remains the registered instrument for any Z3 output; Track 2's machinery is
the commissioning gate R9 rides on.

**Do not (R8-specific):**
- Do not re-run the old campaign at n≈376/group without redesign — scale without redesign is
  a precise answer to the wrong question (teacher redraw, stability saturation, small effect
  size, and quarantined resource records all survive an n bump).
- Do not read claim-grade task_loss as a learning claim — metric honesty ≠ construct validity.
- Do not read the stability axis's saturation as evidence of no stability tradeoff — it did
  not discriminate (imp-36); that is a measurement defect, not a null.
- Do not resurrect pre-fix r5b_b/r51c resource records for resource claims — resource
  recommissioning happens under the fixed compute/energy semantics only.
- Do not interpret the old registered null — it is explained (underpowered + non-stationary),
  not confirmed, and not refuted.

## 🎯 R9 — Surgical Stress Tests (discovery phase; successor to R8) — OPEN (planned 2026-09-01)

**Premise (2026-09-01 strategic review, adopted):** R8 built the most honest
microscope in the business — but a microscope discovers nothing by looking at
slides with nothing on them. Backprop is the undisputed king of i.i.d.,
stationary, unconstrained optimization; asking "which algorithm learns MNIST
fastest with unlimited memory and exact gradients" returns Backprop, or a tie,
forever. The discovery phase changes the environment so **Backprop's
superpowers become liabilities** (exact global gradients, global clocks,
activation storage, stationary streams) **and the ontology's axes become
strictly necessary** (locality, plasticity, energy, O(1) memory credit).
Grid campaigns map known territory; R9 runs **surgical, deeply powered,
single-hypothesis stress tests**, with the AutoScientist mapping boundary
conditions only *after* an effect exists.

**R8 is the prerequisite and the gate, not the casualty:** every R9 trial
commissions through the R8 machinery — power preregistration + label gate
(R8.4), embedded planted-effect control with quarantine (R8.5), stationary/
structured task streams (R8.3), fidelity manifest re-evaluation on the new
commission (R5b-D). An unpowered or leaky stress test just produces noisy,
uninterpretable drama.

| # | Trial | Hypothesis | Design | Metrics | Claim scope |
|---|------|-----------|--------|---------|-------------|
| R9.1 | **Catastrophic Forgetting Trial** (M-axis) | Routing/FastWeight plasticity retains Task A while learning Task B because ψ isolates/stores episode-local pathways; Null collapses to chance on A | **Structured task-sequence stream A→B(→C)** — e.g. digits 0–4 → 5–9 → Fashion-MNIST. Within-segment stationarity (accumulation representable per segment, R8.3 machinery per segment) + across-segment shift (retention measurable). *This is NOT the imp-54 stream — that stream is degenerate per-episode noise (unlearnable), not structured non-stationarity; a task sequence is the environment continual learning is defined on.* Prior art to revive under the R8 gates: `computronium/experiments/joint/continual_learning.py` + `configs/preregistrations/cl_backward_transfer_matched_memory.json` + `cl_retest_discriminating_probe.json` (Split-MNIST, memory-matched, verified arms). **Z3 pivot:** extend the Z3 gate with a retention arm (Task A → B → A, θ frozen, ψ switches back; retention metric) — upgrades Z3 from capability (ψ can switch) to utility (switching prevents forgetting) | Backward transfer (retention of A after B), forward transfer, per-segment adaptation time; embedded lr=0 control must sit at chance throughout | retention under structured non-stationarity (new scope alongside the claim-scope rule — `retention` joins per-episode adaptation / accumulated learning / resource-efficiency / stability / M-axis plasticity) |
| R9.2 | **Physical Constraint Trial** (S/D axes) | Under severe substrate constraints (memristive IR-drop, analog precision caps, noise, memory ceilings), exact-global Backprop degrades or collapses while local rules (EqProp, FA, local goodness) degrade gracefully and dominate the 𝒞-Pareto frontier | The **same powered design run twice**: (a) unconstrained Digital — Backprop expected to win (the honest baseline); (b) constrained — Memristive/noisy substrate, precision-capped, memory-budgeted (no activation storage for BPTT). Resource axes are trustworthy post-imp-45 (work-derived compute/energy, state split) — the frontier shift is measurable, not fictional. Map the "Goldilocks zone" where locality beats global optimality | 𝒞-Pareto frontier shift (compute/energy/memory/latency vs accuracy), collapse boundary of the Backprop arm, graceful-degradation curves | resource-efficiency under physical constraints |
| R9.3 | **Deep Credit Trial** (C-axis) | ThermodynamicContrast (EqProp) / PredictiveSettling (PC) learn a 50+ step temporal dependency with O(1) activation memory where BPTT OOMs or vanishes | Long-horizon recurrent task (deep parity over 50+ steps; state-space realization). Memory-profiled arms: BPTT's memory grows O(depth) and its gradients vanish; energy-based arms settle to a fixed point and credit locally | Learned temporal dependency at fixed memory; BPTT failure boundary (OOM/vanishing); settling-time cost of the local alternative | credit assignment at depth (validates the C-axis mathematics where shallow tasks cannot) |

**R9 method rules:**
- One hypothesis per trial; preregister through the R8.4 gate with the
  embedded control (lr=0 arm at chance; ψ-frozen arm where ψ is the
  treatment) — a moving control quarantines the trial, always.
- Power comes from **effect-enlarging design**, not n-scaling: the pilot
  showed a well-aimed contrast carries d≈1.8 (n≈6/group) where the old
  design's d=0.205 needed n≈376/group. Task structure, budget, and
  constraint severity are the levers.
- The AutoScientist maps **boundary conditions after the effect exists**
  (e.g., at what switch rate does routing's retention advantage disappear;
  at what IR-drop does the local-rule Pareto dominance begin).
- "If it works it will be obvious" is unchanged: a retention gap of 90%-vs-
  chance, a Pareto frontier that visibly shifts, or a learned 50-step
  dependency at O(1) memory needs no statistics to be seen — the gates exist
  to make sure the seeing is honest.

**Execution order:** R9.1 is the flagship (M-axis is the framework's
differentiator; prior art exists; the pilot's ψ-stability signature — imp-58 —
points exactly there). R9.2 next (substrate machinery exists; needs the
constraint-arm harness). R9.3 last (needs the long-horizon task + memory
profiling; EqProp-on-recurrent depth machinery exists from the native
research directions).

## 🔁 Pull-Based Backlog (non-blocking; pull when a campaign manifest or suite needs it)

| Item | Trigger / pull condition |
|------|--------------------------|
| R2.2 residual | Substrate facade-merge consideration (`ontology/_substrate.py` impl vs `ontology/substrate/` facade — facade is fine, consider merge) + grep for other parallel legacy/new pairs |
| R3.4 | Tile × dynamics matrix (tile_ep/pc/gnn/snn device-dynamics incompatibility; tile_fa/tp/hebbian) — fix or document as permanent xfail with precise reasons |
| R3.5 | `_AdaptedSystem._infer_geometry` hardcoded (784→256,128→10) — recover heuristics from deleted `adapter/` package |
| R3.6 | `_TaskTrainer` gaps — scheduler wiring, energy tracking, honor `tracker`/`safety_config` (wire when hyperopt trials need them) |
| R3.7 | Substrate fidelity nits — Neuromorphic: real spike dropout or drop cosmetic `sparsity` field; Memristive: conductance-range semantics (pairs with RESEARCH3 substrate work) |
| R3.8 | Stretch — `natural_language_query` TF-IDF weighting; derive `V_nudged = free energy + β·loss` to strengthen the predictive-coding Lyapunov xfail |
| R4.1 | FA feedback projection through the Substrate operator API (validates non-settle paths) |
| R4.2 | Register `SubstrateSettleKernel` in `KernelRegistry` for the EQPROP family |
| R4.3 | MEP Triton kernels (Muon, Fisher whitening) → Substrate update operator |
| R4.4 | Sparse substrate transpose-mask handling; ternary `init_scale` param (un-xfail ternary equivalence); optional per-step `inject_state_noise` during settle |
| imp-4 | Pyright full `strict` on ontology = 131 findings (torch `Unknown` tracking); annotation work in `_dynamics`/`geometry`/`update` |
| imp-8 | `compute_energy` duplication across Energy/Spike/Instantaneous/Diffusion dynamics — extract one `_energy_from_state(state, geometry)` helper next touch |
| imp-19 | `FrontierRecord.seed` legacy default 42 — make required at the next schema break |
| imp-23 | `substrate_coupled` plasticity was engagement-verified only — probe for fixed-dim `step` assumptions on next touch |
| imp-26 | Params-moved learning locks for the remaining README-table factories (FA lock exists; probe the rest on next touch) |
| imp-27 | `settle` protocol: canonical return-state contract is now documented + caller-census locked (R7 #5); remaining: rename any rebuilder-style implementations if their names mislead on next touch |
| imp-29 | Nudge-unwired settle paths (predictive_settling target clamp; diffusion target term) — repair only when a campaign manifest needs them |
| imp-30 | Deployments' `family="tile"` registrations CLI-orphaned — fold into `family="equitile"` or drop the metamodel member at next touch |
| imp-36 | Campaign stability axis non-discriminative — wire a cheap per-episode proxy if a manifest needs stability contrast |
| imp-37 | Latency objective is wall-clock noise — repeated-timing methodology or deterministic proxy before any task-scale claim |
| imp-41 | `demo/tests/` 28 stale failures — rewrite or delete on next demo-test touch |

## ⏸ Deferred (carried from TODO8, unchanged)

| Item | Reason |
|------|--------|
| ConvGeometry / GraphGeometry / AttentionGeometry / 3D Spatial Lattice | Science runs on Feedforward/Recurrent/Tile at MLP scale; geometry-DEFERRED skips stay skips |
| Coverage floor (~16.8%) | opt-in `--cov`; raise after API stabilizes |
| Rocq general-case formalization | CP-B pull-based; diagonal case done with paper proof |
| `test_ontology_parity.py` decomposition | Slow-marked; split fast/slow only if gate iteration speed demands |
| Physical hardware (PR-3b / CP-D) | Latency-gated procurement — starts per RESEARCH3 day-one, not here |

## ⚠️ Watch

- axis_probe `[2-0]` flake — no recurrence since 2026-08-31; still watching
- CUDA tolerance boundaries shift xfail edges — CPU/GPU tests kept separate, construction seeding in place
- Shakedown suites' harness arms are M-axis-identical (see PR-7 notes + imp-43) — pipeline-level engagement locks are green, but L1/L3 *claims* still need the harness rewired onto the ψ-engaged path (or params-moved locks) first
- Z3 flagship gate (R8 Gate A / R8.1–R8.2): **landed 2026-09-01** — every `evaluate_z3`
  run self-validates via its embedded `psi_gate` (exact θ invariance, ψ non-constancy +
  task-conditioning, ψ→gate wiring, RNG-aligned frozen-ψ control, above-chance probe acc)
  and emits `claims_scope` (`psi_engaged` iff the gate passes). Do not read any Z3 output
  whose gate failed; the registered flagship run (CP-A) is unblocked and should re-verify
  the gate per seed at its own scale
- Power gate (R8.4 / imp-55): commissions must state expected n vs MDE@80% in their preregistration; below-floor commissions are labeled `pilot`/`plumbing`/`instrument-check`, never claim-grade
- Smoke-scale campaign deltas are capped at chance by the non-stationary synthetic stream (imp-54 / R8.3) — never read pooled smoke task_loss/accuracy deltas as accumulated-learning evidence; the stationary-teacher design (`stationary_teacher=True`) is the accumulation-capable path and its pilot variance is now measured (`benchmark_results/stationary_pilot_*.json`) — but note the CampaignStack path rebuilds θ per episode (per-episode-adaptation scope even with stationary teachers); accumulated-learning/retention claims run the persistent-θ chain
- **R9 (open):** the discovery phase runs surgical stress tests (R9.1 forgetting / R9.2 constraint / R9.3 deep credit), every trial gated by R8.4/R8.5 machinery. R9.1 must build a *structured* task-sequence stream (stationary within segments, shifting across) — not the imp-54 degenerate stream; Z3's retention pivot (A→B→A) is open; R9.2 needs the constraint-arm harness; R9.3 needs the long-horizon task + memory profiling. No R9 commission interprets anything without its embedded control passing and the label gate at claim-grade
- Control-band sizing (imp-59): at-chance embedded controls quarantine on sampling noise if the band is not sized to the control arm's scored-sample count — preregistrate the band from the registered N

## 💡 Improvement Ledger (continues TODO8's imp-N from imp-42)

42. **A suite can run green its whole life printing a chance-level headline (2026-09-01, PR-7).**
`structural_robustness.py` saved "original model state after pre-training" with no pre-training
loop anywhere — `pre_damage_accuracy` was measured on an untrained net (≈1/10 for 10 classes) and
`recovery_ratio = final/pre_damage` was degenerate up to ~10×. Nobody had read the number.
Fixed: real pre-train loop (the suite's `epochs` param) + per-scenario optimizer rebuild
(PR-1). Post-fix: pre-damage 0.926, recovery ratio ≈ 1.22. *Lesson: shakedown means reading the
printed numbers against their chance levels, not just collecting populated fields — the imp-40
"compile-clean ≠ runtime contract" class extends to "runs-and-emits ≠ measures-anything."*
43. **Benchmark-harness ψ engagement is unproven in L1/L3 (2026-09-01, PR-7).** `adaptation_efficiency`
and `structural_robustness` report bit-identical means across M-arms (null/routing/fast_weights/
substrate_coupled): the harness models construct ψ but adaptation/recovery timing is driven by the
θ-optimizer loop, so plasticity cannot express itself in the measured metrics. Claims-scoped today
(`PSI_WIRED_UNCONTROLLED`/`PLUMBING_ONLY`), so not a defect for shakedown — but any L1/L3 *claim*
needs either the ontology-pipeline path (`run_train_step` + imp-22 ψ wiring) or a params-moved/
ψ-engagement lock in the harness first (pairs with imp-26). `compute_efficiency` is the exception:
its gate-entropy/effective-FLOPs metrics genuinely discriminate routing (1.0 vs 8.0 active routes).
44. **R7 suspect #1 — task-rotation structural parity (2026-09-01).** The R5b-B ancestor (0/48
replication from even grid cycles) was fixed rev d (visit-count family alternation), but the
invariant "every coordinate visits both task families across its scheduled visits" has no dedicated
lock — the replication gate implies it only indirectly. Probe: assert the invariant over the
commissioned artifacts and pin it as a lock test so an even-cycle regression cannot silently
reproduce the 0/48 collapse.
45. **R7 suspect #2 — resource-vector variance (2026-09-01).** imp-17 wired the 𝒞 axes and imp-35
killed zero-span MC boxes, but nobody has verified the five axes actually vary across the
heterogeneous r5b_b grid. Probe: per-axis span/variance over `r5b_b/records/episodes.json`; a
constant axis is a fiction (the imp-17/imp-35 collapse signature), not a finding.
46. **R7 suspect #4 — metric-honesty census (2026-09-01).** imp-20 fixed `free_accuracy` only.
Probe: for every metric emitted by campaign/benchmark paths, trace the state it is computed on —
target-free or supervision-contaminated? Any leaky metric quarantines the evidence chain that
consumes it.
47. **R7 suspect #5 — settle() caller census (2026-09-01).** imp-27 documented the two mutation
contracts; the fidelity probe already ate this bug once. Probe: every `settle()` caller must use
the returned state — a caller reading the input state silently compares pre-settle activations
(manufactures "non-descent"/"no-effect" results).
48. **R7 suspect #6 — bare-`else:` dispatch census (2026-09-01).** imp-24/imp-25 fixed three
dispatches and left the lesson "grep for bare `else:` before the next axis value lands". Probe:
run that census now over all dispatch tables — any silent fallback manufactures fake ablation arms.
49. **R7 suspect #7 — construction-seeding universality (2026-09-01).** Campaign paths seed via
`episode_seed`; parity classes seed via `construction_seed`. Probe: every factory call in every
campaign/benchmark/harness — seeded, or riding ambient RNG? Unseeded arms contaminate M-axis
comparisons where ψ is supposed to be the only difference.
50. **R7 suspect #8 — fidelity-probe meta-validation (2026-09-01).** The gate is only as good as
its probes. Probe: for each fidelity probe, engineered ground truth (a deliberately broken
implementation must fail it, a correct one must pass) — a probe that passes a broken instrument
wrong-foots the entire defect-filtering pipeline.
51. **R7 suspect #9 — statistical specification audit (2026-09-01).** The 0.05 claimable floor,
stratification, and direction-merge have not been power-checked against the campaign's actual
effect sizes/variance. Probe: is the test correctly specified for the current scale? A
mis-specified test converts "no evidence of effect" into "evidence of no effect" — the exact
error the fidelity policy forbids.

**R7 probe outcomes (2026-09-01 — first pass + second-pass closures):** the sweep produces signal.
- #1 PASS + LOCKED: the rev d rotation fix holds structurally (48/48 coords × 240/240 strata,
  both families, uniform visits); engine-level lock pinned in `TestTaskRotation` — a sampler
  change cannot silently reintroduce the even-cycle collapse.
- #2 FIXED: `backward_flops` was constant 0 across all 480 r5b_b records (the compute axis was a
  forward-only MAC proxy, understating learning cost by the entire backward pass) and `energy_j`
  held the target-free settled *state* energy (min −3.84) inside a consumption vector. Now:
  total train-step MACs + a documented 2× backward estimate; consumption energy work-derived and
  monotone (`MAC_ENERGY_J`); state energy split out (`state_energy_j`). Pre-fix r5b_b/r51c
  resource records are quarantined for resource claims — never silently reinterpreted.
  ψ-capacity was already a real M-axis discriminator (0/128/512 by primitive).
- #6 DEFECT FIXED + EXPOSURE AUDITED: 12 silent dispatch fallbacks eradicated (see table).
  *Lesson: imp-24 fixed one entry point and the lesson said "grep before the next axis value
  lands" — the census should have been run the same day; the class survived in the flagship
  5-D round-trip path (`spec.py`/`factory.py`) where the L0 schema lock is enforced.*
  Exposure audit across all six commissioned campaigns (r51c/r5b_b artifacts: 1200 records;
  quick_gpu×3/smoke_cpu SQLite: 62 coords): **zero records executed unknown axis values** —
  the defect was latent, nothing historically contaminated.
- #7 PASS: all construction sites seeded (`torch.manual_seed` at every suite entry,
  `episode_seed` on the campaign path).
- Confirmed with data: imp-36's stability saturation is literal — one distinct
  (ρ, lyapunov, settling, basin) tuple across 480 records.

**R7 probe outcomes (2026-09-01 — third pass, sweep complete):** the microscope now sees.
- #4 DEFECTS FOUND + FIXED + LOCKED: distributed trainer's `train_acc` silently constant 0.0
  (imp-53); `nudged_fit_accuracy` quarantine + closed pipeline schema + strict free-only claim
  reads; provenance table is `docs/metric_provenance.md`. Task-loss attribution is now
  claim-grade for the campaign chain.
- #5 PASS + LOCKED: 26/26 settle call sites bind the returned state; AST lint lock with
  probe-the-probe self-checks.
- #3 LOCKED (pipeline level): ψ moves → modulate reaches activations → frozen-ψ control shifts
  metrics, for routing + fast_weights. Suite-level verdicts pinned; Z3 still gated.
- imp-52 PASS + LOCKED: the instrument detects the planted lr effect at ceiling through the
  campaign claim chain; non-stationary smoke-stream property discovered and pinned (imp-54).
- #9 UNDERPOWERED: MDE@80% 0.314 vs top observed 0.205 (imp-55) — the null is not evidence of
  no effect; scale or redesign first.
- #8 PASS + LOCKED: every fidelity probe fails its engineered broken variant and passes its
  correct variant (18 meta-cases).

52. **R7 positive control — planted-effect instrument self-test (2026-09-01).** The termination
criterion ("if it works it will be obvious") needs a concrete detector: plant an obvious synthetic
effect (lr=0 vs lr>0, trained vs untrained, correct vs deliberately inert credit, ψ engaged vs ψ
frozen where ψ modulates output) and require the instrument to detect it with high confidence
before any campaign is interpreted. An instrument self-test, not a scientific claim — it proves
the microscope can see *something*, which the null alone cannot.
**Outcome (same day): PASS.** Through the real `evaluate_episode` claim chain, lr=0 stays at
chance while lr=0.1 reaches acc 1.0 on a stationary task (stable across 3 seeds; locks in
`tests/property/test_positive_control.py`). Building it surfaced a task-design property with
interpretation consequences (imp-54).

53. **Metric-honesty census found a silently dead metric, not just names (2026-09-01, R7 #4).**
`DistributedSystemTrainer` epoch `train_acc` read `free_state.metrics["accuracy"]` — a key only
`task_loss` writes, and the distributed settle path never calls it — so it reported exactly 0.0
for its whole life. Fixed with `_accuracy_from_state` (computed from free-state activations).
Same-day renames: pipeline `accuracy` → `nudged_fit_accuracy` (quarantined; schema closed via
`METRIC_SCHEMA`), `evaluate_episode` claim reads made strict free-only (a leaky emitter now
KeyErrors — proven by a meta-test), trainer/lightning/tradeoff-track consumers switched
free-first. *Lesson: imp-20 fixed the campaign's headline key and stopped there; the same leak
class survived in every consumer of the shared pipeline schema — the census needed to walk the
whole claim chain, not the emitter.*

54. **The smoke-scale campaign task stream is non-stationary by design (2026-09-01, imp-52).**
`episode_batch`'s `synthetic` family redraws a fresh random teacher every episode, so a single θ
cannot accumulate learning across a smoke campaign — pooled smoke-scale task_loss/task_accuracy
deltas are capped at chance-level by construction for every arm, including a learning one.
Pinned as `test_per_episode_stream_is_non_stationary_by_design`. *Consequence: smoke-campaign
deltas measure per-episode adaptation under continual teacher change, never accumulation; any
"learning" claim from pooled smoke artifacts is structurally impossible, and their null deltas
say nothing about learning machinery. Stationary-task or held-out-probe designs are required
for accumulation claims.*

55. **The registered null is underpowered, not just unstable (2026-09-01, R7 #9).**
`scripts/power_audit.py`: MDE@80% = 0.314 at n=160/group vs top observed d = 0.205 (power 0.45;
pooled 0.138, power 0.23). Even the largest effect the campaign exhibited was beyond its
detectable range — "stratified-unstable at the 0.05 floor" was the readable edge of an
underpowered design. *Consequence: the null stays uninterpreted, and the first scale question
is quantitative — ≈376/group at current variance, or a redesign that enlarges the effect (e.g.
stationary tasks per imp-54, ψ-gated arms per imp-43) — before re-commissioning.*

56. **An embedded control must align the RNG stream, not just the seed (2026-09-01, R8.1).**
The first frozen-ψ control arm ran after the treatment arm in the same process, so its
batches and probes came from a later RNG position — the "ψ effect" it measured conflated ψ
stepping with sampling noise, and on the ψ-disabled arm the control differed from treatment
purely by batch noise (false-positive engagement: the gate item that must be False was True).
Fix: snapshot the global RNG streams at the treatment arm's entry point and replay them
before the control arm (`_snapshot_rng`/`_restore_rng`) — arms now see bit-identical batches
and probes; on the disabled run they are bit-identical, which is itself the probe-the-probe
assertion. *Lesson: "same seed" is not "same stream" — a control arm must inherit the stream
position its treatment started from, or the comparison measures sampling noise. Same-day
corollary: the ψ wiring probe needed the same isolation (a dedicated `torch.Generator`) or
it would have shifted every downstream draw.*

57. **Z3's registered "exact" θ invariance was actually a 1e-6 tolerance (2026-09-01, R8.1).**
The preregistration decision rule said "max abs Δθ < 1e-6" and the suite emitted
`theta_invariant = report.is_within(1e-6)` — a tolerance, not the claim's exact-zero
invariant. R8.1 tightened the emission to bitwise equality (`ThetaAuditReport.invariant`;
max-abs == 0.0 is elementwise-exact) and added `theta_sha256` for artifact-level identity
checks (used to prove engaged/ψ-disabled arms share θ — the warm-up phase is ψ-independent
and phase 2 freezes θ, so same-seed arms are bitwise identical). *Lesson: when a registered
claim states an exact invariant, the emitting field must be exact too — a tolerance field
invites a tolerance-level pass on an exactness claim. Discovered while wiring the R8 gate;
no historical Z3 record ever showed nonzero drift (all 0.0), so nothing is quarantined.*

58. **The stationary pilot's first signal: ψ-mediated accumulation is ~10× more
seed-stable than θ accumulation — and effect size is a design property (2026-09-01, R8.3).**
Persistent-θ arms walking the stationary stream (40 ep × 3 seeds, registered shape):
fast_weights' late-window accuracy varied across seeds by sd ≈ 0.010–0.011 while null's
varied by sd ≈ 0.085–0.118 (both difficulty variants; routing 0.006–0.025). Hypothesis-grade
(3 seeds), but it is exactly the signature the M-axis hypothesis needs: the plastic state,
not the θ trajectory, is carrying the stable part of the accumulated behavior. Power lesson
from the same run: the routing-vs-null deficit carried d≈1.8 (n≈6/group for 80% power)
where the old campaign's best effect (d=0.205) needed n≈376/group — **well-aimed questions
enlarge effects; n-scaling cannot rescue a mis-aimed one** (the quantitative form of "stop
measuring slides with nothing on them"; drove the R8.6 → R9 re-scope). *Caveat pinned: d
from 3 seeds is order-of-magnitude only — a registered commission re-estimates variance at
its own scale.*

59. **An at-chance control band must be sized for the record count, or small
pilots self-quarantine (2026-09-01, R8.5).** The first 8-episode pilot smoke "failed" its
lr=0 embedded control (mean acc 0.242 vs band 0.075–0.175): with only 128 scored samples,
per-batch chance noise alone (σ ≈ 0.029) plus init-to-init variation puts the frozen arm
outside a ±0.05 band ~regularly. The registered pilot (40 ep × 3 seeds = 1920 samples)
passed at 0.146/0.150. *Lesson: the control band is a statistical instrument too — width
must scale with √N of the control arm's scored samples, or the quarantine fires on sampling
noise (the R8.5 gate manufacturing a false defect of exactly the class it exists to catch).
R9 trials must size the band from the registered record count at preregistration time.*

## 🔧 Quick Commands

```bash
uv run pytest -q                       # gate (~90s): unit+property; slow/benchmark/llm auto-deselected
uv run pytest tests -m slow            # slow tier (~25min; `tests` arg required)
uv run pyright computronium/ontology   # type policy: elevated-standard on ontology, basic repo-wide

# PR-7 shakedown suites (smoke scale; drop --quick for full registered runs):
uv run python -m computronium.experiments.joint.algorithm_migration --quick
uv run python -m computronium.experiments.joint.adaptation_efficiency --quick
uv run python -m computronium.experiments.joint.compute_efficiency --quick
uv run python -m computronium.experiments.joint.structural_robustness --quick
# Z3 flagship (Level 4) — GATED: needs suite-level engagement lock + positive control first:
comp benchmark run --suite z3_fixed_weights --seeds 5 --device cuda

# R5b campaign stack (built in TODO8; discovery locks pin the registered null):
uv run pytest tests/property/test_discovery_locks.py tests/property/test_campaign_fidelity.py -q
uv run scripts/fidelity_gate_report.py --campaign-dir autoscientist_campaigns/r5b_b

# R7 instrument locks (third pass):
uv run pytest tests/property/test_metric_provenance.py tests/property/test_settle_caller_census.py \
  tests/property/test_psi_engagement.py tests/property/test_positive_control.py \
  tests/property/test_fidelity_meta_validation.py -q
uv run python scripts/power_audit.py   # MDE@80% vs observed effects over r5b_b task_loss

# R8 locks (Z3 gate + stationary teacher):
uv run pytest tests/property/test_z3_engagement.py tests/property/test_stationary_teacher.py -q
# Z3 gate runs inside every evaluate_z3; registered-scale shakedown (~8 s/seed CPU):
uv run python -m computronium.experiments.joint.z3_fixed_weights --meta-train-epochs 50 --eval-epochs 20 --seeds 3

# R8 Track 2 (stationary pilot + power prereg + embedded controls):
uv run python -m computronium.experiments.joint.stationary_pilot \
  --episodes 40 --seeds 0,1,2 --output benchmark_results/stationary_pilot.json
uv run pytest tests/property/test_power_preregistration.py tests/property/test_stationary_teacher.py -q

# NOTE: sync with `uv sync --extra dev --extra lightning` (plain dev sync removes
#   lightning -> 4 collection errors). Serial pytest only — xdist hangs in this env.
```
