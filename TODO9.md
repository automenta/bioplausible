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
> eradicated; gate 59 passed, ruff/pyright net-zero) · 1 DEFECT FOUND, fix pending
> (`backward_flops` ≡ 0 fiction; energy records state energy, not consumed) · data-confirms
> imp-36 saturation. The sweep produces signal.
>
> **Numbering:** improvement items continue TODO8's append-only ledger from **imp-42**;
> imp-1..41 remain canonical in TODO8.md.

## Policy (carried from TODO8, unchanged)

- Zero backwards compatibility · GPU-first for all training paths · serial pytest only (xdist hangs in this env)
- No new tests for broken capability — xfail with precise reasons
- **The System's own ParameterUpdate owns Δθ — external torch optimizers must not drive composed systems** (custom-loss harnesses route through `core.pipeline.apply_autograd_update`)
- **No scientific conclusion from any campaign delta until both arms pass an implementation-fidelity check.** A failed fidelity check is *inconclusive*, never a refutation. Deltas on known-defective axes are quarantined from attribution, not interpreted.
- Observed-but-unregistered deltas are never interpreted (pre-registration precedes comparison)
- **A null is a symptom, not a result.** No null is read as evidence of no effect while an R7 instrument suspect remains unprobed; each probe either finds a defect (fix it, re-run the affected campaign tier) or passes (pin it where cheap, move on)
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

## 🎯 R7 — Instrument-Honesty Sweep (the null is a symptom) — ACTIVE

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
| 1 | Task rotation structurally broken (R5b-B 0/48 ancestor) | Every coordinate in a multi-family grid campaign appears in **both** task families across its scheduled visits — assert over commissioned artifacts, pin as lock test (rev d fixed the rotation; the invariant itself is unpinned) | imp-44 | ✅ PASS 2026-09-01 — 48/48 coords, 240/240 (coord, seed) strata hit both families, uniform 10 visits; pin the lock test on next touch |
| 2 | Resource accounting fake (imp-17 ancestor) | `compute`/`memory`/`energy`/`latency`/`plastic_state_capacity` must **vary** across coordinates in the heterogeneous r5b_b grid — per-axis span over `records/episodes.json`; a constant axis is a fiction (imp-17/imp-35 collapse signature), not a finding | imp-45 | ⚠️ DEFECTS 2026-09-01 — 4/5 axes genuinely vary (ψ-capacity discriminates null=0/routing=128/fast_weights=512 — real M-axis signal), **but** `backward_flops` ≡ 0 across 480 records (compute axis is forward-only fiction) and `energy_j` records *state* free energy (goes negative) rather than consumed energy. Wiring fix + artifact-semantics note pending; recorded r5b_b resource claims stay quarantined until re-commissioned |
| 3 | ψ engagement unproven in benchmark harnesses | In each suite: does ψ actually change, and does that change alter the measured behavior? (engagement lock per harness) | imp-43 | pending |
| 4 | Metric honesty still leaky beyond `free_accuracy` (imp-20 ancestor) | For every metric emitted by campaign/benchmark paths: trace the state it is computed on — target-free, or supervision-contaminated? Any leaky metric quarantines the evidence chain that consumes it | imp-46 | pending |
| 5 | Settle mutation contracts ambiguous (imp-27 ancestor) | Every `settle()` caller uses the **returned** state — a caller reading the input state silently compares pre-settle activations (manufactures "non-descent"/"no-effect" results; the fidelity probe already ate this once) | imp-47 | pending |
| 6 | Silent dispatch fallbacks remain (imp-24/imp-25 ancestor) | Grep **all** dispatch tables for bare `else:`; each must raise on unknown values — any silent fallback manufactures fake ablation arms | imp-48 | ✅ DEFECT FOUND + FIXED 2026-09-01 — the imp-24 fix covered only the joint compose path; `spec.py` (geometry/dynamics/update/plasticity), `factory.py` (both round-trip sites), and `joint.py` geometry still silently substituted defaults (a `diffusion` config ran Instantaneous; a typo'd update ran SGD; a typo'd plasticity ran Null — fake M-arms). 12 raise-conversions; verified gate 59 passed, ruff/pyright net-zero. Legitimate else-sites audited and left: `_credit_from_config` (already raises), MEP binary mode, activations explicit-default, pareto metadata fallback (documented, imp-35-mitigated) |
| 7 | Construction seeding not universal (R1.4 ancestor) | Every factory call in every campaign/benchmark/harness is seeded, or θ init rides ambient RNG? Unseeded arms contaminate M-axis comparisons where ψ is supposed to be the only difference | imp-49 | ✅ PASS 2026-09-01 — every joint suite seeds θ-init (`torch.manual_seed(seed)` at each `evaluate_*` entry; structural_robustness also seeds `random`), campaign path is `episode_seed`-seeded (imp-11 lock) |
| 8 | Fidelity probes themselves wrong | Probe-the-probe: for each fidelity probe, engineered ground truth — a deliberately broken implementation must fail it, a correct one must pass; a probe that passes a broken instrument wrong-foots the entire defect-filtering pipeline | imp-50 | pending |
| 9 | Statistical test mis-specified | Is the 0.05 claimable floor correct? Is stratification/direction-merge implemented correctly? Is the test powered for the effect sizes/variance at this scale? A mis-specified test converts "no evidence of effect" into "evidence of no effect" — the exact error the fidelity policy forbids | imp-51 | pending |
| 10 | The "obvious" result is missing | Not a probe — the sweep's termination criterion (Policy): keep probing until the result is obvious | — | — |

**Proposed order** (cheap + high-information first): 1 → 2 → 6 → 7 (data checks + grep censuses)
→ 4 → 5 (code audits) → 3 (engagement locks) → 9 (stats audit) → 8 (meta-validation, hardest).
Re-run the affected campaign tier after any defect fix.

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
| imp-27 | `settle` protocol has two mutation contracts (in-place vs new-state) — document on the Protocol and make one canonical, or rename the rebuilders |
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
- Shakedown suites' harness arms are M-axis-identical (see PR-7 notes + imp-43) — any L1/L3 *claim* must route through the ψ-engaged path first

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

**R7 probe outcomes (2026-09-01, first pass — probes 1/2/6/7):** the sweep produces signal.
- #1 PASS: the rev d rotation fix holds structurally (48/48 coords × 240/240 strata, both
  families, uniform visits). Remaining: pin the invariant as an engine-level lock test.
- #2 DEFECTS: `backward_flops` constant 0 across all 480 r5b_b records — the compute axis is a
  forward-only MAC proxy, so "compute" understates learning cost by the entire backward pass;
  `energy_j` is the target-free settled **state** energy (min −3.84), not consumed energy — a
  state variable recorded in a consumption vector. ψ-capacity is the one clean discriminator
  (0/128/512 by primitive). *Follow-up: wire backward MACs (≈2× forward) + a consumed-energy
  proxy in `_episode_resources`/`measure_suite_resources`; r5b_b resource claims stay
  quarantined until re-commissioned under the fixed semantics.*
- #6 DEFECT FIXED: 12 silent dispatch fallbacks eradicated (see table). *Lesson: imp-24 fixed
  one entry point and the lesson said "grep before the next axis value lands" — the census
  should have been run the same day; the class survived in the flagship 5-D round-trip path
  (`spec.py`/`factory.py`) where the L0 schema lock is enforced.*
- #7 PASS: all construction sites seeded.

## 🔧 Quick Commands

```bash
uv run pytest -q                       # gate (~75s): unit+property; slow/benchmark/llm auto-deselected
uv run pytest tests -m slow            # slow tier (~25min; `tests` arg required)
uv run pyright computronium/ontology   # type policy: elevated-standard on ontology, basic repo-wide

# PR-7 shakedown suites (smoke scale; drop --quick for full registered runs):
uv run python -m computronium.experiments.joint.algorithm_migration --quick
uv run python -m computronium.experiments.joint.adaptation_efficiency --quick
uv run python -m computronium.experiments.joint.compute_efficiency --quick
uv run python -m computronium.experiments.joint.structural_robustness --quick
# Z3 flagship (Level 4):
comp benchmark run --suite z3_fixed_weights --seeds 5 --device cuda

# R5b campaign stack (built in TODO8; discovery locks pin the registered null):
uv run pytest tests/property/test_discovery_locks.py tests/property/test_campaign_fidelity.py -q
uv run scripts/fidelity_gate_report.py --campaign-dir autoscientist_campaigns/r5b_b

# NOTE: sync with `uv sync --extra dev --extra lightning` (plain dev sync removes
#   lightning -> 4 collection errors). Serial pytest only — xdist hangs in this env.
```
