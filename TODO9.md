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
> **Numbering:** improvement items continue TODO8's append-only ledger from **imp-42**;
> imp-1..41 remain canonical in TODO8.md.

## Policy (carried from TODO8, unchanged)

- Zero backwards compatibility · GPU-first for all training paths · serial pytest only (xdist hangs in this env)
- No new tests for broken capability — xfail with precise reasons
- **The System's own ParameterUpdate owns Δθ — external torch optimizers must not drive composed systems** (custom-loss harnesses route through `core.pipeline.apply_autograd_update`)
- **No scientific conclusion from any campaign delta until both arms pass an implementation-fidelity check.** A failed fidelity check is *inconclusive*, never a refutation. Deltas on known-defective axes are quarantined from attribution, not interpreted.
- Observed-but-unregistered deltas are never interpreted (pre-registration precedes comparison)

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
