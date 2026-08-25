# Computronium Sprint Plan: TODO4 — Sprint Close-Out & Research Foundation

## Status: TODO3 Remnants CARRIED FORWARD | Phase 7 (Close-Out) OPEN | Phase 8 (Research Prerequisites) NOT STARTED

> Consolidates all unchecked work from `TODO3.md` with the preliminary infrastructure defined in `RESEARCH3.md`. After Phase 7 + 8, work hands off to the RESEARCH3 catalog (15 items, 5 critical paths) under its Execution Protocol (E-1…E-11).

---

## Phase 7: TODO3 Sprint Close-Out

### 7.1 Rocq Proof Artifact (was TODO3 §4.3.4, PARTIAL)
**State**: statements repaired & compiling via `make` in `rocq/`. Proved: `Utils.v` (8 Qed, 0 admits), `gradE_diagonal`, `energyFunction_diagonal`, `stationary_is_fixed_point`.

- [ ] **7.1.1** Prove `energy_decreases_diagonal` — **NEXT, plumbing only**
  - Paper derivation complete: per-index Δ = −(η/2)(2−ηu)t² ≤ 0 with u = 1−Wᵢᵢ > 0, t = u·hᵢ − bᵢ, ηu < 2
  - Recipe (in STUB comment): remember u/t → rewrite Hstep/Hb → difference by `field` → sign chain `Rmult_le_pos` + `sq_nonneg` + `lra`
  - Watch-outs learned: no `nra`/`nlinarith` in this install (use `lra` + factored atoms + explicit sign certificates); `ring` cannot clear `/2` (use `field`); `remember` already substitutes (skip follow-up rewrites); `assert .. := tactic.` invalid (use `{ tactic }` or `by`)
- [ ] **7.1.2** General-case `energy_decreases`: Cauchy-Schwarz descent inequality on symmetrized form *(currently admitted w/ paper proof — CP-B pull-based)*
- [ ] **7.1.3** `settle_converges`: classical coercivity/completeness argument (fixed-point half already proved) *(CP-B pull-based)*
- [ ] **7.1.4** EqProp module split-out → new `rocq/EqProp.v` importing EnergyDynamics: controlLyapunov + nudgedSettleStep + locality axiom stub (numeric counterpart exists: `tests/property/test_eqprop_locality.py`) *(CP-B pull-based)*
- [~] **7.1.5** Optional CI: `rocq-prover` apt job `-I` flags vs. real runner — **SUPERSEDED by RESEARCH3 Theory Program ψ-coverage proposition**; job already exists (`ci.yml:91–97`) — touch only if it fails on a real runner

### 7.2 EqProp Competitive Verification (was TODO3 §5.1.1 — FINAL TASK)
**Config**: `hidden_dims=(512,512,512)`, `beta=0.1`, `inference_steps=20`, `lr=0.001`, `grad_clip=1.0`; auto-gradient checkpointing enabled (512×3 fits 10GB VRAM).

Locked-in fixes: separate free/nudged state objects in `train_step`; multi-layer settling w/ top-down pass in `EnergyMinimizationDynamics.settle`; small random recurrent init; `grad_clip` in `ParameterUpdateConfig` → `EuclideanUpdate.step()`.

- [ ] **7.2.1** Fix gradient clipping/settling loop and lock it in (first epoch NaN-free on CPU verified; stabilize across full epoch)
- [ ] **7.2.2** Run full 20-epoch MNIST training
- [ ] **7.2.3** Target >80% test accuracy — or document why not (either closes the task)

### 7.3 CI Gates (TODO3 DoD remainder)
Gate = current *configured* baseline (`pyproject.toml [tool.pyright]` — `strict` was deliberately relaxed to a per-rule profile; do NOT reintroduce), not aspirational strictness:

- [ ] pytest (full suite) green; coverage ≥15% measured on **full suite** only (partial runs read ~13%)
- [ ] `make` in `rocq/` compiles clean
- [ ] `ruff format --check .` + no *new* violations on touched files (9,578 existing findings are parked debt — burn-down deferred to the post-settlement dead-code purge, which shrinks the problem first)
- [ ] Pyright green at its configured per-rule profile — requires 7.4

**De-prioritized by design**: lint burn-down and coverage growth wait until the architecture settles and dead code is purged; both get cheaper then.

### 7.4 Hard Type Errors in Core (blocks the 7.3 pyright gate)
Genuine correctness errors surfacing even under the relaxed profile — invariants, not cosmetics:

- [ ] `computronium/core/ontology.py` (~52): `ExperimentConfig.ontology` unknown attr (:1539); `update_map` possibly unbound (:1648); `_param_name` assigned on `Tensor`/`Parameter` ×12 (:1951–:2284); `SystemState` list-invariance assignments `free_state`/`nudged_state`/`activations` (:2587–:2590 → widen to `Sequence`); `Tensor | float` returned as `Tensor` (:2670); `_AdaptedSystem` missing `to_spec`/`from_spec` protocol members (:2795); `int(Tensor | Module)` (:3004)
- [ ] `computronium/core/registry.py` (2): phantom `ComponentCategory.PROPAGATOR` attribute (:547); unbound generic params `num_layers`/`topology_type`/`connectivity`/`recurrent_weight` (:637)
- Policy: prefer deletion over patching where the code path is already dead (architecture settling); each fix either restores an invariant or removes the path

---

## Phase 8: Research Prerequisites (from RESEARCH3 §Factored Prerequisites)

Built once; consumed by the RESEARCH3 catalog. Startup sequence per RESEARCH3 §Team Allocation.

| ID | Prerequisite | Contents | Unblocks | When |
|----|--------------|----------|----------|------|
| **PR-0** | Verification gate | `docs/baseline.md` gates at-or-better (pytest / pyright strict / ruff) + TIER 0/digits campaign green | Every empirical item | **Day 1** |
| **PR-1** | Optimizer-phase hygiene | Rebuild Adam between meta-train and ψ-adaptation; `evaluate_z3` currently carries momentum buffers over frozen θ → contaminates exact-zero Δθ claim | Z3, Algorithm Migration | Days 2–3 |
| **PR-2** | θ-invariance audit harness | Snapshot → freeze → run → re-snapshot → exact-diff as reusable context manager, per-seed reports | Z3, Algorithm Migration, continual learning | Days 2–3 |
| **PR-3a** | Software resource instrumentation | Canonical `ResourceUsage` = `core/profiling.py:38` — **consolidate the two duplicate definitions first** (`core/stability/frontier.py:9`, `core/campaign/resource_vector.py:18`), then wire into every suite runner (proxy FLOPs/memory/latency; no hardware needed) | Z3 proxy-tier energy, L2 effective-FLOPs, AutoScientist frontier | Days 3–4 |
| **PR-3b** | Physical calibration anchor | One *measured* Joule/FLOP anchor workload (board sensor / wall meter / RAPL per `docs/hardware_targets.md`); calibrates proxies → measured tier w/ error bars | Measured-tier energy claims, Edge/Green AI, Hardware pilot | Procurement **Day 1** (lead-time-gated) |
| **PR-4** | Pre-registration & statistics kit | Seed count ≥5, bootstrap-CI utility, paired-comparison harness, threshold-registration template in repo | Z3, L1–L3.5, benchmark contract, discovery replication gates | Days 4–5 |
| **PR-5** | Calibrated stability guard | ROC-calibrated kill thresholds (<5% false-kill on known-good, >95% kill on unstable, <10% overhead); `_fast_proxy` vs full-Jacobian disagreement rate quantified | Unattended campaigns, discovery | After PR-7 |
| **PR-6** | Evaluation fairness contract | One pre-registered doc: per-rule tuning budgets (**GPU-hours, not epochs**), early-stopping, seeds, data splits, ICL-bridge scale-matching rule (performance-gated qualification ≥95%/task) | Benchmark paper, discovery pre-reg, edge comparisons, ICL bridge | Waiting periods (writing only) |
| **PR-7** | Switching-machinery shakedown | L3.5 two-task migration + L1 adaptation (+ L2/L3 smokes) as *instrumentation tests* before Z3: validates ψ reset, temperature schedule, diversity entropy, Δθ audit end-to-end on cheapest settings; harvests known-good/bad configs | Z3 (directly de-risked), PR-5 calibration, PR-9 smoke configs | Week 2 |
| **PR-8** | Export pipeline parity | ONNX/ternary export round-trip verified (accuracy delta ≤ noise) on one representative model | Edge/Green AI, Hardware pilot | Pull-based (CP-D) |
| **PR-9** | Campaign commissioning | One tiny AutoScientist campaign completing full iterate → interrupt → checkpoint → resume cycle (`autoscientist_campaigns/` empty today) | Frontier campaign, Algorithm discovery | After PR-5 |

### 8.1 Execution Order (startup sequence, first two weeks)

1. **Day 1**: PR-0 verification gate + place hardware orders (CP-D lead time is the constraint, not difficulty)
2. **Days 2–3**: PR-1 optimizer hygiene (verify no momentum carry-over) + PR-2 θ-invariance harness (test on trivially frozen model)
3. **Days 3–4**: PR-3a software instrumentation into suite runners
4. **Days 4–5**: PR-4 statistics kit checked in
5. **Week 2**: PR-7 shakedown in cost order — L3.5 (`algorithm_migration.py`: ψ reset, temperature schedule, Δθ audit) → L1 reduced-dims (`adaptation_efficiency.py`: switching stream, adaptation half-life) → L2/L3 smokes (`compute_efficiency.py`, `structural_robustness.py`: metrics populate). **Day-10 checkpoint** reviews all output — fix plumbing bugs at ~0.1% of full cost.
6. **Waiting periods** (any CP-A block, per E-8): draft PR-6; PyTorch wrapper API sketch (interface design only); Rocq scaffold compile check (done — 7.1 remains pull-based)

### 8.2 Dependency Chain (Phase 8 internal)

```
PR-0 ──→ PR-1 ──→ PR-2 ──→ PR-4 ──→ PR-7 (shakedown)
                                  ├─→ PR-5 (guard calibration ← known-good/bad configs)
                                  └─→ PR-9 (commissioning ← smoke-scale configs)
PR-3a ──→ (parallel; feeds Z3/frontier resource metrics)
PR-3b ←─ procurement Day 1 (latency-gated, off-spine)
PR-6, PR-8 (waiting-period / pull-based, no hard consumers until fan-out)
```

Exit criterion: PR-7 green + PR-5 calibrated + PR-9 commissioned ⇒ RESEARCH3 catalog unblocked end-to-end (CP-A proceeds to Z3 flagship; CP-B/C/D/E per spines).

---

## Handoff: Out of Scope Here (lives in RESEARCH3.md)

All 15 catalog items (Z3 flagship, ICL bridge, L1–L3.5 full runs, AutoScientist frontier, guard manifesto dataset, continual learning, physics proof, theory ψ-coverage + contraction propositions, benchmark paper, algorithm discovery, wrapper implementation, edge/green AI, hardware pilot, biological twin) execute under RESEARCH3's Critical Paths CP-A…CP-E, Team Allocation (~1.5 FTE: CP-A 70%, CP-C 15%, CP-B/D/E 15% shared), Execution Protocol E-1…E-11, and Publication Map. This file ends where the prerequisites end.

Bridge points from Phase 7:
- 7.1.2–7.1.4 feed CP-B (verification spine); ψ-coverage proposition is the next statement after diagonal plumbing closes
- 7.2's EqProp result anchors the eqprop coordinate in the future benchmark paper (PR-6 contract applies to any rerun)
- Energy-tracking + grad-clip fixes are load-bearing for L1/L3.5 shakedown runs

---

## Key References

| Artifact | Location |
|----------|----------|
| Working EqProp config | `computronium/experiments/eqprop_vision_parity.py::MODEL_CONFIGS["eqprop"]` |
| Parity tests | `tests/property/test_ontology_parity.py` |
| EqProp locality tests | `tests/property/test_eqprop_locality.py` |
| Rocq formalization | `rocq/` (canonical; Lean retired) |
| Z3 substrate | `computronium/experiments/joint/z3_fixed_weights.py`, `computronium/core/plasticity/rule_state.py` |
| Shakedown suites | `algorithm_migration.py`, `adaptation_efficiency.py`, `compute_efficiency.py`, `structural_robustness.py` (all in `computronium/experiments/joint/` unless noted) |
| Profiling / resources | `computronium/core/profiling.py:38` (canonical `ResourceUsage`; dupes in `stability/frontier.py`, `campaign/resource_vector.py` — consolidate per PR-3a) |
| Stability stack | `computronium/core/stability/` (`SpectralRadiusEstimator`, `_fast_proxy`) |
| Failure manifesto | `computronium/analysis/failure_manifesto.py` |
| Campaign stack | `autoscientist_campaigns/` (empty — see PR-9) |
| Hardware targets | `docs/hardware_targets.md`; baseline gates `docs/baseline.md` |
| Research catalog & protocol | `RESEARCH3.md` (items, CP-A…CP-E, E-1…E-11) |

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Gradient explosion in recurrent weights | Blocks 7.2 | `grad_clip` locked in; settling-loop fix is 7.2.1 gate |
| GPU OOM on 512×3 EqProp | Blocks 7.2 | Auto-gradient checkpointing (fits ~100MB on 10GB) |
| Rocq proof friction (no `nra`/`nlinarith`) | Delays 7.1 | Recipe + watch-outs recorded; admits acceptable past 7.1.1 per hard-stop policy |
| Momentum carry-over contaminates Δθ claims | Invalidates Z3 headline | PR-1 before any ψ-adaptation run; PR-2 audits mid-run |
| PR-3b hardware lead time | Gates measured-tier claims only | Procure Day 1; PR-3a keeps proxy tier unblocked |
| PR-5 false-kill rate | Blocks unattended campaigns | ROC calibration on PR-7-harvested config sets |
| PR-9 never exercised | Blocks frontier/discovery | Commissioning cycle is small + cheap; run right after PR-5 |

---

## Definition of Done (TODO4 Complete)

- [ ] **7.1.1** `energy_decreases_diagonal` proved (0-admit diagonal case); 7.1.2–7.1.4 either closed or explicitly parked under CP-B pull-based policy
- [ ] **7.2** EqProp 20-epoch MNIST >80% accuracy (or documented why not)
- [ ] **7.4** Hard type errors cleared from `ontology.py`/`registry.py` — fixed or dead paths deleted
- [ ] **7.3** CI green at configured baseline: pytest full suite, coverage ≥15% (full-suite measure), `make` in `rocq/`, `ruff format --check`, pyright per-rule profile; full lint burn-down explicitly deferred to post-settlement purge
- [ ] **PR-0…PR-4** merged and green
- [ ] **PR-7** shakedown green w/ harvested good/bad config sets
- [ ] **PR-5** calibrated + **PR-9** commissioned
- [ ] **PR-6** drafted; PR-3b procured/order placed; PR-8 parked pending CP-D
- [ ] Handoff: RESEARCH3 catalog unblocked — no further planning docs; execution moves to CP-A spine
