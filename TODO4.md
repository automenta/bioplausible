# Computronium Sprint Plan: TODO4 — Sprint Close-Out & Research Foundation

> Consolidates all unchecked work from `TODO3.md` with the preliminary infrastructure defined in `RESEARCH3.md`. After Phases 7 + 8, work hands off to the RESEARCH3 catalog (15 items, 5 critical paths) under its Execution Protocol (E-1…E-11). Session Log at the bottom is reverse-chronological.

## Status — Phase 9 EXECUTED (session 7); RESEARCH3 sweeps UNBLOCKED at harness scale

| Track | State |
|---|---|
| Phase 7 close-out — 7.1.1, 7.2, 7.3, 7.4 | ✅ done |
| Phase 8 prerequisites — PR-0…PR-7, PR-9 (+ PR-6 draft) | ✅ done (PR-3b procurement-pending, PR-8 pull-based) |
| §7 debt — pyright/lint/F-E triage, CLI de-mock, EqProp drift fix | ✅ cleared session 5 |
| ⚡ Phase 9 family-neutral pipeline (9.1–9.5) | ✅ executed session 7 (eqprop seed-42 MNIST rerun pending, see log) |
| RESEARCH3 catalog execution | ✅ UNBLOCKED at instrumentation+harness scale — campaign sweeps may compose any axis value |

### Execution queue (next session, in order)
1. **EqProp seed-42 MNIST parity rerun** (`experiments/joint/eqprop_mnist.py`) on the 3080 under the canonical pipeline — confirm best/final acc within noise of the 81.32 %/79.30 % record (metrics keys unchanged; step cost drops ~2× since local_goodness-style single-phase savings don't apply but thermo path is identical work minus dead branches). Recalibrates PR-6 budgets.
2. **RESEARCH3 CP-A / Z3 flagship** — Z3 smoke re-verified green post-Phase-9 (`theta_change=0.0`, `theta_invariant=true`); proceed per catalog.
3. Only if differentiable-through-settle is ever needed: root-cause residual graph retention (growth dropped 4.1→1.6 MB/step after out-of-place adds; `no_grad` masks it entirely — checkpointing path remains unused/vestigial meanwhile).

---

## ⚡ Phase 9: Family-Neutral Training Pipeline — EXECUTED (2026-08-25 session 7)

**Goal:** support ALL algorithms without biasing toward any specific one (EqProp included). Discovered during session-5 review; the current pipeline hardcodes EqProp's two-phase ritual for every family. Numbered after Phase 8 (discovered post-close-out) but executes ahead of everything else, including RESEARCH3 sweeps.

**Owner's directive:** no algorithm is second-class; capabilities are declared, not assumed.

**Outcome (session 7):** all of 9.1–9.5 landed. Canonical loop lives in **`core/pipeline.py`** (`run_train_step`/`run_forward`); every composed generation (5-D `_ComposedSystem`, 6-D `_JointSystem`, `_AdaptedSystem`, `System` protocol default) delegates to it — the four duplicated hand-rolled loops (~300 lines) are gone, which also closes the 9.5 consolidation decision (option A: one canonical loop; `SystemTrainer` remains pure epoch bookkeeping). Per-axis probe: **30/30 accepted combos green + 3 pairwise fences raise with reason + 11 prior exclusions removed**. Commissioning rerun: **PASSED incl. bit-exact resume**. Z3 smoke: θ-invariance unchanged.

### Evidence base (gathered session 5 — do not re-derive)

*Free/nudged usage inside each credit's `compute_pseudo_gradient`:*

| Credit | free refs | nudged refs | Verdict under today's pipeline |
|---|---|---|---|
| ThermodynamicContrast | 6 | 6 | native two-phase ✓ |
| HomeostaticCredit | 7 | 7 | two-phase ✓ |
| RandomProjectionsCredit | 6 | 3 | two-phase, mostly used |
| TargetInversionCredit | 1 | 2 | partially wasted settles |
| LocalGoodnessCredit | 2 | **0** | nudged settle = pure waste |
| BackpropCredit | **0** | 2 | free settle = pure waste (+ crashes detached, see 9.2) |
| TemporalTraceCredit | **0** | **0** | **both settles pure waste** |

*Structural bias markers:* `core/credit/adapters.py` star topology routes all cross-family interop THROUGH the EqProp credit. **Session 7 enumeration: adapters have ZERO live call sites** outside the package `__init__` re-export — star topology already dormant; adapters remain only for deliberate hybrid-composition experiments. `_train_step_spiking`'s per-dynamics dispatch is superseded by phase negotiation (and its spike-recording was a literal `pass` placeholder — `record_spikes` has no callers repo-wide).

### Code anchors (stable paths only — re-resolve at edit time)

| Concern | Location |
|---|---|
| Composed `train_step`, trainer-side generation | `_ComposedSystem` built inside `SystemTrainer` (`core/system_trainer.py`) — incl. `_train_step_spiking` dispatch precedent |
| Composed `train_step`, adapted-system generation | `_AdaptedSystem.train_step` (`core/ontology.py`) |
| Raw-loop training pattern | `TrainingMixin.train_step` (`core/training_mixin.py`) — candidate canonical for the 9.5 consolidation decision |
| Credit protocol + native family classes | `CreditAssignment` protocol + `ThermodynamicContrast` et al. in `core/ontology.py` (`compute_pseudo_gradient` per class) |
| Cross-family adapter star | `core/credit/adapters.py` |
| Composition entry + substrate selection | `compose_joint_system_from_configs` (`core/system_trainer.py`); `SubstrateConfig` in `core/ontology.py` (+ its `from_spec` selector) |
| Axis fences | `_EXCLUDED_AXES` in `core/campaign/evaluation.py` |

*Axis probe (session 5):* 16/21 probed combos (build + one `train_step`) green; the 5 failures fall under the fences below. `_EXCLUDED_AXES` records a reason per fenced axis value — 11 total: substrates ×5 (`analog`, `memristive`, `neuromorphic`, `sparse`, `ternary`), gradient-credit spellings ×2 (`gradient`, `backprop`), non-euclidean updates ×4 (`riemannian_orthogonal`, `spectral_constrained`, `natural_gradient`, `elastic_consolidation`).

### 9.1 Capability-declared phase negotiation — **DONE (session 7)**
- [x] `Phase` StrEnum (`free`/`nudged`, extensible) in `core/ontology.py`; credits declare `phases: ClassVar[tuple[Phase, ...]]` — thermodynamic_contrast / homeostatic / random_projections / target_inversion = `(FREE, NUDGED)`; local_goodness = `(FREE,)`; backprop = `(NUDGED,)` + `requires_autograd=True`; temporal_trace = `()`
- [x] All three composed `train_step` generations + the `System` protocol default delegate to `pipeline.run_train_step`, which settles exactly `credit.phases` and passes credits a phase-keyed `Mapping[Phase, SystemState]` (uniform signature `compute_pseudo_gradient(states, loss, geometry)`)
- [x] `_train_step_spiking`/`_is_spiking_system` dispatch deleted — generalized by negotiation (temporal_trace settles 0 phases; STDP reads recorded spikes only; live spike recording remains RESEARCH3 scope)
- [x] Regression tests: settle-count == declared phases per family + metrics parity keys (`tests/unit/core/test_family_neutral_pipeline.py`)
- [x] Adapter call-site enumeration: zero live sites (see outcome note); adapters updated to new signature with hybrid capability declarations

### 9.2 Autograd-capable path for gradient credit — **DONE (session 7)**
- [x] `requires_autograd: ClassVar[bool]`; `run_train_step` runs under `nullcontext()` only when flagged, else the default `torch.no_grad()` (7.2.1 semantics preserved for everyone else). BackpropCredit additionally guards on `loss.requires_grad` and computes grads over learnable weights only (bias grads discarded — consistent contract)
- [x] Memory-leak regression gate: flat live-graph-tensor census over steps on CPU (hard gate) + flat CUDA MB variant when available
- [x] `gradient`/`backprop` un-excluded from `_EXCLUDED_AXES`; wired into `_CREDIT_FACTORIES`

### 9.3 Non-euclidean update fixes — **DONE (session 7)**
- [x] Root cause: all four updates paired `pseudo_grads[i]` with `params.items()` **by index**; bias interleaving made `[hidden] - lr·[out,in]` broadcast-crash. Fixed once via shared `apply_pseudo_gradients(params, grads, transform)` choke point (pairs by learnable-weight order — the same predicate every credit uses to emit; biases pass through; grads detached at the single choke point). EuclideanUpdate's shape-skip guard removed (it had been silently masking dead credit rules — see log)
- [x] Unit tests per update over composed params incl. biases (+ clip/momentum/detach behavior): `tests/unit/core/test_update_rules.py`
- [x] Un-excluded from `_EXCLUDED_AXES`

### 9.4 Substrate-type tag — **DONE (session 7)**
- [x] `SubstrateType` StrEnum + explicit `substrate_type` field on `SubstrateConfig` (default digital); all nine config factories set it
- [x] Single selector `substrate_from_config(config)` (match/case; sparse/ternary import lazily from `core/substrates/`) used by both `compose_*_from_configs` and `_ComposedSystem.from_spec`. Also fixed the credit dispatch: explicit match with `ValueError` on unknown — a `homeostatic` config previously fell into the silent `else → BackpropCredit` mislabel
- [x] Behavioral fidelity test: analog noise fires under composed coordinates; per-value class-selection assertions incl. Sparse/Ternary
- [x] Un-excluded the 5 substrates (plus optical/quantum now wired too)

### 9.5 Neutrality verification harness — **DONE (session 7)**
- [x] Permanent parametrized probe: `tests/unit/core/test_axis_probe.py` — every accepted per-axis combo builds + trains one real step w/ parity metric keys; pairwise fences raise `UnsupportedCoordinateError("dynamics", "…requires layered geometry…")` (`tile_mesh` × energy_minimization/predictive_settling/spike_integration — discovered session 7); any future `_EXCLUDED_AXES` entry is asserted to actually raise
- [x] Metrics schema parity: `{"loss","energy","accuracy"}` ⊆ every family's step metrics, float extras merged from output state (spike stats survive loss computation now)
- [x] Trainer consolidation decided: ONE canonical loop in `core/pipeline.py`; all generations delegate; raw-loop duplication retired

### Constraints (hard) — all verified session 7
- ✅ 7.2.1 stability/memory semantics preserved (no_grad default; detach at the single `apply_pseudo_gradients` choke point; memory-flatness gate green)
- ✅ Commissioning bit-exact redo determinism stays green (rerun PASSED)
- ✅ Z3 `theta_invariant=true` semantics unchanged (smoke re-verified)
- Rocq proofs untouched
- Backwards compatibility NONE — signatures broken freely; all call sites swept (credits, adapters, gradient_check, distributed_trainer, cli/lab, grpc_seam test)
- ⏳ PR-6 recalibration: relative step costs shifted (one-phase families settle half; thermo path identical minus dead branches) — final numbers ride on the pending seed-42 rerun

### Dependency chain
```
9.1 (phase negotiation) ──→ 9.2 (autograd flag rides the negotiation)
        └──→ 9.5 harness (grows alongside, gates exit)
9.3, 9.4 independent ──→ un-exclusions land as each closes
Exit: probe fully green-or-reasoned + parity reruns + harness merged ⇒ RESEARCH3 sweeps UNBLOCKED
```

---

## Phase 7: TODO3 Sprint Close-Out — CLOSED *(residual: 7.1.2–7.1.4 parked CP-B pull-based; 7.1.5 superseded)*

### 7.1 Rocq Proof Artifact (was TODO3 §4.3.4, PARTIAL)
**State**: statements repaired & compiling via `make` in `rocq/`. Proved: `Utils.v` (8 Qed, 0 admits), `gradE_diagonal`, `energyFunction_diagonal`, `stationary_is_fixed_point`.

- [x] **7.1.1** Prove `energy_decreases_diagonal` — **DONE (2026-08-25)**
  - Closed via new scalar lemma `per_index_descent` (`rocq/EnergyDynamics.v`): per-index difference = −(η/2)(2−ηu)t² with u = 1−Wᵢᵢ > 0, t = u·h−b; identity discharged by `field`, sign chain by `Rmult_le_pos` ×2 + `sq_nonneg` + `lra`.
  - Main theorem lifts pointwise via `sum_R_le`; `settleStep i` rewritten through `gradE_diagonal`.
  - Verified: `make` clean; `Print Assumptions energy_decreases_diagonal` shows only stdlib axioms (classical choice via `Rle_lt_dec` in `sq_nonneg`, funext from Reals) — **0 admits** in the dependency chain.
  - Recipe note for future proofs: scalar-lemma-first beats inline `remember` plumbing — keeps `field`/`lra` goals small and reusable.
- [ ] **7.1.2** General-case `energy_decreases`: Cauchy-Schwarz descent inequality on symmetrized form *(currently admitted w/ paper proof — CP-B pull-based)*
  - Hint from 7.1.1 close-out: the scalar-lemma + `sum_R_le` lifting pattern applies here too once the Cauchy–Schwarz estimate is factored as its own lemma.
- [ ] **7.1.3** `settle_converges`: classical coercivity/completeness argument (fixed-point half already proved) *(CP-B pull-based)*
- [ ] **7.1.4** EqProp module split-out → new `rocq/EqProp.v` importing EnergyDynamics: controlLyapunov + nudgedSettleStep + locality axiom stub (numeric counterpart exists: `tests/property/test_eqprop_locality.py`) *(CP-B pull-based)*
- [~] **7.1.5** Optional CI: `rocq-prover` apt job `-I` flags vs. real runner — **SUPERSEDED by RESEARCH3 Theory Program ψ-coverage proposition**; job already exists (`ci.yml:91–97`) — touch only if it fails on a real runner

### 7.2 EqProp Competitive Verification (was TODO3 §5.1.1 — FINAL TASK)
**Config**: `hidden_dims=(512,512,512)`, `beta=0.1`, `inference_steps=20`, `lr=0.001`, `grad_clip=1.0`; auto-gradient checkpointing enabled (512×3 fits 10GB VRAM).

Locked-in fixes: separate free/nudged state objects in `train_step`; multi-layer settling w/ top-down pass in `EnergyMinimizationDynamics.settle`; small random recurrent init; `grad_clip` in `ParameterUpdateConfig` → `EuclideanUpdate.step()`.

- [x] **7.2.1** Fix gradient clipping/settling loop and lock it in — **DONE (2026-08-25 session 2)**. Four root causes found & fixed; see Session Log §stability for the full chain:
  1. checkpointed settle path compared `all_acts[-1]` with *itself* (`torch.dist(a,a)`≡0) → always broke at `convergence_start`; now tracks `prev_output`
  2. `StateDynamicsConfig.step_size` existed but `_settle_step` never applied it → un-damped bidirectional settling diverged (energy → −1e12 within an epoch); now relaxes `h ← h + η·(f(·)−h)`
  3. `EuclideanUpdate` clipped **per-tensor**, rescaling every gradient to norm exactly 1.0 → erased near-equilibrium decay, constant-size random walk; now global-norm clip (`clip_grad_norm_` semantics)
  4. optimizer momentum 0.9 amplified noise ×10 past the accuracy peak (80% @ep0 then collapse); `create_eqprop_system(update_momentum=…)` parameterized, 7.2 uses 0.0
  - Plus the blocking memory leak: settle autograd graphs accumulated ~4 MB/step → CUDA OOM at epoch 4. Pseudo-grads are consumed as plain values (no backward anywhere in the pipeline) → detached in `EuclideanUpdate.step`; `_ComposedSystem.train_step` runs under `torch.no_grad()`; `forward_with_intermediates` bias/recurrent adds made out-of-place. Verified flat 16.6 MB over 400 steps.
- [x] **7.2.2** Run full 20-epoch MNIST training — **DONE (2026-08-25 session 3)**. Clean 20-epoch record, no divergence/OOM/NaN abort (558.9 s on the 3080). `results/eqprop_mnist/results.json`: best_val_acc=**81.17 %** (~ep7), final_val_acc=57.14 % (late-run drift — see session log). Runner's strict `target_met` flag (final-epoch gate) is False by design; the registered claim ("reaches ≥80 % within schedule") is met by the best.
- [x] **7.2.3** Target >80% test accuracy — **DONE**: best 81.17 % crosses target (prior session's ep7 81.2 % reproduced under seed 42). Known limitation recorded as new work item: val accuracy decays after ~ep10 (energy keeps dropping to −6e4 while acc falls → late-phase objective misalignment); fix vocabulary = LR decay, early stopping on val, or weight-norm regularization before any rerun for a paper-grade final-epoch number. **→ Late-drift FIXED session 5**: per-epoch LR decay (γ=0.9 from ep3) + val-based early stopping (patience 4); rerun gave best **81.32 %** @ep7, final **79.30 %** @ep11 (early-stopped) vs prior 57.14 % collapse at ep19 — drift absent in this rerun (single seed; multi-seed confirmation pending), both numbers reported per PR-6 contract (`results/eqprop_mnist/results.json`).

### 7.3 CI Gates (TODO3 DoD remainder)
Gate = current *configured* baseline (`pyproject.toml [tool.pyright]` — `strict` was deliberately relaxed to a per-rule profile; do NOT reintroduce), not aspirational strictness:

- [x] pytest (full suite) green at baseline; coverage ≥15% — **DONE (2026-08-25 session 3)**: 1043 passed / 59 F / 18 E / 66 s / 11 xfailed / 3 xpassed; **coverage 47.13 %** vs floor 15 %. All 77 failure/error lines proven pre-existing by `git stash` A/B against HEAD (identical sets). Excludes two known-hang modules (`test_ontology_parity.py`, `test_grpc_seam_subprocess.py`) — both documented as work items. Proto codegen repair this session unblocked dht/grpc_seam/p2p_constraints collection.
- [x] `make` in `rocq/` compiles clean *(re-verified after 7.1.1)*
- [x] `ruff format --check .` **green repo-wide** — re-verified session 3 after excluding generated `*_pb2*.py` via `pyproject.toml` `[tool.ruff] exclude`
- [x] Pyright green at its configured per-rule profile on touched files — **session 3**: validation/ + new tests = 0 errors; repo-wide count 3853 → **3837**

**De-prioritized by design**: lint burn-down and coverage growth wait until the architecture settles and dead code is purged; both get cheaper then.

### 7.4 Hard Type Errors in Core — **DONE (2026-08-25)**
Genuine correctness errors surfacing even under the relaxed profile — invariants, not cosmetics. All cleared; `pyright core/ontology.py core/registry.py` = **0 errors** (was 61 errors incl. ~52 enumerated).

**registry.py (2 → 0):**
- `ComponentCategory.PROPAGATOR` (:547): `check_compatibility` had zero callers repo-wide → **deleted** (dead path per policy)
- Unbound generic params (:637): call switched to `GeometryConfig.feedforward(...)` factory (matches the classmethod pattern used everywhere else)

**ontology.py (~52 → 0):**
- `ExperimentConfig.ontology` + `update_map` unbound: `SystemConfig.from_experiment` was written for a config shape that no longer exists and would `AttributeError` at runtime; its sole caller `SystemTrainer.from_configs` also had zero live callers (docstring mention only) → **both deleted**, plus now-unused TYPE_CHECKING imports
- `_param_name` ×12: new module helper `_set_param_name(tensor, name)` (`setattr`) — single choke point, readers already use `getattr(...,"_param_name","default")`
- SystemState assignments: root cause was `[state.activations]` wrapping (nested-list bug, not just typing); fixed to pass-through. NOTE: widening fields to `Sequence[Tensor]` was tried and **reverted** — consumers do tensor ops on these fields and the widening cascaded ~30 new errors; keep `list[Tensor] | Tensor | None`
- `surrogate_objective` ×2 + homeostatic variant: `torch.as_tensor(...)` wrap (no-op on Tensor, lifts float)
- LocalGoodness goodness sum: empty-range int-0 case handled by same wrap
- `_AdaptedSystem`: real `to_spec()` added (5 axis configs via `fields()`, schema_version 1.0); `from_spec` raises `NotImplementedError` with pointer to ModelAdapter (wrapped legacy nn.Module is not spec-reconstructable)
- `int(Tensor | Module)` in `_infer_input_dim/_infer_output_dim`: `hasattr` on nn.Module can't narrow (submodule hazard); replaced with `getattr` + `isinstance(int)`/1-element-Tensor checks
- `momentum` missing ×2: required `StateDynamicsConfig` field added (`momentum=0.0`) to the two raw constructor calls
- "Expected 0 positional arguments" ×2: caused by annotating maps as `dict[str, type[CreditAssignment]]` / `list[tuple[..., type[CreditAssignment]]]` — protocol `type[...]` constructors expose 0 params; dropped annotations so pyright infers the concrete union
- `Tensor not callable`: `self.model.train_step` under `hasattr` → `getattr` + `callable()` narrowing
- Geometry `_layers`/`_recurrent_weight` reaches (~20 errors): new `_layer_stack(geometry)` / `_recurrent_weight(geometry)` helpers; **non-checkpointed settle path deduplicated into a direct `self._settle_step(...)` call** (−70 lines of duplicated loop body — checkpointed/non-checkpointed now share one implementation)
- Spike-count metrics contract violation: producer stored `list[Tensor]` in `metrics: dict[str, float]`; consumer summed it. Fixed both ends: producer writes pre-aggregated float `avg_spikes_per_neuron`; consumer reads it directly (also fixes 2 latent pyright errors in `system_trainer.py`)
- `_estimate_layer_lipschitz(i, None)` crash path: geometry param widened to `Geometry | None`, neutral 1.0 return

**Verification:** targeted suites (unit/core, test_refactor, ontology_locks, axis_certifications): 352 passed, failures identical to baseline; parity suite timeout pre-exists at baseline. Repo-wide pyright −65 errors.

- Policy: prefer deletion over patching where the code path is already dead (architecture settling); each fix either restores an invariant or removes the path

### 7.5 New work item (discovered during 7.4): repo-wide pyright burn-down to gate
Remaining ~3853 errors are concentrated in `acceleration/compile.py`, `acceleration/contrastive_kernels.py`, `acceleration/eqprop_kernel_backend.py` (attribute-in-init patterns like the old `_param_name` issue), and `experiments/`. Same fix vocabulary applies: `getattr`+`isinstance` narrowing over `hasattr`, declared-instance helpers instead of duck-typed attribute writes, deletion of dead branches. Suggest tackling per-module after the post-settlement dead-code purge shrinks the surface.

---

## Phase 8: Research Prerequisites (from RESEARCH3 §Factored Prerequisites) — COMPLETE except PR-3b *(procurement-pending)* & PR-8 *(pull-based CP-D)*

Built once; consumed by the RESEARCH3 catalog. Startup sequence per RESEARCH3 §Team Allocation.

| ID | Prerequisite | Contents | Unblocks | When |
|----|--------------|----------|----------|------|
| **PR-0** ✅ | Verification gate | `docs/baseline.md` gates at-or-better (pytest / pyright strict / ruff) + TIER 0/digits campaign green — **DONE 2026-08-25**: baseline.md refreshed w/ full-suite numbers (47.13 % cov; 77 pre-existing F/E proven via stash A/B) | Every empirical item | **Day 1** |
| **PR-1** ✅ | Optimizer-phase hygiene | Rebuild Adam between meta-train and ψ-adaptation; `evaluate_z3` currently carries momentum buffers over frozen θ → contaminates exact-zero Δθ claim | Z3, Algorithm Migration | ~~Days 2–3~~ Done |
| **PR-2** ✅ | θ-invariance audit harness | Snapshot → freeze → run → re-snapshot → exact-diff as reusable context manager, per-seed reports | Z3, Algorithm Migration, continual learning | ~~Days 2–3~~ Done |
| **PR-3a** ✅ | Software resource instrumentation | Canonical `ResourceUsage` = `core/profiling.py:38` — duplicates deleted (`stability/frontier.py`, `campaign/resource_vector.py`), wired into every suite runner via `measure_suite_resources` (proxy FLOPs/memory/latency; no hardware needed) | Z3 proxy-tier energy, L2 effective-FLOPs, AutoScientist frontier | ~~Days 3–4~~ Done |
| **PR-3b** | Physical calibration anchor | One *measured* Joule/FLOP anchor workload (board sensor / wall meter / RAPL per `docs/hardware_targets.md`); calibrates proxies → measured tier w/ error bars | Measured-tier energy claims, Edge/Green AI, Hardware pilot | Procurement **Day 1** (lead-time-gated) |
| **PR-4** | Pre-registration & statistics kit | Seed count ≥5, bootstrap-CI utility, paired-comparison harness, threshold-registration template in repo — **DONE 2026-08-25** (`validation/preregistration.py` + `docs/preregistration_template.md` + `configs/preregistrations/eqprop_mnist_80pct.json` + 9 unit tests incl. hypothesis property test; fixed latent `cohens_d` crash via new one-sample `cohens_dz`) | Z3, L1–L3.5, benchmark contract, discovery replication gates | Days 4–5 | ✅ |
| **PR-5** ✅ 2026-08-25 session 4 | Calibrated stability guard | `core/stability/guard.py`: `StabilityGuard` (two statistic modes: `fast_proxy`, `windowed_growth`), `calibrate_threshold` (max-margin feasible ROC point), `quantify_proxy_disagreement`, `measure_guard_overhead`; driver `scripts/calibrate_stability_guard.py` → artifact `benchmark_results/stability_guard_calibration/calibration.json`. **Result**: windowed_growth τ=1.029, FKR=0 % (≤5 %), KR=100 % (≥95 %); fast_proxy INFEASIBLE on non-normal systems (median rel err ≈50 %) — see session log | Unattended campaigns, discovery | Done |
| **PR-6** ✅ 2026-08-25 session 4 | Evaluation fairness contract | `docs/evaluation_fairness_contract.md` drafted: GPU-hour budgets per rule family, best-val early stopping w/ both numbers reported, ≥5 seeds + PR-4 kit stats, fixed splits seed 42, ICL-bridge ≥95 %/task qualification on measured FLOPs/step | Benchmark paper, discovery pre-reg, edge comparisons, ICL bridge | Drafted; binding at rerun |
| **PR-7** ✅ full-scale 2026-08-25 session 4 | Switching-machinery shakedown | All four suites rerun at configured budgets (`--device cuda`, sequential): exit 0, populated JSON in `benchmark_results/*/` incl. per-seed `resources`. **Day-10-checkpoint finding**: suites are instrumentation shells — toy `forward()` never touches plasticity/ψ and A1 training updates θ freely (θ-change≈0.72≠0); true ψ-reset/Δθ machinery lives in the Z3 path. See session log | Z3 de-risked, PR-5 calibration, PR-9 smoke configs | Done at instrumentation scale; real-data runs = RESEARCH3 L1–L3.5 |
| **PR-8** | Export pipeline parity | ONNX/ternary export round-trip verified (accuracy delta ≤ noise) on one representative model | Edge/Green AI, Hardware pilot | Pull-based (CP-D) |
| **PR-9** ✅ 2026-08-25 session 4 | Campaign commissioning | `autoscientist_campaigns/commission.py`: tiny REAL campaign (composed 6-D system via `compose_joint_system_from_configs`) through iterate → checkpoint → interrupt → resume → complete. All checks green incl. **bit-exact redo determinism** (redone episode loss identical to 16 digits). Report: `autoscientist_campaigns/commission_report.json`. Fixed latent gaps the CLI runner had (checkpointing was commented-out mock; see session log) | Frontier campaign, Algorithm discovery | Done |

### 8.1 Execution Order (startup sequence, first two weeks)

1. **Day 1**: PR-0 verification gate + place hardware orders (CP-D lead time is the constraint, not difficulty) — **PR-0 DONE 2026-08-25** (baseline.md refreshed)
2. **Days 2–3**: PR-1 optimizer hygiene (verify no momentum carry-over) + PR-2 θ-invariance harness (test on trivially frozen model) — **PR-1 + PR-2 DONE 2026-08-25** (see Session Log); **Z3 smoke verified session 3**: `evaluate_z3("digital/recurrent/energy_minimization/rule_state/thermodynamic_contrast/euclidean", 5/2 epochs, cpu)` → `theta_change=0.0, theta_invariant=true`, schema unchanged
3. **Days 3–4**: PR-3a software instrumentation into suite runners — **DONE 2026-08-25** (`ResourceUsage` consolidated into `core/profiling.py`; `measure_suite_resources` wired into all four joint suite evaluators)
4. **Days 4–5**: PR-4 statistics kit checked in — **DONE 2026-08-25 session 3** (template doc + example registration JSON + unit tests; latent crash fixed)
5. **Week 2**: PR-7 shakedown in cost order — L3.5 (`algorithm_migration.py`: ψ reset, temperature schedule, Δθ audit) → L1 reduced-dims (`adaptation_efficiency.py`: switching stream, adaptation half-life) → L2/L3 smokes (`compute_efficiency.py`, `structural_robustness.py`: metrics populate). **Smoke level GREEN session 3** (all four `--quick`, exit 0, JSON populated). **Day-10 checkpoint** reviews all output — fix plumbing bugs at ~0.1% of full cost; full-scale runs remain.
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
| 7.2 MNIST runner | `computronium/experiments/joint/eqprop_mnist.py` (results: `results/eqprop_mnist/results.json`) |
| θ-invariance harness (PR-2) | `computronium/core/plasticity/theta_audit.py`; Z3 consumer in `experiments/joint/z3_fixed_weights.py::evaluate_z3` |
| Pre-registration kit (PR-4) | `computronium/validation/preregistration.py`; template `docs/preregistration_template.md`; example `configs/preregistrations/eqprop_mnist_80pct.json`; tests `tests/unit/validation/test_preregistration.py` |
| Profiling / resources (PR-3a) | `computronium/core/profiling.py:38` canonical `ResourceUsage` (+ `measure_suite_resources`); duplicate definitions deleted (`stability/frontier.py` now imports/re-exports, `campaign/resource_vector.py` removed) |
| Parity tests | `tests/property/test_ontology_parity.py` |
| EqProp locality tests | `tests/property/test_eqprop_locality.py` |
| Rocq formalization | `rocq/` (canonical; Lean retired) |
| Campaign episode machinery (session 5) | `computronium/core/campaign/evaluation.py` (`build_coordinate_system`, `evaluate_episode`, `episode_batch`, `activity_transition`); consumed by `autoscientist_campaigns/commission.py` + `computronium/cli/campaign.py` |
| Credit adapter star (Phase 9 target) | `computronium/core/credit/adapters.py` (`*ToThermodynamic*` hub); phase-usage evidence in ⚡ Phase 9 header |
| Z3 substrate | `computronium/experiments/joint/z3_fixed_weights.py`, `computronium/core/plasticity/rule_state.py` |
| Shakedown suites | `algorithm_migration.py`, `adaptation_efficiency.py`, `compute_efficiency.py`, `structural_robustness.py` (all in `computronium/experiments/joint/` unless noted) |
| Stability stack | `computronium/core/stability/` (`SpectralRadiusEstimator`, `_fast_proxy`); **guard (PR-5): `core/stability/guard.py`** + driver `scripts/calibrate_stability_guard.py` → `benchmark_results/stability_guard_calibration/calibration.json`; tests `tests/unit/core/test_stability_guard.py` |
| Commissioning (PR-9) | `autoscientist_campaigns/commission.py` (+ `campaign.db`, `checkpoints/`, `commission_report.json` artifacts) |
| Fairness contract (PR-6) | `docs/evaluation_fairness_contract.md` |
| Failure manifesto | `computronium/analysis/failure_manifesto.py` |
| Campaign stack | `autoscientist_campaigns/commission.py` (PR-9) + `computronium/cli/campaign.py` (real runner post-session 5); artifacts `campaign.db`, `checkpoints/`, `commission_report.json` |
| **Canonical training loop (Phase 9)** | `computronium/core/pipeline.py` (`run_train_step`/`run_forward`/`task_loss`/`phase_states`); capabilities on credits in `core/ontology.py` (`Phase`, `phases`, `requires_autograd`) |
| **Update-rule pairing (Phase 9)** | `core/ontology.py::apply_pseudo_gradients` — single choke point; bias-safe; detaches |
| **Substrate selection (Phase 9)** | `core/ontology.py::substrate_from_config` + `SubstrateType` tag on `SubstrateConfig`; sparse/ternary via lazy imports from `core/substrates/` |
| **Neutrality harness (Phase 9)** | `tests/unit/core/test_axis_probe.py` + `test_family_neutral_pipeline.py` + `test_update_rules.py` |
| Hardware targets | `docs/hardware_targets.md`; baseline gates `docs/baseline.md` |
| Research catalog & protocol | `RESEARCH3.md` (items, CP-A…CP-E, E-1…E-11) |

---

## Risk Mitigation

| Risk | Impact | Mitigation | State |
|------|--------|------------|-------|
| Gradient explosion in recurrent weights | Blocked 7.2 | Global-norm `grad_clip` locked in (7.2.1 fix chain) | ✅ retired |
| GPU OOM on 512×3 EqProp | Blocked 7.2 | Auto-gradient checkpointing + leak fixes (flat 16.6 MB / 400 steps verified) | ✅ retired |
| Autograd through settle reintroduces memory leak | Blocks 9.2 | `requires_autograd` flag gates grad-enable + census/MB regression gates in `test_family_neutral_pipeline.py` | ✅ retired (session 7) |
| Non-euclidean updates crash on composed params | Fenced 4 axes (9.3) | Index-pairing root-caused; shared `apply_pseudo_gradients` pairing; unit tests incl. biases | ✅ retired (session 7) |
| Substrate mislabeling via precision-only class selection | Fenced 5 substrates (9.4) | Explicit `SubstrateType` tag + single `substrate_from_config` selector + fidelity tests; also killed the silent homeostatic→BackpropCredit fallback | ✅ retired (session 7) |
| Rocq proof friction (no `nra`/`nlinarith`) | Delays CP-B items | Scalar-lemma-first recipe recorded (7.1.1); admits acceptable past 7.1.1 per hard-stop policy | 🟡 managed |
| Momentum carry-over contaminates Δθ claims | Invalidates Z3 headline | PR-1 rebuilds Adam post-freeze; PR-2 audits mid-run; Z3 smoke green | ✅ mitigated |
| PR-3b hardware lead time | Gates measured-tier claims only | Procure Day 1 (still pending); PR-3a keeps proxy tier unblocked | 🟡 external |
| PR-5 false-kill rate | Blocks unattended campaigns | ROC-calibrated τ=1.029 (FKR 0 %, KR 100 %); runner kill-decision wiring still open (RESEARCH3 CP-A/Z3 smoke) | 🟡 mitigated at calibration scale |
| Foreign git stash makes `git stash` A/B unsafe | Corrupts working tree (~330 paths splattered once, session 5 incident) | Baseline A/B only via `git worktree add /tmp/x HEAD` until that stash is claimed/dropped | 🔴 live |

---

## Definition of Done (TODO4 Complete — reopened by ⚡ Phase 9)

- [x] **⚡ Phase 9 family-neutral pipeline** — 9.1–9.5 closed session 7; probe green-or-fenced-with-reason; commission bit-exact ✅ + Z3 θ-invariant ✅ reruns; eqprop seed-42 MNIST rerun pending (GPU, next-session queue item 1); PR-6 budget numbers pending that run
- [x] **7.1.1** `energy_decreases_diagonal` proved (0-admit diagonal case) ✅ 2026-08-25; 7.1.2–7.1.4 remain explicitly parked under CP-B pull-based policy
- [x] **7.2** EqProp 20-epoch MNIST >80% accuracy — ✅ 2026-08-25 session 3: full 20-epoch record, best_val_acc **81.17 %** (target crossed), no divergence/OOM/NaN; final-epoch drift (57.14 %) recorded as new work item
- [x] **7.4** Hard type errors cleared from `ontology.py`/`registry.py` — fixed or dead paths deleted ✅ 2026-08-25 (0 pyright errors on both files; dead paths `from_experiment`/`from_configs`/`check_compatibility` removed)
- [x] **7.3** CI green at configured baseline — ✅ 2026-08-25 session 3: full-suite pytest 1043 passed / coverage **47.13 %** (floor 15 %) / all 77 F+E pre-existing via stash A/B; `ruff format --check .` green (`*_pb2*.py` excluded); pyright touched-files clean, repo count 3837
- [x] **PR-0…PR-4 merged and green**: PR-0 ✅ (baseline.md refreshed) · PR-1 ✅ + Z3 smoke green · PR-2 ✅ · PR-3a ✅ · PR-4 ✅ complete (tests + template + example JSON)
- [x] **PR-7** full-scale (instrumentation budgets) green w/ JSON artifacts regenerated in `benchmark_results/` ✅ session 4; Day-10-checkpoint review produced the suite-shell finding below
- [x] **PR-5** calibrated ✅ session 4: windowed_growth τ=1.029 / FKR 0 % / KR 100 %; fast_proxy proven insufficient alone on non-normal systems
- [x] **PR-9** commissioned ✅ session 4: full fault-tolerance cycle, bit-exact redo, report JSON committed
- [x] **PR-6** drafted ✅ session 4 (`docs/evaluation_fairness_contract.md`); PR-3b procurement still external/pending; PR-8 parked pending CP-D
- [x] **Handoff: RESEARCH3 catalog unblocked** ✅ session 5 — pre-RESEARCH3 engineering debt from §7.5 + session logs cleared (see Session Log session 5): campaign CLI runs real composed systems w/ fault-tolerant checkpointing; `ResourceUsage.measure` device-honest; stale `profiling.py` stability call fixed & live; suite ψ-wiring **audited per-suite** (L1/L2 `psi_wired_uncontrolled`, L3.5/L3 `plumbing_only`; rewiring stays open); EqProp late-drift fixed.
- [ ] **GATE LIFTED (session 7): RESEARCH3 campaign-scale sweeps UNBLOCKED** — Phase 9 exit criteria met at harness scale (probe green/fenced + parity reruns green + harness merged). Remaining parity item: eqprop MNIST seed-42 rerun. Anything touching new axis values must keep `_EXCLUDED_AXES`/pairwise fences honest via `test_axis_probe.py`.

### Exit criterion status
PR-7 green ✅ + PR-5 calibrated ✅ + PR-9 commissioned ✅ ⇒ RESEARCH3 catalog unblocked end-to-end at instrumentation scale (CP-A proceeds to Z3 flagship; CP-B/C/D/E per spines). PR-3b measured-tier claims remain gated on hardware arrival.

---

## Session Log & Future-Work Notes

### 2026-08-25 session 7 (⚡ Phase 9 executed: family-neutral pipeline) — **GATE LIFTED**

**Executed (9.1→9.5 dependency order):**
1. **Canonical loop extracted** (`core/pipeline.py`, new): `run_train_step` settles exactly `credit.phases` (phase-keyed `Mapping[Phase, SystemState]` into credits), enables autograd only under `requires_autograd`, computes loss/energy/accuracy once on the output state (nudged → free → bare-forward fallback), and returns parity-guaranteed `{"loss","energy","accuracy",…float extras}`. `run_forward` + `task_loss` + `phase_states` helper dedupe the remaining copies. All four duplicated loops deleted: `_ComposedSystem._train_step_inner`+`_train_step_spiking`+`_is_spiking_system`, `_JointSystem.train_step` body, `_AdaptedSystem` fallback, `System` protocol default (~300 lines).
2. **Capabilities declared** on every credit (`phases`/`requires_autograd` ClassVars) per the session-5 evidence table. Settle-count negotiation verified per family: thermo/RP/homeostatic/target_inversion = 2, local_goodness/backprop = 1, temporal_trace = 0.
3. **Update-rule consolidation (9.3)**: shared `apply_pseudo_gradients(params, grads, transform)` in ontology.py; Euclidean/Spectral/Natural/EWC/Riemannian all delegate. **Root cause of the 4 crashes: index-based pairing** (`pseudo_grads[i]` ↔ i-th `params.items()`) vs bias interleaving. Euclidean's old shape-skip guard was silently DROPPING mismatched gradients — which had masked three latent credit bugs (below).
4. **Latent credit bugs exposed & fixed** (all previously silent no-ops/garbage under the composed pipeline):
   - RandomProjectionsCredit appended `grad.T` — transposed pseudo-gradients, never applied.
   - LocalGoodnessCredit emitted per-activation `(batch,dim)` "gradients" — never shape-matched any weight. Now emits the layer-local goodness direction `dG/dW_l = δ_{l+1}^T @ pre_l / batch` over its declared FREE phase.
   - BackpropCredit differentiated w.r.t. ALL params incl. biases and returned them bias-interleaved — first weight got the first bias's gradient via broadcast. Now differentiates learnable weights only, contract-ordered.
5. **Substrate tag (9.4)**: `SubstrateType` StrEnum + `substrate_type` field (default digital); factories set it; single selector `substrate_from_config` replaces three ad-hoc maps (two of which held *string* class names that always fell back to Digital). Credit dispatch made explicit match/case with `ValueError` on unknown — a `homeostatic` config previously instantiated **BackpropCredit** silently. Memristive fidelity fixes surfaced by real runs: int8 precision now applies to conductances only (states stay analog floats in `inject_state_noise` and crossbar I/O).
6. **Un-fenced everything**: `_EXCLUDED_AXES` is empty; substrate/credit/update factory sets extended (8 substrates incl. sparse/ternary/optical/quantum; +gradient, +homeostatic credits; +4 non-euclidean updates at step_size=0.01). New pairwise fence discovered & wired: settling dynamics × `tile_mesh` raise with reason (`_check_pairwise` + `_LAYERED_ONLY_DYNAMICS`) — tile-mesh routing exposes no layer sequence for settle to iterate.
7. **Harness merged (9.5)**: `tests/unit/core/test_axis_probe.py` (30 accepted per-axis combos train one real episode w/ metric-key parity; fenced pairs raise; `_EXCLUDED_AXES` entries asserted to raise; substrate class-selection table; analog-noise behavioral fidelity), `test_family_neutral_pipeline.py` (phase-declaration table, settle-count regression, backprop autograd e2e with weights-move/biases-frozen assertion, no-grad default check, canonical-loop equivalence, CPU census + CUDA MB memory-flatness gates), `test_update_rules.py` (biases untouched / weights shaped / clip / momentum / detach).

**Verification:** new harness 73 passed; ontology_locks/axis_certifications/eqprop_locality/gradient_equivalence/test_ontology/checkpoint/stability_guard re-run — remaining failures proven pre-existing at HEAD via `/tmp/bio-head` worktree runs with HEAD's own test code (`test_d_spike_integration_lyapunov`, eqprop_locality scale-free/free-energy tests). pyright 0 errors on pipeline/adapters/evaluation/system_trainer + new tests; ruff clean on all touched files (repo debt files reduced vs HEAD: ontology −61, system_trainer −61, lab −18 findings). Commissioning rerun end-to-end **PASSED** (bit-exact resume losses to 16 digits). Z3 smoke green post-change (`theta_change=0.0`, `theta_invariant=true`). ⚠️ LSP false alarm: `autoscientist_campaigns/commission.py` diagnostics about `INPUT_DIM/_episode_batch/_evaluate_episode` were stale-buffer noise — file imports and passes cleanly.

**⚠️ Incident (recovered):** a mid-session scripting error truncated `ontology.py` to 0 lines (wrote `s[:0] or ""` instead of aborting). Recovered by `git checkout -- computronium/core/ontology.py` + deterministic replay of every edit from this log. Rule: never write a file inside a "guard" branch; assert-and-exit before any open-for-write.

**New work items discovered this session:**
| Item | Detail | Suggested attack |
|---|---|---|
| EqProp seed-42 MNIST parity rerun | Only unmet Phase 9 exit item; metrics keys unchanged so runner should work as-is | Next session queue item 1; then PR-6 budget numbers |
| Live spike recording for STDP | temporal_trace declares `()`; `record_spikes` still has zero callers (was already dead — old path settled twice then read nothing) | RESEARCH3 scope: wire SpikeIntegrationDynamics.settle → credit.record_spikes, then revisit declaration |
| TargetInversionCredit returns `[]` | Declares both phases but produces no gradients (learned inverse maps never implemented) | RESEARCH3 algorithm work; harness keeps it honest |
| `metrics["free_energy_per_iter"]` dropped | List value violated the float-only metrics contract and had zero consumers; dynamics still track history internally | Re-add as a proper top-level field if Control-Lyapunov consumers need it |
| grpc_seam test updated but still hangs | Signature fixed for collection safety; hang is pre-existing | Existing parked item |

### 2026-08-25 session 6 (review → Phase 9 inserted) — **TODO4 REOPENED, IMMEDIATE PRIORITY SET**

**Owner review of session 5 surfaced two directives:**
1. *Don't prematurely condemn possibilities* — the ψ-wiring demotion was re-audited (per-suite `claims_scope` replaced the blanket "instrumentation_shell"; empirical differential test proved L1/L2 wiring differentiates), and a full axis probe of the new composition path found 5 broken combos (gradient credit + 4 non-euclidean updates) plus a silent substrate-mislabeling gap — all fenced with reasons in `_EXCLUDED_AXES` rather than assumed away.
2. *Family-neutrality is now a first-class requirement.* Audit evidence: every composed train_step runs EqProp's free+nudged ritual regardless of family (LocalGoodness ignores nudged; Backprop ignores free; TemporalTrace ignores both); all cross-family interop routes through ThermodynamicContrast adapters (star topology); the 6-D path self-documents as a 5-D port. `SystemTrainer` itself is clean (pure delegation).

**Decision:** inserted **⚡ Phase 9 — Family-Neutral Training Pipeline** at the top of this file with immediate priority; RESEARCH3 campaign-scale sweeps are gated on its exit criteria (instrumentation-scale work may proceed). Capability-declared phase negotiation is the core fix; autograd flag, update-rule fixes, substrate tag, and a permanent neutrality harness ride alongside. Evidence tables embedded in Phase 9 header so execution needs no archaeology.

### 2026-08-25 session 5 (pre-handoff debt clear: campaign CLI real, EqProp drift fix, suite demotion) — **TODO4 CLOSED**

**Executed (in next-session-checklist order):**
1. **Campaign CLI de-mocked** (`computronium/cli/campaign.py::_run_campaign`): mock FrontierRecords + commented-out checkpoint call replaced with the commission.py machinery, which was first **extracted into a shared module** `computronium/core/campaign/evaluation.py` — `build_coordinate_system(coordinate)` composes any 6-D coordinate via ontology config factories + `compose_joint_system_from_configs`; `evaluate_episode(...)` runs one deterministic-batch `train_step`, probes windowed growth, fills `FrontierRecord` stability fields **and now also real `registry_signature`/`composite_state_shape`** (via `compute_registry_signature`/`compute_composite_state_shape`) replacing "mock_signature". Checkpointing follows the PR-9 lessons: snapshot ENTERING an interval episode; resume restores θ into geometry params + RNG states and redoes unrecorded episodes since the latest checkpoint. Unsupported coordinates raise `UnsupportedCoordinateError(axis, value)` → logged + skipped per experiment. Verified: fresh 7-iteration run on CPU (25 episodes/9 iterations in DB incl. real snapshots), checkpoint at ep5, resume clean; commissioning rerun end-to-end **PASSED** with bit-exact redo after the refactor.
2. **EqProp late-drift fix** (`experiments/joint/eqprop_mnist.py`): `_decay_lr` scales `system.update.config.step_size` ×0.9/epoch from ep3 (frozen `ParameterUpdateConfig` handled via `dataclasses.replace`); early stopping at patience 4 on val_acc; result record adds `best_epoch` + `early_stopped`. Rerun (3080, seed 42): **best 81.32 % @ep7, final 79.30 % @ep11 (early-stopped), 337.5 s** vs prior best 81.17 %/final 57.14 % @ep19 — post-peak drift absent under this rerun (single seed; treat as promising, not proven general). `target_met=False` is still the strict final-epoch gate; registered ≥80 % claim met by best.
3. **Suite ψ-wiring audit** (`experiments/joint/`): per-suite inspection + empirical test replaced the session-4 blanket finding. **Correction to the earlier "suites ignore ψ" claim — it was overbroad:** `adaptation_efficiency` and `compute_efficiency` (routing variant) DO step ψ inside forward and modulate computation; a quick differential run showed routing vs fast_weights trajectories are NOT identical (final loss 0.0122 vs 0.0096, real ψ keys populated). The unwired case is specific: `algorithm_migration`'s forward ignores ψ by documented in-code simplification ("here we allow full training"), and `structural_robustness` is a plain-MLP damage test. Remaining caveat for L1/L2 is control, not wiring: θ trains concurrently, `plasticity.step` gets a `None` context, no frozen-θ phase. Each suite result now carries an honest per-suite `claims_scope`: L1/L2 = `psi_wired_uncontrolled`, L3.5/L3 = `plumbing_only` (constants in `_claims.py`; docstrings state these are audit status, not verdicts, and that frozen-θ rewiring via `ThetaInvarianceAudit` would upgrade L1/L2). No suite rerun needed.
4. **Small work items cleared**: `ResourceUsage.measure` device default `"cuda"` → `None` (infers from model params; CPU-only callers measure honestly); stale `profiling.py` call `estimate_spectral_radius(system, x, y)` replaced via shared `activity_transition` — and found to be **live-crashing for non-square geometries** once actually executed (perturbation is in-place) → new `_activity_spectral_radius` helper returns None unless input_dim==output_dim (square case verified ρ≈0.27).
5. **Type-contract fix surfaced by real data**: `FrontierRecord.composite_state_shape` widened to nested `dict[str, dict[str, tuple[int, ...]]]` matching what `compute_composite_state_shape` produces; `to_dict`/`from_dict` made losslessly round-trippable (was `list(v)` key-flattening).
6. **Axis-coverage probe of the new composition path** (prompted by review — "don't assume what you haven't run"): swept every accepted axis value with build + one `train_step` on CPU. **16/21 combos verified green** (all 5 dynamics, all 5 plasticities, 4/5 credits, euclidean). **5 combos fail and are now explicitly excluded** via `_EXCLUDED_AXES` in `evaluation.py` so campaigns reject them with a reason instead of silently misrunning:
   - `credit=gradient/backprop`: needs autograd through settle; composed `train_step` is deliberately detached/no_grad → "element 0 does not require grad". Fix direction: an autograd-enabled train path or a gradient-credit variant over settle-phase differences.
   - All 4 non-euclidean updates crash: shape `[hidden]` vs `[batch, hidden]` inside their `step` on composed params (likely bias-vector handling). Fix direction: normalize param iteration/shape handling in each update rule.
   These are **composition gaps to fix, not judgments on the methods** — the exclusions keep possibilities open while preventing dishonest records.
7. **Substrate-fidelity gap found & fenced**: `compose_joint_system_from_configs` selects the substrate class by `config.precision`, which cannot distinguish analog (`"float32"` = digital's) and silently falls back to `DigitalSubstrate`, dropping behavioral overrides (`AnalogSubstrate.inject_state_noise` etc. — confirmed they differ). Root cause: `SubstrateConfig` has no substrate-type field, so intent is unrecoverable from config alone; note `from_spec` has its own working selector keyed on device+precision. `build_coordinate_system` therefore accepts only `digital` today and rejects the rest with a reason. Fix = add an explicit substrate-type tag to `SubstrateConfig` (ontology change, ripples into to_spec/from_spec) — deferred as its own item below rather than half-fixed here.

**Verification:** pyright 0 errors on all touched files (profiling 9→6, remainder parked 7.5 debt); ruff zero new findings (cli/campaign −16 net vs HEAD; complexity counts reduced via helper extraction); `tests/unit/core/test_checkpoint.py` 9/9; unit/core+validation failures proven pre-existing via `git worktree` A/B at HEAD (identical 5-F set).

**⚠️ Incident + gotcha (read before any A/B testing):** the old foreign stash (`stash@{0}`, tools/-era WIP, ~319 files — documented session 4) makes `git stash` A/B **unsafe**: a scoped `stash push` + `pop` popped the WRONG entry, splattering conflict markers across 330 paths. Recovery used: backup touched files → `git reset` + `git checkout -- .` → restore from backup → delete pop-created untracked junk (`bioplausible/`, `examples/`, `tools/`, stray test files identified by mtime). The foreign stash itself survived untouched. **Rule: until that stash is claimed/dropped, do baseline A/B only via `git worktree add /tmp/x HEAD`.**

**Remaining open items (Phase 9 owns the composition gaps; rest are RESEARCH3 scope):**
| Item | Detail | Where |
|---|---|---|
| **⚡ Family-neutral pipeline** | Phase 9 (above) — phase negotiation, autograd flag, update fixes, substrate tag, neutrality harness | **TODO4 Phase 9 — immediate priority** |
| Guard kill-decisions in runners | Probes now feed FrontierRecords (CLI/commission) but no runner *decides* on τ=1.029 yet | RESEARCH3 CP-A/Z3 smoke |
| fast_proxy disagreement on real systems | Ginibre ≈50 % median is family-specific | RESEARCH3 CP-A |
| Suite ψ-wiring upgrade path | L1/L2 are `psi_wired_uncontrolled` (ψ works, no frozen-θ control); frozen-θ phases + real step context would make them citable. L3.5 forward is unwired by design choice | RESEARCH3 L1–L3.5 full runs |
| PR-3b hardware anchor / PR-8 export parity | Unchanged, external/pull-based | CP-D |

### 2026-08-25 session 4 (PR-7 full-scale + PR-5 calibration + PR-9 commissioning + PR-6 draft) — **TODO4 EXECUTION COMPLETE**

Note: `benchmark_results/` had been deleted by the user before this session; all artifacts were regenerated from scratch (smoke artifacts were never committed, so nothing was lost).

**Executed:**
1. **PR-7 full-scale**: all four suites at configured budgets on the 3080, sequential background job (`setsid nohup` recipe held). Total wall ≈37 s — the suites are toy-scale *by construction* (synthetic tasks, tiny MLPs; each "epoch" = one 64-sample batch). All exit 0; JSONs populated incl. per-seed `resources` (PR-3a wiring verified in situ).
2. **PR-5**: new `core/stability/guard.py` + `scripts/calibrate_stability_guard.py`; 12 unit tests green. Calibration family: non-normal Ginibre linear maps (gain sweep 0.7–1.4 ×4 seeds), labels by unrolled divergence (norm >1e3× over 200 steps).
   - **Headline result**: the one-step `_fast_proxy` statistic CANNOT separate good/unstable on non-normal systems — good max 1.03 vs bad min 0.97 overlap → no feasible threshold exists. Median proxy-vs-full-Jacobian relative error ≈50 % (= σ_max/ρ gap of the Ginibre ensemble), correlation undefined (state-independent Jacobian).
   - Fix vocabulary: added `windowed_growth` statistic (peak whole-tensor norm growth over a settling window; rides transitions the system already executes). It separates cleanly: **τ=1.029 → FKR=0 %, KR=100 %**, gap [0.94, 1.06].
   - Overhead accounting: toy-family ratios (≈9× fast_proxy / ≈19× windowed per probe) are Python-overhead-dominated and not representative; real semantics: windowed costs ~zero marginal (norm checks along existing trajectory), fast_proxy costs 2 extra transition evals/probe. Measure on real settling workloads before quoting numbers.
3. **PR-9**: `autoscientist_campaigns/commission.py` — first REAL campaign cycle (the CLI runner's checkpointing was a commented-out mock; `autoscientist_campaigns/` was empty). Uses `compose_joint_system_from_configs` (square feedforward geometry so activity recirculation is shape-valid), guard probes feed `rho_jacobian`/`basin_stability` in `FrontierRecord`.
   - **All checks green**: checkpoint_valid, theta_fidelity (torch.equal), state_fidelity, rng_canary_match, **bit-exact redo determinism** (redone ep3 loss identical to 16 digits), final_iteration=6.
   - Two correctness lessons baked into the design: (a) checkpoint must snapshot state **entering** an episode — post-episode snapshots replay with the wrong RNG stream position; (b) resumed joint must reload θ from the checkpoint into geometry params before redoing work.
4. **PR-6**: `docs/evaluation_fairness_contract.md` drafted (GPU-hour budgets per rule family, best-val selection w/ both best & final reported, ≥5 seeds via PR-4 kit, fixed split seed 42, ICL ≥95 %/task qualification on measured FLOPs/step, supersede-based deviation policy).

**Code changes to core (zero new lint/type debt, A/B-verified):**
- `JointSystem` protocol gained `context` property; implemented on `_JointSystem` and `_NullJointSystem` (was `_make_context` only) — replaces private access for consumers like the commissioning harness.
- `core/stability/__init__.py` re-exports guard API.

**New work items discovered this session:**
| Item | Detail | Suggested attack |
|---|---|---|
| Shakedown suites don't exercise ψ/θ separation | Toy `PlasticityModel.forward()` ignores `self.plasticity`/`self.psi`; A1 training updates θ freely → `theta_change≈0.72≠0` while the suite header claims "ψ switches strategy without θ update". routing/fast_weights rows are numerically identical (per-call manual_seed + plasticity unused). | Rewire toy models through actual ψ-mediated path (rule_state/route gates in forward, freeze θ for A1 phase, use `ThetaInvarianceAudit`) OR demote suites to plumbing tests explicitly and move ψ-claims entirely to Z3. **→ audited session 5**: per-suite `claims_scope` recorded (L1/L2 `psi_wired_uncontrolled`, L3.5/L3 `plumbing_only`); frozen-θ rewiring stays open (RESEARCH3 L1–L3.5) |
| Guard not wired into runners | PR-5 guard is standalone; suite/campaign runners don't call it yet | Add `StabilityGuard(windowed_growth, τ=1.029)` probe-per-K-steps to suite evaluators + campaign episodes; log decisions into results JSON. **Still open** — RESEARCH3 CP-A/Z3 smoke |
| Campaign CLI mock | `cli/campaign.py::_run_campaign` still evaluates mock records + TODO comments; `create_checkpoint` call commented out | Port commission.py's evaluation+checkpoint loop into the CLI (or make CLI delegate to it) before frontier campaigns. **→ done session 5** |
| `ResourceUsage.measure` defaults cuda | Commissioning had to bypass it; CPU-only runs can't measure honestly | Parameterize device by availability or caller config. **→ done session 5** (default `None`, inferred from model device) |
| Stale stability call site | `core/profiling.py:787` calls `estimate_spectral_radius(joint_system, Tensor, Tensor)` — wrong signature, would crash if reached (7.5 bucket) | Fix during profiling.py pyright burn-down. **→ fixed session 5** (`_activity_spectral_radius`, square geometries only) |
| fast_proxy bias quantification | Ginibre disagreement ≈50 % median is family-specific; no non-normal REAL-system measurement yet | Rerun `quantify_proxy_disagreement` on a real settling coordinate during Z3 smoke. **Still open** — RESEARCH3 CP-A |

**Gotchas for future sessions:**
- Suite "epochs" are single-batch steps on synthetic data — do NOT extrapolate wall times to real-data budgets.
- Windowed-growth statistic uses whole-tensor norm ratio; per-row max variant overlaps classes (verified empirically) — keep the whole-tensor definition when recalibrating.
- Checkpoint placement rule: snapshot ENTERING episode k replays exactly; snapshotting after k's training draws shifts the RNG stream and silently breaks redo equality.
- `git stash` A/B still requires clean-committed HEAD; note there is an old unrelated stash entry (tools/ changes, 319 files) in the repo — leave it alone unless the owner claims it.

### 2026-08-25 session 3 (Phase 7 close + PR-0/4 finish + Z3 & PR-7 smoke) — **PHASE 7 CLOSED**

**Executed (in checklist order):**
1. **7.2 closed**: full 20-epoch run completed in 558.9 s (3080). best_val_acc=81.17 % (~ep7), final=57.14 %. No divergence/OOM/NaN — stability fixes hold through the entire schedule. `target_met=false` is the runner's *final-epoch* gate, not a miss of the registered "reaches ≥80 %" claim.
2. **7.3 closed**: single full-suite `uv run pytest -q` (addopts carry `--cov-fail-under=15`): 1043 passed / 59 F / 18 E / 66 skipped / 11 xfailed / 3 xpassed, **coverage 47.13 %**, 114 s. All 77 FAILED/ERROR lines proven pre-existing: `git stash` A/B vs HEAD produced an **identical failure set** (diff = ∅; HEAD alone can't even collect — see proto fix below). Two modules excluded as known hangs (below).
3. **Z3 smoke green**: PR-1/PR-2 refactor verified end-to-end on CPU (5 meta / 2 eval epochs): `theta_change=0.0`, `theta_invariant=true`, per-task adaptation losses populate, results schema unchanged.
4. **PR-4 finished**: `docs/preregistration_template.md`, `configs/preregistrations/eqprop_mnist_80pct.json`, `tests/unit/validation/test_preregistration.py` (9 tests incl. hypothesis property test). **Latent crash found & fixed**: `paired_comparison` called two-sample `cohens_d(diffs, [0.0])` → always raises (size-1/zero-variance group). Added one-sample `cohens_dz` to `validation/statistics.py`; harness reports dz=0.0 for degenerate identical-arm input instead of crashing.
5. **PR-0 closed**: `docs/baseline.md` refreshed with session numbers (see its new top section).
6. **PR-7 smoke-green**: all four shakedown suites (`--quick --device cuda`, sequential) exit 0 with populated JSON in `benchmark_results/*/`. L3.5 migration table shows per-coordinate task accuracies + Δθ audit columns; L1 shows adaptation half-life; L2/L3 show FLOP ratios/metrics.

**Proto codegen repair (unblocks 4 integration modules):**
- Symptom: collection crash — `TypeError: Couldn't build proto file into descriptor pool: couldn't resolve name '.computronium.p2p.TileActivationRequest'`.
- Root cause: checked-in `tile_mesh_pb2.py` was internally inconsistent — serialized descriptor declared package `bioplausible.p2p` while all cross-references pointed at `.computronium.p2p.*` (stale hand-edit or partial regen).
- Fix: `uv run python -m grpc_tools.protoc -I. --python_out/--grpc_python_out computronium/p2p/proto computronium/p2p/proto/tile_mesh.proto`. ⚠️ protoc mirrors the input path under `--*_out` — outputs land in `proto/computronium/p2p/proto/`; move them up one level and delete that mirror dir. No hardcoded service-name strings exist repo-wide, so importers are unaffected by the package value.
- Also added `*_pb2.py`/`*_pb2_grpc.py` to `[tool.ruff] exclude` (generated code broke the format gate).

**Full-suite A/B methodology (reusable):**
- Only 2 known-hang modules excluded: `tests/property/test_ontology_parity.py` (baseline hang), `tests/integration/test_grpc_seam_subprocess.py` (spawns `_grpc_worker` subprocess that sits at 0 CPU while pytest blocks on unix-socket read forever). Everything else runs, including the newly fixed proto modules.
- Baseline comparison must stash tracked edits AND ignore untracked new test files (an import error anywhere aborts collection for the whole suite).
- Failure taxonomy of the 77 pre-existing: 18 E = all of `test_settle_protocol_models.py` (fixture-level); rest spread over ontology_locks, axis certifications (`test_membrane_boundedness` hypothesis cases), determinism_extended, validation_all learning tests, gradient_equivalence.

**New work items discovered this session:**
| Item | Detail | Suggested attack |
|---|---|---|
| EqProp late-training drift | val acc peaks ~81 % @ep7 then decays to 57 % @ep19 while energy keeps falling (−6e4) — late-phase objective misalignment, not divergence | LR decay or val-based early stopping first (runner already tracks best); weight-norm regularization second. Needed before any paper-grade final-epoch number. **→ FIXED session 5** (LR decay γ=0.9 from ep3 + patience-4 early stop; rerun best 81.32 % @ep7, final 79.30 %) |
| grpc subprocess worker deadlock | `test_grpc_seam_subprocess.py`: spawned worker at 0 CPU, pytest blocked on pipe read; no timeout infra (no pytest-timeout dep) | Debug worker startup (stdin inheritance/port binding); consider adding `pytest-timeout` as suite-level guard. **Still open** |
| `test_ontology_parity.py` hang | pre-existing, noted since session 1 | Investigate before using parity suite as verification tooling. **Still open** |
| Full-suite F/E burn-down | 59 F / 18 E pre-existing; biggest chunk is settle_protocol fixture errors | After dead-code purge (same bucket as 7.5 lint/pyright burn-down). **Still open** |
| Pyright burn-down continues | repo count 3853 → 3837; concentration unchanged (`acceleration/compile.py`, `contrastive_kernels.py`, `eqprop_kernel_backend.py`, `experiments/`) | 7.5 vocabulary applies. **Still open** (repo count 3837) |

**Gotchas for future sessions:**
- Background GPU/CPU jobs from this agent shell need `setsid nohup … < /dev/null & disown` — plain `nohup cmd &` gets killed when the shell tool times out.
- protoc output-path mirroring (above) — always check for the nested mirror dir after regeneration.
- `git stash` A/B against HEAD only works because sessions 1–2 were committed; keep the tree committed before starting baseline comparisons.
- `ruff check <dir>` recurses into parked-debt subdirs (`validation/tracks/*`) — scope checks to touched files when judging *new* findings.

### 2026-08-25 session 2 (7.2.1 + PR-1/2/3a/4 + training launch)

**7.2.1 stability chain (all fixed, in fix order):**
- Convergence self-comparison in checkpointed settle (`torch.dist(a,a)`≡0) → track `prev_output` (`core/ontology.py`).
- `step_size` relaxation wired into `_settle_step`: `h ← h + η·(f(·)−h)` with config `step_size=0.1`. Without it the bidirectional (bottom-up + top-down + recurrent) settle loop has gain >1 and diverges: energy hit −9.5e11 within one epoch, −2.7e23 by epoch 4.
- Per-tensor grad clip **rescaled every gradient to norm exactly 1.0**, destroying magnitude decay near equilibrium → constant-size coherent random walk that degraded a converged solution (80.6% @ep0 → 10% @ep3). Replaced with global-norm clipping (`torch.stack(norms)` → vector_norm). Lesson: per-tensor normalization on pseudo-gradients is pathological; keep relative magnitudes.
- Optimizer momentum 0.9 on noisy contrastive pseudo-grads drifts past the optimum (buf amplifies stale directions ×10). `create_eqprop_system(update_momentum=…)` added; 7.2 uses 0.0 → monotone climb (39.6→77.0% over 5 epochs).
- **Memory leak (blocked everything)**: settle builds an autograd graph per phase-step but *no backward ever runs* — graphs were retained ~4 MB/step → CUDA OOM at epoch 4. Fixes: detach pseudo-grads at the `EuclideanUpdate.step` choke point; wrap `_ComposedSystem.train_step` in `torch.no_grad()` (semantically exact — pseudo-grads are correlation values); out-of-place bias/recurrent adds in both `forward_with_intermediates` impls. Verified flat 16.6 MB / 400 steps. Debug recipe that worked: bisect pipeline stages → `gc.get_objects()` CUDA-tensor census (thousands of live (512,512) graph tensors) → minimal-repro differential (plain-init settle flat vs geometry-init growing).

**New artifact map:**
| Artifact | What |
|---|---|
| `computronium/experiments/joint/eqprop_mnist.py` | 7.2 runner; config derived from `MODEL_CONFIGS["eqprop"]`; NaN guard; JSON results |
| `computronium/core/plasticity/theta_audit.py` | PR-2 `ThetaInvarianceAudit` ctx mgr + `ThetaAuditReport` + `require_frozen`; tests `tests/unit/core/test_theta_audit.py` (green) |
| `computronium/validation/preregistration.py` | PR-4 kit: `MIN_SEEDS=5`, `require_min_seeds`, `ThresholdRegistration` (JSON round-trip), `paired_comparison` (bootstrap CI + permutation p + Cohen's dz), `.passes(reg)` |
| `core/profiling.py::ResourceUsage` | PR-3a canonical merged class (vector core compute/memory/energy/latency/ψ + measurement detail; `__add__`=sum w/ peak-max, `/`, `to_dict`/`from_dict` campaign-keyed, `measure()`); `measure_suite_resources()` helper wired into all four joint evaluators (`"resources"` key per seed result) |
| deleted | `core/campaign/resource_vector.py`; dup class in `core/stability/frontier.py` (importers repointed to profiling; `stability/__init__` still re-exports) |

**PR-1**: `evaluate_z3` now rebuilds Adam post-freeze over trainable-only params (no meta-train momentum survives into ψ-adaptation) and wraps the whole switching/adaptation phase in `ThetaInvarianceAudit`; results schema unchanged (`theta_change`, `theta_invariant`). ⚠️ NOT yet smoke-run at the time — **verified green in session 3** (`theta_change=0.0`, `theta_invariant=true`).

**Gotchas for future sessions:**
- Frozen-check must read **live** params; `p.detach().clone().requires_grad` is always False (bit me in the audit itself).
- `from_dict` accepts legacy `parameter_count` key for pre-consolidation campaign records; old plain-key stability dicts ("memory"/"energy") are NOT read — backwards compat NONE by policy.
- Divide-by-zero in `ResourceUsage.__truediv__` now raises (old frontier class returned zeros); `test_stability_metrics.py` 33/33 green with new semantics.
- `eqprop_vision_parity.py` repaired (dead `CoreTrainer` import removed; routes through ontology factories; only `eqprop` + `backprop_mlp` supported, others logged+skipped; latent `n_permutations=` kwarg bug fixed). Its pandas/numpy pyright findings (~25) pre-date this work — part of 7.5 debt.
- Pre-existing errors left untouched (7.5 scope): `profiling.py` F821 `SystemConfig`:608, pynvml imports; `z3_fixed_weights.py` "Tensor not callable" stub artifacts; eqprop_vision_parity aggregation block.

### 2026-08-25 session 1 (7.1.1 + 7.4 + partial 7.3)
- **Rocq**: `per_index_descent` lemma added; `energy_decreases_diagonal` closed with zero admits (stdlib classical axioms only). The scalar-lemma-first pattern is the reusable recipe for 7.1.2.
- **Dead code deleted**: `SystemConfig.from_experiment`, `SystemTrainer.from_configs` (~170 lines), `Registry.check_compatibility`. If a config-driven trainer factory is needed again, rebuild it against the *current* `ExperimentConfig.system: SystemConfig` field rather than resurrecting these.
- **Settle-path dedup**: non-checkpointed settling now calls `_settle_step` directly; future dynamics changes have exactly one place to edit. Gradient-checkpointing semantics unchanged (`use_reentrant=False` path untouched).
- **Contract fix to remember**: `SystemState.metrics` is `dict[str, float]`; spike stats are now pre-aggregated (`avg_spikes_per_neuron`) by the producer. Do not store tensors/lists in metrics dicts.
- **Type-narrowing vocabulary that worked**: `getattr`+`isinstance`/`callable()` instead of `hasattr` (nn.Module submodule hazard); drop `type[Protocol]` annotations on class-tables (0-positional-arg artifact); module-level instance helpers (`_layer_stack`, `_recurrent_weight`, `_set_param_name`) over duck-typed attribute access.
- **Test hygiene**: baseline A/B via `git stash` proved all current property-test failures pre-exist (hypothesis-generated `test_membrane_boundedness` cases + ontology_locks). `tests/property/test_ontology_parity.py` hangs at baseline too — investigate before relying on it for 7.2 verification.
- **Next cheapest wins**: (1) full-suite pytest run to close the remaining 7.3 checkbox; (2) 7.2.1 gradient-clip/settle stabilization then 20-epoch run; (3) PR-0 gate doc + PR-4 statistics kit (pure code, no hardware dependency).
