# Computronium Sprint Plan: TODO4 — Sprint Close-Out & Research Foundation

> Consolidates all unchecked work from `TODO3.md` with the preliminary infrastructure defined in `RESEARCH3.md`. After Phases 7 + 8, work hands off to the RESEARCH3 catalog (15 items, 5 critical paths) under its Execution Protocol (E-1…E-11). Session Log at the bottom is reverse-chronological.

## Status — Phases 7–9 EXECUTED; session 12 closed queue items 1–3; session 13 executed Z3 order-robustness redesign → CP-A blocker moved from parity redesign to per-phase anneal (registered) — both attempt 1 (400-step budget) and attempt 2 (anneal + 400) triaged, residual stochastic tail remains

| Track | State |
|---|---|
| Phase 7 close-out — 7.1.1, 7.2, 7.3, 7.4 | ✅ done |
| Phase 8 prerequisites — PR-0…PR-7, PR-9 (+ PR-6 draft) | ✅ done (PR-3b procurement-pending, PR-8 pull-based) |
| §7 debt — pyright/lint/F-E triage, CLI de-mock, EqProp drift fix | ✅ cleared session 5 |
| ⚡ Phase 9 family-neutral pipeline (9.1–9.5) | ✅ executed session 7; seed-42 parity rerun ✅ session 8 (bit-level match) |
| Guard kill-decisions in runners (PR-5 → CP-A) | ✅ wired session 8; kill set EMPTY (session 11); **τ=1.029 confirmed lossless on all 16 real settling coordinates (session 12 sweep)** |
| Z3 flagship | 🟡 capability claim CITABLE but ORDER-SCOPED: v2 (canonical order, 5/5 seeds, gates green) stands for its design; v3 randomized-order proportion endpoint NOT confirmed; v4 redesign (per-task Adam rebuild + entropy floor + gate-history instrumentation + budget 400 + anneal) triaged across two confirmatory attempts — residual stochastic discovery tail at parity-last & criterion window truncation; meta-training variance dominates remaining failures. Speed-vs-finetune: honest NULL. |
| RESEARCH3 catalog execution | 🟡 Z3 v2 result citable with explicit design scope; v4 redesign artifacts + mechanistic instrumentation live; CP-A blocker narrowed to within-phase temperature anneal tuning (registered design change #3). |
### Execution queue (next session, in order)

1. **Per-phase temperature anneal tuning (CP-A blocker, registered design change #3)** — attempt 1 (budget 240→400) solved parity-last accuracy but criterion windows truncated (discovery @335/369); attempt 2 (anneal + 400) narrowed but residual stochastic tail remains (seed 9 threshold never discovered in either arm — meta-training variance; two seeds censored by window). Registered: `adapt_temp_end=0.5` linear anneal; need to decide whether to (a) anneal further to sharpen locking, (b) extend budget to 600, or (c) redefine criterion to trailing-window-from-discovery. All changes E-1 pre-registration required.
2. **Mechanistic read of controller drift across phases** — session 12 proved offline data insufficient (no gate histories persisted). Session 13 **wired gate-history instrumentation into `_run_adaptation`** — both confirmatory attempts now carry full per-step operator distributions. Offline replay of `benchmark_results/z3_order_robust/` artifacts available immediately for the anneal decision.
3. **τ recalibration follow-ups (optional):** optical/quantum fast-proxy relative errors are denominator-dominated (~1800–4400×) — `median_absolute_error` + `mean_absolute_error` + `median_reference_norm` fields added to `DisagreementReport` and `scripts/guard_family_sweep.py` (session 13). Artifact regeneration needed for complete sweep record.
4. PR-3b hardware anchor / PR-8 export parity — unchanged, external/pull-based (CP-D).

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
| **Guard kill-decisions (session 8)** | `core/campaign/evaluation.py::DEFAULT_GUARD_TAU`/`GuardKillError`/`evaluate_episode(guard_threshold=…)`; CLI skip paths in `computronium/cli/campaign.py`; kill-set pinned in `test_axis_probe.py::_GUARD_KILLED_SUBSTRATES` |
| **Z3 metrics & controls (session 8)** | `experiments/joint/z3_fixed_weights.py` (`TaskShape`, `_adapt_all_tasks`, `_run_baselines`; baselines a/b/c + steps-to-criterion + soft-eval + collapse flag in results JSON) |
| **Z3 registered metric & prereg (session 9)** | `_windowed_criterion_step` (100-step window) + `_fixed_probe`/`_probe_accuracy` in `z3_fixed_weights.py`; registration `configs/preregistrations/z3_psi_vs_finetune_steps.json`; decision log `DECISIONS.md`; null-run artifacts `benchmark_results/z3_pilot/` (+ `manifest.json`) |
| **Z3 meta-training repair (session 10)** | `z3_fixed_weights.py`: `MetaRecipe`, `step_plasticity`/pure `forward`, episode-structured `_meta_train`, `TASK_OPERATOR_MAP`, two-phase orchestration in `evaluate_z3`; round driver `scripts/z3_meta_repair.py`; artifacts `benchmark_results/z3_meta_repair/round{2,3}.json` + repaired pilot `benchmark_results/z3_pilot_rerun/` (+ manifest) |
| **Z3 differential rounds & capability run (session 11)** | `z3_fixed_weights.py` (`entropy_end` curriculum, `replay_steps` distillation + `_replay_pass`, pre-adapt probe accuracies, all-arm curves); driver rounds R4/R5 in `scripts/z3_meta_repair.py`; window re-analysis `scripts/z3_window_analysis.py`; full-run driver `scripts/z3_full_run.py`; v2 registration `configs/preregistrations/z3_psi_capability_vs_random.json`; artifacts `benchmark_results/z3_full/` (+ manifest), `benchmark_results/z3_r4_probe/round4.json` |
| **Z3 v3 proportion endpoint & randomized order (session 12)** | v3 registration `configs/preregistrations/z3_capability_proportion_vs_random.json`; `evaluate_z3(task_order=…)` + `_apply_task_order` in `z3_fixed_weights.py` (realized order echoed as `results["task_order"]`; both arms share the per-seed order); exact `fisher_exact_p_one_sided` in `validation/statistics.py` (pure stdlib, no scipy); driver `scripts/z3_full_run.py` (per-seed `_seed_task_order`, proportion analysis + descriptive paired gap + `_order_broken_stats`, default out `benchmark_results/z3_proportion/`) |
| **Guard family sweep (session 12)** | `scripts/guard_family_sweep.py` → `benchmark_results/stability_guard_calibration/family_sweep.json` (8 substrates × 2 settling dynamics; windowed growth, proxy-vs-Jacobian disagreement, overhead per coordinate) |
| **Substrate fixes (session 11)** | `core/substrates/ternary_substrate.py::_get_or_create_params` (α from `mean(|w|)·alpha_init`); `core/ontology.py::OpticalSubstrate.get_forward_operator` (quadrature `sin_half`); kill set `_GUARD_KILLED_SUBSTRATES = frozenset()` in `tests/unit/core/test_axis_probe.py` |
| **E-11 decision log** | `DECISIONS.md` (append-only; prereg timestamps, kill/death decisions, deviations) |
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
| PR-5 false-kill rate | Blocks unattended campaigns | τ=1.029 wired into runners session 8; zero false kills on 29 healthy composed coordinates (growth=1.000 exactly); ternary/optical caught as designed | ✅ mitigated (recalibrate on more families at Z3 pilot) |
| Foreign git stash makes `git stash` A/B unsafe | Corrupts working tree (~330 paths splattered once, session 5 incident) | Baseline A/B only via `git worktree add /tmp/x HEAD` until that stash is claimed/dropped | 🔴 live |
| Z3 non-convergence (RESEARCH3 named risk) | Gates CP-A fan-out (ICL bridge, frontier M-axis seed) | Sessions 9–12 walked it: null → repaired → pilot positive → capability gates green at fixed canonical order (v2, 5/5 seeds) → **v3 randomized-order proportion run NOT confirmed: differential does not generalize; all 7 failures across both arms are parity-only and order-governed**. Residual risk is a TASK-DESIGN problem (parity self-revelation), not algorithmic — redesign must be pre-registered before any rerun | 🟡 narrowed to parity task design |

---

## Definition of Done (TODO4 Complete — reopened by ⚡ Phase 9)

- [x] **⚡ Phase 9 family-neutral pipeline** — 9.1–9.5 closed session 7; probe green-or-fenced-with-reason; commission bit-exact ✅ + Z3 θ-invariant ✅ reruns; **eqprop seed-42 MNIST parity rerun ✅ session 8** (best 81.32 % @ep7 / final 79.30 %, bit-level match to record); PR-6 budgets may cite `results/eqprop_mnist_rerun/results.json`
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
- [x] **GATE LIFTED (session 7): RESEARCH3 campaign-scale sweeps UNBLOCKED** — Phase 9 exit criteria met at harness scale (probe green/fenced + parity reruns green + harness merged). Parity item CLOSED session 8: eqprop seed-42 MNIST rerun bit-level matches the record. Anything touching new axis values must keep `_EXCLUDED_AXES`/pairwise fences honest via `test_axis_probe.py`.
- [x] **Z3 pilot rung (session 9): executed honestly to an E-7 null** — prereg committed pre-run (`z3_psi_vs_finetune_steps.json`, unevaluated by design until a promoted full run); registered 100-step-window metric + probe curves + E-3 manifests live; two plumbing defects fixed (ψ integrator, soft/hard mismatch); promotion DENIED with autopsy. Z3 remains the open CP-A blocker; everything else in this file stays closed.
- [x] **Z3 meta-training repair (session 10): E-2 rounds executed → promoted recipe found; pilot rerun POSITIVE vs null** — solver-map correction (threshold→Identity, probe-verified), feedback ψ channel with episode structure, temp anneal + entropy bonus, two-phase forced warm-up; ψ-only criterion on parity+lastsym @~107–130 steps, threshold 0.84 (censored), Δθ exact, diversity 1.42. Scope caveat recorded: random-ψ control ≈ meta-ψ → closing the meta-training differential is the next CP-A gate before the full run.
- [x] **Z3 differential rounds + capability full run (session 11): capability gates 3/3 green on 5 seeds** — Δθ exact, registered criterion reached on ALL THREE tasks in every seed, worst-task acc ≥0.9789. Speed-vs-finetune: honest NULL (descriptive, ±0.17 log-ratio). Differential-vs-random: dz=1.08 but endpoint INCONCLUSIVE at n=5 (bimodal control) — owner decision queued. v1 speed registration retired via E-1 deviation; v2 capability registration committed pre-run.
- [x] **Substrate divergence fixes (session 11): guard kill set EMPTY** — ternary fan-in-scaled α and optical quadrature forward both settle at ρ=1.000; `_GUARD_KILLED_SUBSTRATES` flipped consciously with harness green.
- [x] **Z3 endpoint decision + order sensitivity + τ recalibration (session 12): queue items 1–3 closed** — option (b) executed as v3 proportion registration (Fisher exact, 10 seeds/arm, randomized per-seed order folded into the design, committed pre-run); outcome E-7 NOT CONFIRMED with decisive autopsy (all failures parity-only, order-governed; v2 claim scoped to its design); `fisher_exact_p_one_sided` added to the PR-4 kit; guard family sweep confirms τ lossless on 16/16 real settling coordinates.

### Exit criterion status
PR-7 green ✅ + PR-5 calibrated ✅ + PR-9 commissioned ✅ ⇒ RESEARCH3 catalog unblocked end-to-end at instrumentation scale (CP-A proceeds to Z3 flagship; CP-B/C/D/E per spines). PR-3b measured-tier claims remain gated on hardware arrival. **CP-A update (session 12): Z3's citable capability claim is ORDER-SCOPED (v2 design); the open blocker is parity task redesign with a fresh E-1 registration — statistical instruments are no longer the bottleneck.**

---

## Session Log & Future-Work Notes

### 2026-08-26 session 12 (queue items 1–3: v3 proportion endpoint + randomized order; τ family sweep) — **E-7 NULL WITH DECISIVE AUTOPSY: ALL FAILURES ARE PARITY-ONLY AND ORDER-GOVERNED**

**Executed (E-1/E-11 order held — registration + DECISIONS entry committed before any data):**

1. **Endpoint decision executed as option (b), hardened:** new registration `configs/preregistrations/z3_capability_proportion_vs_random.json` supersedes v2's endpoint. Event = arm fails a seed iff worst-task final accuracy < 0.95; primary = exact one-sided **Fisher** test on failure counts across **10 seeds/arm**; α=0.05; rejection region given 0 treatment failures = ≥4/10 control failures (power ≈95% at observed rates, ≈62% if the true rate were 0.4 — registered up front). Rationale: v2 showed the control is Bernoulli-mixture, so a mean-difference test against margin 0.25 tested the wrong null family; more seeds alone can't fix a mean pinned AT the margin.
2. **Task-order sensitivity folded into the same design** (carried session-10 item, zero extra compute): `evaluate_z3(task_order=…)` reorders the whole switching stream via `_apply_task_order` (adaptation, baselines, diversity all follow; both arms share the per-seed order; realized order echoed in results). Driver derives orders from `random.Random(seed)` and reports `_order_broken_stats`. All 10 seeds are fresh draws under this design — the v2 fixed-order run is not reused for the new endpoint (no double-dipping).
3. **`fisher_exact_p_one_sided` added to `validation/statistics.py`** (exact rational hypergeometric tail via `math.comb`/`Fraction`; no scipy dependency) + unit tests (known values: p(0 vs 6/10)=1001/184756; rejection boundary at 4/10; input validation).
4. **Confirmatory run** (`scripts/z3_full_run.py`, seeds {0..9}, GPU ~41 s/seed, artifacts `benchmark_results/z3_proportion/` + manifest w/ registration sha256):
   - **NOT CONFIRMED.** Fisher p=0.5 — z3 fails 3/10 seeds {1,2,3}, random fails 4/10 {4,6,7,8}. Gates correctly fail on the same seeds. Descriptive paired gap collapses +0.258→**+0.046** [−0.203,+0.295], dz 1.08→**0.11**: the v2 differential does NOT survive order randomization.
   - **The load-bearing structural finding:** every seed-level task failure in the run — **7/7 across both arms — is on PARITY alone**; last_symbol and threshold are solved by BOTH arms under EVERY order (60/60 arm-task solves). Coverage structure:
     - parity FIRST ⇒ both arms solve everything (seeds 0,5,9);
     - z3 fails iff order = (lastsym → threshold → parity) — deterministically 3/3 seeds (parity ≈0.48–0.51 after two preceding controller-training phases);
     - random control fails parity in 4/6 non-parity-first cells (threshold-first prefixes plus last→parity→threshold) and SOLVES parity exactly where z3 fails ((l,t,p)×3).
   - Pre-adapt routing priors: parity ≈0.49–0.51 everywhere (nothing installed); threshold 0.61–0.97 yet never fails — prior erosion is recovered by adaptation for threshold but parity has no prior to erode; its bandit lock-in is fragile to whatever routing basin earlier phases leave in the shared controller.
   - Speed-vs-finetune null unchanged (descriptive log ratios at windows {20,50,100} persisted per seed).
5. **τ recalibration sweep (queue item 3, confirmatory as predicted):** new driver `scripts/guard_family_sweep.py` → `benchmark_results/stability_guard_calibration/family_sweep.json`; 16 coordinates = 8 substrates × {energy_minimization, predictive_settling} at rule_state/thermo/euclidean. **windowed_growth = 1.0000 exactly on all 16 ⇒ τ=1.029 lossless (FKR 0%)** on real families across the full substrate grid. fast_proxy disagreement per family: memristive 0.00, analog/sparse/ternary/digital 0.40–0.62, neuromorphic 0.86–1.13 median rel-err, Pearson ≈0 — PR-5's "fast_proxy cannot gate non-normal settling" confirmed on real systems; optical/quantum relative errors are denominator-dominated (~1800–4400×) and need an absolute-error field before quoting.

**Verification:** ruff histograms byte-equal-or-better vs HEAD worktree A/B on all touched files (`z3_fixed_weights.py`, `statistics.py`, driver, sweep script, tests) + format clean; pyright **0 errors** on touched files (warnings pre-existing untyped-dict patterns); targeted suites green — preregistration 11 passed (incl. 2 new Fisher tests), z3_criterion_window + stability_guard 19 passed, integration Z3 smoke 1 passed (canonical no-order path exercised at runtime); CPU determinism smoke of `task_order` (same seed+order ⇒ identical results; invalid order raises; θ-invariant). Full pytest suite NOT rerun (touched surface: one experiment module param, statistics addition, two scripts, tests — per instruction to minimize redundant executions).

**New work items discovered this session:**
| Item | Detail | Suggested attack |
|---|---|---|
| Parity task redesign (**CP-A blocker**) | All Z3 order-robustness failures are parity-only; self-revealing operator makes its coverage fragile to controller drift from preceding phases | Make the label require a TRAINED decoding head (kills self-revelation), or swap the third task, or scope all claims to canonical order explicitly. E-1 register BEFORE any run |
| Controller-drift mechanistic read | Why does the bandit fail to lock parity only under specific prefixes? Curves already saved per arm/seed | Offline replay of `z3_proportion_results.json` curves + operator histograms during the parity phase (z3-fail cells vs random-solve cells at (l,t,p)); no rerun needed |
| Order-broken criterion censoring | Seed 5/7 pass the 0.95 floor but miss windowed criterion on some task — floor and criterion disagree at the margin | Report both consistently in future registrations (already dual-recorded here); consider registering on the stricter of the two |
| Optical/quantum proxy disagreement metric | Relative error explodes when the full-Jacobian reference ≈ 0 | Add absolute-error field / floored reference norm in `quantify_proxy_disagreement` reporting |

### 2026-08-26 session 11 (queue items 1–3: differential rounds R4/R5 → capability full run; substrate divergence fixes) — **CP-A CAPABILITY CLOSED; SPEED NULL RECORDED; KILL SET EMPTY**

**Executed (E-2 → E-1/E-11 order held):**

1. **Differential machinery** (`z3_fixed_weights.py`): `MetaRecipe` gained `entropy_end` (curriculum: linear β anneal high→low within the controller phase — attack b) and `replay_steps` (attack c: per-epoch supervised distillation of episode-best operators from a FIFO trajectory buffer — `(ψ snapshot, batch-mean input) → argmin-mean-loss op`, plain CE, no Gumbel noise). `_meta_train` decomposed into `_forced_episode`/`_controller_episode`/`_episode_best_op`/`_replay_pass`. Diagnostics added to every arm: per-task **pre-adaptation probe accuracy** (routing quality at ψ=0 before any step), and accuracy **curves persisted for all three arms** (driver `_seed_curves`). Recipe echo extended (`entropy_end/replay_steps/adapt_temp`); CLI flags added.
2. **R4 + probe autopsy (attacks a/b falsified as-is):** with `adapt_temp` unset, adaptation inherits end-of-anneal T≈0.5 ("cold"): cold PRESERVES priors (threshold reached criterion where the hot pilot had censored) but starves discovery (parity died at chance). Pre-adaptation accuracies at meta-300/wu40/curriculum = 0.49/0.60/0.64 — attack (a)'s premise is structurally impossible: at ψ=0 every task presents identical inputs, so ONE shared default routing can match AT MOST one task's solver.
3. **Parity trilemma (load-bearing structural finding):** the parity operator emits the label itself as a feature (verified: forced-op parity scores 1.000 even with an UNTRAINED trunk). Any broad sampler — including a fresh random controller — therefore sits at the steps-to-criterion metric floor (~window size) on parity. Worst-task SPEED margins vs the random control are unwinnable at ANY window size; worst-task margins vs fine-tune are dominated by window-floor ties on parity/lastsym.
4. **R5 (replay × temperature, ≤8-config discipline):** replay distillation fixed meta's parity anti-learning (replay_hot parity 1.000); temperature swept {0.75/1.25/2.0}: cold/mid solve lastsym+threshold but drop parity; hot solves all three. **Promoted `wu60_hot`** (= R3 winner + adapt_temp=2.0 at meta-300): 1.000/0.996/0.992, criterion ALL tasks both seeds, simplest recipe (replay unnecessary).
5. **E-1 re-registration BEFORE the full run** (`configs/preregistrations/z3_psi_capability_vs_random.json`; DECISIONS.md entry first): v1 speed endpoint retired UNEVALUATED with a three-part instrument-redesign rationale (window floors adaptation-time; parity floor asymmetry makes worst-task-vs-random unwinnable; v1 aggregation text self-contradictory). New primary endpoint: per-seed worst-task final hard-selection accuracy, z3 − random control, margin +0.25, PR-4 paired harness, ≥5 seeds; gates = Δθ exact + all-task criterion coverage + ≥0.95 floor.
6. **Confirmatory full run** (`scripts/z3_full_run.py`, seeds {0..4}, GPU ~95 s/seed, artifacts `benchmark_results/z3_full/` + manifest w/ registration sha256):
   - **Gates 3/3 green on every seed** — Δθ exact; criterion on ALL tasks in all 5 seeds; worst-task acc ≥0.9789. The Z3 capability claim is demonstrated at registered scale.
   - **Primary endpoint INCONCLUSIVE (recorded, not fished):** mean gap 0.2577 > margin but CI [0.076, 0.439] straddles it (p=0.13, dz=1.08). Autopsy: the random control is BIMODAL — solved everything in 2/5 seeds (~0.99), failed lastsym in 3/5 (0.52–0.60). Pilot n=2 couldn't see the ~40% luck rate; expected gap ≈0.26 sits AT the margin. Owner options recorded in DECISIONS.md (descriptive acceptance / proportion-endpoint re-registration / more seeds).
   - **Speed vs baseline-(a): NULL confirmed descriptively** — log step ratios within ±0.17 at windows {20,50,100} across 5 seeds.
7. **Substrate divergence fixes (queue item 3, both root-caused & verified):**
   - *Ternary*: α initialized from latent-weight magnitude (`mean(|w|)·alpha_init`, fan-in-scaled) instead of fixed 1.0 — unit-magnitude quantized weights gave settling gain ρ≈2×10⁸; now ρ=1.000 exactly with healthy loss/accuracy.
   - *Optical*: forward used the MZI bar term `cos(φ/2)` which maps w=0 to FULL-strength coupling (cos 0=1) → settle overflow to inf; switched to quadrature term `sin(φ/2)` (vanishes at zero phase). ρ: inf → 1.000.
   - `_GUARD_KILLED_SUBSTRATES` flipped to EMPTY in `test_axis_probe.py` with rationale comment; axis-probe suite re-run green (79 passed).

**Verification:** targeted suites green — test_axis_probe 79 passed / stability_guard 33 / family-neutral pipeline / update_rules / z3_criterion_window 7/7; benchmark integration failures proven pre-existing via `/tmp/bio-head` worktree A/B (identical 3 CLI-subprocess fails at HEAD). ruff histograms byte-equal vs HEAD on touched core files (ontology 295=295); pyright 0 new errors (ternary's 3 FunctionCtx artifacts pre-exist at HEAD). Full pytest suite NOT rerun (touched surface: one experiment module, two scripts, ternary α-init, 2-line ontology operator swap, harness kill-set flip — per instruction to minimize redundant executions).

**New work items discovered this session:**
| Item | Detail | Suggested attack |
|---|---|---|
| Z3 differential endpoint decision | Random control bimodal (~40% solves-all); mean-diff instrument mis-specified; gates all green so capability claim citable NOW | Owner picks (a) descriptive / (b) Fisher-exact re-registration / (c) more seeds (insufficient alone). **→ EXECUTED session 12**: option (b) as v3 proportion registration + randomized order; outcome E-7 null with parity/order autopsy |
| Optical stochastic crosstalk | `photonic_forward` draws crosstalk noise per call without a seeded generator — transitions are nondeterministic across identical replays | Route through the substrate's precision/noise config or a torch.Generator; verify bit-reproducibility of episodes |
| Parity task design ceiling | Self-revealing operator makes parity free for any sampler; caps Z3's discriminative power for routing studies | Design change: make the parity feature require trained decoding (register first!) or drop parity from differential-style claims. **→ UPGRADED TO CP-A BLOCKER session 12** (all order-randomized failures are parity-only) |
| Replay attack under-tuned | Single setting (replay_steps=4) tested; helped parity but wasn't needed by promoted recipe | If endpoint redesign revives speed claims, sweep replay around curriculum+hot |

### 2026-08-26 session 10 (queue item 1: Z3 meta-training repair E-2 rounds 1–3 + queue item 2: pilot rerun) — **CP-A REPAIRED; PILOT POSITIVE**

**Executed (E-2 → E-1 order held):**
1. **Repair implemented in `z3_fixed_weights.py`** (attacks a–c from the session-9 autopsy):
   - *Feedback ψ channel:* `forward` is now PURE — plastic state moves only via explicit `step_plasticity(loss)`: ψ ← tanh(decay·ψ + scale·proj([mean-gates ; loss])), fixed random projection (`feedback_proj`), decay=0.9 / scale=0.15. Purity also fixes two latent defects at once: probe/eval passes can no longer corrupt adaptation dynamics, and the batch-shaped-ψ wart (`_probe_accuracy` had to chunk to the training batch) is gone — ψ lives as canonical `[1, hidden]`, expanded at read. The old `psi_operator_logits` buffer was dead post-fix-1 and is deleted.
   - *Episode structure:* `_meta_train` runs per-task episodes (episode_len consecutive batches, ψ reset at boundaries) instead of interleaving one batch per task per epoch — without this ψ has no within-task segment to summarize.
   - *Attack b:* linear temp anneal temp_start→temp_end across controller episodes + gate-entropy bonus −β·H(g).
   - *Attack c:* two-phase recipe — forced-operator θ warm-up (TASK_OPERATOR_MAP, warmup_lr=3e-3) → θ-frozen controller-only ST phase with fresh Adam between phases.
   - Recipe knobs consolidated into frozen dataclass `MetaRecipe` (episode_len/feedback/entropy_beta/temp_start/temp_end/warmup_fraction/warmup_lr/adapt_temp); CLI flags added; results schema additive (`meta_recipe` echo). `with_baselines=False` skips E-10 arms for triage rounds.
2. **Load-bearing bug found by round-1 nulls: TASK_OPERATOR_MAP was WRONG.** threshold→Threshold-op cannot work: the label is sum(values)>0 while that op's features keep only signs (linear-probe solvability: sign features ≈ chance; Identity features ≈0.99). Corrected map: parity→4, last_symbol→3, **threshold→Identity(0)**. Round-1's all-chance results (kept in `benchmark_results/z3_meta_repair/` … note: R1 JSON crashed on Path serialization after compute — logs only; fixed for later rounds) were the tell; a single-task forced-warm-up control isolated it.
3. **Second mid-repair fix: ψ saturation.** First feedback design used raw O(1)-norm projections; tanh railed within a few steps → ψ froze as constant context → flat-at-chance adaptation curves. Decay+scale keeps the running summary responsive.
4. **E-2 rounds** (driver `scripts/z3_meta_repair.py`, gate = seed-mean ≥0.7 on ALL tasks): R2 (post-map-fix): `full_b02`/`full_longep` pass. R3 (compose winners, meta-100): **promoted `b02_longep_wu60`** = entropy_beta 0.2, episode_len 16, warmup_fraction 0.6 → 1.000/0.988/0.808. Stopped at 3 rounds per plan.
5. **Pilot rerun** (seeds {0,1}, adapt 240/task, GPU): flat failing-task curves diagnosed as exploration failure (solver op never sampled) → added `adapt_temp` (gating temperature during adaptation), set 2.0. Final artifact `benchmark_results/z3_pilot_rerun/`: **Δθ exact both seeds; diversity H=1.42 (null: ≤0.003); criterion reached on parity @107–112 and last_symbol @107–130; threshold 0.83–0.85, censored at budget**. Registered endpoint remains UNEVALUATED (both ψ and baseline-(a) censored on threshold).

**Honest scope caveat:** random_psi control ≈ meta-trained controller (≈1.0/0.99/0.82) — the repair's mechanism is feedback-driven bandit exploration over the warmed-up trunk, not meta-learned routing. Frozen floor shows meta-training DOES install a correct threshold prior (~0.99 fresh-ψ) which sequential adaptation then erodes (final 0.84 < floor). Closing the differential = next-session queue item 1. Full autopsy + deviations (epoch-semantics change justifying meta-100; adapt_temp addition) in DECISIONS.md.

**Verification:** z3-relevant suites green (`test_z3_criterion_window` 7/7, integration `test_z3_fixed_weights_runs`); pyright 0 errors on touched files (also cleared the old register_parameter typing artifact); ruff histogram vs HEAD net-improved (C901/ERA001×2/F841/SIM102 gone; operator-output shaping extracted to `_operator_feature`); CPU smoke of `evaluate_z3` green incl. θ-invariance. Full suite NOT rerun (single experiment module + new driver; per instruction to minimize redundant executions).

**New work items discovered this session:**
| Item | Detail | Suggested attack |
|---|---|---|
| Meta-vs-random differential ≈ 0 | Random-ψ adapts as fast as meta-trained controller; Z3 headline ("meta-training buys speed") undemonstrated even though absolute performance is strong | Queue item 1: longer controller phase / entropy curriculum / replay-style policy sharpening; verify pre-adaptation routing converges |
| Threshold criterion censoring | Threshold plateaus ~0.84 at 240-step budget (identity found late; floor erosion from earlier task phases) | Either longer per-task budget (re-register first) or per-phase Adam reset to stop cross-task drift; check whether routing prior survives when task order varies |
| Task-order sensitivity unquantified | Adaptation always runs parity→lastsym→threshold; shared-controller drift makes order matter (floor erosion evidence) | Randomize task order per seed in next run; report order-broken stats. **→ DONE session 12** (folded into v3 design; answer: order is THE governing variable — see session 12 log) |
| Round-1 artifact loss | R1 JSON write crashed (Path not serializable) AFTER compute — only stdout logs survive | Driver fixed; keep artifacts for any rerun |

### 2026-08-26 session 9 (queue item 1: Z3 pilot rung — prereg committed first; E-7 null with autopsy) — **CP-A RUNG FAILED, LOOPED BACK**

**Executed (E-1/E-11 order held):**
1. **Pre-registration committed BEFORE the pilot ran** (stricter than E-1's post-promotion slot, per queue directive): `configs/preregistrations/z3_psi_vs_finetune_steps.json` — primary endpoint = worst-task mean log step ratio log(steps_finetune/steps_z3), threshold log(1.25)≈0.2231, α=0.05, ≥5 seeds, paired via the PR-4 kit; Δθ-exact and all-tasks-≥95 % as gate conditions; censoring policy (never-reached → scored at budget) registered explicitly. `DECISIONS.md` created as the E-11 append-only log.
2. **Registered steps-to-criterion implemented** (`z3_fixed_weights.py`): `_windowed_criterion_step` = first step whose trailing **100-step window** mean probe accuracy ≥0.98 (replaces session-8 batch-window proxy). Per-task fixed probe sets (16 batches × 64, generated once before adaptation → deterministic + disjoint from the fresh training stream); per-step `accuracy_curve` recorded for every arm (Fig. 1 deliverable data); baseline-(a) now also reports per-stage `steps_to_criterion`; suite writes an **E-3 manifest.json** (config sha256 + git commit + UTC timestamp) next to results; CLI gained `--seq-len/--input-dim/--probe-batches`.
3. **Pilot run** (2 seeds, meta 50 / adapt 240 steps/task, GPU): ~3 min, θ-invariance exact both seeds. **Outcome: every arm at chance (~0.50) on every task; diversity entropy ≤0.003; criterion never met.** Even baseline-(a) θ-fine-tuning sat at chance on its own training task ⇒ plumbing-class failure, not a science result yet.

**Root-cause chain (all empirically isolated, CPU diagnostics in DECISIONS.md):**
| # | Defect | Evidence | Fix |
|---|---|---|---|
| 1 | ψ-logit integrator: `new_logits = psi_logits + update` is an unbounded random walk | ‖ψ‖ 1→157 in 60 steps, H 2.08→0.036, loss pinned at ln 8 — softmax saturation kills all gradients everywhere | Removed; forward now matches RESEARCH3's gating equation `g_k=softmax(controller(ψ_t,x_t))` |
| 2 | Soft-mixture steering: controller classifies by weighting the mixture, a solution that vanishes under eval argmax | train loss → ln 2 while hard-eval stays chance (pre- and post-fix-1) | **Straight-through Gumbel**: hard selection forward (= eval semantics), soft gradients |
| 3 | Task-identity acquisition: joint from-scratch meta-training collapses onto an arbitrary operator before the decoder means anything | ST + fix 1 still → ln 2 / chance; forced-operator controls are HEALTHY (parity 100 %, last-symbol ≥95 %, threshold ~87 %) so features+decoder+loss path are fine; parity label is invisible in x stats and scalar ψ carries too little history | **OPEN** — next-session queue item 1 (ranked attacks a–d) |

**Verification:** new window-metric unit tests 7/7 (hypothesis oracle-equality property + boundary table); ruff findings on z3 file byte-identical to HEAD (checked via `/tmp/bio-head` worktree diff of rule histograms); pyright errors unchanged at the 9 pre-existing "Tensor not callable" artifacts; smoke (--quick, cuda) green end-to-end incl. manifest write. Full pytest suite NOT rerun (touched surface = one experiment module + one new test file; per instruction to minimize redundant executions).

**New work items discovered this session:**
| Item | Detail | Suggested attack |
|---|---|---|
| Z3 meta-training repair | Queue item 1 above; null memo material for M-axis boundary publication if attacks stall | Loss-feedback ψ channel > temp-anneal+entropy-bonus > two-phase curriculum > task redesign. **→ DONE session 10** (all three composed + solver-map correction; pilot rerun positive) |
| `Z3Model.forward` persists batch-shaped ψ state | Any batch-size change mid-stream crashes (`expand` mismatch); worked around by chunking probe evals to the training batch | Store ψ as `[1, …]` canonical (mean or first-row), expand at read; flip consciously + rerun smoke. **→ FIXED session 10** (pure forward; ψ = `[1, hidden]`; probes chunk-free) |
| Parity task observability | All 3 tasks share identical randn inputs; parity (order-n counting mod 2) has no first-moment signature → controller cannot condition selection on input statistics alone | Either give tasks distinguishable input distributions (design change — register first!) or make task ID enter via ψ adaptation history only, and prove it suffices. **→ RESOLVED session 10**: ψ adaptation-history feedback suffices — identity acquisition works from selection-consequence memory alone |

### 2026-08-25 session 8 (queue 1–2: EqProp parity ✅, guard kill-decisions wired, Z3 smoke rung) — **CP-A ADVANCED**

**1. EqProp seed-42 MNIST parity rerun — PASSED, bit-level match.**
`--quick` smoke first (E-1 hygiene, 4.1 s, caught nothing — pipeline healthy), then the full schedule on the 3080: **best 81.32 % @ep7, final 79.30 % @ep11 (early-stopped), 341.2 s** — identical to the session-5 record to 4 decimals under the canonical Phase-9 loop. `results/eqprop_mnist_rerun/results.json` (session-5 baseline preserved at `results/eqprop_mnist/`). PR-6 budget recalibration can now cite this artifact.

**2. Guard kill-decisions wired into runners (open since PR-5).**
- `core/campaign/evaluation.py`: `DEFAULT_GUARD_TAU = 1.029` (PR-5 ROC point) + `GuardKillError(coordinate, statistic, threshold)`; `evaluate_episode(..., guard_threshold=None | τ)` now *decides*: decision recorded in `FrontierRecord.metadata["guard_kill"]`, and a kill raises AFTER logging so runners skip the coordinate exactly like an unsupported one. `None` = record-only mode for capability harnesses.
- CLI campaign (`run` + `_redo_unrecorded_episodes`) catches `GuardKillError` → log+skip; commissioning rerun end-to-end **PASSED** (kill=False logged per episode; bit-exact resume intact).
- **Empirical calibration check on real systems:** all 29 healthy composed coordinates read windowed growth = exactly 1.000 (settling is contractive) → τ=1.029 has zero false-kill margin there. Two genuine catches: **ternary diverges (growth ≈ 4×10⁵)** and **optical overflows to inf** during settle windows — both pass build+one-train-step (capability probe), only the guard sees it. This is precisely the unattended-campaign failure class PR-5 was built for.
- Randomized sweep over `joint_full` space (150 coords) exposed three cross-axis crash kinds the per-axis probe cannot reach; two real bugs fixed:
  1. `PredictiveSettlingDynamics.compute_energy` tested ``not layer_acts`` BEFORE its own isinstance-list check → ambiguous-boolean crash whenever activations hold a bare output Tensor (temporal_trace settles 0 phases). One-line operator reorder.
  2. `NeuromorphicSubstrate` float16 leaked into host-facing state I/O and the float32 output projection (`Half != Float` crashes across tile_mesh/diffusion and feedforward/predictive_settling). Fixed to the memristive contract from session 7: device-native precision stays internal (`_to_precision` inside the forward op), boundary activities return float32.
  - Post-fix sweep: **100 ok / 24 guard-killed / 26 fenced / 0 crashes.**
- Harness (`test_axis_probe.py`): capability probe moved to `guard_threshold=None` (stability gating is orthogonal to capability); new `test_guard_kill_status_matches_known_unstable_set` pins {ternary, optical} as the kill set at default τ (fixing either flips the set consciously); `_CROSS_AXIS_REGRESSIONS` parametrized test replays the crashing coordinates; dtype-contract test for neuromorphic I/O.

**3. Z3 flagship E-1 smoke rung complete — metrics & controls now exist.**
`experiments/joint/z3_fixed_weights.py` extended from accuracy/θ-audit/diversity-only to the catalog's full metric set:
- Per task: `steps_to_criterion` (batch-window proxy for the registered 100-step definition — upgrade at pilot), `soft_eval_accuracy` (control d: hard-vs-mixture discretization gap), adaptation losses retained.
- Flags/timing: `diversity_collapsed` (H < log 2), `wall_clock_s.psi_adaptation`.
- E-10 control set implemented: (a) `finetune_forgetting` — sequential θ fine-tune at identical step budget producing the stage×task accuracy matrix + per-task forgetting tax; (b) `random_psi` — controller re-initialized post-meta-training, isolating what meta-training bought; (c) `frozen_floor` — trunk-only, no ψ adaptation.
- Refactor: `TaskShape` frozen dataclass dedupes the batch/seq/dim/device quadruple across helpers; `_meta_train`/`_adapt_all_tasks`/`_run_baselines` extracted (evaluate_z3 shrank; results schema purely additive so CLI aggregation is untouched).
- Smoke verified on GPU (5 meta / 3 eval epochs): every key populates, `theta_change=0.0`/`theta_invariant=true` preserved, baselines behave sensibly (floor≈chance, forgetting matrix well-formed).

**Verification:** axis-probe + family-neutral pipeline suites green (100 passed); stability-guard suite green; commissioning PASSED; ruff clean on all touched files (remaining findings in cli/campaign.py + z3 file are pre-existing parked debt); pyright unchanged on z3 file (pre-existing artifacts documented session 2).

**New work items discovered this session:**
| Item | Detail | Suggested attack |
|---|---|---|
| Ternary/optical settle divergence | ternary α_init=1.0 → unit-magnitude quantized weights → settling gain≫1; optical overflow inf. Honest guard-kills today, but they shrink the sweepable space by 2 substrates | Fan-in-scaled α (Xavier-style) for ternary; root-cause optical overflow. Flip `_GUARD_KILLED_SUBSTRATES` consciously after. **→ DONE session 11** (α from `mean(|w|)·alpha_init`; optical quadrature `sin(φ/2)` — cos mapped w=0 to full coupling; ρ=1.000 both; kill set empty) |
| Z3 pilot prereg | E-1/E-11: commit thresholds before pilot-promoted full config; steps-to-criterion proxy must be replaced by registered definition for citable numbers | Use PR-4 template + `configs/preregistrations/` |
| τ recalibration on real families | Ginibre-calibrated τ=1.029 happens to be lossless on composed systems (growth=1.000 exactly), but one family ≠ calibration set | Fold `quantify_proxy_disagreement` on real settling coordinates into Z3 pilot (already an open CP-A item) |

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

---

### 2026-08-26 session 13 (TODO4 queue items 1–3): Z3 order-robustness redesign → per-task Adam rebuild + entropy floor + gate-history instrumentation + budget 400 + per-phase anneal; two confirmatory attempts triaged; CP-A blocker narrowed to anneal tuning

**Executed (E-1/E-11 discipline — registration + DECISIONS before every data collection):**

1. **Offline mechanistic read (queue item 2, corrected):** wrote `scripts/z3_drift_analysis.py` mining `benchmark_results/z3_proportion/`. **Found:** v2/v3 artifacts persist final accuracies/pre-adapt priors/speed windows only — NO raw curves or gate histograms (TODO4's claim was optimistic; R4/R5 repair rounds have curves but at canonical order). Gate-history instrumentation now wired into `_run_adaptation` for all future runs. Findings: pre-adapt prior carries ZERO parity outcome signal (0.496 solved vs 0.492 failed); R4 cold config shows flat-at-chance exploration-failure signature; R5 hot solves parity (1.0). Artifact: `benchmark_results/z3_drift_analysis/findings.json`.

2. **Parity redesign E-2 triage matrix (queue item 1, revised scope):** tested three design families on stress cells (seed 0 order p,t,l; seed 1 order l,t,p):
   - v3 raw emission: (p,t,l) all solve; (l,t,p) parity FAILS 0.48 (deterministic 3/3 v3).
   - coded quadrature + entropy floor β=0.1: (p,t,l) l/t FAIL; (l,t,p) all solve.
   - coded antipodal + floor: same pattern — coding moves failure to post-parity phases.
   - raw + floor (no coding): (l,t,p) parity FAILS 0.48 (floor inert).
   - **raw + floor + per-task Adam rebuild**: (l,t,p) all solve (parity 1.0 @229; l/t ≥0.993 @100/102). **Mechanism:** stale Adam second moments carried across task boundaries starved later phases — PR-1 hygiene extended to every phase boundary fixed the v3-killer order.
   - Coded-emission variants REVOKED pre-data: they deepened the exclusive-op4 basin, making post-parity phases starve WORSE; scale-invariance of linear decoder ruled out margin fixes analytically. See `DECISIONS.md` for the three-way triage record.

3. **Registered design v4 (amended pre-data):**
   - per-task Adam rebuild in `_adapt_all_tasks` + fine-tune baseline (identical protocol both arms)
   - adaptation entropy floor β=0.1 (`adapt_entropy_beta`)
   - gate-history rider (mean gates / hard-op histogram / entropy per step, all arms)
   - budget 240→400 (first amendment, after discovery-latency census: max observed 239 + 100-step window > 240)
   - per-phase temperature anneal `adapt_temp_end=0.5` (second amendment, after attempt 1 censoring)
   - registration: `configs/preregistrations/z3_capability_order_robust.json`

4. **Confirmatory attempt 1 (10 seeds/arm, 400 steps): NOT CONFIRMED.** Accuracy floor passed 9/10 (seed 9 z3=0.82); seeds 2,3 solved parity @335/369 but criterion censored; random arm 10/10. Mechanism: per-task rebuild shifted failure from deterministic (l,t,p) to stochastic tail — discovery latencies span 1–369 steps, two races lost within 400, two windows truncated by budget.

5. **Confirmatory attempt 2 (10 seeds/arm, 400 steps, anneal): NOT CONFIRMED.** Accuracy floor failed 1/10 (seed 9 threshold 0.82 in BOTH arms — meta-training variance); seeds 2,3 criterion censored (discovery 335/369); anneal reduced mid-phase entropy but late-lock tail persists. Mechanism: constant β=0.1 entropy floor + flat temp 2.0 means "explore forever, sharpen never" — anneal to 0.5 narrows but doesn't eliminate the tail; None-discovery cases (seed 9 threshold in both arms) unaffected.

6. **Guard τ follow-up (queue item 3):** added `mean_absolute_error`, `median_absolute_error`, `median_reference_norm` to `DisagreementReport` (`core/stability/guard.py`) and `scripts/guard_family_sweep.py`. Family sweep regeneration needed for complete artifact.

**Verification:** ruff format clean; ruff errors 26 in `z3_fixed_weights.py` (pre-existing class, better than HEAD 29); 0 in all other touched files; pyright 0 errors on touched files; targeted tests 36 passed; integration Z3 smoke passed.

**Artifacts produced:**
- `scripts/z3_drift_analysis.py` + `benchmark_results/z3_drift_analysis/findings.json`
- `configs/preregistrations/z3_capability_order_robust.json` (twice amended)
- `benchmark_results/z3_order_robust_attempt1/` (first confirmatory, 240→400)
- `benchmark_results/z3_order_robust/` (second confirmatory, anneal + 400)
- `DECISIONS.md` entries for both amendments
- `tests/unit/validation/test_z3_redesign.py` (12 tests: entropy floor, gate history, evaluate_z3 persistence)

**New work items discovered:**
| Item | Detail | Suggested attack |
|---|---|---|
| Anneal tuning / budget extension (CP-A blocker) | Residual stochastic discovery tail at parity-last (latency P99 ~369, one None-discovery case); criterion window truncation on late discoveries | Decide: (a) anneal further (e.g., 2.0→0.25, or cosine), (b) extend budget to 600, (c) redefine criterion to trailing-window-from-discovery. Register BEFORE run. Gate-history artifacts from both attempts available for offline census. |
| Guard family sweep regeneration | Absolute-error fields added to `DisagreementReport`; `family_sweep.json` lacks them | Re-run `scripts/guard_family_sweep.py` → `benchmark_results/stability_guard_calibration/family_sweep.json` (confirmatory, ~2 min GPU). |
| Meta-training variance diagnosis | Seed 9 failed threshold in BOTH arms — controller quality varies per seed; 10/10 control arm suggests protocol is fine, meta-training is the variance source | Add meta-training quality gate (pre-adapt accuracy threshold) to registration; or increase seed count for statistical averaging. |

---

### 2026-08-25 session 1 (7.1.1 + 7.4 + partial 7.3)
- **Rocq**: `per_index_descent` lemma added; `energy_decreases_diagonal` closed with zero admits (stdlib classical axioms only). The scalar-lemma-first pattern is the reusable recipe for 7.1.2.
- **Dead code deleted**: `SystemConfig.from_experiment`, `SystemTrainer.from_configs` (~170 lines), `Registry.check_compatibility`. If a config-driven trainer factory is needed again, rebuild it against the *current* `ExperimentConfig.system: SystemConfig` field rather than resurrecting these.
- **Settle-path dedup**: non-checkpointed settling now calls `_settle_step` directly; future dynamics changes have exactly one place to edit. Gradient-checkpointing semantics unchanged (`use_reentrant=False` path untouched).
- **Contract fix to remember**: `SystemState.metrics` is `dict[str, float]`; spike stats are now pre-aggregated (`avg_spikes_per_neuron`) by the producer. Do not store tensors/lists in metrics dicts.
- **Type-narrowing vocabulary that worked**: `getattr`+`isinstance`/`callable()` instead of `hasattr` (nn.Module submodule hazard); drop `type[Protocol]` annotations on class-tables (0-positional-arg artifact); module-level instance helpers (`_layer_stack`, `_recurrent_weight`, `_set_param_name`) over duck-typed attribute access.
- **Test hygiene**: baseline A/B via `git stash` proved all current property-test failures pre-exist (hypothesis-generated `test_membrane_boundedness` cases + ontology_locks). `tests/property/test_ontology_parity.py` hangs at baseline too — investigate before relying on it for 7.2 verification.
- **Next cheapest wins**: (1) full-suite pytest run to close the remaining 7.3 checkbox; (2) 7.2.1 gradient-clip/settle stabilization then 20-epoch run; (3) PR-0 gate doc + PR-4 statistics kit (pure code, no hardware dependency).
