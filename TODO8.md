# TODO8.md — Consolidated Plan

> **Rev 2026-08-31 (b).** P0–P5 session logs consolidated away (full history in `git log`).
> Research catalog lives in [RESEARCH3.md](RESEARCH3.md); this doc owns the engineering that unblocks it.
>
> **State:** P0–P5 **complete incl. pyright policy** · R1 **complete incl. construction seeding**
> (device threading, placement guard, runner auto-device, EqProp CUDA epoch ≈ 5.6 s) · U-bypass
> sweep complete (see audit below) · **R5.1a CPU smoke campaign commissioned** (mid-flight
> SIGKILL → resume → complete, artifacts in `autoscientist_campaigns/smoke_cpu/`) ·
> gate `pytest -q` green (1221 passed / 66 skipped / 25 xfailed / 4 xpassed, ~74 s) ·
> placement guard `tests/property/test_native_device_placement.py` 31 green on CUDA.
>
> **Policy:** zero backwards compatibility · GPU-first for all training paths · no new tests for broken
> capability (xfail with precise reasons) · serial pytest only (xdist hangs in this env) ·
> **the System's own ParameterUpdate owns Δθ — external torch optimizers must not drive composed systems**
> (custom-loss harnesses route through `core.pipeline.apply_autograd_update`).

## ✅ Completed Record (2026-08-30/31)

| Phase | Outcome |
|-------|---------|
| P0 | Registry lazy auto-population (28 native + aliases), KB constructor, PARAM_UPDATE registrations, module boundary — 1455 passed / 0 failed |
| P1 | `_TaskTrainer` rewrite, EWC consolidate, ModelAdapter real BPTT/metrics, geometry flattening — every triage failure fixed, none left as xfail |
| P2 | Native smoke 28/28 (pass/xfail w/ reasons), settle protocol 24+ pass, validation skips → xfails, 4 xpassed resolved, property locks green |
| P3 | Quarantine emptied (2 deleted, 6 enabled), free-energy tracking implemented, gRPC worker API drift fixed, 5 timeout victims fixed, parity flake root-caused — 1499 passed / 0 failed |
| P4 | `SubstrateSettleKernel` ported; `EnergyMinimizationDynamics.settle` substrate-native (Digital bitwise-equal to legacy); 10-test equivalence suite |
| P5 | Campaign schema freeze (migrations + `SchemaVersionError`), replication gate, counterfactual attribution, `CampaignStack` facade, 8 rankable objectives, migration smoke in CI, CLI validated end-to-end |
| P5 close | `pyrightconfig.json` (basic repo-wide, elevated-standard on `computronium/ontology`: 0 errors) + pre-commit hook `uv run pyright computronium/ontology`. Fixed 11 latent NameErrors in `ontology/system.py` (unimported substrate classes / `ComponentMetadata`), `GradientCredit` protocol conformance, `_settle_kernel` optional-subscript. Dead code deleted: `ontology/utils/state.py`, `ontology/dynamics/primitives.py`, 9 dead state helpers in `_dynamics.py`, duplicate `_layer_stack`/`_recurrent_weight`/`_learnable_weight_names`/`_set_param_name` copies, `_AdaptedSystem` dead `apply_pseudo_gradients` |
| R1 | `device` explicit on all native factories (unknown kwargs now raise), `compose_system`/`compose_joint_system`/`build_coordinate_system` take `device`; `_ComposedSystem.to()` **was a silent no-op** (dict reassignment never moved module params) — fixed via `nn.Module.to`; `device` property on both system shapes; runner auto-device (`AutoScientist._execute_proposal`, `CampaignStack` incl. checkpoints, `evaluate_episode` batch placement, `evaluate_migration` + joint suites via `get_device`); ψ init (`RoutingPlasticity`/`FastWeightPlasticity`) device-aware; placement guard over all 28 factories; EqProp MNIST epoch ≈ 5.6 s on CUDA |
| U-sweep | External-optimizer audit of every `torch.optim.*`/`create_optimizer` site: 4 ontology violations fixed (tradeoff ×2, hardware [18a], application [21c]), 2 dead strays deleted (core_tracks), 8 fake `use_spectral_norm`/`max_steps` kwargs removed (silently ignored → SN "ablations" compared identical models); new primitive `core.pipeline.apply_autograd_update` for custom-loss harnesses |
| R1.4 | Suite-wide construction seeding for the parity classes (`construction_seed()` helper in `test_ontology_parity.py`; applied to the 4 presets-vs-native pairs + threshold-bearing credit composition). Parity file green with seeded inits: 30 passed / 1 skipped / 2 xfailed — all parity thresholds hold |
| R5.1a | CPU smoke campaign commissioned via `scripts/commission_smoke_campaign.py`: mid-flight SIGKILL (process group) at 1 durable episode of iteration 1 → CLI `--resume` → completed through iteration 6, 13 episodes. Artifacts in `autoscientist_campaigns/smoke_cpu/records/`: `manifest.json` (git commit, torch/CUDA, seed, budget, kill/resume timeline), `report.md` (episodes, Pareto frontier, counterfactual attribution, replication gate), `run_first.txt`/`run_resume.txt`, YAML checkpoint copy. DB + `checkpoints/` gitignored by design |

Old P6 checklist items already satisfied: EqProp anchor 81.32% MNIST · ComputroniumLinear (26 tests) · torch.jit → torch.export migration in `deployment.py`.

## 🔁 What Changed in This Revision

1. **GPU-first is the critical path, not an optimization.** CPU settle paths stall campaigns (P5 model sweep stalled; GPU directive 2026-08-31). R1 precedes everything.
2. **Campaign commissioning promoted to headline deliverable.** The P5 stack is built but `autoscientist_campaigns/` is empty — zero completed runs (RESEARCH3 PR-9). Demonstrability = one commissioned run.
3. **Five scattered "improvement opportunities" sections folded into R2–R4** — nothing lost; resolved items dropped.
4. **Old P6 (Research Phases 4/5/6) absorbed by RESEARCH3's critical paths.** This doc ends where RESEARCH3 begins; R6 is the handoff.
5. **Sequencing:** R1 → R5.1a (CPU smoke, **not** GPU-gated) → R5.1b/c; R2/R3/R4 interleave but never block the first commissioned campaign; R6 last.
6. **Bitwise determinism deprioritized:** discovery/replay locks use tolerance + environment-locked manifests on GPU; bitwise equality is an opt-in extra (CPU reference or explicit deterministic mode), never a requirement.
7. **U-axis ownership enforced (2026-08-31):** codebase-wide optimizer sweep — external torch optimizers may no longer drive composed Systems; audit table below documents the fixed and the legitimately-external sites.

---

## 🔍 U-Axis Bypass Audit (2026-08-31, complete)

**Rule:** a composed System (`compose_system`/`compose_joint_system`/native factory) is only ever
updated through its own ParameterUpdate axis. Custom-loss harnesses call
`core.pipeline.apply_autograd_update(system)` (autograd grads → pseudo-grads → `update.step` →
`geometry.update_params`); plain-`nn.Module` baselines are out of scope.

**Fixed this session (violations — external optimizer drove a composed System):**

| Site | Was | Now |
|------|-----|-----|
| `validation/tracks/tradeoff_tracks.py` [57a/57b] | `optim.Adam` + `loss.backward()` loop on native EqProp/Backprop systems; "EqProp" arm never settled (D/C axes dead) | `train_and_measure` drives `model.train_step` (full 5-axis pipeline) |
| `validation/tracks/hardware_tracks.py` [18a] | `create_optimizer(sgd)` + `optimizer.step()` on native EqProp | `apply_autograd_update(model)` |
| `validation/tracks/application_tracks.py` [21c] EWC | `create_optimizer(adam)` + manual EWC penalty + `optimizer.step()` on native EqProp | EWC penalty loss → autograd → `apply_autograd_update(model)` (canonical route remains an `ElasticConsolidationUpdate` coordinate — future switch) |
| `validation/tracks/core_tracks.py` `_train_model`/`_train_model_sn` | `create_optimizer` created, never stepped (dead stray) | deleted |

Also removed: 8 silently-ignored `use_spectral_norm`/`max_steps` factory kwargs (see R3.9 note —
those "SN ablation" tracks compared identical models; the comparison signal was fake).

**Audited legitimate (no action; listed so future sweeps don't re-litigate):**

| Site | Why it's not a violation |
|------|--------------------------|
| `experiments/joint/*` (`PlasticityModel`, `PlasticityModulatedModel` + `torch.optim.Adam`) | benchmark harness models, not composed Systems; no U-axis exists to bypass. Optimizer-phase hygiene (rebuild Adam between meta-train and ψ-adaptation) = RESEARCH3 PR-1 |
| `ontology/system.py` `_AdaptedSystem.train_step` fallback SGD | Strangler-Fig seam: legacy `nn.Module` models can't be driven by the ontology update; System `EuclideanUpdate` supplies `step_size` |
| `core/trainer.py` `dispatch_train_step` | BPTT fallback requires an optimizer only when the model has no learning rule; System models take the `train_step` path |
| `core/local_learning/**` (`BioOptimizer` etc.) | legacy model layer; the optimizer *is* the learning-rule implementation |
| `zoo/**` | deprecated (R2.1), scheduled for deletion — do not fix |
| `training/rl.py`, `sklearn_interface.py`, `graph/training.py`, `domains/trainer.py`, `benchmarks/rigorous.py`, `benchmarks/algorithm_migration.py`, `deployment/quantization.py`, `lightning_/module.py`, `core/ebm.py`, `core/dynamics/adapters.py`, `core/nebc.py`, `scripts/z3_reverification_audit.py` | plain `nn.Module` baselines / infra / distillation targets; no composed System in the loop |

---

## 🎯 R1 — GPU-First Runners + Close P5 (do first) — ✅ COMPLETE

*Verified: `create_native_*` factories had no `device` param (`**kwargs` silently ignored → tensors always construct on CPU even under CUDA trainers); `autoscientist/campaign.py:717`, `core/system_trainer/factory.py` + `joint.py` hardcoded `device="cpu"`. Additionally discovered: `_ComposedSystem.to()` moved params into a throwaway dict view (no-op) — fixed.*

| # | Task | Status |
|---|------|--------|
| 1.1 | `device` through native factories | ✅ explicit `device: str \| torch.device = "cpu"` on all factories via `compose_system(device=…)`; `**kwargs` removed → unknown kwargs raise `TypeError` |
| 1.2 | Auto-device in runners | ✅ `get_device()` (single resolver, `core/utils/device.py`) in `AutoScientist._execute_proposal`, `CampaignStack(device="auto")` + checkpoint restore, `evaluate_episode` (batches follow the joint's parameter device), `evaluate_migration` (default `"auto"`), joint suites |
| 1.3 | CUDA placement guard | ✅ `tests/property/test_native_device_placement.py` — all 28 factories + buffers + substrate-metadata agreement + kwargs-rejection + CPU default |
| 1.4 | Suite-wide construction seeding | ✅ `construction_seed()` helper applied to parity classes (`test_ontology_parity.py`): 4 presets-vs-native pairs seed identically per construction (the P3 CUDA flake pattern), credit composition seeded before compose. Crash-freedom-only tests (`acc >= 0.0`) left unseeded — no threshold to flake |
| 1.5 | Close P5: pyright policy | ✅ `pyrightconfig.json`: basic repo-wide, elevated-standard on `computronium/ontology` (0 errors); pre-commit hook gated. Note: pyright's `strict` array cannot be downgraded per-rule, so full `strict` on ontology (131 findings, mostly torch `Unknown` tracking) is deferred — see improvement opportunities |

**Done when:** ✅ every factory accepts explicit `device` and rejects unknown kwargs · ✅ placement guard green over all 28 factories · ✅ EqProp single-epoch MNIST on CUDA ≈ 5.6 s (256-batch, 20 settle steps) · ✅ runners default to CUDA when available · ✅ pyright policy enforced in pre-commit · ✅ parity classes construction-seeded (R1.4)

## 🧹 R2 — Retirement & Signal Honesty (stability; interleavable, never blocks R5)

| # | Task | Detail |
|---|------|--------|
| 2.1 | Zoo retirement | Audit first (grep zoo for `@register`/`Registry.register`/presets/PARAM_UPDATE entries), extract still-live registrations (MEP presets, MEP PARAM_UPDATE) into first-class ontology modules, full suite → delete `computronium/zoo/**` incl. `tile_models.py`/`tile_fa.py`/`tile_lm.py` → full suite again. User directive: zoo deprecated for the ontology API — don't fix zoo components |
| 2.2 | Dead/duplicate sweep | **Partially done (2026-08-31):** `ontology/dynamics/primitives.py` deleted, `ontology/utils/state.py` deleted, dead state helpers + duplicate helper copies removed. Remaining: `Substrate` naming (`ontology/_substrate.py` impl vs `ontology/substrate/` facade — facade is fine, consider merge); grep for other parallel legacy/new pairs |
| 2.3 | Registry API unification | `Registry.list()` vs `list_models()` alias asymmetry (module-boundary test pins the raw view); alias `get_metadata` projects from canonical |
| 2.4 | xpass noise fix | Native smoke tile tests are xfail-but-xpass (smoke checks crash-freedom only). Split: crash-free smoke (strict pass) + learning-capability test (true xfail). **Must precede R5b discovery locks** |
| 2.5 | Skip census (one pass, then done) | 89 skips → fixed categories: missing optional dep / CUDA unavailable / DEFERRED geometry → `skip(reason=…)` · known broken capability → `xfail(reason=…)` · env flake → `flaky` + ticket · dead legacy → delete. After the census the skip count becomes meaningful instead of suspicious |
| 2.6 | Small items | `uv add kademlia` (DHT actually exercised in slow tier); fold `test_grpc_seam_multi_process` skip into the working subprocess pattern; watch axis_probe `[2-0]` flake for recurrence |

## 🧩 R3 — Capability Completeness (smallest blocker first)

| # | Blocker | Fix shape |
|---|---------|-----------|
| 3.1 | `DiffusionDynamics` autograd bug — energy tensor loses grad history during settle → `torch.autograd.grad` fails | Smallest known native blocker; un-xfails 2 integration files + smoke entry |
| 3.2 | FA + `InstantaneousDynamics`: no error signal (free=nudged) | Derive proper error signal or restrict FA to compatible dynamics |
| 3.3 | PEPITA `LocalGoodnessCredit`: empty pseudo-gradients | Debug credit path |
| 3.4 | Tile × dynamics matrix (tile_ep/pc/gnn/snn device-dynamics incompatibility; tile_fa/tp/hebbian) | Fix or document as permanent xfail with precise reasons |
| 3.5 | `_AdaptedSystem._infer_geometry` hardcoded (784→256,128→10) | Recover heuristics from deleted `adapter/` package |
| 3.6 | `_TaskTrainer` gaps | Scheduler wiring, energy tracking, honor `tracker`/`safety_config` (wire when hyperopt trials need them) |
| 3.7 | Substrate fidelity nits | Neuromorphic: real spike dropout or drop the cosmetic `sparsity` field; Memristive: restore conductance-range semantics (pairs with RESEARCH3 substrate work) |
| 3.8 | Stretch | `natural_language_query` TF-IDF weighting; derive `V_nudged = free energy + β·loss` to strengthen the predictive-coding Lyapunov xfail |
| 3.9 | Coordinate validity matrix | Classify the above: implementation bug (fix) vs conceptual incompatibility (declare invalid). Enumerate supported/unsupported Tile×Dynamics×Credit combos; raise `IncompatibleCoordinateError` for conceptually invalid coordinates; keep xfail only for combos that should work but currently fail. **Some coordinates are invalid, not broken** |

## ⚙️ R4 — Substrate API Breadth (P4 continuation; opportunistic — pulled by campaign needs)

| # | Task |
|---|------|
| 4.1 | FA feedback projection through the Substrate operator API (validates non-settle paths) |
| 4.2 | Register `SubstrateSettleKernel` in `KernelRegistry` for the EQPROP family (currently `EqPropKernelBackend` covers all targets) |
| 4.3 | MEP Triton kernels (Muon, Fisher whitening) → Substrate update operator |
| 4.4 | Sparse substrate transpose-mask handling; ternary `init_scale` param (un-xfail ternary equivalence); optional per-step `inject_state_noise` during settle for analog/memristive substrates |

## 🎬 R5 — Demonstrable Product Surface (RESEARCH3 PR-9)

| # | Task | Detail |
|---|------|--------|
| 5.1a | ✅ CPU smoke campaign | **Commissioned 2026-08-31** — `autoscientist_campaigns/smoke_cpu/`: 2–5 coords/iter from `joint_smoke`, seed 0, synthetic task (8-dim, batch 16). Lifecycle: start → checkpoint (ENTERING-episode, interval 1) → **mid-flight SIGKILL at 1 durable episode** (iteration 0 in DB) → CLI `--resume` → completed through iteration 6 (13 episodes; interrupted iteration 1 re-ran its deterministic coordinate stream → documented overlap). Manifest + report + logs + YAML checkpoint in `records/`. Regenerate: `uv run scripts/commission_smoke_campaign.py --fresh` |
| 5.1b | GPU quick campaign | 5–20 coordinates, 2–3 seeds on CUDA: no silent CPU fallback (placement guard), visible speedup, metrics/resources recorded |
| 5.1c | **Commissioned campaign** | 30–100 coordinates, ≥5 seeds for winners, ≥2 task families (replication gate), frontier + counterfactual + golden manifest persisted into `autoscientist_campaigns/`. The first real artifact — nothing consumes the stack until this exists |
| 5.2 | One-command demo on GPU | `comp campaign run` → Pareto + replication + counterfactual report, end-to-end on CUDA, documented. The commissioning script already renders that report on CPU; R5.1b reuses it (`--device cuda`), R5.2 reduces to documenting the one-liner |
| 5.3 | Demo/quickstart truth-check | `scripts/quickstart.py`, `demo/main.py`, README factory examples verified against current API (the autoscientist package was unimportable until P5 — demos may be stale); add slow-tier smoke |
| 5.4 | CLI/docs polish | `comp` prog-name prints "biopl" (cosmetic); align README's "0 errors in strict mode" claim with the actual pyright policy |
| 5.5 | Export pin | torch.export (PT2) round-trip test for FeedforwardGeometry + RecurrentGeometry (migration done; pin it) |

### R5b — Discovery Demo Package (the "prove it" milestone)

*Demonstrate the ontology flywheel making a discovery — end to end on Digital (CPU/GPU) substrate, every claim locked in by tests. No hardware needed: the D×C×U×M axes carry the discovery space on CPU/GPU (momentum vs plain EqProp settle convergence, spectral/Riemannian U-axis stabilizers, FastWeight/Routing adaptation vs Null, ternary/sparse substrate trade-offs). Smoke-scale L1 runs are feasible on CPU today; GPU (R1) makes sweeps credible.*

| # | Item | Detail |
|---|------|--------|
| A | Pre-register one toy-scale hypothesis | From RESEARCH3 L1/L2/substrate-ablation catalog (e.g. "FastWeight cuts post-switch re-adaptation ≥30% vs Null at matched compute"); thresholds committed before any full run |
| B | Locked grid campaign on GPU | `CampaignStack.run_campaign` over ~30–100 coordinates, matched budgets, ≥5 seeds; replication gate (≥2 task families) must pass |
| C | Evidence chain | Pareto frontier over 𝒞 (compute/memory/energy/latency/plasticity) + counterfactual attribution table naming the axis that owns each knee |
| D | **Discovery locks (tests)** | ① winner-must-replicate: pinned-seed test asserting the discovered gap within tolerance — failing test = capability regression; ② attribution lock: `analysis/counterfactual.py` ranks the discovered axis first, stable across seeds; ③ replay lock: same `(seed, campaign_id, iteration)` re-derives the discovery **within tight tolerance on GPU** — bitwise replay is deprioritized (opt-in: CPU reference or explicit deterministic mode); manifest records torch/CUDA/GPU versions and deterministic flags. ⚠️ Prereqs: R2.4 (xpass split) and improvement-11 θ-init determinism — episode replay currently re-draws init on resume |
| E | Golden manifest (mandatory from R5.1c on) | Every commissioned campaign writes: git commit, config hash, dependency lock, torch/CUDA versions, device, deterministic mode, seed list, task family, budget, replication summary → `results/<item>/<seed>/manifest.json` + checked-in figure-regeneration script (RESEARCH3 E-3) |
| F | Discovery Report UI | Stage 1: static HTML/JSON report from the stack (frontier, replication, attribution, timeline) — snapshot-tested. Stage 2: live Campaign tab in `demo/main.py` (719-line NiceGUI app, currently **zero** campaign/autoscientist wiring) |

**Ordering:** R2.4 (xpass smoke split) precedes writing discovery locks — ambiguous test semantics must not be encoded into the scientific artifact.

**Honesty ladder:** rung 1 = flywheel completes with evidence chain (any outcome demoable — nulls are results); rung 2 = validation discovery (rediscovers known non-trivial structure, attributed); rung 3 = novel discovery (RESEARCH3 Algorithm Discovery item, kill-criterion governed — never promised on a date).

## 🔬 R6 — Handoff to RESEARCH3 (old P6 absorbed)

Old P6 phases map onto RESEARCH3's critical paths — execute there, not here:

| Old P6 item | RESEARCH3 home |
|-------------|----------------|
| Phase 4: Regime Discovery (Bandit Router, Memristive IR-Drop sweep, Photonic Epistemology Swap) | AutoScientist campaign types (substrate ablation, epistemology swap) |
| Phase 5: Family-Coverage Benchmark (coordinate lock ≥30, Resource-Vector Runner, Dynamical Phylogeny) | De Facto Non-Backprop Benchmark + PR-3a |
| Phase 6: Frontier Certification (M-Axis Frontier, Goldilocks Map, Manifesto Dataset) | CP-A tail: frontier campaign + failure manifesto |

**First moves after R1** (RESEARCH3 startup sequence): PR-1 optimizer-phase hygiene (rebuild Adam between meta-train and ψ-adaptation — momentum carry-over contaminates the Δθ=0 claim) → PR-2 θ-invariance audit harness → PR-3a resource instrumentation → PR-7 shakedown (L3.5 → L1 → L2/L3). `docs/baseline.md` (PR-0) and `docs/evaluation_fairness_contract.md` (PR-6) already exist.

---

## 🗓 Execution Order (7 sessions)

1. ~~**Close P5** — `pyrightconfig.json` + pre-commit hook~~ ✅ done 2026-08-31
2. ~~**R1 device threading** — factories, kwargs rejection, placement guard, runner auto-device~~ ✅ done 2026-08-31
3. ~~**R1 validation** — construction seeding~~ ✅ 1.4 done 2026-08-31 (parity classes seeded; parity file 30 pass) · ✅ EqProp MNIST epoch 5.6 s on CUDA · ✅ zero silent CPU fallback (placement guard)
4. ~~**R5.1a CPU smoke campaign**~~ ✅ commissioned 2026-08-31 — start → checkpoint → **mid-flight kill** → resume → complete → artifacts (`autoscientist_campaigns/smoke_cpu/`)
5. **R5.1b GPU quick campaign** — placement + speedup + replication/frontier output (unblocked: CampaignStack runs end-to-end on CUDA; verified 2026-08-31). Command in `autoscientist_campaigns/README.md` — **next**
6. **R2 signal honesty** — xpass split, skip census; prepare zoo extraction list
7. **R5.1c commissioned campaign** — lock config, run, replicate, persist to `autoscientist_campaigns/`, report

Then R5b and RESEARCH3 become real. Everything else (zoo deletion, kernel breadth, capability xfails) is important but must not block the first commissioned campaign.

## ⚠️ Risks

| Risk | Mitigation |
|------|------------|
| GPU-first exposes numerical flakiness (convergence, tolerances, xfail boundaries shift on CUDA) | CPU-equivalence and GPU-tolerance tests kept separate; construction seeding; device recorded in every test/report |
| Stack mechanically complete but scientifically empty | R5.1 must produce artifacts; thresholds pre-registered (R5b-A); replication gate non-bypassable |
| Discovery locks brittle — too tight flakes, too loose prove nothing | Pre-registered effect thresholds; multi-seed robust deltas; tolerance + env lock on GPU, never bitwise |
| Zoo deletion breaks hidden registrations | Audit → extract live registrations → full suite → delete → full suite |
| R3 becomes capability creep | Track as xfail/invalid-coordinate unless the commissioned grid needs that coordinate |

## 💡 New Improvement Opportunities (2026-08-31 session)

1. **`_ComposedSystem.to()` was a silent no-op** — it reassigned entries in the `geometry.params` dict view instead of moving the underlying `nn.Module` parameters. Anyone "training on GPU" via `system.to("cuda")` was on CPU. Fixed (delegates to `nn.Module.to`); pinned by the placement guard. *Lesson: parameter-dict views hide mutation bugs — prefer module-level placement.*
2. **`use_spectral_norm` was a fake knob** — accepted-and-ignored by factories, so SN "ablations" ([56b], [1a/1b], negative_results) compared identical models and reported fabricated contrasts. kwargs are now rejected; making those ablations real means composing `SpectralConstrainedUpdate` U-axis coordinates (pairs with R3.9 / MEP Kinetics campaigns).
3. **ψ-device coupling** — `initial_psi` for Routing/FastWeight created CPU tensors regardless of θ device; now derives device from `SystemContext.device`. RuleState already device-aware. When adding plasticity primitives, always key ψ off `context.device`.
4. **Pyright policy floor** — `strict` on ontology surfaces 131 findings (mostly torch `Unknown` tracking + private-import usage); pyright forbids downgrading rules inside `strict` paths, so the policy uses elevated-standard there (0 errors). Raising to full `strict` = annotation work in `_dynamics`/`geometry`/`update`; repo-wide basic is ~2.5k findings (pre-existing, never gated).
5. **Dead helper consolidation** — `_layer_stack`/`_learnable_weight_names`/`_set_param_name` each had 2–3 live copies across `geometry.py`/`utils/`/`system.py`/`_substrate.py`; consolidated (canonical: `geometry.py` for `_layer_stack`/`_set_param_name`, `utils/params.py` for `_learnable_weight_names`). `_layer_stack` renamed public (`layer_stack`) — it is cross-module API.
6. **R2.4 xpass split still open** — 4 xpassed in native smoke (tile tests crash-free but xfail-marked); must precede R5b discovery locks.
7. ~~**R1.4 construction seeding**~~ ✅ done — parity classes seeded via `construction_seed()` helper; see Completed Record.
8. **`compute_energy` duplication** — Energy/Spike/Instantaneous/Diffusion dynamics each carry near-identical `compute_energy` bodies over duck-typed states; extract to one `_energy_from_state(state, geometry)` helper next touch.
9. **SIGKILL to `uv` orphans the worker python (2026-08-31)** — killing the `uv` wrapper leaves the `comp` child alive, which kept executing the campaign to completion in the background. Campaign runners must `killpg` the session (`scripts/commission_smoke_campaign.py::_kill_tree`, `start_new_session=True`). Same hazard applies to spot-instance teardown and the gRPC worker launcher.
10. **Resume can duplicate episodes (2026-08-31)** — `add_episode` has no uniqueness on (campaign_id, iteration, coordinate); a crash mid-iteration + resume re-runs the whole iteration from the deterministic coordinate stream, so already-recorded coordinates land twice (observed: iteration 1 with 3 episodes). Harmless for lifecycle, but it skews counterfactual `n_pairs` means and would contaminate R5b-D winner-replication locks. Fix shape: per-(campaign, iteration, coordinate) uniqueness or skip-recorded-coordinates when re-running an interrupted iteration.
11. **θ init draws ride the ambient RNG across resume (2026-08-31)** — repeated coordinates after resume get different metrics (observed: same coordinate 2.1026 pre-kill vs 2.1247 post-resume). Coordinate + batch streams are deterministic per (seed, campaign_id, iteration); parameter construction is not. R5b-D replay locks therefore need either per-episode construction seeding derived from (seed, campaign_id, episode) inside `run_campaign`, or redo executed from the checkpointed θ — decide before writing discovery locks.

---

## 🚫 Explicitly Deferred (unchanged)

| Item | Reason |
|------|--------|
| ConvGeometry / GraphGeometry / AttentionGeometry / 3D Spatial Lattice | Science runs on Feedforward/Recurrent/Tile at MLP scale; geometry-DEFERRED skips stay skips |
| Coverage floor | ~16.8%; opt-in `--cov`; raise after API stabilizes |
| Rocq general-case formalization | CP-B pull-based; diagonal case done with paper proof |
| `test_ontology_parity.py` decomposition | Slow-marked (P3); split fast/slow only if gate iteration speed demands |
| Physical hardware (PR-3b / CP-D) | Latency-gated procurement — starts per RESEARCH3 day-one, not here |

## ✅ Definition of Done

- **R1:** ✅ EqProp MNIST epoch on CUDA in ~5.6 s · ✅ parity classes construction-seeded · `pytest -q` green · pyright policy in pre-commit → **R1 complete**
- **R2:** `computronium/zoo/**` deleted · no dead stubs or duplicate Substrate · 0 xpass noise · skip census recorded
- **R3:** DiffusionDynamics un-xfailed · every remaining xfail has a precise reason · no hardcoded geometry inference
- **R4:** ≥2 operator families beyond settle through the Substrate API · equivalence test per port
- **R5:** ≥1 commissioned campaign (iterate→interrupt→resume) in `autoscientist_campaigns/` with golden manifest · demo runs end-to-end on CUDA · discovery locks green (winner-replication + attribution + tolerance replay)
- **R6:** RESEARCH3 PR-1 + PR-2 merged · PR-7 shakedown green

## 🔧 Quick Commands

```bash
# Gate (default, ~65s): unit+property; slow/benchmark/llm auto-deselected; 60s per-test timeout
uv run pytest -q
# Slow tier (~25min; `tests` arg required — testpaths limits bare pytest to unit+property)
uv run pytest tests -m slow
# Full (~23min): uv run pytest tests -m ""
# Coverage (opt-in): uv run pytest tests --cov=computronium --cov-report=term-missing

# Fast gates (seconds): property locks + registry + boundary
uv run pytest tests/property/test_ontology_locks.py tests/unit/core/test_registry.py \
  tests/unit/core/test_module_boundary.py tests/unit/test_refactor.py -q

# CUDA placement guard (all 28 native factories; skips without CUDA)
uv run pytest tests/property/test_native_device_placement.py -q

# Commission a smoke campaign: start → mid-flight kill → resume → artifacts
uv run scripts/commission_smoke_campaign.py --fresh
# `comp campaign run` accepts --device (default auto) / --seed / --tasks

# Native smoke / settle protocol / joint benchmarks
uv run pytest tests/property/test_native_smoke.py -v
uv run pytest tests/integration/test_settle_protocol_models.py -q
uv run pytest tests/integration/joint/test_benchmarks.py -v

# Type check (policy: strict on ontology/, basic elsewhere)
uv run pyright computronium/ontology

# NOTE: sync with `uv sync --extra dev --extra lightning` (plain dev sync removes
#   lightning -> 4 collection errors). Serial only — xdist hangs in this env.
#   Installed script is `comp`; help output prints prog-name "biopl" (cosmetic).
```
