# TODO8.md — Consolidated Plan

> **Rev 2026-08-31 (c).** P0–P5 session logs consolidated away (full history in `git log`).
> Research catalog lives in [RESEARCH3.md](RESEARCH3.md); this doc owns the engineering that unblocks it.
>
> **State:** P0–P5 **complete incl. pyright policy** · R1 **complete incl. construction seeding**
> (device threading, placement guard, runner auto-device, EqProp CUDA epoch ≈ 5.6 s) · U-bypass
> sweep complete (see audit below) · **R5.1a CPU smoke campaign commissioned** (mid-flight
> SIGKILL → resume → complete, artifacts in `autoscientist_campaigns/smoke_cpu/`) ·
> **R5.1b GPU quick campaign commissioned** (seeds 0/1/2 in `quick_gpu*/`, 33 episodes each,
> kill→resume lifecycle intact) · **R2 signal-honesty pass complete** (R2.4 xpass split,
> R2.5 skip census, R2.6 kademlia+grpc seam; R2.1 extraction list prepped, deletion pending) ·
> **R5.3/5.4/5.5 complete** (quickstart truth-checked + slow smoke, prog-name fixed, PT2 pin) ·
> **R5.1c commissioned replication campaign complete** (72-coord grid × 5 seeds × 2 task
> families = 720 episodes, **72/72 replicated**, kill→resume on seed 0 with zero duplicate
> episodes — `autoscientist_campaigns/r51c/`) · **improvement-10/11 fixed** (resume dedup +
> per-episode construction seeding → replay-safe campaigns; R5b-D prereqs cleared) ·
> gate `pytest -q` green (1236 passed / 54 skipped / 33 xfailed / **0 xpassed**, ~75 s) ·
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
| R5.1b | **GPU quick campaign commissioned 2026-08-31** — 3 seeds (0/1/2) × `quick_gpu{,_seed1,_seed2}`/`campaign-id quick_r51b_s*`: 4 coords/iter, 8 iterations, **33 episodes each**, kill@1-episode → resume lifecycle (~5.5 s + ~7.5 s per seed on RTX 3080). Manifest records `device_requested=cuda`/`cuda_available=true`; metrics/resources recorded; report renders frontier + counterfactuals + replication table. **Speedup caveat:** at smoke scale (8-dim synthetic, batch 16) episodes are kernel-launch bound — GPU ≈ CPU latency or worse per coordinate (measured from artifacts; e.g. recurrent/energy_minimization 2.2 ms CPU vs 9.3 ms GPU). Speedup evidence remains training-scale (EqProp MNIST epoch ≈ 5.6 s CUDA). Multi-seed via separate output dirs (script assumes one campaign per DB) — fold into R5.1c script extension if convenient |
| R2.4 | **xpass split complete 2026-08-31.** Native smoke tile tests split by measured ground truth (param-move probe): `tile_fa/tp/hebbian/pc` crash-free → strict-pass smoke; new `test_native_tile_learning_capability` locks learning signal (params must move after `train_step`) as strict xfail — all four leave 17/17 params frozen (the old xfail reasons were substantively right about learning, wrong about crash-freedom). `tile_ep/gnn` crash (`Energy-based settling requires a layered geometry` → candidate invalid coordinate, R3.9), `tile_snn` crash (tensor 16 vs 212 → implementation bug, R3.4) — precise strict xfails. Gate now: 0 xpassed |
| R2.5 | **Skip census complete 2026-08-31** (one pass). Dead legacy deleted: `tests/integration/test_algorithms_integration.py` (Sprint-7 removals), `tests/integration/test_triton_integration.py` (placeholder-only), `tests/unit/test_verify_bias.py` (`_MODEL_SPECS` gone), `TestC_SurrogateLocks` shells in `test_ontology_locks.py`, placeholder `test_equilibrium_gradients_match_bptt_looped_mlp` (native gradient equivalence already covered by `test_gradient_equivalence.py`), dead `test_feedback_alignment_eqprop` in `test_validation_all.py`. Skips→xfails: 5× `EnergyToInstantaneousAdapter modifies frozen config` (adapter bug, strict; includes J6 lock) — capability recovered: 5× `test_substrate_with_riemannian_update` un-skipped and passing on all substrates (the "known limitations" were stale). Legitimate residuals: axis_probe exhausted/pairwise-fenced (34), DEFERRED geometry (11), research-directions `build()` conditional (9), adapter coverage-gap params (7), biology/scaling conditional weight-lookup (5), optional deps (wandb/Triton-CPU) (4). Skip count now meaningful: 66→54 in gate |
| R2.6 | `uv add kademlia` (main dep — p2p DHT skips now exercise); `test_grpc_seam.py` rewritten: dead `test_grpc_seam_multi_process` stub + orphaned server scaffolding deleted (real-transport multi-process seam lives in `test_grpc_seam_subprocess.py`, slow-marked); fault-injection contract pinned locally. axis_probe `[2-0]` flake: no recurrence observed this session — still watching |
| R5.3 | Quickstart truth-checked **end-to-end on CUDA**: Backprop 93.3% / FF 92.9% @ 3 epochs (45 s / 66 s) — matches documented expectations; dead never-called `train_backprop`/`train_forward_forward` helpers deleted; `EPOCHS` env knob (`QUICKSTART_EPOCHS`, default 3) added for smoke use. `demo/main.py` compiles clean against current API (719-line NiceGUI app, still zero campaign wiring — R5b-F Stage 2). Slow-tier smoke added: `tests/slow/test_quickstart_smoke.py` (1-epoch subprocess run, asserts output contract) |
| R5.4 | `comp` prog-name fixed: `prog="biopl …"` → `comp …` across all CLI adapters; docstring/help `biopl` references normalized (module-entry commands point at `python -m computronium.cli.export_kernel` etc.); README "0 errors in strict mode" claim aligned with actual policy (elevated-standard on ontology, repo-wide basic). cli/ ruff diagnostics 268→264 (net negative) |
| R5.5 | PT2 export round-trip pin: `tests/integration/test_pt2_export_roundtrip.py` — FeedforwardGeometry (backprop) + RecurrentGeometry (EqProp) export via `deployment.export_to_pt2`, reload with `torch.export.load`, outputs bitwise-equal to eager on CPU. Note: `_ComposedSystem` is not an `nn.Module` — the geometry module is the export unit |
| R3-adjacent | **`core/construction.py` reflection bug fixed**: `resolve_consumption` reflected on `model_cls.__init__` — for plain function factories that resolves to `object.__init__`, reporting a spurious `**kwargs` catch-all, so hyperopt trials forwarded stale sampled kwargs (`learning_rate`, `max_steps`, …) into native factories and crashed against R1's kwargs rejection. Fixed via `_construction_callable` (reflect on the function itself); `tests/integration/test_hyperopt_integration.py` + `test_phase2_integration.py` now pass (were failing outside the gate) |
| Campaign determinism | **Improvement-10 + 11 fixed 2026-08-31.** (10) resume dedup: `run_campaign` snapshots recorded `(iteration, coordinate, task)` rows on resume and re-proposed slots emit `already_recorded` instead of re-executing; the crash-between-`add_episode`-and-`update_iteration` race can no longer duplicate rows (pinned by `test_resume_skips_already_recorded_episodes`; verified live: seed-0 kill→resume lands exactly 144/144 episodes). (11) per-episode construction seeding: θ init now derives from `(seed, campaign_id, iteration, coordinate)` via blake2b (`episode_seed`) and is applied with `torch.manual_seed` before composition — rebuilds replay identically (pinned by `test_rebuild_replays_identical_metrics`, the CPU-scale template for the R5b-D replay lock). Also: `FrontierRecord.seed` is now stamped from the campaign seed (was always the dataclass default 42 → the replication gate could never count seeds); `evaluate_episode` takes `seed` + passes `task_name` into `episode_batch`; redo path uses `checkpoint.task_name` instead of `task_cycle[0]` |
| Batch families + grid | `episode_batch` is task-family aware: `synthetic` (legacy stream pinned `1000+episode` — R5.1a/b artifacts reproduce) and new `parity` (nonlinear sign-bit sum; independent blake2b stream); unknown families raise (a real-dataset label can no longer masquerade as smoke data). New `space_grid` + `grid_sampler` (stateless round-robin over a sorted grid, keyed by the `(iteration, experiment)` slot — resume- and multi-seed-safe) + `CoordinateSampler` Protocol (positional-only slots) replacing the Callable alias; `comp campaign run --layout grid` and the 72-coordinate `joint_grid` search space. Task assignment rotates `(experiment + iteration) % len(tasks)` so a coordinate's repeat visits cover both families |
| R5.1c | **Commissioned replication campaign 2026-08-31** — `autoscientist_campaigns/r51c/`: `joint_grid` (72 coords) × 5 seeds × 2 task families, 8 coords/iter × 18 iterations (two full grid passes per seed; the +9-iteration repeat flips the task), 8-dim/batch-16 smoke scale, CUDA. Per-seed stores `seed_0..4/` (one campaign per seed, same script invocation — the multi-seed/multi-family script extension R5.1c asked for). Seed 0: kill@1-episode → resume → **exactly 144 episodes, zero duplicates**; seeds 1-4 clean (9.3 s each). Result: **720/720 episodes persisted, 72/72 coordinates replicated** (≥5 seeds × ≥2 families). Golden manifest (git commit, config hash, uv.lock hash, torch/CUDA, determinism flags, budget, per-seed detail, replication summary) + report + merged `episodes.json` + per-seed YAML checkpoints in `records/`. First evidence chain from the flywheel: D-axis `instantaneous → energy_minimization` +0.82 mean accuracy (800 pairs); M-axis `null → fast_weights` +0.018; Pareto frontier degenerate at smoke scale (see improvement-17) |

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
| 2.1 | Zoo retirement | Audit done (2026-08-31). **Extraction list:** (a) MEP presets + strategies: `zoo/mep/_registration.py` registers `smep`/`smep_fast`/… as CREDIT_ASSIGNMENT and MEP optimizers (Dion/Fisher/Muon/Plain) as PARAM_UPDATE — `computronium/__init__.py` lazy table routes `muon_backprop`/`smep`/`smep_fast` here; (b) `zoo/optimizers/` PARAM_UPDATE registrations `ewc`/`spectral`/`sgd`/`adam`/`adamw`/`rmsprop` — check alias-redundancy against ontology U-axis (`ElasticConsolidationUpdate`/`SpectralConstrainedUpdate`/`EuclideanUpdate`) before extraction; (c) `zoo/models/tile_models.py`+`tile_fa.py`+`tile_lm.py` `register_model` entries — superseded by native tile factories, but `TileLM` is imported by `cli/shared.py` (needs first-class home or deletion); (d) `zoo/models/deployments/{vision,rl,timeseries,graph}.py` `register_model` blocks + `DeploymentConfig`. **Fan-in to preserve:** `get_model_spec`/`load_weights` consumed by `cli/shared.py`, `hyperopt/*`, `core/audit.py`, `config/experiment.py`, `execution/{_guards,robustness}.py`, `benchmarks/{compare_nanoGPT,rigorous}.py`. Then: extract → full suite → delete `computronium/zoo/**` → full suite. User directive: zoo deprecated for the ontology API — don't fix zoo components |
| 2.2 | Dead/duplicate sweep | **Partially done (2026-08-31):** `ontology/dynamics/primitives.py` deleted, `ontology/utils/state.py` deleted, dead state helpers + duplicate helper copies removed. Remaining: `Substrate` naming (`ontology/_substrate.py` impl vs `ontology/substrate/` facade — facade is fine, consider merge); grep for other parallel legacy/new pairs |
| 2.3 | Registry API unification | `Registry.list()` vs `list_models()` alias asymmetry (module-boundary test pins the raw view); alias `get_metadata` projects from canonical |
| 2.4 | xpass noise fix | ✅ **Done 2026-08-31** — native smoke tile tests split into crash-free strict smoke + strict-xfail learning-capability locks (see Completed Record). Gate 0 xpassed. Discovery-lock prereq satisfied |
| 2.5 | Skip census (one pass, then done) | ✅ **Done 2026-08-31** — dead legacy deleted, adapter-bug skips→strict xfails, stale Riemannian restrictions lifted (tests now pass), residuals categorized (see Completed Record). Skip count now meaningful: gate 66→54 |
| 2.6 | Small items | ✅ **Done 2026-08-31** — `uv add kademlia`; grpc-seam stub folded into the working subprocess pattern (dead scaffolding deleted); axis_probe `[2-0]` flake: no recurrence this session |

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
| 5.1b | ✅ GPU quick campaign | **Commissioned 2026-08-31** — 3 seeds in `quick_gpu{,_seed1,_seed2}` (campaign-ids `quick_r51b_s*`), 4 coords/iter × 8 iters, **33 episodes each**, kill→resume intact, manifest `device_requested=cuda`, metrics/resources recorded, frontier + counterfactual + replication rendered. Placement guard green; **no speedup at smoke scale** (launch-bound toy task — measured, documented in README + Completed Record); speedup evidence = training-scale (EqProp MNIST ≈ 5.6 s/epoch CUDA). Improvement-10 note: iteration 1 re-ran post-kill, duplicating recorded coordinates (uniqueness fix still open) |
| 5.1c | ✅ **Commissioned campaign** | **Done 2026-08-31** — 72 coords × 5 seeds × 2 families, 720 episodes, 72/72 replicated, kill→resume with zero duplicates, golden manifest + report + `episodes.json` in `autoscientist_campaigns/r51c/`. Script extended for multi-seed + multi-family in one campaign dir (per-seed stores, shared grid, family-aware batches). Task-scale caveat: smoke-scale episodes are launch-bound (improvement-14) — compute-efficiency contrasts need task-scale variation. Next: R5b discovery locks (prereqs cleared) |
| 5.2 | One-command demo on GPU | `comp campaign run` → Pareto + replication + counterfactual report, end-to-end on CUDA, documented. The commissioning script renders that report (R5.1a/b prove it on CPU/CUDA); R5.2 reduces to documenting the one-liner — fold into R5.1c write-up |
| 5.3 | ✅ Demo/quickstart truth-check | **Done 2026-08-31** — quickstart verified end-to-end on CUDA (93.3%/92.9%), dead helpers deleted, `QUICKSTART_EPOCHS` knob, slow-tier smoke `tests/slow/test_quickstart_smoke.py` added; `demo/main.py` compiles vs current API (campaign wiring still zero — R5b-F Stage 2) |
| 5.4 | ✅ CLI/docs polish | **Done 2026-08-31** — prog-name `biopl`→`comp` across CLI adapters, docstring entry-points normalized, README pyright claim aligned |
| 5.5 | ✅ Export pin | **Done 2026-08-31** — `tests/integration/test_pt2_export_roundtrip.py`: PT2 round-trip bitwise-equal for Feedforward + Recurrent geometry modules |

### R5b — Discovery Demo Package (the "prove it" milestone)

*Demonstrate the ontology flywheel making a discovery — end to end on Digital (CPU/GPU) substrate, every claim locked in by tests. No hardware needed: the D×C×U×M axes carry the discovery space on CPU/GPU (momentum vs plain EqProp settle convergence, spectral/Riemannian U-axis stabilizers, FastWeight/Routing adaptation vs Null, ternary/sparse substrate trade-offs). Smoke-scale L1 runs are feasible on CPU today; GPU (R1) makes sweeps credible.*

| # | Item | Detail |
|---|------|--------|
| A | Pre-register one toy-scale hypothesis | From RESEARCH3 L1/L2/substrate-ablation catalog (e.g. "FastWeight cuts post-switch re-adaptation ≥30% vs Null at matched compute"); thresholds committed before any full run |
| B | Locked grid campaign on GPU | `CampaignStack.run_campaign` over ~30–100 coordinates, matched budgets, ≥5 seeds; replication gate (≥2 task families) must pass |
| C | Evidence chain | Pareto frontier over 𝒞 (compute/memory/energy/latency/plasticity) + counterfactual attribution table naming the axis that owns each knee |
| D | **Discovery locks (tests)** | ① winner-must-replicate: pinned-seed test asserting the discovered gap within tolerance — failing test = capability regression; ② attribution lock: `analysis/counterfactual.py` ranks the discovered axis first, stable across seeds; ③ replay lock: same `(seed, campaign_id, iteration)` re-derives the discovery **within tight tolerance on GPU** — bitwise replay is deprioritized (opt-in: CPU reference or explicit deterministic mode); manifest records torch/CUDA/GPU versions and deterministic flags. ⚠️ Prereqs: ~~R2.4 (xpass split)~~ ✅ done 2026-08-31; ~~improvement-11 θ-init determinism~~ ✅ done — construction seeding + resume dedup landed 2026-08-31 (`test_rebuild_replays_identical_metrics` is the CPU-scale template) |
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
5. ~~**R5.1b GPU quick campaign**~~ ✅ commissioned 2026-08-31 — 3 seeds, 33 episodes each, CUDA placement verified, lifecycle intact; smoke-scale speedup honestly measured as absent (launch-bound toy task) — see Completed Record
6. ~~**R2 signal honesty** — xpass split, skip census, zoo extraction list~~ ✅ done 2026-08-31 (R2.4/2.5/2.6 complete; 2.1 extraction list prepped, deletion next)
7. ~~**R5.1c commissioned campaign**~~ ✅ done 2026-08-31 — 72 coords × 5 seeds × 2 families, 720 episodes, 72/72 replicated, kill→resume zero-duplicate, golden manifest in `autoscientist_campaigns/r51c/` (script extension: multi-seed + multi-family per campaign dir, grid layout, family-aware batches)

Then R5b and RESEARCH3 become real: **R5b-A/B/C/D are unblocked** (pre-register a hypothesis, lock the grid campaign, evidence chain, discovery locks). Everything else (zoo deletion, kernel breadth, capability xfails) is important but must not block R5b.

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
6. ~~**R2.4 xpass split still open**~~ ✅ done 2026-08-31 — tile smoke split into strict crash-free pass + strict-xfail learning-capability locks; gate 0 xpassed. See Completed Record.
7. ~~**R1.4 construction seeding**~~ ✅ done — parity classes seeded via `construction_seed()` helper; see Completed Record.
8. **`compute_energy` duplication** — Energy/Spike/Instantaneous/Diffusion dynamics each carry near-identical `compute_energy` bodies over duck-typed states; extract to one `_energy_from_state(state, geometry)` helper next touch.
9. **SIGKILL to `uv` orphans the worker python (2026-08-31)** — killing the `uv` wrapper leaves the `comp` child alive, which kept executing the campaign to completion in the background. Campaign runners must `killpg` the session (`scripts/commission_smoke_campaign.py::_kill_tree`, `start_new_session=True`). Same hazard applies to spot-instance teardown and the gRPC worker launcher.
10. ~~**Resume can duplicate episodes (2026-08-31)**~~ ✅ fixed 2026-08-31 — `run_campaign` now snapshots recorded `(iteration, coordinate, task)` keys on resume and re-proposed slots emit `already_recorded` instead of re-executing; also closes the add-episode/update-iteration crash race. Pinned by `test_resume_skips_already_recorded_episodes`; verified live in R5.1c (seed 0: exactly 144/144 after kill→resume).
11. ~~**θ init draws ride the ambient RNG across resume (2026-08-31)**~~ ✅ fixed 2026-08-31 — per-episode construction seeding derived from `(seed, campaign_id, iteration, coordinate)` via blake2b (`evaluation.episode_seed`), applied with `torch.manual_seed` before composition in `_evaluate`. Same rebuild ⇒ same θ ⇒ same metrics (`test_rebuild_replays_identical_metrics`). Chose construction seeding over checkpoint-θ redo: replay needs no live checkpoint, and the redo path (which restores checkpointed θ) remains the fault-tolerance route.
12. **`resolve_consumption` mis-reflected function factories (2026-08-31, fixed)** — it reflected on `model_cls.__init__`, which for plain function factories resolves to `object.__init__` → spurious `**kwargs` catch-all → hyperopt forwarded stale sampled kwargs (`learning_rate`, `max_steps`, …) into native factories, crashing trials against R1's kwargs rejection. Fixed via `_construction_callable` (reflect on the function itself); hyperopt/phase2 integration tests green. *Lesson: `obj.__init__` reflection lies for functions — pair it with `inspect.isroutine`.*
13. **Skips hid recovered capability (2026-08-31)** — `test_substrate_with_riemannian_update` was skip-marked for "known limitations" that no longer exist; un-skipped, all 5 substrates pass. *Lesson: the census isn't paperwork — stale skips mask fixed capabilities and dead tests mask removed APIs.*
14. **Smoke-scale campaigns cannot demonstrate GPU speedup (2026-08-31)** — 8-dim/batch-16 episodes are kernel-launch bound; measured GPU ≈ CPU or worse per coordinate (recurrent/settle worst at ~0.24×). Any R5.1c/R5b claim involving speedup must be task-scale-qualified (MNIST-scale settle loops, larger batches) or scoped to lifecycle/placement only. Consequence: R5.1c should vary task family *and* scale if compute-efficiency contrasts are wanted from campaign artifacts alone.
15. **Don't mutate the tree while the gate runs (2026-08-31, process note)** — a mid-run `git stash`/`pop` for a ruff HEAD-comparison killed a census pytest run mid-flight (truncated output, no summary). Run comparisons before/after test batches, never concurrently.
16. **`_ComposedSystem` is not an `nn.Module` (2026-08-31)** — torch.export requires a module; the geometry module is the export unit (pinned by `test_pt2_export_roundtrip.py`). If whole-system export is ever wanted, `_ComposedSystem` needs an `nn.Module` facade — currently deliberate (dataclass system, module per axis).
17. **Pareto frontier is degenerate at smoke scale (2026-08-31, R5.1c)** — stable digital coordinates all read growth ≈ 1.000 and carry near-identical resource vectors, so `stability_score` and `energy` don't discriminate: the frontier collapses to 1 point (loss minimizer) and hypervolume reads 0. Any R5b-C frontier claim needs non-degenerate objective signals — per-coordinate energy/FLOPs accounting (R4.3 pairs), or task-scale variation (improvement-14). Until then, Pareto evidence from campaign artifacts is stability/loss only.
18. **Counterfactual pooling caveat (2026-08-31, R5.1c)** — `attribute_axis_effects` pools minimal pairs across seeds and task families (n_pairs 800/1000 in the R5.1c table). Fine for ranking axes; an R5b-D attribution lock that asserts a stable *ranking across seeds* should either stratify pairs per (seed, family) or assert rank stability over per-seed attributions rather than the pooled mean.
19. **`FrontierRecord.seed` legacy default 42 (2026-08-31, now harmless)** — the field always stamped 42 until `evaluate_episode` grew a `seed` param; hand-built records in analysis code still default to it. If a consumer ever counts seeds over hand-built records, it will read fiction — consider making `seed` required at the FrontierRecord boundary when the next schema break happens.

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
- **R2:** `computronium/zoo/**` deleted (extraction list ready; extraction + deletion remain) · no dead stubs or duplicate Substrate · ✅ 0 xpass noise · ✅ skip census recorded → **R2.4/2.5/2.6 complete, 2.1/2.2/2.3 remain**
- **R3:** DiffusionDynamics un-xfailed · every remaining xfail has a precise reason (✅ enforced this session) · no hardcoded geometry inference
- **R4:** ≥2 operator families beyond settle through the Substrate API · equivalence test per port
- **R5:** ✅ R5.1a+b commissioned campaigns (kill→interrupt→resume) in `autoscientist_campaigns/` with manifests · ✅ R5.1c commissioned campaign (golden manifest, replication gate) **done 2026-08-31** · ✅ demo/quickstart verified on CUDA · discovery locks green (winner-replication + attribution + tolerance replay) — **unblocked**: prereqs (5.1c + improvement-10/11) cleared
- **R6:** RESEARCH3 PR-1 + PR-2 merged · PR-7 shakedown green

## 🔧 Quick Commands

```bash
# Gate (default, ~75s): unit+property; slow/benchmark/llm auto-deselected; 60s per-test timeout
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

# Native smoke / settle protocol / joint benchmarks
uv run pytest tests/property/test_native_smoke.py -v
uv run pytest tests/integration/test_settle_protocol_models.py -q
uv run pytest tests/integration/joint/test_benchmarks.py -v

# Commission a smoke campaign: start → mid-flight kill → resume → artifacts
uv run scripts/commission_smoke_campaign.py --fresh
# GPU quick campaign (3 seeds commissioned; see autoscientist_campaigns/README.md):
uv run scripts/commission_smoke_campaign.py --fresh --device cuda \
  --output-dir autoscientist_campaigns/quick_gpu --campaign-id quick_r51b \
  --experiments-per-iter 4 --iterations-first 6 --iterations-resume 8
# `comp campaign run` accepts --device (default auto) / --seed / --tasks / --layout

# R5.1c commissioned replication campaign (72-coord grid x 5 seeds x 2 families):
uv run scripts/commission_smoke_campaign.py --fresh --device cuda \
  --output-dir autoscientist_campaigns/r51c --campaign-id r51c \
  --space joint_grid --objective pareto_replication --layout grid \
  --seeds 0,1,2,3,4 --tasks synthetic,parity \
  --experiments-per-iter 8 --iterations-first 4 --iterations-resume 18

# Quickstart smoke (slow tier, ~1 min on CUDA via QUICKSTART_EPOCHS=1)
uv run pytest tests/slow/test_quickstart_smoke.py -m slow -q
# PT2 export round-trip pin
uv run pytest tests/integration/test_pt2_export_roundtrip.py -q

# Type check (policy: elevated-standard on ontology/, basic elsewhere)
uv run pyright computronium/ontology

# NOTE: sync with `uv sync --extra dev --extra lightning` (plain dev sync removes
#   lightning -> 4 collection errors). Serial only — xdist hangs in this env.
#   kademlia is a main dependency (DHT exercised in slow tier).
```
