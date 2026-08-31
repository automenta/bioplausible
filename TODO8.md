# TODO8.md — Consolidated Plan

> **Rev 2026-08-31.** P0–P5 session logs consolidated away (full history in `git log`).
> Research catalog lives in [RESEARCH3.md](RESEARCH3.md); this doc owns the engineering that unblocks it.
>
> **State:** P0–P4 complete · P5 complete (minus pyright policy) · gate `pytest -q` green (~65s) ·
> full suite 0 failed since 2026-08-30 (latest cert: 1499 passed / 89 skipped / 41 xfailed / 4 xpassed).
>
> **Policy:** zero backwards compatibility · GPU-first for all training paths · no new tests for broken
> capability (xfail with precise reasons) · serial pytest only (xdist hangs in this env).

## ✅ Completed Record (2026-08-30/31)

| Phase | Outcome |
|-------|---------|
| P0 | Registry lazy auto-population (28 native + aliases), KB constructor, PARAM_UPDATE registrations, module boundary — 1455 passed / 0 failed |
| P1 | `_TaskTrainer` rewrite, EWC consolidate, ModelAdapter real BPTT/metrics, geometry flattening — every triage failure fixed, none left as xfail |
| P2 | Native smoke 28/28 (pass/xfail w/ reasons), settle protocol 24+ pass, validation skips → xfails, 4 xpassed resolved, property locks green |
| P3 | Quarantine emptied (2 deleted, 6 enabled), free-energy tracking implemented, gRPC worker API drift fixed, 5 timeout victims fixed, parity flake root-caused — 1499 passed / 0 failed |
| P4 | `SubstrateSettleKernel` ported; `EnergyMinimizationDynamics.settle` substrate-native (Digital bitwise-equal to legacy); 10-test equivalence suite |
| P5 | Campaign schema freeze (migrations + `SchemaVersionError`), replication gate, counterfactual attribution, `CampaignStack` facade, 8 rankable objectives, migration smoke in CI, CLI validated end-to-end |

Old P6 checklist items already satisfied: EqProp anchor 81.32% MNIST · ComputroniumLinear (26 tests) · torch.jit → torch.export migration in `deployment.py`.

## 🔁 What Changed in This Revision

1. **GPU-first is the critical path, not an optimization.** CPU settle paths stall campaigns (P5 model sweep stalled; GPU directive 2026-08-31). R1 precedes everything.
2. **Campaign commissioning promoted to headline deliverable.** The P5 stack is built but `autoscientist_campaigns/` is empty — zero completed runs (RESEARCH3 PR-9). Demonstrability = one commissioned run.
3. **Five scattered "improvement opportunities" sections folded into R2–R4** — nothing lost; resolved items dropped.
4. **Old P6 (Research Phases 4/5/6) absorbed by RESEARCH3's critical paths.** This doc ends where RESEARCH3 begins; R6 is the handoff.
5. **Sequencing:** R1 → R5.1 can start immediately after; R2/R3/R4 interleave opportunistically; R6 follows R1.

---

## 🎯 R1 — GPU-First Runners + Close P5 (do first)

*Verified: `create_native_*` factories have no `device` param (`**kwargs` silently ignored → tensors always construct on CPU even under CUDA trainers); `ontology/system.py` (5 sites), `autoscientist/campaign.py:717`, `core/system_trainer/factory.py` + `joint.py` hardcode `device="cpu"`. Threading `device` through native factories is the #1 acceleration target (2026-08-30 tiering discovery).*

| # | Task | Detail |
|---|------|--------|
| 1.1 | `device` through native factories | Add `device` param to all `create_native_*`; propagate into configs; stop silently ignoring kwargs |
| 1.2 | Auto-device in runners | `"cuda" if torch.cuda.is_available() else "cpu"` for `SystemTrainerConfig` default, `AutoScientist._execute_proposal`, `CampaignStack`, benchmark suites, `evaluate_migration`. CPU only for tiny equivalence/determinism probes |
| 1.3 | CUDA placement guard | Test asserting params land on CUDA after GPU construction (kills the silent-CPU failure mode permanently) |
| 1.4 | Suite-wide construction seeding | `torch.manual_seed` before every factory call in parity tests (P3 Backprop flake pattern — other parity classes share the unseeded-construction bug) |
| 1.5 | **Close P5: pyright policy** | Strict on `ontology/`, basic elsewhere; wire into pre-commit. Satisfies RESEARCH3 PR-0's typing gate |

**Done when:** EqProp single-epoch MNIST fit runs on CUDA in seconds; gate green; pyright policy enforced.

## 🧹 R2 — Retirement & Signal Honesty (stability; interleavable)

| # | Task | Detail |
|---|------|--------|
| 2.1 | Zoo retirement | Extract still-live registrations (MEP presets, MEP PARAM_UPDATE) into first-class ontology modules → delete `computronium/zoo/**` incl. `tile_models.py`/`tile_fa.py`/`tile_lm.py`. User directive: zoo deprecated for the ontology API — don't fix zoo components |
| 2.2 | Dead/duplicate sweep | Delete `ontology/dynamics/primitives.py` stub; resolve `Substrate` duplication (`ontology/_substrate.py` vs `ontology/substrate/`); grep for other parallel legacy/new pairs |
| 2.3 | Registry API unification | `Registry.list()` vs `list_models()` alias asymmetry (module-boundary test pins the raw view); alias `get_metadata` projects from canonical |
| 2.4 | xpass noise fix | Native smoke tile tests are xfail-but-xpass (smoke checks crash-freedom only). Split: crash-free smoke (strict pass) + learning-capability test (true xfail) |
| 2.5 | Skip census | 89 skips → classify deferred-vs-gap; geometry-DEFERRED stay skipped, everything else gets a reason or a fix ticket |
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

## ⚙️ R4 — Substrate API Breadth (P4 continuation; opportunistic)

| # | Task |
|---|------|
| 4.1 | FA feedback projection through the Substrate operator API (validates non-settle paths) |
| 4.2 | Register `SubstrateSettleKernel` in `KernelRegistry` for the EQPROP family (currently `EqPropKernelBackend` covers all targets) |
| 4.3 | MEP Triton kernels (Muon, Fisher whitening) → Substrate update operator |
| 4.4 | Sparse substrate transpose-mask handling; ternary `init_scale` param (un-xfail ternary equivalence); optional per-step `inject_state_noise` during settle for analog/memristive substrates |

## 🎬 R5 — Demonstrable Product Surface (RESEARCH3 PR-9)

| # | Task | Detail |
|---|------|--------|
| 5.1 | **Commission one campaign** | Full iterate → interrupt → checkpoint → resume cycle via `CampaignStack` into `autoscientist_campaigns/` (empty today). The single most demonstrable missing artifact — nothing consumes the campaign stack until this passes |
| 5.2 | One-command demo on GPU | `comp campaign run` → Pareto + replication + counterfactual report, end-to-end on CUDA, documented |
| 5.3 | Demo/quickstart truth-check | `scripts/quickstart.py`, `demo/main.py`, README factory examples verified against current API (the autoscientist package was unimportable until P5 — demos may be stale); add slow-tier smoke |
| 5.4 | CLI/docs polish | `comp` prog-name prints "biopl" (cosmetic); align README's "0 errors in strict mode" claim with the actual pyright policy |
| 5.5 | Export pin | torch.export (PT2) round-trip test for FeedforwardGeometry + RecurrentGeometry (migration done; pin it) |

## 🔬 R6 — Handoff to RESEARCH3 (old P6 absorbed)

Old P6 phases map onto RESEARCH3's critical paths — execute there, not here:

| Old P6 item | RESEARCH3 home |
|-------------|----------------|
| Phase 4: Regime Discovery (Bandit Router, Memristive IR-Drop sweep, Photonic Epistemology Swap) | AutoScientist campaign types (substrate ablation, epistemology swap) |
| Phase 5: Family-Coverage Benchmark (coordinate lock ≥30, Resource-Vector Runner, Dynamical Phylogeny) | De Facto Non-Backprop Benchmark + PR-3a |
| Phase 6: Frontier Certification (M-Axis Frontier, Goldilocks Map, Manifesto Dataset) | CP-A tail: frontier campaign + failure manifesto |

**First moves after R1** (RESEARCH3 startup sequence): PR-1 optimizer-phase hygiene (rebuild Adam between meta-train and ψ-adaptation — momentum carry-over contaminates the Δθ=0 claim) → PR-2 θ-invariance audit harness → PR-3a resource instrumentation → PR-7 shakedown (L3.5 → L1 → L2/L3). `docs/baseline.md` (PR-0) and `docs/evaluation_fairness_contract.md` (PR-6) already exist.

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

- **R1:** EqProp MNIST epoch on CUDA in seconds · `pytest -q` green · pyright policy in pre-commit
- **R2:** `computronium/zoo/**` deleted · no dead stubs or duplicate Substrate · 0 xpass noise · skip census recorded
- **R3:** DiffusionDynamics un-xfailed · every remaining xfail has a precise reason · no hardcoded geometry inference
- **R4:** ≥2 operator families beyond settle through the Substrate API · equivalence test per port
- **R5:** ≥1 commissioned campaign (iterate→interrupt→resume) in `autoscientist_campaigns/` · demo runs end-to-end on CUDA
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
