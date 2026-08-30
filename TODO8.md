# TODO8.md — Unblock Tests → Restore Parity → Research Phases

> **Scope:** Honest, sequenced plan based on `run_all_tests.sh` (67 failed) + TODO2-7.md review.  
> **Principle:** Zero backwards compatibility. No deprecation paths. Fix mechanical onboarding failures first.

---

## 🎯 The Real Critical Path (Execute In Order)

### P0 — Unblock Test Suite (Week 1) — **DO FIRST, NOTHING ELSE**

| # | Task | Root Cause | Files to Touch | Status |
|---|------|------------|----------------|--------|
| 1 | **Registry auto-population** | 28 native models only register on explicit `import registration` | `computronium/__init__.py` — add `from computronium.models.native import registration` | ✅ Done (m0034, redesigned m0830: lazy via `__getattr__` hook) |
| 2 | **KnowledgeBase constructor** | Tests pass `str`/`Path`; impl requires `KnowledgeBaseConfig` | `computronium/knowledge/kb.py` — accept `str | Path | KnowledgeBaseConfig` | ✅ Done (m0830: + kwargs overrides) |
| 3 | **Module boundary tests** | `SystemTrainer` eagerly imported in `__init__.py` | `computronium/__init__.py` — lazy load `SystemTrainer` | ✅ Done (m0830: light import stays torch-free) |
| 4 | **Legacy model aliases** | Tests use old names (`backprop_mlp`, `eqprop_mlp`, etc.) | `computronium/core/registry.py` — add `_ALIASES` entries | ✅ Done (m0052; m0830: `Registry.list()` surfaces aliases w/ registered targets) |
| 5 | **PARAM_UPDATE registrations** | `adam`, `sgd`, etc. not registered for lightning | ~~`computronium/zoo/optimizers/standard.py`~~ → `computronium/ontology/optimizers.py` (zoo deprecated) | ✅ Done (m0830: sgd/adam/adamw/rmsprop, imported via `__getattr__` hook) |
| 6 | **Re-run full suite** | Verify failures drop from registry/constructor cascade | `./run_all_tests.sh` | ✅ Done (m0830: **0 failed, 1455 passed**; was 67F baseline) |
| 7 | **Triage remaining failures** | Separate real capability gaps from downstream effects | Update this doc with actual counts | ✅ Done (m0830: all 19 triaged + fixed, see P1 log) |

**Expected:** >1300 passing, <20 meaningful failures after P0.
**Actual:** 1455 passing, **0 failures** (2026-08-30). ~30s runtime for property+unit gates; full suite 17.5min (tiering needed, see Notes).

---

### P1 — Integration Recovery (Week 1-2) — ✅ **COMPLETE (2026-08-30, ahead of plan)**

All P1 areas resolved as side effects of the P0 triage + targeted capability fixes:

| Area | Root Cause Found | Fix |
|------|------------------|-----|
| Lightning integration | PARAM_UPDATE registry empty at lookup | `computronium/ontology/optimizers.py` (sgd/adam/adamw/rmsprop) |
| Hyperopt/Optuna bridges | `CoreTrainer` removed but `_TaskTrainer` still delegated to it | `_TaskTrainer` rewritten self-contained (`compute_loss` + inline validation) |
| Smoke all tasks (11 tests) | (a) dead `CoreTrainer` import chain; (b) `RLTrainer` blocked by dead `BaseTrainer` import in `training/__init__.py`; (c) tuple `input_dim` → `nn.Linear` crash | (a) `_TaskTrainer` rewrite; (b) removed dead import; (c) flatten in `construct_model` + `FeedforwardGeometry.forward` |
| Continual learning (EWC) | `consolidate()` signature: runner passes 1 arg, U-lock passes 2 | `old_params=None` optional; baseline-drift Fisher semantics (2-arg path unchanged) |
| ModelAdapter tests | plain `nn.Sequential` has no `train_step` → empty legacy metrics | `_standard_metrics` fallback (eval-mode forward + CE + accuracy); `_AdaptedSystem.train_step` real BPTT instead of `{"loss": 0.0}` stub |
| Refactor tests | aliases invisible in `Registry.list()`; registry empty on light import | aliases listed when target registered; explicit `import registration` in test |

**Triage rule applied:** every remaining failure after P0 was root-caused and fixed directly — none were left as xfails.

---

### P2 — Native Capability Parity (Week 2-3)

*Continuation of TODO7 Phase B*

| Target | Current | Required |
|--------|---------|----------|
| **Native smoke tests** | 20 pass, 4 skip, 4 xfail | 28/28 `forward()` + `train_step()` |
| **Settle protocol integration** | 21 pass | 29+ pass (restore missing, xfail true failures) |
| **Validation all** | 2 pass, 14 skip | Reduce skips → xfails with precise reasons |
| **Property test updates** | See below | All 5 files passing/xfail with reasons |
| **Known native issues** | Documented as skips | Track as named blockers with xfails |

**Property test files (from TODO7):**

| Test File | Status | Required |
|-----------|--------|----------|
| `test_ontology_parity.py` | 30 passed, 1 skipped, 2 xfailed | ✅ Mostly done |
| `test_biology_axioms.py` | 7/9 passing | ✅ Mostly done |
| `test_scaling_invariants.py` | 5 passed, 3 skipped, 3 xfailed | Resolve xfail/skip |
| `test_settle_protocol.py` | 6 passed | ✅ Done |
| `test_validation_all.py` | 2 passed, 14 skipped | Reduce skips → xfails |

**Explicit blockers (from TODO7):**

| Model / Component | Issue | Status |
|-------------------|-------|--------|
| `native_tile_ep` | Device/dynamics incompatibility | xfail with reason |
| `native_tile_pc` | Device/dynamics incompatibility | xfail with reason |
| `native_tile_gnn` | Device/dynamics incompatibility | xfail with reason |
| `native_tile_snn` | Device/dynamics incompatibility | xfail with reason |
| `DiffusionDynamics` | Gradient bug | xfail with reason |
| FA + InstantaneousDynamics | No proper error signal | xfail with reason |
| PEPITA | Empty pseudo-gradients | xfail with reason |

**No new tests for broken capability.** Only xfail with precise reasons.

---

### P3 — Ignored Test Files: Explicit Resolution (Week 2-3)

*These 8 files are permanently ignored in `run_all_tests.sh`. Pick ONE outcome per file.*

| File | Outcome | Action |
|------|---------|--------|
| `test_hardware_aware.py` | **DELETE** or migrate to native API | No legacy imports |
| `test_benchmarks.py` (joint) | **ENABLE** or mark `@pytest.mark.slow` | If fast, enable; if slow, mark |
| `test_diffusion_integration.py` | **XFAIL** with reason | Gradient bug in DiffusionDynamics |
| `test_energy_invariants.py` | **ENABLE** (should pass per TODO2) | Fix to native API, re-enable |
| `test_equitile_sparsity_robustness.py` | **DELETE** or migrate | Legacy imports |
| `test_dht.py` | **MARK INFRA/SLOW** | Environment-dependent |
| `test_grpc_seam.py` | **MARK INFRA/SLOW** | gRPC infra issues |
| `test_grpc_seam_subprocess.py` | **MARK INFRA/SLOW** | gRPC infra issues |

**No file stays ignored without explicit status.**

---

### Reference: Untested Functionality Coverage Targets

*Add coverage after P0-P2 stable. Not a current priority.*

| Area | Missing Coverage |
|------|-----------------|
| JointSystemTrainer | 6-D joint training loop |
| Plasticity primitives | Routing, FastWeight, RuleState, SubstrateCoupled |
| AutoScientist | Campaign execution, proposal generation, KB integration |
| P2P distributed training | gRPC worker, Kademlia DHT, fault tolerance |
| Model export | ONNX, TorchScript (PT2), INT8, Ternary |
| Inference server | FastAPI, TensorRT, dynamic batching |
| Stability monitoring | Spectral radius, Lyapunov, basin stability |
| Frontier analysis | Pareto computation, knee detection |

---

### P4 — Kernel Porting: One Family First (Week 3-4)

*Legacy kernels must move to Substrate operator API:*

```python
Substrate.get_forward_operator()
Substrate.get_weight_update_operator()
Substrate.quantize_weights()
```

| Priority | Kernel Family | Reason |
|----------|---------------|--------|
| 1 | EqProp settle kernel | Core 6-D path, high leverage |
| 2 | FA feedback projection | Core 5-D path, validates Substrate API |
| 3 | MEP CUDA kernels | Port to Substrate or custom Autograd Function |
| 4 | Sparse/Ternary quantization | Port to `Substrate.quantize_weights()` |

**Do not port all at once.** Port one end-to-end, use as template.

---

### P5 — Campaign Infrastructure Stabilization (Week 4)

*Blocking research phases. Do not run campaigns on unstable test suite.*

| Item | Target |
|------|--------|
| CampaignStore DB schema freeze | Alembic or custom migrations |
| ProposalObjective expansion | Add `STABILITY`, `ENERGY`, `LATENCY`, `PLASTICITY_CAPACITY` |
| Replication gate | Auto-verify ≥5 seeds + ≥2 task families |
| Counterfactual attribution | Integrate `analysis/counterfactual.py` |
| `CampaignStack.run_campaign(...)` | Deliverable |
| effective-FLOPs → 𝒞 vector | Verify wiring complete |
| Algorithm migration benchmark | CI smoke test |

---

### P6 — Research Phases 4/5/6 (Week 5+)

*Only after P0-P2 stable.*

| Phase | Scope |
|-------|-------|
| **Phase 4: Regime Discovery** | Bandit Router, Memristive IR-Drop sweep, Photonic Epistemology Swap |
| **Phase 5: Family-Coverage Benchmark** | Coordinate lock (≥30), Resource-Vector Runner, Dynamical Phylogeny |
| **Phase 6: Frontier Certification** | M-Axis Frontier, Goldilocks Map, Manifesto Dataset |

**Gate:** P0 complete, P1 mostly complete, P2 smoke/settle/validation stable.

---

## 🚫 Explicitly Deferred (Do Not Work On)

| Item | Reason |
|------|--------|
| `ConvGeometry` | Phase 5/6 science runs on Feedforward/Recurrent/Tile at MLP scale |
| `GraphGeometry` | Same |
| `AttentionGeometry` | Same |
| 3D Spatial Lattice | Same |
| Pyright strict mode | Deprioritized behind functional work; apply policy: strict on ontology/, basic elsewhere |
| Coverage ≥25% | Current ~16.8% (floor 15%); raise after API stable |
| Rocq formalization | CP-B pull-based; diagonal case done (0-admit); general case admitted w/ paper proof |

---

## ✅ Definition of Done (Per Phase)

### P0 Done
- [x] `from computronium import Registry; Registry.list(MODEL)` → 28 native models (+ aliases)
- [x] `KnowledgeBase("path.db")` works (also kwargs: `db_path=`, `auto_embed=`)
- [x] `./run_all_tests.sh` → **0 meaningful failures** (1455 passed)

### P1 Done
- [x] Lightning/Hyperopt/Optuna/Smoke pass (31 passed, 2 skipped)
- [x] No accidental constructor/import failures (`test_module_boundary` green)
- [x] Continual learning EWC + ModelAdapter + Refactor tests pass

### P2 Done
- [ ] 28/28 native smoke tests pass
- [ ] Settle protocol ≥29 pass (remaining xfails documented)
- [ ] Validation all skips reduced → xfails with reasons
- [ ] Property test files (5) passing/xfail with reasons
- [ ] Known Tile/Diffusion/FA/PEPITA issues tracked as xfails
- [ ] **Property locks: 32 ontology + 33 joint stability tests passing**
- [ ] **Joint benchmarks runnable via `biopl benchmark`**

### P3 Done
- [ ] 0 ignored test files without explicit status

### P4 Done
- [ ] At least one kernel family ported to Substrate operator API
- [ ] Kernel equivalence test exists for ported operator

### P5 Done
- [ ] CampaignStore schema frozen
- [ ] CLI commands validated end-to-end
- [ ] Pyright policy applied (strict on ontology/, basic elsewhere)
- [ ] **Campaign persistence: resume from SQLite + YAML checkpoints**
- [ ] **Pareto frontier computed over loss, resources, stability**

### P6 Done
- [ ] Phase 4 regime discovery run
- [ ] Phase 5 coordinate locked
- [ ] Phase 6 frontier campaign with checkpoint/resume
- [ ] **EqProp competitive anchor: 81.32% MNIST cited in benchmark**
- [ ] **ComputroniumLinear: 26 tests, bit-for-bit backprop fallback**
- [ ] **torch.export (PT2) round-trip for FeedforwardGeometry + RecurrentGeometry**

---

## 🔧 Quick Commands

```bash
# P0 verification (light path: torch-free, unregistered)
uv run python -c "import sys, computronium.core.registry as r; print('torch' in sys.modules, bool(r.list_models()))"
# → False False

# P0 verification (top-level attr access: populated)
uv run python -c "
from computronium import Registry
from computronium.core.registry import ComponentCategory as C
print(len(Registry.list(C.MODEL)['model']))"
# → 46 (28 native + tile variants + registered-target aliases)

# Full suite (~17min — tier this, see Notes)
./run_all_tests.sh

# Fast gates (seconds): property locks + registry + boundary
uv run pytest tests/property/test_ontology_locks.py tests/unit/core/test_registry.py tests/unit/core/test_module_boundary.py tests/unit/test_refactor.py -q

# Native smoke tests
uv run pytest tests/property/test_native_smoke.py -v

# Settle protocol
uv run pytest tests/integration/test_settle_protocol_models.py -v

# Smoke all tasks (vision/LM/RL trainer path)
uv run pytest tests/integration/test_smoke_all_tasks.py -q

# Joint benchmarks
uv run pytest tests/integration/joint/test_benchmarks.py -v

# Energy invariants
uv run pytest tests/integration/test_energy_invariants.py -v

# Type check (policy: strict on ontology only)
uv run pyright computronium/ontology
```

---

## 📝 Notes

- **Legacy Zoo**: ~200K lines removed. 3 thin wrappers (`tile_models.py`, `tile_fa.py`, `tile_lm.py`) → **DELETE** (no deprecation, zero users). **2026-08-30: zoo work paused entirely per user directive** ("zoo is being deprecated for the ontology API — don't waste time fixing Zoo components"). Standard optimizers were registered ontology-side instead (`computronium/ontology/optimizers.py`).
- **Native Models**: 28 registered with explicit 5-D axes. Accessible via `Registry.get()` once `registration` loads.
- **Zero-Extension Invariant**: `M=NullPlasticity` slice formally verified (J1 test).
- **EqProp competitive**: 81.32% MNIST anchored via 20-epoch run (grad_clip + checkpointing).
- **ComputroniumLinear (CP-C)**: Drop-in `nn.Linear` wrapper complete, 26 tests.
- **torch.jit → torch.export**: Migration complete in `deployment.py`.

### P0 Progress Log (2025-08-30)
- ✅ Registry auto-population: Added native registration import to `computronium/__init__.py:4`
- ✅ Legacy model aliases: Added 20 entries to `Registry._ALIASES` mapping old names → `native_*` variants
- ✅ Module boundary: `SystemTrainer` already lazy-loaded via `_LAZY` dict
- ✅ KnowledgeBase constructor: Done 2026-08-30 (see below)
- ✅ PARAM_UPDATE registrations: Done 2026-08-30 (see below)
- ✅ Optuna bridge: Resolved 2026-08-30

### P0/P1 Progress Log (2026-08-30) — suite green (0 failed / 1455 passed)

**Registration architecture (redesigned — single lazy mechanism):**
- `computronium/core/registry.py`: `_ensure_native_registered()` (idempotent via `_registration_state` dict, no `global`). **Registry reads are pure** — `get/list/query/get_metadata` call the ensure hook, but `Registry.clear()` sets `blocked=True` permanently so test isolation is respected.
- `computronium/__init__.py`: `__getattr__` hook calls `_ensure_native_registered()` on first top-level attribute access. Eager imports removed.
- **Light-import invariant** (`tests/unit/core/test_module_boundary.py`): `import computronium.core.registry` → no torch, no zoo, `list_models()` empty. Three valid registration triggers: (1) top-level attr access, (2) explicit `from computronium.models.native import registration`, (3) import of `computronium.models.native` / `computronium.ontology` packages (their `__init__`s import registration eagerly).
- Tests that touch the Registry via the light path must trigger registration explicitly (done in `tests/unit/test_refactor.py`).
- `Registry.list()` now includes aliases **only when their canonical target is registered** (exact-list tests in `test_registry.py` stay green).

**Capability fixes (not test-masking):**
- `computronium/ontology/optimizers.py` (NEW): standard torch optimizers as PARAM_UPDATE components; zoo untouched (deprecated per user directive).
- `computronium/knowledge/kb.py`: constructor accepts `str | Path | KnowledgeBaseConfig | None` + `**overrides` (dataclass `replace`); added `natural_language_query()`.
- `computronium/knowledge/vector_store.py`: keyword-search fallback actually implemented (term-overlap score, recency tie-break via rowid, `min_similarity` honored) — was a stub returning `[]`.
- `computronium/domains/trainer.py`: `_TaskTrainer` rewritten self-contained (canonical `compute_loss`, grad-clip, inline val, `train_*`/`val_*` metric shape for hyperopt). Removed dead `CoreTrainer.from_task` delegation.
- `computronium/training/__init__.py`: removed dead `CoreTrainer as BaseTrainer` import (blocked `RLTrainer` import entirely).
- `computronium/core/plasticity/fast_weights.py`: projection matrix is now an **orthonormal-basis random-subspace projection** (zero-padded, QR-seeded) → deterministically non-expansive (`||Pv|| ≤ ||v||`), making the Hebbian decay bound exact. Test bound fixed to use the full outer-product norm (its own docstring semantics).
- `computronium/ontology/update.py`: `ElasticConsolidationUpdate.consolidate(params, old_params=None)` — 2-arg U-lock path unchanged; single-arg boundary call anchors current snapshot and derives Fisher from drift since previous baseline.
- `computronium/ontology/system.py`: `ModelAdapter._standard_metrics` fallback for models without `train_step`; `_AdaptedSystem.train_step` performs a real BPTT step (SGD from `update.config.step_size`) instead of returning `{"loss": 0.0}`.
- `computronium/ontology/geometry.py`: `FeedforwardGeometry.forward`/`forward_with_intermediates` flatten inputs with `dim() > 2` (image-shaped batches).
- `computronium/core/construction.py`: `construct_model` canonicalizes tuple `input_dim` → `math.prod` (vision tasks expose `(C,H,W)`).
- `computronium/core/system_trainer/factory.py`: `_ComposedSystem` gained `.to(device)` and an `optimizer` slot (hyperopt mirrors `trainer.optimizer` onto the model).
- **Deleted**: `computronium/ontology/adapter/` (1213 lines, zero references — parallel dead ModelAdapter implementation; the live one is `computronium/ontology/system.py:710`).
- Tests updated to canonical identity: `test_scientist.py` + `test_phase2_integration.py` use `native_backprop_mlp`.

**Verification:** full suite `./run_all_tests.sh` → **1455 passed, 96 skipped, 33 xfailed, 4 xpassed, 0 failed**. Ruff format clean on all touched files; ruff check adds 0 new findings vs baseline; pyright (ontology + touched modules) 153 → 149 errors (no new).

### New Improvement Opportunities (Discovered 2026-08-30)
- **Test suite tiering (blocking, per user directive: tests must run in minutes)**: full suite is 17.5min. Split `run_all_tests.sh` into `--fast` (property+unit, ~30s) and `--full`; add `pytest.mark.slow` markers. Profile with `--durations=25`. Heavy hitters are smoke/integration tests training real models.
- **4 xpassed** (non-strict xfail): investigate and convert to strict xfails or remove the markers — they hide working functionality.
- **96 skipped**: triage into xfails-with-reasons vs. legitimately-skipped (P2/P3 work).
- **`Registry.list()` vs `list_models()` asymmetry**: alias inclusion differs; document or unify (module-boundary test pins `list_models()` to the raw view).
- **`Registry.get_metadata` on aliases**: resolves via `resolve_alias`, but alias entries have no metadata of their own — consider metadata projection from canonical.
- **`_TaskTrainer` gaps vs old CoreTrainer.from_task**: no scheduler wiring, no energy tracking, `tracker`/`safety_config` accepted but ignored. Wire when hyperopt trials need them.
- **`_AdaptedSystem` inference is stubbed**: `_infer_geometry` returns hardcoded `FeedforwardGeometry(784→256,128→10)` regardless of the wrapped model. Real ontology inference was in the deleted `adapter/` package — recover the useful parts (heuristics/inference) if adapter fidelity matters.
- **Dead code sweep continues**: `computronium/zoo/**` is deprecated (user directive) but still registered/importable via direct import; plan an extraction of any still-live registrations (MEP presets, MEP PARAM_UPDATE entries) into first-class modules before deleting zoo.
- **Duplicate ModelAdapter classes existed** (`ontology/system.py` vs deleted `adapter/adapter.py`) — grep for other parallel legacy/new implementations (e.g. `Substrate` in `ontology/_substrate.py` vs `ontology/substrate/`).
- **`natural_language_query` retrieval quality**: term-overlap is crude; consider TF-IDF weighting over fields once entries grow.

### P2 Sanity Check (2026-08-30)
- `tests/property/test_plasticity_properties.py`: **27/27 passing** (was 1 failing) — FastWeight decay bound is now a real invariant, not a truncated-norm approximation.
- `tests/property/test_ontology_locks.py`: 33 passed, 2 skipped (unchanged).
- `tests/property/joint/` + `tests/integration/joint/`: green (in full suite).
- Remaining P2 items (native smoke 28/28, settle protocol, validation skips→xfails) unchanged — still open.