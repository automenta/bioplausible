# TODO8.md — Unblock Tests → Restore Parity → Research Phases

> **Scope:** Honest, sequenced plan based on `run_all_tests.sh` (67 failed) + TODO2-7.md review.  
> **Principle:** Zero backwards compatibility. No deprecation paths. Fix mechanical onboarding failures first.

---

## 🎯 The Real Critical Path (Execute In Order)

### P0 — Unblock Test Suite (Week 1) — **DO FIRST, NOTHING ELSE**

| # | Task | Root Cause | Files to Touch |
|---|------|------------|----------------|
| 1 | **Registry auto-population** | 28 native models only register on explicit `import registration` | `computronium/models/native/__init__.py` — add `from . import registration` |
| 2 | **KnowledgeBase constructor** | Tests pass `str`/`Path`; impl requires `KnowledgeBaseConfig` | `computronium/knowledge/kb.py` — accept `str | Path | KnowledgeBaseConfig` |
| 3 | **Module boundary tests** | `SystemTrainer` eagerly imported in `__init__.py` | `computronium/__init__.py` — lazy load `SystemTrainer` |
| 4 | **Re-run full suite** | Verify failures drop from registry/constructor cascade | `./run_all_tests.sh` |
| 5 | **Triage remaining failures** | Separate real capability gaps from downstream effects | Update this doc with actual counts |

**Expected:** >1300 passing, <20 meaningful failures after P0.

---

### P1 — Integration Recovery (Week 1-2)

| Area | Likely Root Cause | Fix Approach |
|------|-------------------|--------------|
| Lightning integration | Registry empty at lookup time | Ensure registry populated before `Registry.get()` |
| Hyperopt/Optuna bridges | Registry empty at factory lookup | Same registry fix |
| Smoke all tasks (vision/RL/LM) | Registry + data loading | Fix registry first, then data issues |
| Continual learning | EWC/update-rule config | Debug after registry stable |
| ModelAdapter tests | Ontology inference mismatch | Fix metadata inference |
| Refactor tests | Leftover refactor inconsistency | Direct fix |

**Triage rule:** If a failure disappears after P0, it was downstream. Do not plan work for it.

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
- [ ] `from computronium import Registry; Registry.all()` → 28 native models
- [ ] `KnowledgeBase("path.db")` works
- [ ] `./run_all_tests.sh` → <20 meaningful failures

### P1 Done
- [ ] Lightning/Hyperopt/Optuna/Smoke pass or have precise xfail reasons
- [ ] No accidental constructor/import failures

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
# P0 verification
uv run python -c "from computronium.models.native import registration; from computronium.core.registry import Registry; print(len(Registry._components.get('model', {})))"
# → 28

uv run python -c "from computronium.knowledge import KnowledgeBase; kb = KnowledgeBase('/tmp/test.db'); print('OK')"

# Full suite
./run_all_tests.sh

# Native smoke tests
uv run pytest tests/property/test_native_smoke.py -v

# Settle protocol
uv run pytest tests/integration/test_settle_protocol_models.py -v

# Joint benchmarks
uv run pytest tests/integration/joint/test_benchmarks.py -v

# Property locks (fast CI gate)
uv run pytest tests/property/test_ontology_locks.py -q
uv run pytest tests/property/joint/test_stability_metrics.py -q

# Energy invariants
uv run pytest tests/integration/test_energy_invariants.py -v

# Type check (policy: strict on ontology only)
uv run pyright computronium/ontology
```

---

## 📝 Notes

- **Legacy Zoo**: ~200K lines removed. 3 thin wrappers (`tile_models.py`, `tile_fa.py`, `tile_lm.py`) → **DELETE** (no deprecation, zero users).
- **Native Models**: 28 registered with explicit 5-D axes. Accessible via `Registry.get()` once `registration` loads.
- **Zero-Extension Invariant**: `M=NullPlasticity` slice formally verified (J1 test).
- **EqProp competitive**: 81.32% MNIST anchored via 20-epoch run (grad_clip + checkpointing).
- **ComputroniumLinear (CP-C)**: Drop-in `nn.Linear` wrapper complete, 26 tests.
- **torch.jit → torch.export**: Migration complete in `deployment.py`.