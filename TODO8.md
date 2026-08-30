# TODO8.md — Comprehensive Test & Capability Parity Plan

> **Scope:** Achieve a functioning, well-tested codebase leveraging the new Ontology (not Legacy "Zoo") API/architecture. Based on complete test run results from `run_all_tests.sh` (67 failed, 1387 passed, 97 skipped, 33 xfailed, 4 xpassed).

---

## 📊 Executive Summary

**Current State:**
- **Core ontology & native models**: Working (28 native models, 5-D compositions functional)
- **Registry**: Empty unless `computronium.models.native.registration` is explicitly imported
- **KnowledgeBase**: Constructor broken (expects `KnowledgeBaseConfig`, tests pass `str`)
- **Lightning integration**: Uses `ComponentCategory.MODEL` which exists but registry is unpopulated
- **Test coverage**: ~16.8% (meets ≥15% floor), but many integration tests failing due to missing registrations
- **Geometry gaps**: No `ConvGeometry`, `GraphGeometry`, `AttentionGeometry` — Conv/Graph/Attention EqProp gone

---

## 🔴 Critical Failures (Blocking)

### 1. Registry Auto-Population Failure
**Impact:** 28 native models registered only on explicit import of `registration.py`. Tests fail because registry is empty.
**Root Cause:** `__init__.py` lazy-imports individual factory functions, not the registration module.
**Fix:** Import registration module at package load time, or auto-import on first registry access.

### 2. KnowledgeBase Constructor Signature Mismatch
**Impact:** 44 test failures in `tests/unit/test_knowledge.py`
```python
# Current: requires KnowledgeBaseConfig
kb = KnowledgeBase(config=KnowledgeBaseConfig(db_path="..."))
# Tests pass: str
kb = KnowledgeBase(db_path="...")
```
**Fix:** Accept both `str` and `KnowledgeBaseConfig` in `__init__`.

### 3. Lightning Integration Registry Lookup Failure
**Impact:** 8 test failures in `tests/integration/test_lightning_integration.py`
```python
# lightning_/module.py line 35
cls = Registry.get(ComponentCategory.MODEL, name)  # Returns empty registry
```
**Fix:** Ensure registry is populated before Lightning module use.

### 4. Hyperopt/Optuna Bridge Failures
**Impact:** 12 test failures across `test_hyperopt_integration.py`, `test_optuna_bridge_integration.py`
**Root Cause:** Model instantiation fails due to empty registry.

### 5. Smoke All Tasks Failures (Vision/RL/LM)
**Impact:** 18 test failures in `test_smoke_all_tasks.py`
**Root Cause:** Same registry issue + data loading issues.

### 6. Continual Learning Test Failure
**Impact:** 1 failure in `test_continual_learning.py::test_ewc_single_task_learning`

### 7. Ontology ModelAdapter Test Failures
**Impact:** 2 failures in `test_ontology.py::TestModelAdapter`

### 8. Refactor Test Failures
**Impact:** 2 failures in `test_refactor.py`

---

## 🟡 High Priority (Capability & Verification)

### 9. Property Test Migrations (from TODO7.md Phase B)
| Test File | Status | Required |
|-----------|--------|----------|
| `test_ontology_parity.py` | 30 passed, 1 skipped, 2 xfailed | ✅ Mostly done |
| `test_biology_axioms.py` | 7/9 passing | ✅ Mostly done |
| `test_scaling_invariants.py` | 5 passed, 3 skipped, 3 xfailed | Migrate remaining |
| `test_settle_protocol.py` | 6 passed | ✅ Done |
| `test_validation_all.py` | 2 passed, 14 skipped | Document skips better |

### 10. Native Model Smoke Tests
**Current:** `test_native_smoke.py` — 20 passed, 4 skipped, 4 xfailed
**Target:** All 28 native models have working `forward()` + `train_step()` smoke tests.

### 11. Settle Protocol Integration
**Current:** 21 passing (was 18, restored from 29)
**Target:** Full 29+ passing with TileAlgorithm convergence fixes.

---

## 🟠 Medium Priority (Architecture & Performance)

### 12. Legacy Kernel Porting to Substrate Operator API (P2b)
| Kernel Type | Status | Target |
|-------------|--------|--------|
| Triton kernels (eqprop, FA, etc.) | **PENDING** | Port to `Substrate.get_forward_operator()` |
| CUDA kernels (MEP) | **PENDING** | Port to Substrate or custom Autograd Function |
| Custom backward (EquilibriumFunction) | **PENDING** | Verify compatibility with native System |
| Sparse/Ternary quantization | **PENDING** | Port to `Substrate.quantize_weights()` |

### 13. Geometry Build-Out Decision (P3 — Explicit Decision Required)
| Geometry | Need For | Effort | Decision |
|----------|----------|--------|----------|
| `ConvGeometry` | Vision (ConvEqProp, ConvTileNet) | Medium | **Defer** unless science needs |
| `GraphGeometry` | Graph EqProp, GNN tasks | Medium | **Defer** |
| `AttentionGeometry` | Transformer EqProp, LM variants | High | **Defer** |
| 3D Spatial Lattice | Neural Cube | High | **Defer** |

**Recommendation:** Defer all. Phase 5/6 science runs on Feedforward/Recurrent/Tile at MLP scale.

### 14. Pyright Strict Mode
**Current:** ~4315 errors
**Target:** ≤1000 (deprioritized behind functional work)

### 15. Ignored Test Files in `run_all_tests.sh`
These 8 files are permanently ignored. Audit each:
| File | Reason Ignored | Action |
|------|----------------|--------|
| `test_hardware_aware.py` | Legacy imports | Migrate or delete |
| `test_benchmarks.py` (joint) | Not implemented | Implement or delete |
| `test_diffusion_integration.py` | DiffusionDynamics gradient bug | Fix or skip |
| `test_energy_invariants.py` | Legacy imports | Migrate |
| `test_equitile_sparsity_robustness.py` | Legacy imports | Migrate |
| `test_dht.py` | Environment issues | Fix infra or skip |
| `test_grpc_seam.py` | gRPC infra issues | Fix infra or skip |
| `test_grpc_seam_subprocess.py` | gRPC infra issues | Fix infra or skip |

---

## 🟢 Lower Priority (Polish & Completeness)

### 16. Coverage Improvements
- Target: ≥25% (currently ~16.8%)
- Focus: Ontology primitives, SystemTrainer, JointSystemTrainer, Registry

### 17. Slow Test Optimization
Identify tests >10s and optimize without sacrificing coverage:
- Hypothesis property tests with large example counts
- Integration tests with full training loops
- Consider `@pytest.mark.slow` separation

### 18. Redundant/Outdated Tests
Audit and remove:
- Legacy zoo model tests (already mostly removed)
- Duplicate parity tests
- Tests for removed capabilities (Conv EqProp, Graph EqProp, Transformer EqProp)

### 19. Untested Functionality
Add coverage for:
- JointSystemTrainer (6-D joint training loop)
- Plasticity primitives (Routing, FastWeight, RuleState, SubstrateCoupled)
- AutoScientist campaign execution
- P2P distributed training
- Model export (ONNX, TorchScript, INT8, Ternary)
- Inference server (FastAPI, TensorRT)
- Stability monitoring (spectral radius, Lyapunov, basin stability)

---

## 📋 Phased Execution Plan

### Phase 0: Critical Fixes (Week 1) — **DO FIRST**
- [ ] Fix Registry auto-population (import registration module in `__init__.py`)
- [ ] Fix KnowledgeBase constructor to accept `str | KnowledgeBaseConfig`
- [ ] Verify Lightning integration works with populated registry
- [ ] Re-run full test suite — expect >1300 passing, <20 failing

### Phase 1: Integration Test Recovery (Week 1-2)
- [ ] Fix hyperopt/optuna bridge tests
- [ ] Fix smoke_all_tasks tests (vision, RL, LM)
- [ ] Fix continual learning test
- [ ] Fix ModelAdapter tests
- [ ] Fix refactor tests
- [ ] Audit and migrate/remove 8 ignored test files

### Phase 2: Property Test Completion (Week 2)
- [ ] Complete `test_scaling_invariants.py` migration
- [ ] Restore settle protocol integration to 29+ passing
- [ ] Improve `test_validation_all.py` skip documentation
- [ ] Achieve 28/28 native model smoke tests passing

### Phase 3: Kernel Porting (Week 2-3)
- [ ] Port Triton kernels to Substrate operator API
- [ ] Port CUDA kernels
- [ ] Verify EquilibriumFunction compatibility
- [ ] Port quantization kernels

### Phase 4: Geometry Decision & Defer (Week 3)
- [ ] Document explicit decision to defer Conv/Graph/Attention geometry
- [ ] Close P3 items with "Deferred" status
- [ ] Update capability map in README

### Phase 5: Coverage & Polish (Week 3-4)
- [ ] Add missing test coverage for JointSystemTrainer, Plasticity, AutoScientist, P2P, Export, Inference, Stability
- [ ] Optimize slow tests
- [ ] Remove redundant tests
- [ ] Push coverage to ≥25%

### Phase 6: Pyright & CI (Week 4)
- [ ] Reduce pyright errors to ≤1000
- [ ] Ensure `ruff format --check`, `ruff check`, `pyright`, `pytest --cov` all pass in CI
- [ ] Update `run_all_tests.sh` to remove unnecessary ignores

---

## ✅ Acceptance Criteria (Definition of Done)

| Criterion | Target |
|-----------|--------|
| **Test Pass Rate** | ≥95% (excluding known xfail/skip with documented reasons) |
| **Critical Failures** | 0 (Registry, KnowledgeBase, Lightning, Hyperopt, Smoke tasks) |
| **Native Model Coverage** | 28/28 smoke tests passing |
| **Property Locks** | All 32 ontology locks passing |
| **Settle Protocol** | ≥29 integration tests passing |
| **Coverage** | ≥25% |
| **Pyright Errors** | ≤1000 |
| **Ignored Test Files** | 0 (all migrated, fixed, or deleted with justification) |
| **CI Gate** | `ruff format --check` → `ruff check` → `pyright` → `pytest --cov` all pass |

---

## 🎯 Success Metrics

1. **`run_all_tests.sh`** exits 0 with <20 failures (all documented xfail/skip)
2. **`comp lab benchmark --domain vision --quick`** runs end-to-end
3. **`comp scientist --campaign ...`** executes AutoScientist campaign
4. **`comp joint-validate --coordinate ...`** validates arbitrary 6-D coordinates
5. All 13 factory functions (`create_*_mlp`) work with `SystemTrainer`
6. Registry queries (`Registry.query_axis(...)`) return native models

---

## 📝 Notes & Context

- **TODO7.md** modularization DoD is ✅ COMPLETE — this plan focuses on verification & capability parity
- **Legacy Zoo**: ~200K lines removed (13 modules). 3 modules kept (`tile_models.py`, `tile_fa.py`, `tile_lm.py`) as thin wrappers
- **Native Models**: 28 registered with explicit 5-D ontology axes. Accessible via `Registry.get()` once registration module loads
- **Zero-Extension Invariant**: `M=NullPlasticity` slice formally verified (J1 test)
- **Science vs Product**: Geometry build-out is a **fork** — defer unless Phase 5/6 science demands it

---

## 🔧 Quick Commands for Verification

```bash
# Full test suite
uv run pytest tests/ -q --tb=no -q 2>&1 | tail -20

# Property locks (fast CI gate)
uv run pytest tests/property/test_ontology_locks.py -q
uv run pytest tests/property/joint/ -q

# Native smoke tests
uv run pytest tests/property/test_native_smoke.py -v

# Integration verification
uv run pytest tests/integration/test_validation_all.py -v
uv run pytest tests/integration/test_settle_protocol_models.py -v

# Type checking
uv run pyright .

# Coverage
uv run pytest --cov=computronium --cov-report=term-missing
```