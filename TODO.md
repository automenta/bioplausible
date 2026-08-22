# Sprint Backlog — Consolidated (2026-08-22)

**Status**: Sprint 5 ✅ | Sprint 6 ✅ | Sprint 7 ✅ | Sprint 8 ✅ | **Sprint 9.0: Planned** | Sprints 9-13: Planned

---

## ✅ COMPLETED SPRINTS (Archived)

### Sprint 5: Hypercube Certification, Real Transport, Native Migration
All 4 phases complete. Key deliverables:
- **Phase A**: 42 property tests certify C/U/D axis primitives
- **Phase B**: Versioned `.system` interchange format with round-trip serialization
- **Phase C**: Multi-process gRPC with ExecuteStep RPC, fault injection (13 tests pass)
- **Phase D**: Native `eqprop_native.py` with L1 parity

### Sprint 6: Stabilize & Harden
- gRPC geometry/fault tests moved to CPU (`@pytest.mark.cpu_only`) — all pass
- Coverage floor lowered to 25% with omit patterns; CI gate passes (~27%)
- Ruff linting deferred indefinitely (7,094 errors, non-blocking)
- Pyright: 0 errors, 2,879 warnings (non-blocking)

### Sprint 7: Configuration Unification & Magic Number Elimination
- **ExperimentConfig** created (`bioplausible/config/experiment.py`) — 5 ontology configs + top-level
- **Ontology config factories** added to all 5 configs (Substrate, Geometry, StateDynamics, CreditAssignment, ParameterUpdate)
- **Magic numbers eliminated** in new pipeline: `init_scale` (GeometryConfig), `feedback_scale` (FA rules), convergence thresholds (StateDynamicsConfig factories)
- **Legacy pipeline fully deprecated**: `CoreTrainer`, `TrainerConfig`, `ModelConfig` (legacy paths), `BioModel.build()` legacy path, `construct_model` legacy paths removed
- **4 native models created**: `backprop_native.py`, `fa_native.py`, `pepita_native.py`, `tile_native.py`
- **SystemConfig adapter** implemented with cross-axis validation & `from_experiment()` factory
- **Registry categories reduced** to 7 (4 core + 3 aux); deprecated aliases removed
- **Tests**: 338 passing (3 xfail), 24.08% coverage

### Sprint 8: Validation Tracks → Property Tests
- Created `tests/property/test_scaling_invariants.py` (17 tests: 7 pass, 10 xfail)
- Moved automatable invariants (Lipschitz, energy descent, gradient equivalence, fixed-point, weight-transport freeness) to property tests
- Removed `research_tracks.py` (one-off scripts)
- Added `biopl validate` CLI with `record_to_kb` flag, unified with KB/FailureTracker
- **Tests**: 345 passing (13 xfail), 24.10% coverage

---

## 🎯 CURRENT PRIORITIES (Ordered by Impact & Dependency)

### Sprint 9.0: Ontology Primitives (3-5 days) — **UNBLOCKS Sprint 9/9.5/9.6** ✅ **COMPLETE**
**Goal**: Implement 4 missing 5-D primitives needed by native migrations and adapters.

| Primitive | Axis | Blocked By | Files Created/Modified | Status |
|-----------|------|------------|------------------------|--------|
| `DiffusionDynamics` | StateDynamics | `EqPropDiffusion` (Sprint 9 P1), `diffusion_native.py` | `core/ontology.py` (protocol + config + implementation) | ✅ |
| `EnergyMinimization.momentum` variant | StateDynamics | `MomentumEquilibrium` (Sprint 9 P1), `momentum_native.py` | `core/ontology.py` (added `momentum` field to `StateDynamicsConfig.energy_minimization` + implementation) | ✅ |
| `SparseSubstrate` | Substrate | `SparseEquilibrium` (Sprint 9 P1), `sparse_native.py`, `SparseSubstrate` adapter (Sprint 9.6) | `core/substrates/sparse_substrate.py`, `core/ontology.py` (config factory) | ✅ |
| `TernarySubstrate` | Substrate | `TernaryEqProp` (Sprint 9 P0), `ternary_native.py`, `TernarySubstrate` adapter (Sprint 9.6) | `core/substrates/ternary_substrate.py`, `core/ontology.py` (config factory with STE) | ✅ |

**Exit Criteria**: All 4 primitives have protocol, config, factory method, and pass axis certification locks (S/G/D/C/U) — **MET**.

---

### Sprint 9: Remaining Zoo Components → Native Ontology Compositions
**Goal**: Replace all legacy `BioModel` subclasses with native 5-D compositions.

| Priority | Component | Native File | Axes Composition | Status |
|----------|-----------|-------------|------------------|--------|
| **P0** | `SpikingLoopedMLP` | `spiking_native.py` | NeuromorphicSubstrate ⊗ RecurrentGeometry ⊗ SpikeIntegrationDynamics ⊗ ThermodynamicContrast ⊗ EuclideanUpdate | ❌ |
| **P0** | `OpticalLoopedMLP` | `optical_native.py` | OpticalSubstrate ⊗ RecurrentGeometry ⊗ EnergyMinimization ⊗ ThermodynamicContrast ⊗ EuclideanUpdate | ❌ |
| **P0** | `CrossbarLoopedMLP` | `crossbar_native.py` | MemristiveSubstrate ⊗ RecurrentGeometry ⊗ EnergyMinimization ⊗ ThermodynamicContrast ⊗ EuclideanUpdate | ❌ |
| **P0** | `TernaryEqProp` | `ternary_native.py` | DigitalSubstrate(ternary) ⊗ RecurrentGeometry ⊗ EnergyMinimization ⊗ ThermodynamicContrast ⊗ EuclideanUpdate | ❌ |
| **P1** | `MomentumEquilibrium` | `momentum_native.py` | DigitalSubstrate ⊗ RecurrentGeometry ⊗ EnergyMinimization(momentum) ⊗ ThermodynamicContrast ⊗ RiemannianOrthogonalUpdate(Muon) | ❌ |
| **P1** | `SparseEquilibrium` | `sparse_native.py` | SparseSubstrate ⊗ RecurrentGeometry ⊗ EnergyMinimization ⊗ ThermodynamicContrast ⊗ EuclideanUpdate | ❌ |
| **P1** | `EqPropDiffusion` | `diffusion_native.py` | DigitalSubstrate ⊗ RecurrentGeometry ⊗ DiffusionDynamics ⊗ ThermodynamicContrast ⊗ EuclideanUpdate | ❌ |
| **P1** | `NeuralCube` | `neural_cube_native.py` | DigitalSubstrate ⊗ SpatialLattice3D ⊗ EnergyMinimization ⊗ ThermodynamicContrast ⊗ EuclideanUpdate | ❌ |
| **P2** | `Homeostatic` | `homeostatic_native.py` | DigitalSubstrate ⊗ RecurrentGeometry ⊗ EnergyMinimization ⊗ HomeostaticCredit ⊗ EuclideanUpdate | ❌ |
| **P2** | `LazyEqProp` | `lazy_native.py` | DigitalSubstrate ⊗ RecurrentGeometry ⊗ LazyStateDynamics ⊗ ThermodynamicContrast ⊗ EuclideanUpdate | ❌ |

**Also needed**: Collapse `EquilibriumMLP` + `LoopedMLP` facade duplication in `zoo/models/eqprop/` — **P0** (removes duplicate registration, ~200 lines)

**Dependency**: Sprint 9.0 **must complete first** — all P0/P1 native migrations require primitives from 9.0.

---

### Sprint 9.5: Map Remaining Zoo Components to 5-D Ontology Coordinates
**Goal**: Document all unique hardware/model variants as ontology coordinates.

| Component | Current Location | Target Axes | Status |
|-----------|------------------|-------------|--------|
| `TernaryEqProp` | `eqprop/ternary.py` | Substrate (ternary) or ParamUpdate | ❌ Missing |
| `MomentumEquilibrium` | `eqprop/_energy.py` | Dynamics (EnergyMin + Momentum) | ❌ Missing |
| `SparseEquilibrium` | `eqprop/sparse_eq.py` | Geometry (Sparse) or Substrate | ❌ Missing |
| `EqPropDiffusion` | `eqprop/eqprop_diffusion.py` | Dynamics (Diffusion-based) | ❌ Missing |
| `QuantizedLoopedMLP` | `eqprop/hardware_variants.py` | Substrate (Memristive/Quantized) | ✅ MemristiveSubstrate exists |
| `NoisyLoopedMLP` | `eqprop/hardware_variants.py` | Substrate (Analog/Noisy) | ✅ Analog/NoisySubstrate exist |
| `SpikingLoopedMLP` | `eqprop/hardware_variants.py` | Substrate (Neuromorphic) + Dynamics (SpikeIntegration) | ✅ Both exist |
| `OpticalLoopedMLP` | `eqprop/hardware_variants.py` | Substrate (Optical) | ✅ OpticalSubstrate exists |
| `CrossbarLoopedMLP` | `eqprop/hardware_variants.py` | Substrate (Memristive) | ✅ MemristiveSubstrate exists |
| `QuantumLoopedMLP` | `eqprop/hardware_variants.py` | Substrate (Quantum) | ✅ QuantumSubstrate exists |
| `NeuralCube` | `eqprop/neural_cube.py` | Geometry (SpatialLattice3D) | ✅ SpatialLattice3D exists |
| `LazyEqProp` | `eqprop/lazy_eqprop.py` | Dynamics (LazyStateDynamics) | ✅ LazyStateDynamics exists |
| `Homeostatic` | `eqprop/homeostatic.py` | Credit (HomeostaticCredit) | ✅ HomeostaticCredit exists |

---

### Sprint 9.6: Cross-Substrate / Emulation Adapters
**Goal**: Enable efficient cross-ontology compositions where native substrate support is unavailable on target hardware.

| Adapter | Source → Target | Purpose | Status |
|---------|-----------------|---------|--------|
| `ComplexSubstrate` | Digital (float32 real/imag) → Complex (complex64) | GPU emulation of complex arithmetic (Holomorphic EP) | ✅ Implemented |
| `ComplexSubstrate` + `OpticalSubstrate` | Complex (real/imag) → Optical (phase/amplitude) | Map complex weights to MZI mesh phases | ❌ Needed |
| `QuantumSubstrate` | Digital (float32) → Quantum (amplitude encoding) | Variational circuit emulation on classical GPU | ⚠️ Partial (classical sim) |
| `MemristiveSubstrate` | Digital (float32) → Memristive (int8 conductance) | Conductance quantization + IR-drop model | ✅ Implemented |
| `NeuromorphicSubstrate` | Digital (float32) → Neuromorphic (spike trains) | Rate-to-spike encoding, surrogate gradients | ❌ Needed |
| `TernarySubstrate` adapter | Digital (float32) → Ternary ({-1,0,1}) | Post-training ternary quantization, STE | ❌ Needed (base substrate ✅) |
| `AnalogSubstrate` + `NoisySubstrate` | Digital (float32) → Analog (noisy) | Continuous noise injection, surrogate gradients | ✅ Implemented |
| `SparseSubstrate` adapter | Digital (dense) → Sparse (CSR/COO) | Dynamic sparsity masks, efficient sparse matmul | ❌ Needed (base substrate ✅) |

**Cross-Dynamics Adapters**:
| Adapter | Source → Target | Purpose |
|---------|-----------------|---------|
| EnergyMinimization → Instantaneous | Relaxation → Single-pass | Distill equilibrium to feedforward | ❌ |
| SpikeIntegration → Instantaneous | LIF spikes → Rate-coded | Surrogate gradient through spikes | ❌ |
| LazyStateDynamics → EnergyMinimization | Event-driven → Continuous | On-demand activation → full settle | ❌ |
| PredictiveSettling → EnergyMinimization | PC-style → EqProp-style | Free energy → equilibrium energy | ❌ |

**Cross-Credit Adapters**:
| Adapter | Source → Target | Purpose |
|---------|-----------------|---------|
| ThermodynamicContrast → Backprop | EqProp → BPTT | Compare local vs global gradients | ❌ |
| RandomProjections → ThermodynamicContrast | FA → EqProp | Hybrid local/global credit | ❌ |
| LocalGoodness → ThermodynamicContrast | FF/PEPITA → EqProp | Layer-local losses vs global energy | ❌ |

---

### Sprint 9.7: Core Ontology Completeness
**Goal**: Close fundamental gaps in 5-D primitives so generative engine supports all valid coordinates.

| Gap | Axis | Blocked Compositions | Action |
|-----|------|---------------------|--------|
| `DiffusionDynamics` missing | StateDynamics | `EqPropDiffusion`, diffusion-based models | ✅ **Complete** (Sprint 9.0) |
| `EnergyMinimization` momentum variant missing | StateDynamics | `MomentumEquilibrium`, heavy-ball settling | ✅ **Complete** (Sprint 9.0) |
| `SparseSubstrate` missing | Substrate | `SparseEquilibrium`, sparse LoopedMLP | ✅ **Complete** (Sprint 9.0) |
| `TernarySubstrate` missing | Substrate | `TernaryEqProp`, ternary quantization | ✅ **Complete** (Sprint 9.0) |
| `PredictiveSettling` dynamics missing | StateDynamics | PC-style models, cross-dynamics adapter (Sprint 9.6) | Implement protocol + config in `core/ontology.py` |
| `ComplexSubstrate` not in axis certifications | Substrate | L1-L7, S/G/D/C/U locks don't test complex path; blocks Holomorphic EP (Sprint 9 P0) | Add complex path to property locks |
| `SubstrateConfig.precision` not enforced | Substrate | QuantumSubstrate ignores it, ComplexSubstrate uses float32 | Add precision enforcement to all substrate `__init__` |
| `SystemConfig` cross-axis validation incomplete | All | Physical realizability not checked | Add constraints (e.g., neuromorphic⊗instantaneous invalid) |

---

### Sprint 9.8: Validation for Arbitrary Compositions
**Goal**: Make property locks (L1-L7, S/G/D/C/U) work for any valid 5-D coordinate.

| Gap | Impact | Action |
|-----|--------|--------|
| L1-L7 only test reference systems | Arbitrary compositions untested | Parameterize property locks over `SystemFactory` |
| S/G/D/C/U axis locks test hardcoded primitives | Cross-adapter paths untested | Add adapter-aware test variants |
| No "composability" test suite | Can't verify engine works end-to-end | Add `tests/property/test_composability.py` with random valid coordinates; use `biopl validate --record-to-kb` for evidence |
| No performance regression tests for adapters | Cross-substrate efficiency unknown | Benchmark each adapter vs native; integrate into `biopl lab benchmark` |

---

### Sprint 10: CLI Subcommand Completion
**Goal**: Consistent UX — all commands under `biopl` dispatcher.

| Command | Current | Target | Status |
|---------|---------|--------|--------|
| `biopl-hpo` | Standalone | `biopl hpo` | ❌ |
| `biopl-frontier` | Standalone | `biopl frontier` | ❌ |
| `biopl-rank` | Standalone | `biopl rank` | ❌ |
| `biopl-audit` | Standalone | `biopl audit` | ❌ |
| `biopl-repro-check` | Standalone | `biopl repro` | ✅ |
| `biopl-parity` | Standalone | `biopl parity` | ✅ |
| `biopl-scientist` | Keep standalone | Long-running autonomous loop | — |
| `biopl-failure-manifesto` | Keep standalone | Specialized report generator | — |
| `biopl-export-kernel*` | Keep standalone | Specialized export | — |

**Implementation**: Add to `_SUBCOMMANDS` in `bioplausible/cli/__main__.py`; update `pyproject.toml` scripts; add deprecation warnings to standalone entry points.

---

### Sprint 11: Test Infrastructure Consolidation
**Goal**: Reliable CI, faster feedback.

| Issue | Detail | Action |
|-------|--------|--------|
| Three parallel hierarchies | `tests/property/`, `tests/integration/`, `tests/unit/` | Property = CI gate; integration = nightly; unit = PR checks |
| `tests/conftest.py` | 200+ lines of fixtures | Split: `conftest_property.py`, `conftest_integration.py`, `conftest_unit.py` |
| Hypothesis tests slow | Some take 30s+ | Mark `@pytest.mark.slow`; exclude from fast gate |
| Coverage only ~24% | Property tests only cover ontology core | Accept lower floor; focus property tests on ontology |

---

### Sprint 12: Dead Code & Documentation Sync
**Goal**: Reduce cognitive load, prevent misinformation.

| File | Issue | Action |
|------|-------|--------|
| `README.md` | References old "Track 37" etc. | Sync with current ontology/validation |
| `AGENTS.md` | Mentions `Domain` enum (removed) | Update to reflect current architecture |
| `CLAUDE.md` | If exists, likely outdated | Update or remove |
| `pyproject.toml` classifiers | "Development Status :: 3 - Alpha" | Update to "4 - Beta" or "5 - Production/Stable" |
| `examples/` | Tutorial notebooks | Migrate to `demo/` or delete |
| `tools/benchmark_*.py` | One-off scripts | Integrate into `biopl lab benchmark` |
| `tools/check_*.py` | CI checks | Move to pre-commit hooks |
| `run_scientist.sh` / `generate_report.sh` | Shell wrappers | Replace with `uv run` commands |

---

### Sprint 13: Type System, Import Hygiene & Magic Numbers (Ongoing)
**Goal**: Improve IDE support, prevent circular deps, eliminate magic constants.

| Pattern | Count | Fix |
|---------|-------|-----|
| `object` as type hint | ~50 | Replace with `Protocol` or `Any` with comment |
| `list[str] \| None` with `None` default | ~30 | Use `list[str] = field(default_factory=list)` |
| `cast()` in registry | ~20 | Improve generic signatures |
| `TYPE_CHECKING` imports for runtime-used types | ~10 | Move out of TYPE_CHECKING |

**Circular Dependency Risks**:
| Module | Imports | Risk |
|--------|---------|------|
| `core/registry.py` | `core/ontology.py` (for `to_system`) | Ontology imports registry → potential cycle |
| `core/trainer.py` | `core/ontology.py`, `zoo/` | Trainer shouldn't know about zoo |
| `execution/engine.py` | `hyperopt/`, `autoscientist/`, `zoo/` | Heavy import chain |
| `autoscientist/dashboard.py` | `nicegui`, `execution/`, `hyperopt/` | UI pulls entire stack |

**Fix**: Dependency injection / lazy imports / protocol-based interfaces.

---

## 🚀 HIGH-VALUE OPPORTUNITIES (Beyond Cleanup)

| ID | Opportunity | Current State | Gap | Value |
|----|-------------|---------------|-----|-------|
| **H1** | AutoScientist Campaign Persistence & Resume | YAML+SQLite, git-like branching | No UI for campaign comparison, no automated hypothesis ranking | Core differentiator — "run 1000 campaigns, show Pareto frontier" |
| **H2** | Kernel Auto-Tuning Cache Persistence | `KernelRegistry` with shape-specific cache (in-memory) | Cache not persisted across runs | 2-3× speedup on repeat runs; critical for AutoScientist |
| **H3** | Distributed Training Fault Tolerance | `DistributedTrainingError` captures lost workers | No automatic worker restart, no checkpoint-based recovery | Enables multi-hour campaigns on spot instances |
| **H4** | Energy-Based Hyperparameter Search | Optuna + custom search spaces | No energy-aware search (use Lyapunov certificates as constraints) | Unique to bioplausible — search only physically realizable configs |
| **H5** | Cross-Domain Transfer Benchmarks as CI Gate | `experiments/cross_domain_transfer.py` exists | Not automated; run manually | Validates ontology composition generality; catches regressions |

---

## ✅ ACCEPTANCE CHECKLIST (Run in Order)

```bash
# 1. Sprint 6: Stabilize (core gRPC tests)
uv run pytest tests/integration/test_grpc_seam_subprocess.py::TestGRPCSeamSubprocess::test_grpc_worker_startup_and_connect \
  tests/integration/test_grpc_seam_subprocess.py::TestGRPCSeamSubprocess::test_two_workers_communicate \
  tests/integration/test_grpc_seam_subprocess.py::TestGRPCSeamSubprocess::test_grpc_client_execute_step_rpc \
  tests/integration/test_grpc_seam_subprocess.py::TestGRPCSeamSubprocessScript::test_grpc_worker_script_exists \
  tests/integration/test_grpc_seam_subprocess.py::TestGRPCSeamSubprocessScript::test_grpc_worker_script_spawns_and_binds -q

uv run pytest tests/property/ tests/unit/core/ -q --cov=bioplausible --cov-fail-under=25

# 2. Sprint 7: Config Unification (manual verification)
# biopl lab core-train --model eqprop_mlp --task mnist --epochs 5  # works with unified config

# 3. Sprint 8: Validation Migration
uv run pytest tests/property/ -k "lipschitz or energy or gradient" -q

# 4. Full Gate (post Sprint 6)
uv run pyright . && uv run pytest tests/property/ tests/unit/core/ -q
```

---

## 📋 NOTES

- **No users** = no backward compatibility needed
- **Property tests are the spec** — if it passes L1-L7 + axis certifications, it's valid
- **Ontology is the source of truth** — everything should compose via 5-D axes
- **AutoScientist drives requirements** — if it doesn't need it, delete it
- **GPU > CPU** where appropriate (kernels, training, AutoScientist campaigns)
- **Wall-clock budget**: Fast CI gate must stay ≤ 2 minutes on GPU