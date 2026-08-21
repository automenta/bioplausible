# Sprint 5 Development Plan — Certify the Hypercube & Upgrade the Seam

**Source**: RECRYSTALLIZE.md feedback + codebase verification  
**Status**: Planning — no implementation started  
**Constraint**: Zero campaigns; all work stays in fast-CI gate (GPU ≤ 5 min, CPU ≤ 10 min)

---

## Coverage Gap Analysis (Verified Against `bioplausible/core/ontology.py`)

| Axis | Certified by L1–L7 Locks | Uncertified (Exist in Ontology, Missing Locks) |
|------|--------------------------|-----------------------------------------------|
| **S** (Substrate) | `DigitalSubstrate`, `MemristiveSubstrate`, `OpticalSubstrate` | `NeuromorphicSubstrate`, `QuantumSubstrate` |
| **G** (Geometry) | `FeedforwardGeometry`, `RecurrentGeometry`, `TileGeometry` | `NeuromorphicFabric`, `SpatialLattice3D` (deferred) |
| **D** (Dynamics) | `InstantaneousDynamics`, `EnergyMinimizationDynamics`, `PredictiveSettlingDynamics` | `SpikeIntegrationDynamics` |
| **C** (Credit) | `BackpropCredit`, `ThermodynamicContrast`, `RandomProjectionsCredit` | `LocalGoodnessCredit`, `TargetInversionCredit`, `TemporalTraceCredit` |
| **U** (Update) | `EuclideanUpdate` | `RiemannianOrthogonalUpdate`, `SpectralConstrainedUpdate`, `NaturalGradientUpdate`, `ElasticConsolidationUpdate` |

**Verification**: All uncertified members exist as implemented classes in `ontology.py` (lines 1880–2973). The gap is strictly in the property-lock suite (`tests/property/test_ontology_locks.py`).

---

## Workstream A — Certify Remaining C & U Members (P1)

**Goal**: Extend the gradient-equivalence gate (`bioplausible/validation/gradient_check.py`) and add property locks for the uncertified CreditAssignment and ParameterUpdate primitives.

### A1 — LocalGoodnessCredit & TargetInversionCredit: Surrogate Objective Locks
**Files**: `bioplausible/core/ontology.py` (add method), `bioplausible/validation/gradient_check.py` (extend), `tests/property/test_ontology_locks.py` (new tests)

- **Problem**: These rules don't estimate the global gradient; finite-differencing the task loss tests the wrong thing.
- **Solution**: Each rule exposes a `surrogate_objective(free_state, nudged_state, geometry) -> Tensor` method. The lock finite-differences **that** (hard gate). Cosine alignment with true loss gradient recorded as KB fingerprint (metric, not gate).
- **Implementation**:
  1. Add `surrogate_objective` abstract method to `CreditAssignment` protocol (`ontology.py:424`)
  2. Implement in `LocalGoodnessCredit` (layer-local goodness: sum of `σ(h)²`) and `TargetInversionCredit` (local target MSE)
  3. Extend `check_gradient_equivalence` to accept optional `surrogate_loss_fn`; when provided, FD the surrogate and gate on that
  4. Add `TestC_SurrogateLocks` class in `test_ontology_locks.py` with parametrized tests for both credits
- **Acceptance**: Both pass FD gate (cosine ≥ 0.95 vs surrogate FD); KB records cosine vs true loss gradient

### A2 — TemporalTraceCredit: STDP Window Property Tests
**Files**: `bioplausible/core/ontology.py` (enhance), `tests/property/test_ontology_locks.py` (new tests)

- **Properties to certify** (all cheap, no training):
  1. **Causal potentiation**: pre→post spike (Δt > 0) ⇒ positive weight change
  2. **Anti-causal depression**: post→pre spike (Δt < 0) ⇒ negative weight change
  3. **Antisymmetry**: `W(Δt) = -W(-Δt)` for symmetric STDP window
  4. **Exponential decay**: `|W(Δt)| ∝ exp(-|Δt|/τ)`
- **Implementation**:
  1. Add `compute_stdp_window(pre_spikes, post_spikes, dt) -> Tensor` method to `TemporalTraceCredit` (returns per-Δt weight change)
  2. Add `TestC_TemporalTraceSTDP` with 4 parametrized tests injecting synthetic spike pairs
  3. Use `seeded()` fixture for determinism; assert mathematical properties on returned window
- **Acceptance**: All 4 properties hold within numerical tolerance (1e-6)

### A3 — U-Axis Step Property Tests
**Files**: `tests/property/test_ontology_locks.py` (new tests)

| Update Rule | Property | Test Method |
|-------------|----------|-------------|
| `RiemannianOrthogonalUpdate` (Muon) | Preserves orthogonality of constrained block | `Q = update.step(params, grads); assert (Q.T @ Q ≈ I).all()` |
| `SpectralConstrainedUpdate` | Lipschitz bound ≤ 1 after step | `svd_max(update.step(...)) ≤ 1.0 + 1e-6` |
| `ElasticConsolidationUpdate` | Protected params bitwise untouched | `torch.equal(old_protected, new_protected)` when `ewc_lambda > 0` |
| `NaturalGradientUpdate` | Whitening idempotent on own output | `F⁻¹(F(g)) ≈ g` (diagonal Fisher) |

- **Implementation**: Add `TestU_StepProperties` class with 4 parametrized tests. Each constructs a tiny 2-layer geometry, runs one `step()`, asserts the property.
- **Acceptance**: All 4 pass; wall-clock < 10s total

---

## Workstream B — Certify Remaining D & S Members (P2)

### B1 — SpikeIntegrationDynamics: Lyapunov Lock
**Files**: `bioplausible/core/ontology.py` (verify implementation), `tests/property/test_ontology_locks.py` (new tests)

- **Property**: Under fixed input, (a) membrane potentials bounded, (b) per-window spike counts converge (variance monotonically decreasing across settling steps).
- **Implementation**:
  1. Verify `SpikeIntegrationDynamics.settle()` tracks per-step spike counts (add `spike_history: list[Tensor]` to state if needed)
  2. Add `TestD_SpikeIntegrationLyapunov` with test injecting constant current, running `max_steps=50`, asserting:
     - `max(potential) < V_thresh + margin` for all steps
     - `var(spike_counts[step:])` non-increasing
- **Acceptance**: Both properties hold; test < 5s

### B2 — NeuromorphicSubstrate: Passivity Lock
**Files**: `bioplausible/core/ontology.py` (verify `inject_state_noise`), `tests/property/test_ontology_locks.py` (new tests)

- **Property**: Noise injection is non-expansive: `‖N(a) − N(b)‖ ≤ ‖a − b‖` for random pairs `a, b`.
- **Implementation**: Add `TestS_NeuromorphicPassivity` generating 100 random tensor pairs, applying `inject_state_noise`, asserting 2-norm contraction.
- **Acceptance**: Holds for all 100 pairs; test < 2s

### B3 — QuantumSubstrate: Parameter-Shift Equivalence
**Files**: `bioplausible/core/ontology.py` (verify `get_weight_update_operator`), `tests/property/test_ontology_locks.py` (new tests)

- **Property**: Parameter-shift gradient ≡ finite difference on a tiny circuit (2 qubits, 1 parameter).
- **Implementation**:
  1. Ensure `QuantumSubstrate.get_weight_update_operator` implements parameter-shift rule
  2. Add `TestS_QuantumParameterShift` comparing parameter-shift gradient vs central FD on a 1-parameter rotation gate
- **Acceptance**: Cosine ≥ 0.999; test < 3s

---

## Workstream C — Upgrade L7 Seam to Real Transport (P2)

**Goal**: The gRPC layer (`bioplausible/p2p/grpc_service.py`) is tested only in-process. First real socket contact.

### C1 — Integration Test: Multi-Process TileMeshService
**File**: `tests/integration/test_grpc_seam.py` (new)

- **Scenario**: Spawn 2–3 subprocesses running `TileMeshService` (via `GRPCServer`), each hosting a shard of a tiny `TileGeometry` (2 layers × 2 tiles/layer, 8 neurons/tile).
- **Test**: Run one distributed training step via `DistributedSystemTrainer` (in-process reference) vs. multi-process gRPC. Compare final metrics (loss, accuracy) within `LOOSE` tolerance (rel_diff ≤ 0.1, abs_diff ≤ 1e-3).
- **Implementation**:
  1. Use `multiprocessing` or `subprocess` to launch servers on `localhost:50051`, `50052`, `50053`
  2. `GRPCConnectionPool` connects client to all peers
  3. Run 1 epoch, 1 batch; collect metrics from rank 0
  4. Compare against single-process `DistributedSystemTrainer` on identical geometry/seed
- **Acceptance**: Metrics match within `LOOSE`; no serialization errors; test < 30s

### C2 — Fault Injection: Worker Kill Mid-Step
**File**: `tests/integration/test_grpc_seam.py` (extend)

- **Scenario**: Same 3-worker setup. At step boundary (after boundary sync, before parameter update), SIGKILL worker 1.
- **Assertion**: Remaining workers either (a) cleanly recover and complete epoch with structured `worker_lost` metric, or (b) halt with explicit `DistributedTrainingError` — **never** silent corruption (NaN metrics, hanging, partial updates applied).
- **Implementation**:
  1. In test, `os.kill(pid, signal.SIGKILL)` at precise synchronization point
  2. Catch `grpc.RpcError` / `asyncio.TimeoutError` in client
  3. Verify structured error or clean recovery path in `DistributedSystemTrainer`
- **Acceptance**: No silent corruption; explicit failure mode documented in test name

---

## Workstream D — First Native Migration: `eqprop_*` Family (P3, Parallel)

**Goal**: Migrate `eqprop_*` models to native 5-D Protocols per Phase-3 table. Strangler Fig: registry names stable, L1 parity gates swap, legacy path tagged deprecated.

### D1 — Inventory & Parity Baseline
**Files**: `bioplausible/zoo/models/eqprop/` (scan), `bioplausible/core/registry.py` (metadata)

- List all `eqprop_*` registered models (`Registry.list_models(ComponentCategory.MODEL)` filter)
- For each, run `ModelAdapter.validate()` (Sprint 4) to establish L1 parity baseline
- Document which already pass (green) vs. need fixes

### D2 — Native Protocol Implementation
**Files**: `bioplausible/zoo/models/eqprop/` (new native modules), `bioplausible/core/registry.py` (re-register)

For each model family, create native implementation composing:
- `S=DigitalSubstrate()`, `G=RecurrentGeometry(...)`, `D=EnergyMinimizationDynamics(...)`, `C=ThermodynamicContrast(...)`, `U=EuclideanUpdate(...)`

| Legacy Model | Native Composition | Effort |
|--------------|-------------------|--------|
| `eqprop` / `standard_eqprop` | `Digital ⊗ Recurrent ⊗ EnergyMinimization ⊗ ThermodynamicContrast ⊗ Euclidean` | Low |
| `deep_ep` / `directed_ep` | Same + deeper `RecurrentGeometry` | Low |
| `finite_nudge_ep` | Same + `beta` config | Low |
| `lazy_eqprop` | Same + `LazyStateDynamics` (new, thin wrapper) | Medium |
| `homeostatic_eqprop` | Same + `HomeostaticCredit` (new, extends `ThermodynamicContrast`) | Medium |
| `conv_eqprop` / `modern_conv_eqprop` | `G=ConvRecurrentGeometry` (new) | High (defer) |
| `transformer_eqprop` / `causal_transformer_eqprop` | `G=AttentionGeometry` (new) | High (defer) |

- **Migration Rule**: New native class registered under **same name**; old class moved to `bioplausible/zoo/models/eqprop/_legacy/` with metadata tag `status:deprecated:superseded_by_native_protocol`
- **Gate**: `ModelAdapter.validate(rtol=0.05, atol=1e-3)` must pass before legacy removal

### D3 — Registry & CLI Stability
- `Registry.get("eqprop")` returns native implementation
- `biopl run --model eqprop` unchanged
- `Registry.to_system("eqprop")` returns identical 5-D coordinate
- Deprecation warning emitted once per process when legacy path touched

---

## Explicitly Deferred (Not in Sprint 5)

- Hypercube campaigns (AutoScientist search over uncertified axes)
- Scaling benchmarks (multi-GPU, multi-node)
- Multi-host P2P (Kademlia bootstrap, NAT traversal)
- Hardware/SPICE validation (memristive IR-drop vs SPICE, optical phase noise vs hardware)
- `fabric/3D` geometries (`NeuromorphicFabric`, `SpatialLattice3D`)
- `ConvRecurrentGeometry`, `AttentionGeometry` (new G-axis members)

---

## Exit Criterion: "Don't Jinx It" → Satisfied Precondition

**Campaigns begin when every coordinate the proposer can name is machine-certified.**

After Sprint 5, a sweep over the hypercube can only compose rules that each carry their own equivalence or Lyapunov proof. The fast-CI gate (`pytest tests/property/test_ontology_locks.py -q`) becomes the certification authority.

### Acceptance Checklist (Extended from RECRYSTALLIZE.md)

```bash
# 1. Property locks (fast CI gate) — NOW INCLUDES A1–A3, B1–B3
uv run pytest tests/property/test_ontology_locks.py -q

# 2. Core + integration suites — NOW INCLUDES test_grpc_seam.py
uv run pytest tests/unit/core/test_ontology.py tests/integration/test_gradient_equivalence.py tests/integration/test_energy_invariants.py tests/integration/test_grpc_seam.py -q

# 3. Type checking
uv run pyright

# 4. Linting & formatting
uv run ruff format --check . && uv run ruff check .

# 5. Full suite
uv run pytest tests/ -q

# 6. Wall-clock budget check (record in PR)
# GPU ≤ 5 min, CPU ≤ 10 min for lock suite (including new tests)
```

---

## Additional Gaps & Opportunities Discovered During Verification

### Gap 1: `CreditAssignment.surrogate_objective` Missing from Protocol
The `CreditAssignment` protocol (`ontology.py:424`) lacks the `surrogate_objective` method required for Workstream A1. **Must add** before implementing LocalGoodness/TargetInversion locks.

### Gap 2: `SpikeIntegrationDynamics` Doesn't Track Spike History
Current implementation (`ontology.py:2547`) returns only final activations. Need to expose per-step spike counts for Lyapunov test. **Minimal change**: add `spike_counts: list[Tensor]` to `SystemState` or return via `state.metrics`.

### Gap 3: `QuantumSubstrate.get_weight_update_operator` Uses Simplified Update
Current implementation (`ontology.py:2207`) does `current_w - 0.01 * pseudo_grad`, not true parameter-shift rule. **Must implement** proper parameter-shift for B3 to be meaningful.

### Gap 4: `DistributedSystemTrainer` Fault Tolerance Path Undefined
No structured `worker_lost` recovery or halt logic exists (`distributed_trainer.py`). Workstream C2 requires designing this. **Decision needed**: fail-fast with `DistributedTrainingError` vs. checkpoint-based recovery.

### Gap 5: `ModelAdapter.validate()` Tolerance for eqprop May Be Too Tight
EqProp's stochastic settling may not achieve `rtol=0.05` vs legacy. May need family-specific tolerances or increased `max_steps` in native composition.

### Opportunity 1: Unify `surrogate_objective` with KB Fingerprinting
The KB (`bioplausible/knowledge/kb.py`) can store `(rule_name, surrogate_cosine, true_gradient_cosine)` tuples. AutoScientist can then query "which credit rules have high surrogate fidelity but low true-gradient alignment?" — a signal for architectural mismatch.

### Opportunity 2: Property Locks as Living Documentation
Each lock test name (`test_l3a_thermodynamic_contrast_local`, `test_u_muon_orthogonality`) is a executable spec. Consider generating a markdown matrix from test discovery for the docs.

### Opportunity 3: gRPC Seam Test as Foundation for Phase 4
`test_grpc_seam.py` establishes the multi-process test harness. Phase 4 (real P2P) can extend it with Kademlia bootstrap, multi-host, and chaos testing without new infrastructure.

---

## File Map for Implementation

```
bioplausible/core/ontology.py
  ├─ Add CreditAssignment.surrogate_objective (A1)
  ├─ Enhance SpikeIntegrationDynamics spike tracking (B1)
  ├─ Implement QuantumSubstrate parameter-shift (B3)
  └─ Verify NeuromorphicSubstrate.inject_state_noise contract (B2)

bioplausible/validation/gradient_check.py
  ├─ Extend check_gradient_equivalence for surrogate_loss_fn (A1)
  └─ Add MetricRule entries for LocalGoodness, TargetInversion (A1)

tests/property/test_ontology_locks.py
  ├─ TestC_SurrogateLocks (A1)
  ├─ TestC_TemporalTraceSTDP (A2)
  ├─ TestU_StepProperties (A3)
  ├─ TestD_SpikeIntegrationLyapunov (B1)
  ├─ TestS_NeuromorphicPassivity (B2)
  └─ TestS_QuantumParameterShift (B3)

tests/integration/test_grpc_seam.py          (NEW — C1, C2)

bioplausible/zoo/models/eqprop/
  ├─ _legacy/                                (D2 — legacy moved here)
  ├─ native_eqprop.py                        (D2 — new native)
  ├─ native_deep_ep.py                       (D2)
  ├─ native_finite_nudge.py                  (D2)
  └─ __init__.py                             (D2 — re-export natives)

bioplausible/core/registry.py
  ├─ Update metadata tags for deprecated legacy (D2)
  └─ Ensure to_system() projects natives correctly (D3)

bioplausible/core/distributed_trainer.py
  └─ Add fault-tolerance path for worker loss (C2)
```

---

## Dependency Graph

```
A1 (surrogate_objective protocol) ──┐
                                    ├─→ A1 tests (LocalGoodness, TargetInversion)
A2 (STDP window method) ────────────┤
                                    ├─→ A2 tests
A3 (U-axis properties) ─────────────┤
                                    ├─→ A3 tests
B1 (SpikeIntegration spike history) ──┤
                                    ├─→ B1 tests
B2 (Neuromorphic passivity) ────────┤
                                    ├─→ B2 tests
B3 (Quantum parameter-shift) ───────┤
                                    └─→ B3 tests

C1 (gRPC multi-process test) ───────┐
                                    └─→ C2 (fault injection) [depends on C1 infra]

D1 (inventory + validate baseline) ──┐
                                    ├─→ D2 (native implementations)
D2 (native eqprop_*) ────────────────┤
                                    └─→ D3 (registry/CLI stability)

All workstreams independent; can run in parallel.
```

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| `surrogate_objective` breaks existing CreditAssignment impls | Low | Medium | Default implementation raises `NotImplementedError`; only LocalGoodness/TargetInversion override |
| gRPC test flaky on CI (port conflicts) | Medium | High | Use `port=0` (OS assigns), read actual port from server; retry logic |
| EqProp native parity fails (stochastic settling variance) | Medium | Medium | Increase `max_steps` in native composition; use `rtol=0.1` for EqProp family |
| Quantum parameter-shift not implemented correctly | Medium | Low | B3 test is the spec; implement until test passes |
| `DistributedSystemTrainer` fault path undefined | High | Medium | Time-box C2: if no clean design in 1 day, fail-fast with explicit error is acceptable for Sprint 5 |

---

## Notes for Implementers

1. **No new dependencies**. All tests use existing `torch`, `pytest`, `grpc`, `multiprocessing`.
2. **Keep tests fast**. Each new property test < 5s. Use tiny geometries (WIDTH=32, DEPTH=2).
3. **Determinism**: All new tests use `seeded()` fixture and `select_device()` from `tests/property/_support.py`.
4. **Protocol signatures**: Do not change `CreditAssignment.compute_pseudo_gradient` or `ParameterUpdate.step` signatures. Add `surrogate_objective` as *new* abstract method with default `raise NotImplementedError`.
5. **Registry stability**: `Registry.get()` and CLIs must not change. Native migration is internal.
6. **Documentation**: Update `docs/api/ontology.md` and `docs/CORRECTNESS_LOCK.md` after each workstream lands.