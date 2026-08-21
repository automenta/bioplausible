# Sprint 5 Development Plan — Certify the Hypercube & Upgrade the Seam (REVISED)

**Source**: RECRYSTALLIZE.md feedback + codebase verification + TODO_REVIEW.md  
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
| **U** (Update) | *(none — see Inconsistency 7)* | `EuclideanUpdate`, `RiemannianOrthogonalUpdate`, `SpectralConstrainedUpdate`, `NaturalGradientUpdate`, `ElasticConsolidationUpdate` |

**Verification**: All uncertified members exist as implemented classes in `ontology.py` (lines 1880–2973). The gap is strictly in the property-lock suite (`tests/property/test_ontology_locks.py`).

**Note**: `EuclideanUpdate` and `BackpropCredit` are used in L1/L7 but lack dedicated property tests — moved to uncertified column per Inconsistencies 7–8.

---

## Phase 0 — Critical Gap Fixes (P0, Sequential, ~3–5 Days)

*Must complete before any workstream can start cleanly.*

### G1 — `CreditAssignment.surrogate_objective` Default Method
**File**: `bioplausible/core/ontology.py:424`

```python
@runtime_checkable
class CreditAssignment(Protocol):
    config: CreditAssignmentConfig
    
    @abstractmethod
    def compute_pseudo_gradient(...): ...
    
    # DEFAULT METHOD — non-breaking, only overridden by LocalGoodness/TargetInversion
    def surrogate_objective(self, free_state, nudged_state, geometry) -> Tensor:
        raise NotImplementedError("Surrogate objective not defined for this credit rule")
```

**Acceptance**: All 6 existing credit classes pass `isinstance(x, CreditAssignment)` without changes.

---

### G2 — Surrogate Test Harness (`check_surrogate_equivalence`)
**File**: `bioplausible/validation/gradient_check.py` (new function)

```python
def check_surrogate_equivalence(
    name: str,
    credit: CreditAssignment,
    free_state: SystemState,
    nudged_state: SystemState,
    geometry: Geometry,
    threshold: float = 0.95,
) -> tuple[float, float]:
    """
    Per-layer surrogate FD check.
    
    1. Get per-layer pseudo-gradients from credit.compute_pseudo_gradient
    2. Get per-layer surrogate losses from credit.surrogate_objective
    3. For each layer: FD the surrogate w.r.t that layer's weights
    4. Compare rule direction vs surrogate FD gradient (cosine ≥ threshold)
    """
    # Implementation: iterate layers, FD each surrogate scalar
    # Returns (mean_surrogate_cosine, mean_true_gradient_cosine) for KB fingerprint
```

**Acceptance**: Function exists, called by A1 tests; returns two cosines for KB.

---

### G3 — `TemporalTraceCredit` STDP Implementation
**File**: `bioplausible/core/ontology.py:2805`

```python
class TemporalTraceCredit:
    def __init__(self, config: CreditAssignmentConfig | None = None):
        self.config = config or CreditAssignmentConfig(credit_type="temporal_trace")
        self._pre_spike_times: dict[int, Tensor] = {}  # layer -> spike times
        self._post_spike_times: dict[int, Tensor] = {}
    
    def record_spikes(self, layer_idx: int, pre_spikes: Tensor, post_spikes: Tensor) -> None:
        """Call from SpikeIntegrationDynamics during settling."""
        self._pre_spike_times[layer_idx] = pre_spikes
        self._post_spike_times[layer_idx] = post_spikes
    
    def compute_stdp_window(self, pre_spikes: Tensor, post_spikes: Tensor, dt: Tensor) -> Tensor:
        """Return weight change per Δt bin. Exponential STDP window."""
        # Δt = post - pre; causal (Δt>0) => LTP; anti-causal (Δt<0) => LTD
        # W(Δt) = A_plus * exp(-Δt/τ_plus) for Δt>0; -A_minus * exp(Δt/τ_minus) for Δt<0
        ...
    
    def compute_pseudo_gradient(...):  # Implement using recorded spikes
        ...
```

**Acceptance**: Class has working STDP; `compute_stdp_window` returns per-Δt weights.

---

### G4 — `QuantumSubstrate` Parameter-Shift Implementation
**File**: `bioplausible/core/ontology.py:2201`

```python
class QuantumSubstrate:
    def get_weight_update_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        def parameter_shift_update(pseudo_grad: Tensor, current_w: Tensor) -> Tensor:
            # Classical simulation of parameter-shift rule for 1-qubit rotation
            # For each parameter θ: ∇f(θ) ≈ [f(θ+π/2) - f(θ-π/2)] / 2
            # pseudo_grad is the "target direction"; we return the parameter-shift estimate
            # Simplified: assume current_w encodes rotation angles; shift each by ±π/2
            shifted_plus = self._evaluate_circuit(current_w + math.pi/2)
            shifted_minus = self._evaluate_circuit(current_w - math.pi/2)
            param_shift_grad = (shifted_plus - shifted_minus) / 2
            return current_w - self.config.step_size * param_shift_grad
        return parameter_shift_update
    
    def _evaluate_circuit(self, params: Tensor) -> Tensor:
        """Classical simulation of parameterized quantum circuit."""
        # Minimal: 1 qubit, RY(θ), measure <Z>
        return torch.cos(params)  # <Z> = cos(θ) for RY(θ)|0>
```

**Acceptance**: `get_weight_update_operator` returns parameter-shift estimate; B3 test passes.

**Note**: No external quantum dependency — classical simulation of 1-qubit circuit suffices for property test.

---

### G5 — `SpikeIntegrationDynamics` Spike History Tracking
**Files**: `bioplausible/core/ontology.py:202` (SystemState), `2547` (settle)

```python
# SystemState (line 227)
@dataclass(frozen=False, slots=True)
class SystemState:
    ...
    spike_counts: list[Tensor] | None = None  # ADD: per-step per-neuron spike counts
    ...

# SpikeIntegrationDynamics.settle() (line 2547)
def settle(self, state, geometry, substrate, target=None):
    ...
    spike_counts = []
    for step in range(self.config.max_steps):
        h_new = geometry.route(h)
        spikes = (h_new > 1.0).float()  # Threshold crossing
        spike_counts.append(spikes.sum(dim=0))  # (n_neurons,) per step
        h_new = torch.where(h_new > 1.0, torch.zeros_like(h_new), h_new)
        ...
    state.spike_counts = spike_counts
    if state.metrics is not None:
        state.metrics['spike_counts'] = spike_counts
    return state
```

**Acceptance**: `state.spike_counts` populated; B1 test can assert variance monotonicity.

---

### G6 — `DistributedSystemTrainer` Fault-Tolerance Design (Fail-Fast)
**File**: `bioplausible/core/distributed_trainer.py`

```python
class DistributedTrainingError(RuntimeError):
    def __init__(self, message: str, lost_workers: list[str], step: int, partial_metrics: dict | None = None):
        self.lost_workers = lost_workers
        self.step = step
        self.partial_metrics = partial_metrics
        super().__init__(message)

# In training loop:
try:
    boundary_sync = await self._sync_boundaries(step)
except (grpc.RpcError, asyncio.TimeoutError) as e:
    lost = self._identify_lost_workers()
    raise DistributedTrainingError(
        f"Worker communication failed at step {step}",
        lost_workers=lost,
        step=step,
        partial_metrics=self._collect_partial_metrics()
    )
```

**Acceptance**: `DistributedTrainingError` defined; raised on gRPC failure with metadata; C2 test can assert structured halt.

---

## Phase 1 — Test Logic Corrections (P1, Can Parallelize with Phase 0)

*Fix test assertions to match actual implementation semantics.*

### C7 — Add `EuclideanUpdate` / `BackpropCredit` Property Tests
**File**: `tests/property/test_ontology_locks.py`

```python
class TestU_EuclideanProperties:
    def test_euclidean_momentum_accumulates(self):
        update = EuclideanUpdate(ParameterUpdateConfig(step_size=0.01, momentum=0.9))
        params = {"w": torch.randn(10, 10)}
        grads = [torch.randn(10, 10)]
        # First step
        p1 = update.step(params, grads, None)
        # Second step with same grad
        p2 = update.step(p1, grads, None)
        # Momentum buffer should cause larger second step
        assert (params["w"] - p2["w"]).norm() > (params["w"] - p1["w"]).norm() * 1.5

class TestC_BackpropCreditProperties:
    def test_backprop_credit_matches_autograd(self):
        # Use existing gradient_check machinery but as property lock
        from bioplausible.validation.gradient_check import check_gradient_equivalence
        ...
```

**Acceptance**: Both have dedicated property tests in lock suite.

---

### C9 — Neuromorphic Passivity Test: Deterministic Noise Comparison
**File**: `tests/property/test_ontology_locks.py`

```python
def test_s_neuromorphic_passivity(self):
    substrate = NeuromorphicSubstrate()
    # Use SAME noise seed for both inputs
    with seeded(42):
        a = torch.randn(4, 32)
        b = torch.randn(4, 32)
        # Capture noise state
        torch.manual_seed(42)
        na = substrate.inject_state_noise(a)
    with seeded(42):
        nb = substrate.inject_state_noise(b)
    # Now ‖na - nb‖ ≤ ‖a - b‖ (deterministic noise cancels)
    assert torch.norm(na - nb) <= torch.norm(a - b) + 1e-6
```

**Acceptance**: Test passes on current implementation.

---

### C10 — Muon Test: Gradient Orthogonalization, Not Param Orthogonality
**File**: `tests/property/test_ontology_locks.py`

```python
def test_u_muon_gradient_orthogonal(self):
    update = RiemannianOrthogonalUpdate(ParameterUpdateConfig(ortho_steps=5))
    params = {"w": torch.randn(10, 10)}
    grads = [torch.randn(10, 10)]
    # Test the internal orthogonalization
    ortho_grad = update._newton_schulz(grads[0])
    assert torch.allclose(ortho_grad.T @ ortho_grad, torch.eye(10), atol=1e-5)
```

---

### C11 — Elastic Test: Params Move Toward Old Params
**File**: `tests/property/test_ontology_locks.py`

```python
def test_u_elastic_moves_toward_old_params(self):
    update = ElasticConsolidationUpdate(ParameterUpdateConfig(ewc_lambda=1000.0))
    params = {"w": torch.randn(10, 10)}
    grads = [torch.randn(10, 10)]
    # Consolidate first
    update.consolidate(params, {"w": torch.ones(10, 10)})  # importance=1
    old_w = params["w"].clone()
    new_params = update.step(params, grads, None)
    # Delta should have negative dot product with (w - old_w)
    delta = new_params["w"] - params["w"]
    assert (delta * (params["w"] - old_w)).sum() < 0
```

---

## Workstream A — Certify Remaining C & U Members (P2, Parallel)

*After Phase 0–1 complete.*

### A1 — LocalGoodnessCredit & TargetInversionCredit: Surrogate Objective Locks
**Files**: `ontology.py` (implement), `gradient_check.py` (use new harness), `test_ontology_locks.py` (tests)

```python
# LocalGoodnessCredit.surrogate_objective:
def surrogate_objective(self, free_state, nudged_state, geometry):
    # Layer-local goodness: sum of σ(h)^2 for positive pass, minimize for negative
    if free_state.activations and isinstance(free_state.activations, list):
        total_goodness = sum((torch.sigmoid(act)**2).sum() for act in free_state.activations[1:])
        return total_goodness
    return torch.tensor(0.0)

# TargetInversionCredit.surrogate_objective:
def surrogate_objective(self, free_state, nudged_state, geometry):
    # Local target MSE: ‖h_l - target_l‖^2 per layer
    ...
```

**Test**: `TestC_SurrogateLocks` calls `check_surrogate_equivalence` for both; asserts cosine ≥ 0.95; KB records both cosines.

---

### A2 — TemporalTraceCredit: STDP Window Property Tests
**Files**: `ontology.py` (add method), `test_ontology_locks.py` (tests)

```python
# In TemporalTraceCredit:
def compute_stdp_window(self, pre_spikes: Tensor, post_spikes: Tensor, dt: Tensor) -> Tensor:
    """Return weight change per Δt bin for given spike trains."""
    # pre_spikes: (n_pre, n_spikes_pre), post_spikes: (n_post, n_spikes_post)
    # Compute all pairwise Δt, bin, apply exponential window
    ...

# Tests (4 parametrized):
@pytest.mark.parametrize("pre_time,post_time,expected_sign", [
    (0.0, 5.0, +1),   # Causal pre→post => potentiation
    (5.0, 0.0, -1),   # Anti-causal post→pre => depression
    (0.0, 0.0, 0),    # Simultaneous => zero (antisymmetry)
])
def test_stdp_causal_potentiation(pre_time, post_time, expected_sign):
    ...
def test_stdp_antisymmetry():
    W_dt = credit.compute_stdp_window(...)
    W_neg_dt = credit.compute_stdp_window(...)
    assert torch.allclose(W_dt, -W_neg_dt, atol=1e-6)
def test_stdp_exponential_decay():
    # Fit exp(-|Δt|/τ) to |W(Δt)|
    ...
```

---

### A3 — U-Axis Step Property Tests (Corrected)
**File**: `tests/property/test_ontology_locks.py`

| Update Rule | Property | Test |
|-------------|----------|------|
| `RiemannianOrthogonalUpdate` | Gradient orthogonalized | `ortho_grad.T @ ortho_grad ≈ I` |
| `SpectralConstrainedUpdate` | Gradient svd_max ≤ 1 | `svd(update._orthogonalize_or_project(grad)) ≤ 1.0` |
| `NaturalGradientUpdate` | Fisher whitening idempotent | `F⁻¹(F(g)) ≈ g` (diagonal) |
| `ElasticConsolidationUpdate` | Params move toward old_params | `(Δw)·(w-old_w) < 0` |

---

## Workstream B — Certify Remaining D & S Members (P2, Parallel)

### B1 — SpikeIntegrationDynamics: Lyapunov Lock
**File**: `test_ontology_locks.py`

```python
def test_d_spike_integration_lyapunov(self):
    sys = _make_spike_system()  # Digital + Feedforward + SpikeIntegration + ...
    x, y = tiny_batch(42)
    state = SystemState(x=x, y=y)
    state.activations = sys.geometry.forward(x, sys.substrate)
    state = sys.dynamics.settle(state, sys.geometry, sys.substrate, target=None)
    
    spike_counts = state.spike_counts  # List[Tensor] per step
    assert spike_counts is not None
    
    # (a) Membrane potentials bounded
    for step_acts in state.activations_history:  # Need to track this too
        assert step_acts.max() < 1.5  # V_thresh + margin
    
    # (b) Spike count variance non-increasing
    totals = [sc.sum().item() for sc in spike_counts]
    for i in range(1, len(totals)):
        assert np.var(totals[i:]) <= np.var(totals[i-1:]) + 1e-6
```

**Note**: Requires `activations_history` in `SystemState` or exposed via `state.metrics`.

---

### B2 — NeuromorphicSubstrate: Passivity Lock (Deterministic)
**File**: `test_ontology_locks.py` — uses C9 fix (same seed)

---

### B3 — QuantumSubstrate: Parameter-Shift Equivalence
**File**: `test_ontology_locks.py`

```python
def test_s_quantum_parameter_shift(self):
    substrate = QuantumSubstrate()
    update_op = substrate.get_weight_update_operator()
    # 1-parameter circuit: current_w = θ, pseudo_grad = 1.0 (arbitrary)
    current_w = torch.tensor([0.5])  # θ = 0.5 rad
    pseudo_grad = torch.tensor([1.0])
    
    # Parameter-shift estimate
    updated = update_op(pseudo_grad, current_w)
    param_shift_step = current_w - updated  # ∝ param_shift_grad
    
    # Finite difference on <Z> = cos(θ)
    eps = 1e-4
    fd_grad = (torch.cos(current_w + eps) - torch.cos(current_w - eps)) / (2*eps)
    
    # Direction alignment
    cos = F.cosine_similarity(param_shift_step.unsqueeze(0), fd_grad.unsqueeze(0)).item()
    assert cos >= 0.999
```

---

## Workstream C — Upgrade L7 Seam to Real Transport (P2, Parallel)

### C1 — Multi-Process gRPC Integration Test
**File**: `tests/integration/test_grpc_seam.py` (NEW)

```python
import multiprocessing
import grpc
import time

def run_server(port, geometry_shard, node_id, ready_event, barrier):
    server = GRPCServer(geometry_shard, node_id, port)
    asyncio.run(server.start())
    ready_event.set()
    barrier.wait()  # Wait for all servers + client ready
    asyncio.run(server.stop())

@pytest.mark.integration
def test_grpc_seam_multi_process():
    # 1. Build tiny TileGeometry (2 layers × 2 tiles × 8 neurons)
    geometry = TileGeometry(...)
    shards = geometry.shard(num_workers=3)  # Need shard method
    
    # 2. Spawn 3 server processes
    ports = [0, 0, 0]  # OS assigns
    ready_events = [multiprocessing.Event() for _ in range(3)]
    barrier = multiprocessing.Barrier(4)  # 3 servers + 1 client
    processes = []
    for i, shard in enumerate(shards):
        p = multiprocessing.Process(
            target=run_server, args=(ports[i], shard, f"node_{i}", ready_events[i], barrier)
        )
        p.start()
        processes.append(p)
    
    # 3. Wait for servers, get actual ports
    for ev in ready_events:
        ev.wait(timeout=10)
    actual_ports = [get_actual_port(p) for p in processes]  # Via shared memory or file
    
    # 4. Client connects via GRPCConnectionPool
    pool = GRPCConnectionPool(geometry, "client", port=actual_ports[0])
    await pool.start_server()
    for i, port in enumerate(actual_ports):
        if i > 0:
            pool.add_peer(f"node_{i}", f"localhost:{port}")
    
    barrier.wait()  # All ready
    
    # 5. Run 1 distributed step
    x, y = tiny_batch(42)
    trainer = DistributedSystemTrainer(pool, ...)
    metrics = await trainer.train_step(x, y)
    
    # 6. Compare vs single-process SystemTrainer
    ref_trainer = SystemTrainer(...)
    ref_metrics = ref_trainer.train_step(x, y)
    
    # LOOSE tolerance
    assert abs(metrics["loss"] - ref_metrics["loss"]) <= 1e-3
    assert abs(metrics["accuracy"] - ref_metrics["accuracy"]) <= 0.1
    
    # 7. Cleanup
    for p in processes:
        p.terminate()
        p.join(timeout=5)
```

**Acceptance**: Metrics match within LOOSE; no serialization errors; test < 30s.

---

### C2 — Fault Injection: Worker Kill Mid-Step
**File**: `tests/integration/test_grpc_seam.py` (extend)

```python
def test_grpc_seam_fault_injection():
    # Same setup as C1 with 3 workers
    ...
    # After boundary sync, before param update, kill worker 1
    p1 = processes[1]
    os.kill(p1.pid, signal.SIGKILL)
    
    # Client should raise DistributedTrainingError
    with pytest.raises(DistributedTrainingError) as exc_info:
        await trainer.train_step(x, y)
    
    assert exc_info.value.lost_workers == ["node_1"]
    assert exc_info.value.step == 1
    assert exc_info.value.partial_metrics is not None
```

---

## Workstream D — First Native Migration: `eqprop_*` Family (P3, Parallel)

### D1 — Inventory & Parity Baseline
**Script**: `scripts/inventory_eqprop.py`

```python
from bioplausible.core.registry import Registry, ComponentCategory
models = Registry.list_models(ComponentCategory.MODEL)
eqprop_models = [m for m in models if m.startswith("eqprop") or "ep" in m]
for name in eqprop_models:
    adapter = ModelAdapter(Registry.get_model(name), Registry.get_metadata(...))
    result = adapter.validate(rtol=0.1, atol=1e-2)  # Family-specific
    print(f"{name}: passed={result['passed']}, diffs={result['differences']}")
```

---

### D2 — Native Protocol Implementation
**Files**: `bioplausible/zoo/models/eqprop/` (new native modules)

| Legacy Model | Native Composition | Status |
|--------------|-------------------|--------|
| `eqprop` / `standard_eqprop` | `Digital ⊗ Recurrent ⊗ EnergyMinimization ⊗ ThermodynamicContrast ⊗ Euclidean` | P2 |
| `deep_ep` / `directed_ep` | Same + deeper `RecurrentGeometry` | P2 |
| `finite_nudge_ep` | Same + `beta` config | P2 |
| `lazy_eqprop` | Same + `LazyStateDynamics` (NEW) | P2 |
| `homeostatic_eqprop` | Same + `HomeostaticCredit` (NEW) | P2 |
| `conv_eqprop` / `modern_conv_eqprop` | `G=ConvRecurrentGeometry` (NEW) | **DEFER** |
| `transformer_eqprop` / `causal_transformer_eqprop` | `G=AttentionGeometry` (NEW) | **DEFER** |

**New Classes Needed** (implement in `ontology.py` or `eqprop/`):
- `LazyStateDynamics` — thin wrapper around `EnergyMinimizationDynamics` with lazy evaluation
- `HomeostaticCredit` — extends `ThermodynamicContrast` with homeostatic regulation

**Migration Rule**: 
- New native class registered under **same name** via `@register_model`
- Old class moved to `_legacy/` with metadata tag `status:deprecated:superseded_by_native_protocol`
- Gate: `ModelAdapter.validate(rtol=0.1, atol=1e-2)` passes

---

### D3 — Registry & CLI Stability
- `Registry.get("eqprop")` returns native implementation
- `biopl run --model eqprop` unchanged
- `Registry.to_system("eqprop")` returns identical 5-D coordinate
- Deprecation warning emitted once per process when legacy path touched

---

## High-Value Opportunities (Integrated)

### O1 — L0 Config Schema Lock (Add to Lock Suite)
**File**: `tests/property/test_ontology_locks.py`

```python
def test_l0_config_schema_roundtrip():
    for config_cls in [SubstrateConfig, GeometryConfig, StateDynamicsConfig, CreditAssignmentConfig, ParameterUpdateConfig]:
        cfg = config_cls()  # Default
        # Round-trip via Registry
        system = compose_system_from_configs(cfg)
        cfg2 = extract_config(system)
        assert cfg == cfg2  # Frozen dataclass equality
```

---

### O2 — KB Integration for Gradient Fingerprints
**File**: `bioplausible/validation/gradient_check.py`

```python
# In check_gradient_equivalence and check_surrogate_equivalence:
from bioplausible.knowledge.kb import KB
kb = KB()
kb.record_gradient_fingerprint(
    family=name,
    fd_cosine=fd_cos,
    rule_cosine=rule_cos,
    surrogate_cosine=surrogate_cos,  # None if not applicable
    threshold=threshold,
    timestamp=datetime.now().isoformat()
)
```

---

### O3 — Lock Matrix Generator (Script)
**File**: `scripts/gen_lock_matrix.py`

```python
# Discovers test_* in test_ontology_locks.py, parses L1/L2/L3a... from names
# Generates docs/CORRECTNESS_LOCK_MATRIX.md with badges
# CI step: diff matrix vs test count
```

---

### O4 — Family-Specific Tolerances in ModelAdapter
**File**: `bioplausible/core/ontology.py:1803`

```python
FAMILY_TOLERANCES = {
    "eqprop": (0.1, 1e-2),
    "fa": (0.05, 1e-3),
    "backprop": (0.01, 1e-4),
    "predictive_coding": (0.1, 1e-2),
}

def validate(self, x=None, y=None, rtol=None, atol=None):
    if rtol is None or atol is None:
        family = self._infer_family()
        rtol, atol = FAMILY_TOLERANCES.get(family, (0.05, 1e-3))
    ...
```

---

### O5 — gRPC Test Port Allocation
**File**: `tests/integration/test_grpc_seam.py`

```python
# Use port=0, read actual port
server = GRPCServer(..., port=0)
await server.start()
actual_port = server._server._port  # Or use socket.getsockname()
```

---

## Explicitly Deferred (Not in Sprint 5)

- Hypercube campaigns (AutoScientist search over uncertified axes)
- Scaling benchmarks (multi-GPU, multi-node)
- Multi-host P2P (Kademlia bootstrap, NAT traversal)
- Hardware/SPICE validation (memristive IR-drop vs SPICE, optical phase noise vs hardware)
- `fabric/3D` geometries (`NeuromorphicFabric`, `SpatialLattice3D`)
- `ConvRecurrentGeometry`, `AttentionGeometry` (new G-axis members)
- Checkpoint-based fault recovery (Option B) — Phase 4

---

## Exit Criterion: "Don't Jinx It" → Satisfied Precondition

**Campaigns begin when every coordinate the proposer can name is machine-certified.**

After Sprint 5, a sweep over the hypercube can only compose rules that each carry their own equivalence or Lyapunov proof. The fast-CI gate (`pytest tests/property/test_ontology_locks.py -q`) becomes the certification authority.

### Acceptance Checklist (Extended)

```bash
# 0. Critical gap fixes verified
uv run pytest tests/property/test_ontology_locks.py::TestL0ConfigSchema -q
uv run pytest tests/unit/core/test_ontology.py -k "surrogate or stdp or quantum or spike" -q

# 1. Property locks (fast CI gate) — NOW INCLUDES A1–A3, B1–B3, C7, C9–C11, O1
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

## Dependency Graph (Revised)

```
G1 (surrogate_objective default) ──┐
                                   ├─→ G2 (check_surrogate_equivalence)
G3 (TemporalTraceCredit STDP) ─────┤
                                   ├─→ A1 (surrogate locks)
G4 (Quantum parameter-shift) ──────┤
                                   ├─→ A2 (STDP window tests)
G5 (SpikeIntegration spike_counts) ──┤
                                   ├─→ A3 (U-axis corrected)
G6 (DistributedTrainingError) ──────┤
                                   ├─→ B1 (Spike Lyapunov)
C7 (Euclidean/Backprop props) ──────┤
                                   ├─→ B2 (Neuromorphic passivity)
C9 (Neuromorphic deterministic) ────┤
                                   ├─→ B3 (Quantum param-shift)
C10 (Muon gradient ortho) ──────────┤
                                   └─→ C1 (gRPC multi-process)
C11 (Elastic toward old_params) ─────┘      └─→ C2 (fault injection)

O1–O5 (opportunities) — independent, can parallelize anytime

D1 (inventory) → D2 (native impls) → D3 (registry)
  (needs LazyStateDynamics, HomeostaticCredit from P2)
```

---

## File Map for Implementation

```
bioplausible/core/ontology.py
  ├─ G1: CreditAssignment.surrogate_objective default method
  ├─ G3: TemporalTraceCredit STDP implementation
  ├─ G4: QuantumSubstrate parameter-shift
  ├─ G5: SystemState.spike_counts + SpikeIntegrationDynamics tracking
  ├─ O4: ModelAdapter.FAMILY_TOLERANCES
  └─ D2: LazyStateDynamics, HomeostaticCredit (new classes)

bioplausible/validation/gradient_check.py
  ├─ G2: check_surrogate_equivalence() new function
  └─ O2: KB integration in both check functions

tests/property/test_ontology_locks.py
  ├─ C7: TestU_EuclideanProperties, TestC_BackpropCreditProperties
  ├─ C9: Neuromorphic passivity with same seed
  ├─ C10: Muon gradient orthogonalization test
  ├─ C11: Elastic toward old_params test
  ├─ A1: TestC_SurrogateLocks (uses check_surrogate_equivalence)
  ├─ A2: TestC_TemporalTraceSTDP (4 parametrized)
  ├─ A3: TestU_StepProperties (corrected)
  ├─ B1: TestD_SpikeIntegrationLyapunov
  ├─ B2: TestS_NeuromorphicPassivity
  ├─ B3: TestS_QuantumParameterShift
  └─ O1: TestL0ConfigSchemaRoundtrip

tests/integration/test_grpc_seam.py (NEW)
  ├─ C1: Multi-process TileMeshService test (port=0)
  └─ C2: Fault injection with DistributedTrainingError

bioplausible/core/distributed_trainer.py
  └─ G6: DistributedTrainingError + fail-fast logic

bioplausible/zoo/models/eqprop/
  ├─ _legacy/ (old classes moved here)
  ├─ native_eqprop.py, native_deep_ep.py, native_finite_nudge.py
  ├─ native_lazy_eqprop.py, native_homeostatic_eqprop.py
  └─ __init__.py (re-export natives)

scripts/
  ├─ inventory_eqprop.py (D1)
  └─ gen_lock_matrix.py (O3)

docs/CORRECTNESS_LOCK_MATRIX.md (generated by O3)
```

---

## Risk Mitigation (Updated)

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| `surrogate_objective` default breaks type checking | Low | Medium | Protocol default methods supported by Pyright 1.1+; test with `pyright` |
| `check_surrogate_equivalence` complexity | Medium | High | Start simple: 1-layer MLP, 1 surrogate scalar; extend |
| TemporalTraceCredit STDP design wrong | Medium | Medium | Implement minimal pairwise Δt first; iterate |
| Quantum parameter-shift needs quantum sim | Low | High | Classical 1-qubit cos(θ) simulation sufficient for property test |
| gRPC test flaky (port=0 race) | Medium | High | Use `multiprocessing.Barrier` + shared `Queue` for port passing |
| EqProp native parity fails (variance) | Medium | Medium | Family tolerance `rtol=0.1`; increase `max_steps` in native |
| `LazyStateDynamics`/`HomeostaticCredit` scope creep | Medium | Low | Time-box: thin wrappers only; defer complex logic |

---

## Notes for Implementers

1. **Phase 0 is sequential** — G1→G2→G3→G4→G5→G6. Do not start workstreams until all 6 pass.
2. **No new dependencies**. All tests use existing `torch`, `pytest`, `grpc`, `multiprocessing`.
3. **Keep tests fast**. Each new property test < 5s. Use tiny geometries (WIDTH=32, DEPTH=2).
4. **Determinism**: All new tests use `seeded()` fixture and `select_device()` from `tests/property/_support.py`.
5. **Protocol signatures**: Do not change `compute_pseudo_gradient` or `step` signatures. `surrogate_objective` is default method.
6. **Registry stability**: `Registry.get()` and CLIs must not change. Native migration is internal.
7. **Documentation**: Update `docs/api/ontology.md`, `docs/CORRECTNESS_LOCK.md` after each workstream lands.
8. **KB writes**: Use `bioplausible.knowledge.kb.KB()` — thread-safe, file-backed.