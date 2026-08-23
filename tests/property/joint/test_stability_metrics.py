"""Property tests for stability metrics (Sprint J3)."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from bioplausible.core.joint.context import SystemContext
from bioplausible.core.joint.state import CompositeState
from bioplausible.core.joint.transition import NullPlasticity, PlasticityConfig
from bioplausible.core.stability import (
    BasinStabilityEstimator,
    FrontierAggregator,
    FrontierRecord,
    LyapunovEstimator,
    ResourceUsage,
    SettlingMonitor,
    SpectralRadiusEstimator,
    estimate_basin_stability,
    estimate_basin_stability_multistart,
    estimate_lyapunov_exponent,
    estimate_spectral_radius,
    measure_settling_time,
)

# ============================================================
# Test Fixtures
# ============================================================


class MockTransition:
    """Mock transition for testing stability metrics."""

    def __init__(self, rho: float = 0.5, dim: int = 32):
        self.rho = rho
        self.dim = dim
        self._step_count = 0

    def __call__(self, z: CompositeState, context: SystemContext) -> CompositeState:
        x = z.activity["x"]
        batch_size = x.shape[0]

        # Simple linear dynamics: x_{t+1} = rho * x_t (deterministic)
        new_x = self.rho * x

        self._step_count += 1
        return CompositeState(
            activity={"x": new_x},
            plastic=z.plastic,
            substrate=z.substrate,
        )

    def reset(self):
        self._step_count = 0


class MockContractingTransition:
    """Mock transition that converges to a fixed point."""

    def __init__(self, fixed_point: Tensor | None = None, rate: float = 0.8):
        self.rate = rate
        self._fixed_point = fixed_point

    def __call__(self, z: CompositeState, context: SystemContext) -> CompositeState:
        x = z.activity["x"]
        if self._fixed_point is None:
            self._fixed_point = torch.zeros_like(x)
        new_x = self._fixed_point + self.rate * (x - self._fixed_point)
        return CompositeState(
            activity={"x": new_x},
            plastic=z.plastic,
            substrate=z.substrate,
        )


@pytest.fixture
def mock_context():
    """Create a minimal SystemContext for testing."""
    from bioplausible.core.joint.state import StateRegistry, StateVariable

    registry = StateRegistry()
    registry.register(StateVariable(name="x", persistent=True))

    theta = {"W": torch.randn(32, 32, requires_grad=True)}
    geometry = type("Geometry", (), {})()
    substrate = type("Substrate", (), {})()
    substrate_config = type("SubstrateConfig", (), {})()
    geometry_config = type("GeometryConfig", (), {})()
    dynamics_config = type("StateDynamicsConfig", (), {})()
    credit_config = type("CreditAssignmentConfig", (), {})()
    update_config = type("ParameterUpdateConfig", (), {})()
    plasticity_config = PlasticityConfig.null()

    return SystemContext(
        theta=theta,
        geometry=geometry,
        substrate=substrate,
        substrate_config=substrate_config,
        geometry_config=geometry_config,
        dynamics_config=dynamics_config,
        credit_config=credit_config,
        update_config=update_config,
        plasticity_config=plasticity_config,
        registry=registry,
    )


@pytest.fixture
def initial_state():
    """Create initial CompositeState."""
    return CompositeState(
        activity={"x": torch.randn(4, 32)},
        plastic={},
        substrate={},
    )


# ============================================================
# ResourceUsage Tests
# ============================================================


class TestResourceUsage:
    def test_default_construction(self):
        ru = ResourceUsage()
        assert ru.compute == 0.0
        assert ru.memory == 0.0
        assert ru.energy == 0.0
        assert ru.latency == 0.0
        assert ru.plastic_state_capacity == 0.0

    def test_construction_with_values(self):
        ru = ResourceUsage(
            compute=100.0,
            memory=50.0,
            energy=10.0,
            latency=5.0,
            plastic_state_capacity=1000,
        )
        assert ru.compute == 100.0
        assert ru.memory == 50.0
        assert ru.energy == 10.0
        assert ru.latency == 5.0
        assert ru.plastic_state_capacity == 1000

    def test_addition(self):
        ru1 = ResourceUsage(compute=100.0, memory=50.0)
        ru2 = ResourceUsage(compute=200.0, memory=30.0)
        ru3 = ru1 + ru2
        assert ru3.compute == 300.0
        assert ru3.memory == 50.0  # max

    def test_division(self):
        ru = ResourceUsage(
            compute=100.0,
            memory=50.0,
            energy=10.0,
            latency=5.0,
            plastic_state_capacity=1000,
        )
        ru2 = ru / 2.0
        assert ru2.compute == 50.0
        assert ru2.memory == 25.0
        assert ru2.energy == 5.0
        assert ru2.latency == 2.5
        assert ru2.plastic_state_capacity == 500

    def test_to_dict_from_dict(self):
        ru = ResourceUsage(
            compute=100.0,
            memory=50.0,
            energy=10.0,
            latency=5.0,
            plastic_state_capacity=1000,
        )
        d = ru.to_dict()
        ru2 = ResourceUsage.from_dict(d)
        assert ru2.compute == ru.compute
        assert ru2.memory == ru.memory
        assert ru2.energy == ru.energy
        assert ru2.latency == ru.latency
        assert ru2.plastic_state_capacity == ru.plastic_state_capacity


# ============================================================
# FrontierRecord Tests
# ============================================================


class TestFrontierRecord:
    def test_construction(self):
        ru = ResourceUsage(compute=100.0)
        fr = FrontierRecord(
            coordinate="digital/recurrent/energy_min/routing/thermo/euclidean",
            task_loss=0.5,
            adaptation_time=100,
            rho_jacobian=0.8,
            lyapunov_local=-0.1,
            settling_time=50.0,
            basin_stability=0.9,
            resources=ru,
            plasticity_primitive="routing",
        )
        assert fr.coordinate == "digital/recurrent/energy_min/routing/thermo/euclidean"
        assert fr.task_loss == 0.5
        assert fr.adaptation_time == 100
        assert fr.rho_jacobian == 0.8
        assert fr.lyapunov_local == -0.1
        assert fr.settling_time == 50.0
        assert fr.basin_stability == 0.9
        assert fr.plasticity_primitive == "routing"

    def test_is_stable(self):
        ru = ResourceUsage()
        fr_stable = FrontierRecord(
            coordinate="test",
            task_loss=0.5,
            adaptation_time=10,
            rho_jacobian=0.9,
            lyapunov_local=-0.1,
            settling_time=10,
            basin_stability=0.9,
            resources=ru,
        )
        fr_unstable = FrontierRecord(
            coordinate="test",
            task_loss=0.5,
            adaptation_time=10,
            rho_jacobian=1.1,
            lyapunov_local=0.1,
            settling_time=10,
            basin_stability=0.1,
            resources=ru,
        )
        assert fr_stable.is_stable()
        assert not fr_unstable.is_stable()

    def test_to_dict_from_dict(self):
        ru = ResourceUsage(compute=100.0)
        fr = FrontierRecord(
            coordinate="test",
            task_loss=0.5,
            adaptation_time=10,
            rho_jacobian=0.8,
            lyapunov_local=-0.1,
            settling_time=10,
            basin_stability=0.9,
            resources=ru,
            plasticity_primitive="routing",
        )
        d = fr.to_dict()
        fr2 = FrontierRecord.from_dict(d)
        assert fr2.coordinate == fr.coordinate
        assert fr2.task_loss == fr.task_loss
        assert fr2.rho_jacobian == fr.rho_jacobian
        assert fr2.plasticity_primitive == fr.plasticity_primitive


# ============================================================
# FrontierAggregator Tests
# ============================================================


class TestFrontierAggregator:
    def test_add_and_len(self):
        agg = FrontierAggregator()
        ru = ResourceUsage()
        for i in range(5):
            fr = FrontierRecord(
                coordinate=f"coord_{i}",
                task_loss=0.5,
                adaptation_time=10,
                rho_jacobian=0.8,
                lyapunov_local=-0.1,
                settling_time=10,
                basin_stability=0.9,
                resources=ru,
            )
            agg.add(fr)
        assert len(agg) == 5

    def test_get_best_by_objective(self):
        agg = FrontierAggregator()
        ru = ResourceUsage()
        for i, loss in enumerate([0.5, 0.3, 0.7, 0.2, 0.6]):
            fr = FrontierRecord(
                coordinate=f"coord_{i}",
                task_loss=loss,
                adaptation_time=10,
                rho_jacobian=0.8,
                lyapunov_local=-0.1,
                settling_time=10,
                basin_stability=0.9,
                resources=ru,
            )
            agg.add(fr)

        best = agg.get_best_by_objective("task_loss", maximize=False)
        assert best is not None
        assert best.task_loss == 0.2

    def test_clear(self):
        agg = FrontierAggregator()
        ru = ResourceUsage()
        fr = FrontierRecord(
            coordinate="test",
            task_loss=0.5,
            adaptation_time=10,
            rho_jacobian=0.8,
            lyapunov_local=-0.1,
            settling_time=10,
            basin_stability=0.9,
            resources=ru,
        )
        agg.add(fr)
        assert len(agg) == 1
        agg.clear()
        assert len(agg) == 0


# ============================================================
# Spectral Radius Tests
# ============================================================


class TestSpectralRadius:
    def test_estimate_spectral_radius_stable(self, mock_context, initial_state):
        transition = MockTransition(rho=0.5)
        rho = estimate_spectral_radius(
            transition, initial_state, mock_context, num_iterations=10
        )
        # Should be close to 0.5
        assert 0.3 < rho < 0.7

    def test_estimate_spectral_radius_unstable(self, mock_context, initial_state):
        transition = MockTransition(rho=1.2)
        rho = estimate_spectral_radius(
            transition, initial_state, mock_context, num_iterations=10
        )
        # Should be close to 1.2
        assert 0.9 < rho < 1.5

    def test_spectral_radius_estimator_class(self, mock_context, initial_state):
        estimator = SpectralRadiusEstimator(num_iterations=10, fast_mode=False)
        transition = MockTransition(rho=0.7)
        rho = estimator(transition, initial_state, mock_context)
        assert 0.5 < rho < 0.9

    def test_spectral_radius_fast_mode(self, mock_context, initial_state):
        estimator = SpectralRadiusEstimator(fast_mode=True)
        transition = MockTransition(rho=0.6)
        rho = estimator(transition, initial_state, mock_context)
        # Fast mode gives rough estimate
        assert rho >= 0.0


# ============================================================
# Lyapunov Exponent Tests
# ============================================================


class TestLyapunovExponent:
    def test_estimate_lyapunov_stable(self, mock_context, initial_state):
        transition = MockTransition(rho=0.5)
        lyap = estimate_lyapunov_exponent(
            transition, initial_state, mock_context, num_steps=20
        )
        # For linear system with rho < 1, Lyapunov exponent = log(rho) < 0
        assert lyap < 0

    def test_estimate_lyapunov_unstable(self, mock_context, initial_state):
        transition = MockTransition(rho=1.2)
        lyap = estimate_lyapunov_exponent(
            transition, initial_state, mock_context, num_steps=20
        )
        # For rho > 1, Lyapunov exponent = log(rho) > 0
        assert lyap > 0

    def test_lyapunov_estimator_class(self, mock_context, initial_state):
        estimator = LyapunovEstimator(num_steps=20, fast_mode=False)
        transition = MockTransition(rho=0.8)
        lyap = estimator(transition, initial_state, mock_context)
        assert lyap < 0

    def test_lyapunov_fast_mode(self, mock_context, initial_state):
        estimator = LyapunovEstimator(fast_mode=True)
        transition = MockTransition(rho=0.7)
        lyap = estimator(transition, initial_state, mock_context)
        assert isinstance(lyap, float)


# ============================================================
# Settling Time Tests
# ============================================================


class TestSettlingTime:
    def test_measure_settling_time_converges(self, mock_context):
        # Start far from fixed point
        z = CompositeState(
            activity={"x": torch.ones(4, 32) * 10.0}, plastic={}, substrate={}
        )
        transition = MockContractingTransition(rate=0.5)
        steps, norms = measure_settling_time(
            transition, z, mock_context, tolerance=1e-3, max_steps=100
        )
        assert steps < 100
        assert len(norms) == steps
        assert norms[-1] < 1e-3

    def test_measure_settling_time_max_steps(self, mock_context):
        # Non-converging transition (oscillating)
        class OscillatingTransition:
            def __init__(self):
                self.step_count = 0

            def __call__(
                self, z: CompositeState, context: SystemContext
            ) -> CompositeState:
                x = z.activity["x"]
                # Alternate between two states
                if self.step_count % 2 == 0:
                    new_x = x * 1.1
                else:
                    new_x = x * 0.9
                self.step_count += 1
                return CompositeState(
                    activity={"x": new_x}, plastic=z.plastic, substrate=z.substrate
                )

        transition = OscillatingTransition()
        z = CompositeState(activity={"x": torch.randn(4, 32)}, plastic={}, substrate={})
        steps, norms = measure_settling_time(
            transition, z, mock_context, tolerance=1e-6, max_steps=10
        )
        assert steps == 10
        assert len(norms) == 10

    def test_settling_monitor_class(self, mock_context):
        monitor = SettlingMonitor(tolerance=1e-3, max_steps=100, record_trajectory=True)
        transition = MockContractingTransition(rate=0.5)
        z = CompositeState(
            activity={"x": torch.ones(4, 32) * 10.0}, plastic={}, substrate={}
        )
        steps, norms, traj = monitor(transition, z, mock_context)
        assert steps < 100
        assert len(norms) == steps
        assert traj is not None
        assert len(traj) == steps + 1

    def test_settling_fast_proxy(self, mock_context):
        monitor = SettlingMonitor()
        transition = MockContractingTransition(rate=0.5)
        z = CompositeState(
            activity={"x": torch.ones(4, 32) * 10.0}, plastic={}, substrate={}
        )
        est = monitor.fast_proxy(transition, z, mock_context)
        assert isinstance(est, int)
        assert est <= monitor.max_steps


# ============================================================
# Basin Stability Tests
# ============================================================


class TestBasinStability:
    def test_estimate_basin_stability_stable(self, mock_context):
        # Attractor at origin, contracting dynamics
        z_attractor = CompositeState(
            activity={"x": torch.zeros(4, 32)}, plastic={}, substrate={}
        )
        transition = MockContractingTransition(rate=0.5)
        stability = estimate_basin_stability(
            transition,
            z_attractor,
            mock_context,
            num_samples=20,
            perturbation_radius=2.0,
            max_steps=50,
        )
        # Should have high basin stability for contracting system
        assert stability > 0.5

    def test_basin_stability_estimator_class(self, mock_context):
        estimator = BasinStabilityEstimator(num_samples=20, fast_mode=False)
        z_attractor = CompositeState(
            activity={"x": torch.zeros(4, 32)}, plastic={}, substrate={}
        )
        transition = MockContractingTransition(rate=0.5)
        stability = estimator(transition, z_attractor, mock_context)
        assert 0.0 <= stability <= 1.0

    def test_basin_fast_mode(self, mock_context):
        estimator = BasinStabilityEstimator(fast_mode=True)
        z_attractor = CompositeState(
            activity={"x": torch.zeros(4, 32)}, plastic={}, substrate={}
        )
        transition = MockContractingTransition(rate=0.5)
        stability = estimator(transition, z_attractor, mock_context)
        assert 0.0 <= stability <= 1.0

    def test_basin_multistart(self, mock_context):
        z_attractor = CompositeState(
            activity={"x": torch.zeros(4, 32)}, plastic={}, substrate={}
        )
        transition = MockContractingTransition(rate=0.5)
        results = estimate_basin_stability_multistart(
            transition,
            z_attractor,
            mock_context,
            num_samples=10,
            perturbation_radii=[0.5, 1.0, 2.0],
        )
        assert len(results) == 3
        # Stability should decrease with radius
        assert results[0.5] >= results[1.0] >= results[2.0]


# ============================================================
# Integration Tests: NullPlasticity with Stability
# ============================================================


class TestNullPlasticityStability:
    def test_null_plasticity_stability_metrics(self, mock_context, initial_state):
        """Test that NullPlasticity system can be analyzed for stability."""
        null_plasticity = NullPlasticity()

        def joint_transition(
            z: CompositeState, context: SystemContext
        ) -> CompositeState:
            # Null plasticity: psi unchanged, activity evolves
            psi_next = null_plasticity.step(z.plastic, z, context)
            # Simple activity dynamics with stronger contraction
            new_x = 0.5 * z.activity["x"]
            return CompositeState(
                activity={"x": new_x},
                plastic=psi_next,
                substrate=z.substrate,
            )

        # Test all stability metrics work
        rho = estimate_spectral_radius(
            joint_transition, initial_state, mock_context, num_iterations=5
        )
        assert 0.4 < rho < 0.6

        lyap = estimate_lyapunov_exponent(
            joint_transition, initial_state, mock_context, num_steps=10
        )
        assert lyap < 0  # Stable

        steps, _ = measure_settling_time(
            joint_transition,
            initial_state,
            mock_context,
            tolerance=1e-3,
            max_steps=100,
            norm_type="absolute",
        )
        assert steps < 100

        # Basin stability at origin
        z_zero = CompositeState(
            activity={"x": torch.zeros(4, 32)}, plastic={}, substrate={}
        )
        stability = estimate_basin_stability(
            joint_transition, z_zero, mock_context, num_samples=10
        )
        assert stability >= 0.0


# ============================================================
# Fast Mode Proxy Tests (CI)
# ============================================================


class TestFastModeProxies:
    """Test that all fast-mode proxies execute without error."""

    def test_all_fast_modes_execute(self, mock_context, initial_state):
        transition = MockTransition(rho=0.7)

        # Spectral radius fast proxy
        spec_est = SpectralRadiusEstimator(fast_mode=True)
        rho = spec_est(transition, initial_state, mock_context)
        assert isinstance(rho, float)

        # Lyapunov fast proxy
        lyap_est = LyapunovEstimator(fast_mode=True)
        lyap = lyap_est(transition, initial_state, mock_context)
        assert isinstance(lyap, float)

        # Settling fast proxy
        settle_mon = SettlingMonitor()
        est = settle_mon.fast_proxy(transition, initial_state, mock_context)
        assert isinstance(est, int)

        # Basin fast proxy
        z_zero = CompositeState(
            activity={"x": torch.zeros(4, 32)}, plastic={}, substrate={}
        )
        basin_est = BasinStabilityEstimator(fast_mode=True)
        stability = basin_est(transition, z_zero, mock_context)
        assert isinstance(stability, float)
        assert 0.0 <= stability <= 1.0


# ============================================================
# Cheap Proxy Properties (Property-based)
# ============================================================


class TestCheapProxyProperties:
    """Property tests for cheap proxy invariants."""

    def test_spectral_radius_fast_proxy_nonnegative(self, mock_context, initial_state):
        """Fast proxy spectral radius should be non-negative."""
        estimator = SpectralRadiusEstimator(fast_mode=True)
        transition = MockTransition(rho=0.5)
        rho = estimator(transition, initial_state, mock_context)
        assert rho >= 0.0

    def test_lyapunov_fast_proxy_real(self, mock_context, initial_state):
        """Fast proxy Lyapunov should be real (not NaN/inf)."""
        estimator = LyapunovEstimator(fast_mode=True)
        transition = MockTransition(rho=0.5)
        lyap = estimator(transition, initial_state, mock_context)
        assert not torch.isnan(torch.tensor(lyap))
        assert not torch.isinf(torch.tensor(lyap))

    def test_settling_fast_proxy_bounded(self, mock_context, initial_state):
        """Fast proxy settling estimate should be bounded by max_steps."""
        monitor = SettlingMonitor(max_steps=100)
        transition = MockTransition(rho=0.5)
        est = monitor.fast_proxy(transition, initial_state, mock_context)
        assert 0 <= est <= 100

    def test_basin_fast_proxy_bounded(self, mock_context):
        """Fast proxy basin stability should be in [0, 1]."""
        estimator = BasinStabilityEstimator(fast_mode=True)
        z_zero = CompositeState(
            activity={"x": torch.zeros(4, 32)}, plastic={}, substrate={}
        )
        transition = MockContractingTransition(rate=0.5)
        stability = estimator(transition, z_zero, mock_context)
        assert 0.0 <= stability <= 1.0
