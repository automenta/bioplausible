"""Standalone test suite for published computronium_stability API.

This test validates that the wheel installs and works identically to internal usage.
Mirrors tests/unit/nn/test_computronium_linear.py pattern.
"""

from __future__ import annotations

import pytest
import torch

# This import should work after `pip install -e .[stability]`
from computronium_stability import (
    BasinConfig,
    BasinStabilityEstimator,
    FrontierAggregator,
    # Frontier
    FrontierRecord,
    GuardConfig,
    GuardDecision,
    GuardHandle,
    LyapunovConfig,
    LyapunovEstimator,
    # Resources
    ResourceUsage,
    SettlingConfig,
    SettlingMonitor,
    # Config + Factories
    SpectralRadiusConfig,
    SpectralRadiusEstimator,
    StabilityGuard,
    StabilityVerdict,
    StepState,
    # Guard API
    attach,
    calibrate_threshold,
    create_basin_estimator,
    create_guard,
    create_lyapunov_estimator,
    create_settling_monitor,
    create_spectral_radius_estimator,
    estimate_basin_stability,
    estimate_basin_stability_multistart,
    estimate_lyapunov_exponent,
    estimate_spectral_radius,
    measure_settling_time,
)
from torch import Tensor

# ============================================================
# Test Fixtures
# ============================================================


class MockCompositeState:
    """Mock CompositeState with clone method."""

    def __init__(self, activity=None, plastic=None, substrate=None):
        self.activity = activity or {}
        self.plastic = plastic or {}
        self.substrate = substrate or {}

    def clone(self):
        return MockCompositeState(
            activity=self.activity.copy(),
            plastic=self.plastic.copy(),
            substrate=self.substrate.copy(),
        )


def make_state(activity=None, plastic=None, substrate=None):
    """Create a mock state with clone method."""
    return MockCompositeState(activity, plastic, substrate)


class MockTransition:
    """Mock transition for testing stability metrics."""

    def __init__(self, rho: float = 0.5, dim: int = 32):
        self.rho = rho
        self.dim = dim

    def __call__(self, z, context):
        x = z.activity["x"]
        batch_size = x.shape[0]
        new_x = self.rho * x
        # Create a simple CompositeState-like object with clone method
        return make_state(
            activity={"x": new_x}, plastic=z.plastic, substrate=z.substrate
        )


class MockContractingTransition:
    """Mock transition that converges to a fixed point."""

    def __init__(self, fixed_point: Tensor | None = None, rate: float = 0.8):
        self.rate = rate
        self._fixed_point = fixed_point

    def __call__(self, z, context):
        x = z.activity["x"]
        if self._fixed_point is None:
            self._fixed_point = torch.zeros_like(x)
        new_x = self._fixed_point + self.rate * (x - self._fixed_point)
        return make_state(
            activity={"x": new_x}, plastic=z.plastic, substrate=z.substrate
        )


@pytest.fixture
def mock_context():
    """Create a minimal SystemContext-like object for testing."""
    registry = type("StateRegistry", (), {"register": lambda self, var: None})()
    registry.register(type("StateVariable", (), {"name": "x", "persistent": True})())

    return type(
        "SystemContext",
        (),
        {
            "theta": {"W": torch.randn(32, 32, requires_grad=True)},
            "geometry": type("Geometry", (), {})(),
            "substrate": type("Substrate", (), {})(),
            "substrate_config": type("SubstrateConfig", (), {})(),
            "geometry_config": type("GeometryConfig", (), {})(),
            "dynamics_config": type("StateDynamicsConfig", (), {})(),
            "credit_config": type("CreditAssignmentConfig", (), {})(),
            "update_config": type("ParameterUpdateConfig", (), {})(),
            "plasticity_config": type(
                "PlasticityConfig", (), {"plasticity_type": "null"}
            )(),
            "registry": registry,
        },
    )()


@pytest.fixture
def initial_state():
    """Create initial state-like object."""
    return type(
        "CompositeState",
        (),
        {
            "activity": {"x": torch.randn(4, 32)},
            "plastic": {},
            "substrate": {},
        },
    )()


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
        assert 0.3 < rho < 0.7

    def test_estimate_spectral_radius_unstable(self, mock_context, initial_state):
        transition = MockTransition(rho=1.2)
        rho = estimate_spectral_radius(
            transition, initial_state, mock_context, num_iterations=10
        )
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
        assert rho >= 0.0

    def test_spectral_radius_config_roundtrip(self):
        config = SpectralRadiusConfig(num_iterations=15, fast_mode=True)
        spec = config.to_spec()
        config2 = SpectralRadiusConfig.from_spec(spec)
        assert config2.num_iterations == 15
        assert config2.fast_mode is True


# ============================================================
# Lyapunov Exponent Tests
# ============================================================


class TestLyapunovExponent:
    def test_estimate_lyapunov_stable(self, mock_context, initial_state):
        transition = MockTransition(rho=0.5)
        lyap = estimate_lyapunov_exponent(
            transition, initial_state, mock_context, num_steps=20
        )
        assert lyap < 0

    def test_estimate_lyapunov_unstable(self, mock_context, initial_state):
        transition = MockTransition(rho=1.2)
        lyap = estimate_lyapunov_exponent(
            transition, initial_state, mock_context, num_steps=20
        )
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

    def test_lyapunov_config_roundtrip(self):
        config = LyapunovConfig(num_steps=30, fast_mode=True)
        spec = config.to_spec()
        config2 = LyapunovConfig.from_spec(spec)
        assert config2.num_steps == 30
        assert config2.fast_mode is True


# ============================================================
# Settling Time Tests
# ============================================================


class TestSettlingTime:
    def test_measure_settling_time_converges(self, mock_context):
        z = make_state(activity={"x": torch.ones(4, 32) * 10.0})
        transition = MockContractingTransition(rate=0.5)
        steps, norms = measure_settling_time(
            transition, z, mock_context, tolerance=1e-3, max_steps=100
        )
        assert steps < 100
        assert len(norms) == steps
        assert norms[-1] < 1e-3

    def test_measure_settling_time_max_steps(self, mock_context):
        class OscillatingTransition:
            def __init__(self):
                self.step_count = 0

            def __call__(self, z, context):
                x = z.activity["x"]
                if self.step_count % 2 == 0:
                    new_x = x * 1.1
                else:
                    new_x = x * 0.9
                self.step_count += 1
                return make_state(
                    activity={"x": new_x}, plastic=z.plastic, substrate=z.substrate
                )

        transition = OscillatingTransition()
        z = make_state(activity={"x": torch.randn(4, 32)})
        steps, norms = measure_settling_time(
            transition, z, mock_context, tolerance=1e-6, max_steps=10
        )
        assert steps == 10
        assert len(norms) == 10

    def test_settling_monitor_class(self, mock_context):
        monitor = SettlingMonitor(tolerance=1e-3, max_steps=100, record_trajectory=True)
        transition = MockContractingTransition(rate=0.5)
        z = make_state(activity={"x": torch.ones(4, 32) * 10.0})
        steps, norms, traj = monitor(transition, z, mock_context)
        assert steps < 100
        assert len(norms) == steps
        assert traj is not None
        assert len(traj) == steps + 1

    def test_settling_fast_proxy(self, mock_context):
        monitor = SettlingMonitor()
        transition = MockContractingTransition(rate=0.5)
        z = make_state(activity={"x": torch.ones(4, 32) * 10.0})
        est = monitor.fast_proxy(transition, z, mock_context)
        assert isinstance(est, int)
        assert est <= monitor.max_steps

    def test_settling_config_roundtrip(self):
        config = SettlingConfig(tolerance=1e-5, max_steps=500, record_trajectory=True)
        spec = config.to_spec()
        config2 = SettlingConfig.from_spec(spec)
        assert config2.tolerance == 1e-5
        assert config2.max_steps == 500
        assert config2.record_trajectory is True


# ============================================================
# Basin Stability Tests
# ============================================================


class TestBasinStability:
    def test_estimate_basin_stability_stable(self, mock_context):
        z_attractor = make_state(activity={"x": torch.zeros(4, 32)})
        transition = MockContractingTransition(rate=0.5)
        stability = estimate_basin_stability(
            transition,
            z_attractor,
            mock_context,
            num_samples=20,
            perturbation_radius=2.0,
            max_steps=50,
        )
        assert stability > 0.5

    def test_basin_stability_estimator_class(self, mock_context):
        estimator = BasinStabilityEstimator(num_samples=20, fast_mode=False)
        z_attractor = make_state(activity={"x": torch.zeros(4, 32)})
        transition = MockContractingTransition(rate=0.5)
        stability = estimator(transition, z_attractor, mock_context)
        assert 0.0 <= stability <= 1.0

    def test_basin_fast_mode(self, mock_context):
        estimator = BasinStabilityEstimator(fast_mode=True)
        z_attractor = make_state(activity={"x": torch.zeros(4, 32)})
        transition = MockContractingTransition(rate=0.5)
        stability = estimator(transition, z_attractor, mock_context)
        assert 0.0 <= stability <= 1.0

    def test_basin_multistart(self, mock_context):
        z_attractor = make_state(activity={"x": torch.zeros(4, 32)})
        transition = MockContractingTransition(rate=0.5)
        results = estimate_basin_stability_multistart(
            transition,
            z_attractor,
            mock_context,
            num_samples=10,
            perturbation_radii=[0.5, 1.0, 2.0],
        )
        assert len(results) == 3
        assert results[0.5] >= results[1.0] >= results[2.0]

    def test_basin_config_roundtrip(self):
        config = BasinConfig(num_samples=50, fast_mode=True)
        spec = config.to_spec()
        config2 = BasinConfig.from_spec(spec)
        assert config2.num_samples == 50
        assert config2.fast_mode is True


# ============================================================
# Guard API Tests
# ============================================================


class TestStabilityGuardAPI:
    def test_guard_decision_creation(self):
        decision = GuardDecision(statistic=1.5, threshold=1.029, kill=True)
        assert decision.statistic == 1.5
        assert decision.threshold == 1.029
        assert decision.kill is True

    def test_stability_guard_internal(self, mock_context, initial_state):
        guard = StabilityGuard(threshold=1.029, statistic="fast_proxy")
        transition = MockTransition(rho=0.5)
        decision = guard(transition, initial_state, mock_context)
        assert isinstance(decision, GuardDecision)
        assert decision.kill is False  # 0.5 < 1.029

    def test_stability_guard_kills_unstable(self, mock_context, initial_state):
        guard = StabilityGuard(threshold=1.029, statistic="fast_proxy")
        transition = MockTransition(rho=1.5)
        decision = guard(transition, initial_state, mock_context)
        assert decision.kill is True

    def test_stability_guard_windowed_growth(self, mock_context, initial_state):
        guard = StabilityGuard(threshold=1.029, statistic="windowed_growth", window=5)
        transition = MockTransition(rho=0.9)
        decision = guard(transition, initial_state, mock_context)
        assert isinstance(decision, GuardDecision)

    def test_calibrate_threshold(self):
        good_stats = [0.8, 0.9, 0.95, 1.0, 1.0]
        bad_stats = [1.5, 1.8, 2.0, 2.5, 3.0]
        report = calibrate_threshold(good_stats, bad_stats)
        assert report is not None
        assert report.false_kill_rate <= 0.05
        assert report.kill_rate >= 0.95

    def test_calibrate_threshold_no_feasible(self):
        good_stats = [1.5, 1.8, 2.0]
        bad_stats = [1.0, 1.1, 1.2]
        report = calibrate_threshold(good_stats, bad_stats)
        assert report is None


# ============================================================
# External Guard API Tests (dict-based state)
# ============================================================


class TestExternalGuardAPI:
    def test_attach_returns_guard_handle(self):
        model = torch.nn.Linear(10, 10)
        handle = attach(model)
        assert isinstance(handle, GuardHandle)
        assert handle.model is model

    def test_guard_handle_check(self):
        model = torch.nn.Linear(10, 10)
        handle = attach(model, statistic="windowed_growth", window=5)
        verdict = handle.check({"x": torch.randn(4, 10)}, step=0)
        assert isinstance(verdict, StabilityVerdict)
        assert hasattr(verdict, "kill")
        assert hasattr(verdict, "max_statistic")
        assert hasattr(verdict, "threshold")
        assert hasattr(verdict, "step")

    def test_stability_verdict_bool(self):
        verdict_kill = StabilityVerdict(
            kill=True,
            decisions=(),
            max_statistic=2.0,
            threshold=1.029,
            step=0,
        )
        verdict_pass = StabilityVerdict(
            kill=False,
            decisions=(),
            max_statistic=0.5,
            threshold=1.029,
            step=0,
        )
        assert bool(verdict_kill) is True
        assert bool(verdict_pass) is False

    def test_guard_kills_expansive_model(self):
        """Guard should kill a model with expansive dynamics."""
        model = torch.nn.Linear(10, 10)
        # Make it expansive
        with torch.no_grad():
            model.weight.data *= 2.0

        handle = attach(model, statistic="windowed_growth", window=10)
        verdict = handle.check({"x": torch.randn(4, 10)}, step=0)
        # With 2x weight, growth should exceed threshold
        assert verdict.kill is True

    def test_guard_passes_contractive_model(self):
        """Guard should pass a model with contractive dynamics."""
        model = torch.nn.Linear(10, 10)
        # Make it contractive
        with torch.no_grad():
            model.weight.data *= 0.5

        handle = attach(model, statistic="windowed_growth", window=10)
        verdict = handle.check({"x": torch.randn(4, 10)}, step=0)
        # With 0.5x weight, growth should be below threshold
        assert verdict.kill is False

    def test_attach_custom_transition_fn(self):
        model = torch.nn.Linear(10, 10)

        def custom_transition(state: StepState) -> StepState:
            x = state["x"]
            with torch.no_grad():
                y = model(x) + 0.1 * x  # Residual
            return {"x": y}

        handle = attach(model, transition_fn=custom_transition)
        verdict = handle.check({"x": torch.randn(4, 10)}, step=0)
        assert isinstance(verdict, StabilityVerdict)

    def test_guard_handle_detach(self):
        model = torch.nn.Linear(10, 10)
        handle = attach(model)
        handle.detach()  # Should not raise


# ============================================================
# Config Factory Tests
# ============================================================


class TestConfigFactories:
    def test_create_spectral_radius_estimator(self):
        config = SpectralRadiusConfig(num_iterations=15, fast_mode=True)
        estimator = create_spectral_radius_estimator(config)
        assert isinstance(estimator, SpectralRadiusEstimator)
        assert estimator.num_iterations == 15
        assert estimator.fast_mode is True

    def test_create_lyapunov_estimator(self):
        config = LyapunovConfig(num_steps=30, fast_mode=True)
        estimator = create_lyapunov_estimator(config)
        assert isinstance(estimator, LyapunovEstimator)
        assert estimator.num_steps == 30
        assert estimator.fast_mode is True

    def test_create_settling_monitor(self):
        config = SettlingConfig(tolerance=1e-5, record_trajectory=True)
        monitor = create_settling_monitor(config)
        assert isinstance(monitor, SettlingMonitor)
        assert monitor.tolerance == 1e-5
        assert monitor.record_trajectory is True

    def test_create_basin_estimator(self):
        config = BasinConfig(num_samples=50, fast_mode=True)
        estimator = create_basin_estimator(config)
        assert isinstance(estimator, BasinStabilityEstimator)
        assert estimator.num_samples == 50
        assert estimator.fast_mode is True

    def test_create_guard(self):
        config = GuardConfig(threshold=1.05, statistic="fast_proxy", window=20)
        guard = create_guard(config)
        assert isinstance(guard, StabilityGuard)
        assert guard.threshold == 1.05
        assert guard.statistic == "fast_proxy"
        assert guard.window == 20

    def test_guard_config_roundtrip(self):
        config = GuardConfig(threshold=1.05, statistic="fast_proxy", window=20)
        spec = config.to_spec()
        config2 = GuardConfig.from_spec(spec)
        assert config2.threshold == 1.05
        assert config2.statistic == "fast_proxy"
        assert config2.window == 20


# ============================================================
# Integration Tests
# ============================================================


class TestIntegration:
    def test_guard_kills_divergent(self, mock_context):
        """Guard kills divergent dynamics."""
        guard = StabilityGuard(threshold=1.029, statistic="windowed_growth", window=10)
        transition = MockTransition(rho=1.5)
        z = make_state(activity={"x": torch.randn(4, 32)})
        decision = guard(transition, z, mock_context)
        assert decision.kill is True

    def test_guard_passes_healthy(self, mock_context):
        """Guard passes contractive dynamics."""
        guard = StabilityGuard(threshold=1.029, statistic="windowed_growth", window=10)
        transition = MockTransition(rho=0.5)
        z = make_state(activity={"x": torch.randn(4, 32)})
        decision = guard(transition, z, mock_context)
        assert decision.kill is False

    def test_all_estimators_fast_mode_execute(self, mock_context):
        """Test that all fast-mode proxies execute without error."""
        z = make_state(activity={"x": torch.randn(4, 32)})
        transition = MockTransition(rho=0.7)

        # Spectral radius fast proxy
        spec_est = SpectralRadiusEstimator(fast_mode=True)
        rho = spec_est(transition, z, mock_context)
        assert isinstance(rho, float)

        # Lyapunov fast proxy
        lyap_est = LyapunovEstimator(fast_mode=True)
        lyap = lyap_est(transition, z, mock_context)
        assert isinstance(lyap, float)

        # Settling fast proxy
        settle_mon = SettlingMonitor()
        est = settle_mon.fast_proxy(transition, z, mock_context)
        assert isinstance(est, int)

        # Basin fast proxy
        z_zero = make_state(activity={"x": torch.zeros(4, 32)})
        basin_est = BasinStabilityEstimator(fast_mode=True)
        stability = basin_est(transition, z_zero, mock_context)
        assert isinstance(stability, float)
        assert 0.0 <= stability <= 1.0


# ============================================================
# Device Management Tests
# ============================================================


class TestDeviceManagement:
    def test_cpu_consistency(self, mock_context):
        """All estimators should work on CPU."""
        z = make_state(activity={"x": torch.randn(4, 32)})
        transition = MockTransition(rho=0.7)

        rho = estimate_spectral_radius(transition, z, mock_context)
        lyap = estimate_lyapunov_exponent(transition, z, mock_context)
        steps, _ = measure_settling_time(
            MockContractingTransition(rate=0.5), z, mock_context
        )
        z_zero = make_state(activity={"x": torch.zeros(4, 32)})
        basin = estimate_basin_stability(
            MockContractingTransition(rate=0.5), z_zero, mock_context
        )

        assert isinstance(rho, float)
        assert isinstance(lyap, float)
        assert isinstance(steps, int)
        assert isinstance(basin, float)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_consistency(self):
        """All estimators should work on CUDA if available."""
        registry = type("StateRegistry", (), {"register": lambda self, var: None})()
        registry.register(
            type("StateVariable", (), {"name": "x", "persistent": True})()
        )

        context = type(
            "SystemContext",
            (),
            {
                "theta": {"W": torch.randn(32, 32, requires_grad=True, device="cuda")},
                "geometry": type("Geometry", (), {})(),
                "substrate": type("Substrate", (), {})(),
                "substrate_config": type("SubstrateConfig", (), {})(),
                "geometry_config": type("GeometryConfig", (), {})(),
                "dynamics_config": type("StateDynamicsConfig", (), {})(),
                "credit_config": type("CreditAssignmentConfig", (), {})(),
                "update_config": type("ParameterUpdateConfig", (), {})(),
                "plasticity_config": type(
                    "PlasticityConfig", (), {"plasticity_type": "null"}
                )(),
                "registry": registry,
            },
        )()

        z = make_state(activity={"x": torch.randn(4, 32, device="cuda")})

        transition = MockTransition(rho=0.7)
        rho = estimate_spectral_radius(transition, z, context)
        assert isinstance(rho, float)
