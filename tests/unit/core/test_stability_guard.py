"""Unit tests for the calibrated stability guard (PR-5)."""

from __future__ import annotations

import pytest
import torch

from computronium.core.joint.transition import PlasticityConfig
from computronium.stability import (
    StabilityGuard,
    calibrate_threshold,
    measure_guard_overhead,
    quantify_proxy_disagreement,
)
from computronium.state import CompositeState, SystemContext

GOOD_STATS = [1.0, 1.1, 0.9]
BAD_STATS = [2.0, 2.2, 2.1]
FEASIBLE_FALSE_KILL = 0.05
FEASIBLE_KILL_RATE = 0.95
WINDOW_THRESHOLD = 1.1
REL_ERROR_BOUND = 0.05
PROBE_COUNT = 5


@pytest.fixture
def context():
    from computronium.ontology import (
        CreditAssignmentConfig,
        DigitalSubstrate,
        GeometryConfig,
        ParameterUpdateConfig,
        RecurrentGeometry,
        StateDynamicsConfig,
        SubstrateConfig,
    )
    from computronium.state import StateRegistry, StateVariable

    geometry_config = GeometryConfig.feedforward(
        input_dim=8, output_dim=2, hidden_dims=(8,)
    )
    geometry = RecurrentGeometry(geometry_config)
    registry = StateRegistry()
    for name in geometry.params:
        registry.register(StateVariable(name=name, persistent=True))
    registry.register(StateVariable(name="x", persistent=True))

    return SystemContext(
        theta=geometry.params,
        geometry=geometry,
        substrate=DigitalSubstrate(),
        substrate_config=SubstrateConfig.digital(),
        geometry_config=geometry_config,
        dynamics_config=StateDynamicsConfig.instantaneous(),
        credit_config=CreditAssignmentConfig.thermodynamic_contrast(),
        update_config=ParameterUpdateConfig.euclidean(step_size=0.01),
        plasticity_config=PlasticityConfig.null(),
        registry=registry,
    )


@pytest.fixture
def state():
    return CompositeState(activity={"x": torch.randn(2, 8)}, plastic={}, substrate={})


def _scaling_transition(scale: float):
    def transition(z: CompositeState, _context: SystemContext) -> CompositeState:
        return CompositeState(
            activity={"x": scale * z.activity["x"]},
            plastic=z.plastic,
            substrate=z.substrate,
        )

    return transition


class TestCalibrateThreshold:
    def test_separated_classes(self):
        report = calibrate_threshold(GOOD_STATS, BAD_STATS)
        assert report is not None
        assert max(GOOD_STATS) <= report.threshold <= min(BAD_STATS)
        assert report.false_kill_rate == pytest.approx(0.0)
        assert report.kill_rate == pytest.approx(1.0)

    def test_infeasible_overlap_returns_none(self):
        report = calibrate_threshold([1.0, 1.5, 2.0], [1.9, 2.0, 2.1])
        assert report is None

    def test_partial_overlap_feasible(self):
        good = list(range(20))
        bad = [19, *range(21, 40)]
        report = calibrate_threshold(good, bad)
        assert report is not None
        assert report.false_kill_rate <= FEASIBLE_FALSE_KILL
        assert report.kill_rate >= FEASIBLE_KILL_RATE

    def test_empty_inputs(self):
        assert calibrate_threshold([], []) is None


class TestStabilityGuard:
    def test_decide_boundaries(self):
        guard = StabilityGuard(threshold=1.0)
        assert guard.decide(0.99).kill is False
        assert guard.decide(1.0).kill is False
        assert guard.decide(1.01).kill is True

    def test_call_probes_and_kills(self, context, state):
        guard = StabilityGuard(threshold=0.5)
        decision = guard(_scaling_transition(2.0), state, context)
        assert decision.kill is True
        assert decision.statistic == pytest.approx(2.0, rel=0.05)

    def test_no_kill_on_contracting(self, context, state):
        guard = StabilityGuard(threshold=1.0)
        assert guard(_scaling_transition(0.3), state, context).kill is False

    def test_windowed_growth_kills_expanding(self, context, state):
        guard = StabilityGuard(
            threshold=WINDOW_THRESHOLD, statistic="windowed_growth", window=8
        )
        decision = guard(_scaling_transition(1.3), state, context)
        assert decision.kill is True
        assert decision.statistic > WINDOW_THRESHOLD

    def test_windowed_growth_spares_contracting(self, context, state):
        guard = StabilityGuard(
            threshold=WINDOW_THRESHOLD, statistic="windowed_growth", window=8
        )
        assert guard(_scaling_transition(0.7), state, context).kill is False


class TestDisagreement:
    def test_identity_map_near_zero_error(self, context, state):
        report = quantify_proxy_disagreement(_scaling_transition(1.0), state, context)
        assert report.mean_relative_error < REL_ERROR_BOUND
        assert report.n_probes > 0
        assert report.proxy_seconds >= 0.0
        assert report.full_jacobian_seconds >= 0.0

    def test_scaling_map_tracks_gain(self, context, state):
        report = quantify_proxy_disagreement(
            _scaling_transition(1.7),
            state,
            context,
        )
        assert report.median_relative_error < REL_ERROR_BOUND

    def test_overhead_finite(self, context, state):
        ratio = measure_guard_overhead(
            _scaling_transition(0.5),
            state,
            context,
            guard=StabilityGuard(threshold=float("inf")),
            n_steps=PROBE_COUNT,
        )
        assert ratio > 0.0
