"""P1 — shared settle primitive + EquilibriumSettleProtocol adoption.

Asserts that ``NeuralCube`` adopts the protocol and that the convergence lever
is a framework property (the §7 win generalised), with hypothesis checks that a
loose threshold always terminates early.
"""

import torch
from hypothesis import given
from hypothesis import strategies as st

from bioplausible.zoo._settling import (
    EquilibriumSettleProtocol,
    settle_state,
)
from bioplausible.zoo.models.eqprop.neural_cube import NeuralCube


def _make(threshold, start=2, max_steps=20, cube=3):
    return NeuralCube(
        cube_size=cube,
        input_dim=8,
        output_dim=4,
        max_steps=max_steps,
        convergence_threshold=threshold,
        convergence_start=start,
    )


def test_neural_cube_adopts_settle_protocol():
    """NeuralCube satisfies the structural settle contract (P1)."""
    model = _make(1e-3)
    assert isinstance(model, EquilibriumSettleProtocol)


def test_loose_threshold_terminates_before_max_steps_and_converges():
    """A convergence_threshold=1.0 model terminates in < max_steps, converged."""
    model = _make(threshold=1.0, start=2, max_steps=20)
    x = torch.randn(4, 8)
    h, steps_taken, converged = settle_state(model, x)
    assert steps_taken < model.max_steps
    assert converged
    assert h.shape == (4, model.cube_size**3)


def test_forward_exposes_steps_and_convergence_probe_metrics():
    """forward() records steps_taken/converged for the probe driver."""
    model = _make(threshold=1.0, start=2, max_steps=20)
    out = model(torch.randn(2, 8))
    assert out.shape == (2, 4)
    assert model._last_settle_steps < model.max_steps
    assert model._last_settle_converged


def test_forward_trajectory_path_still_works():
    """Visualization path returns (out, trajectory) and converges early."""
    model = _make(threshold=1.0, start=2, max_steps=20)
    out, trajectory = model(torch.randn(2, 8), return_trajectory=True)
    assert out.shape == (2, 4)
    assert len(trajectory) > 1
    assert len(trajectory) <= model.max_steps + 1


@given(
    threshold=st.floats(min_value=0.9, max_value=1.0, allow_nan=False),
    max_steps=st.integers(min_value=5, max_value=30),
)
def test_loose_threshold_always_early_stops(threshold, max_steps):
    """Property: with a near-maximum threshold the settle always converges early."""
    model = _make(threshold=threshold, start=2, max_steps=max_steps)
    h, steps_taken, converged = settle_state(model, torch.randn(3, 8))
    assert converged
    assert steps_taken < max_steps
    assert h.shape == (3, model.cube_size**3)


def test_gradient_flows_through_settled_forward():
    """Backprop through the shared (checkpointed) settle reaches the weights."""
    model = _make(threshold=1e-3, start=2, max_steps=15)
    out = model(torch.randn(2, 8))
    out.sum().backward()
    assert model.W_in.weight.grad is not None
    assert model.W_out.weight.grad is not None
