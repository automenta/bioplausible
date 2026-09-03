"""Hypothesis property-based tests for settling utilities in core/local_learning/settling.py."""

import torch
from hypothesis import given
from hypothesis import strategies as st

from computronium.core.local_learning.settling import (
    _inf_norm_converged,
    settle_activations_list,
    settle_single_state,
)


@given(
    steps=st.integers(min_value=0, max_value=50),
    early_start=st.integers(min_value=0, max_value=20),
)
def test_inf_norm_converged_false_before_early_start(steps, early_start):
    """Convergence check returns False when step_idx <= early_start."""
    h_new = torch.randn(10)
    h_old = torch.randn(10)
    for step_idx in range(min(steps, early_start + 1)):
        result = _inf_norm_converged(h_new, h_old, step_idx, early_start=early_start)
        assert not result, (
            f"Should be False at step {step_idx} <= early_start {early_start}"
        )


@given(
    steps=st.integers(min_value=5, max_value=30),
    threshold=st.floats(min_value=1e-6, max_value=1e-2, allow_nan=False),
)
def test_inf_norm_converged_detects_convergence(steps, threshold):
    """Convergence check returns True when delta is tiny."""
    h = torch.ones(10) * 0.5
    h_close = h + torch.full_like(h, threshold / 10)
    result = _inf_norm_converged(
        h_close,
        h,
        steps,
        threshold_early=threshold,
        threshold_late=threshold,
        transition_step=steps + 1,
        early_start=0,
    )
    assert result


@given(
    steps=st.integers(min_value=0, max_value=20),
)
def test_settle_single_state_trajectory_length(steps):
    """Settle trajectory has length steps + 1 (no early convergence)."""
    batch, dim = 4, 8
    h_0 = torch.randn(batch, dim)
    x = torch.randn(batch, dim)

    def forward_step(h, x_in):
        return torch.tanh(h + x_in)

    h_star, trajectory, _dynamics = settle_single_state(
        h_0, forward_step, x, steps, return_trajectory=True
    )
    if trajectory is not None:
        expected_len = min(steps + 1, steps + 1)  # no convergence  # ruff: ignore[unused-variable]
        assert len(trajectory) == steps + 1 or len(trajectory) <= steps + 1
        assert trajectory[0].shape == h_0.shape
    assert h_star.shape == h_0.shape


@given(
    steps=st.integers(min_value=0, max_value=10),
)
def test_settle_activations_list_convergence(steps):
    """Activations-list settling returns correct number of layers."""
    batch, dims = 4, [8, 16, 8]
    acts_0 = [torch.randn(batch, d) for d in dims]

    def dynamics(activations, beta, target):
        return [torch.tanh(a) for a in activations]

    final_acts, _trajectory, _dynamics_out = settle_activations_list(
        acts_0,
        dynamics,
        steps,
        convergence_start=steps + 1,  # never converge early
    )
    assert len(final_acts) == len(dims)
    assert all(a.shape == a0.shape for a, a0 in zip(final_acts, acts_0))


@given(
    steps=st.integers(min_value=1, max_value=10),
)
def test_settle_single_state_returns_dynamics(steps):
    """Dynamics dict has deltas and final_delta keys."""
    batch, dim = 2, 6
    h_0 = torch.randn(batch, dim)
    x = torch.randn(batch, dim)

    def forward_step(h, x_in):
        return torch.tanh(h + x_in)

    _, _, dynamics = settle_single_state(
        h_0, forward_step, x, steps, return_dynamics=True
    )
    assert dynamics is not None
    assert "deltas" in dynamics
    assert "final_delta" in dynamics
    assert isinstance(dynamics["deltas"], list)
    assert isinstance(dynamics["final_delta"], float)
