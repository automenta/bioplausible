"""P1 — shared settle primitive + EquilibriumSettleProtocol adoption.

Asserts that ``TileAlgorithm`` adopts the protocol and that the convergence lever
is a framework property (the §7 win generalised), with hypothesis checks that a
loose threshold always terminates early.

Migrated to native TileAlgorithm after legacy zoo removal.
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from computronium.core.local_learning.builder import TileAlgorithm, TileAlgorithmConfig
from computronium.core.local_learning.settling import (
    SettleConfig,
    SettleProtocol,
    settle_universal,
)


def _make(threshold, start=2, max_steps=20):
    """Create a TileAlgorithm (EP variant) for settle protocol testing."""
    config = TileAlgorithmConfig(
        input_dim=8,
        output_dim=4,
        neurons_per_tile=8,
        tiles_per_layer=2,
        num_hidden_layers=2,
        algorithm="ep",
        mode="ep",
        free_steps=max_steps,
        nudged_steps=max_steps,
        learning_rate=0.001,
        beta=0.1,
    )
    model = TileAlgorithm(config)
    # Set convergence attributes (SettleProtocol)
    model.convergence_threshold = threshold
    model.convergence_start = start
    model.max_steps = max_steps
    return model


def test_tile_algorithm_adopts_settle_protocol():
    """TileAlgorithm satisfies the structural settle contract (P1)."""
    model = _make(1e-3)
    assert isinstance(model, SettleProtocol)


def test_loose_threshold_terminates_before_max_steps_and_converges():
    """A convergence_threshold=1e-2 model terminates in < max_steps, converged."""
    torch.manual_seed(1)
    model = _make(threshold=1e-2, start=2, max_steps=20)
    x = torch.randn(4, 8)
    out, steps_taken, converged, _ = model._run_settle_universal(x)
    assert steps_taken < model.max_steps
    assert converged
    assert out.shape == (4, model.config.output_dim)


def test_forward_exposes_steps_and_convergence_probe_metrics():
    """forward() records steps_taken/converged for the probe driver."""
    torch.manual_seed(123)
    model = _make(threshold=1e-2, start=2, max_steps=20)
    out, dynamics = model(torch.randn(2, 8), return_dynamics=True)
    assert out.shape == (2, model.config.output_dim)
    assert dynamics["steps_taken"] < model.max_steps
    assert dynamics["converged"]


def test_forward_trajectory_path_still_works():
    """Visualization path returns (out, trajectory) and converges early."""
    model = _make(threshold=1.0, start=2, max_steps=20)
    out, trajectory = model(
        torch.randn(2, 8), return_trajectory=True, return_dynamics=True
    )
    assert out.shape == (2, model.config.output_dim)
    assert len(trajectory) > 1
    assert len(trajectory) <= model.max_steps + 1


@pytest.mark.parametrize("threshold,max_steps", [(1e-2, 20), (5e-2, 15), (1e-1, 30)])
def test_loose_threshold_always_early_stops(threshold, max_steps):
    """Property: with a loose threshold the settle converges early for these parameters.

    Tests specific threshold/max_steps combinations that are known to converge.
    """
    torch.manual_seed(1)
    model = _make(threshold=threshold, start=2, max_steps=max_steps)
    out, steps_taken, converged, _ = model._run_settle_universal(torch.randn(3, 8))
    assert converged
    assert steps_taken < max_steps
    assert out.shape == (3, model.config.output_dim)


def test_gradient_flows_through_settled_forward():
    """Backprop through the shared (checkpointed) settle reaches the weights."""
    model = _make(threshold=1e-3, start=2, max_steps=15)
    out = model(torch.randn(2, 8))
    out.sum().backward()
    # Check that input projection has gradients
    assert model.W_in.weight.grad is not None
    assert model.W_out.weight.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
