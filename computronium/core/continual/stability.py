"""Stability helpers for continual learning."""

from __future__ import annotations

import torch
from torch import Tensor

from computronium.core.joint.state import CompositeState


def create_stability_guard(
    threshold: float = 1.029,
    statistic: str = "fast_proxy",
    window: int = 10,
):
    """Create stability guard."""
    from computronium.core.stability import StabilityGuard
    from computronium.core.stability.spectral_radius import SpectralRadiusEstimator
    estimator = SpectralRadiusEstimator(fast_mode=True)
    return StabilityGuard(
        threshold=threshold,
        estimator=estimator,
        statistic=statistic,  # type: ignore[arg-type]
        window=window,
    )


def make_transition_fn(model: torch.nn.Module):
    """Create a simple transition function for stability checking.

    Returns a CompositeState with activity, plastic, and substrate.
    """
    def transition_fn(state, context=None):
        """Transition function that takes a CompositeState and returns CompositeState."""
        x = state.activity.get("x")
        if x is None:
            return CompositeState.empty()
        with torch.no_grad():
            y = model(x)
        # Return CompositeState: activity contains x and y, plastic is empty, substrate is empty
        return CompositeState(
            activity={"x": y, "y": y},
            plastic={},
            substrate={},
        )
    return transition_fn


def make_composite_state(x: Tensor):
    """Create a simple CompositeState for stability checking."""
    return CompositeState(
        activity={"x": x},
        plastic={},
        substrate={},
    )


def check_stability(
    guard,
    transition_fn,
    x: Tensor,
    step: int,
    context=None,
):
    """Check stability at current step."""
    state = make_composite_state(x)
    return guard(transition_fn, state, context)


__all__ = [
    "check_stability",
    "create_stability_guard",
    "make_composite_state",
    "make_transition_fn",
]
