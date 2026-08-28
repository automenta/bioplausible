"""Spectral Radius Estimation: ρ(J) for dynamical system stability.

Framework-agnostic: works with any callable transition function that maps
state dicts to state dicts, where state dicts contain PyTorch tensors.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

# Type aliases for framework-agnostic state
StepState = dict[str, Tensor]
TransitionFn = Callable[[StepState], StepState]


def estimate_spectral_radius(
    transition_fn: TransitionFn,
    state: StepState,
    num_iterations: int = 20,
    perturbation_scale: float = 1e-4,
    activity_key: str = "x",
) -> float:
    """Estimate spectral radius ρ(J) of the transition Jacobian.

    Uses power iteration on the Jacobian-vector product.

    Args:
        transition_fn: State transition function F(state) -> next_state.
        state: Base state to evaluate Jacobian at.
        num_iterations: Number of power iterations.
        perturbation_scale: Scale for finite-difference perturbations.
        activity_key: Key in state to perturb (default: "x").

    Returns:
        Estimated spectral radius ρ(J).
    """
    x_base = state.get(activity_key)
    if x_base is None or not isinstance(x_base, Tensor):
        # Fallback: find first tensor
        for v in state.values():
            if isinstance(v, Tensor):
                x_base = v
                break
        if x_base is None:
            return 0.0

    _batch_size, _dim = x_base.shape

    # Initialize random vector for power iteration
    v = torch.randn_like(x_base)
    v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)

    for _ in range(num_iterations):
        # Compute J * v via finite differences
        x_perturbed = x_base + perturbation_scale * v

        state_perturbed = {**state, activity_key: x_perturbed}

        with torch.no_grad():
            next_base = transition_fn(state)
            next_perturbed = transition_fn(state_perturbed)

        next_base_act = next_base.get(activity_key)
        next_pert_act = next_perturbed.get(activity_key)
        if next_base_act is None or next_pert_act is None:
            return 0.0

        delta = next_pert_act - next_base_act
        Jv = delta / perturbation_scale

        # Power iteration update
        v = Jv / (Jv.norm(dim=-1, keepdim=True) + 1e-8)

    # Final estimate: ||J * v||
    x_perturbed = x_base + perturbation_scale * v
    state_perturbed = {**state, activity_key: x_perturbed}

    with torch.no_grad():
        next_base = transition_fn(state)
        next_perturbed = transition_fn(state_perturbed)

    next_base_act = next_base.get(activity_key)
    next_pert_act = next_perturbed.get(activity_key)
    if next_base_act is None or next_pert_act is None:
        return 0.0

    delta = next_pert_act - next_base_act
    Jv = delta / perturbation_scale

    # Spectral radius estimate
    rho = Jv.norm(dim=-1).mean().item()

    return rho


@dataclass(slots=True)
class SpectralRadiusEstimator:
    """Configurable spectral radius estimator.

    Supports both full power iteration (accurate) and fast proxy (CI).
    """

    num_iterations: int = 20
    perturbation_scale: float = 1e-4
    activity_key: str = "x"
    fast_mode: bool = False

    def __call__(
        self,
        transition_fn: TransitionFn,
        state: StepState,
    ) -> float:
        """Estimate spectral radius."""
        if self.fast_mode:
            return self._fast_proxy(transition_fn, state)
        return estimate_spectral_radius(
            transition_fn,
            state,
            num_iterations=self.num_iterations,
            perturbation_scale=self.perturbation_scale,
            activity_key=self.activity_key,
        )

    def _fast_proxy(
        self,
        transition_fn: TransitionFn,
        state: StepState,
    ) -> float:
        """Fast proxy: single-step Jacobian-vector product."""
        x = state.get(self.activity_key)
        if x is None or not isinstance(x, Tensor):
            return 0.0

        eps = self.perturbation_scale
        v = torch.randn_like(x)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)

        x_perturbed = x + eps * v
        state_perturbed = {**state, self.activity_key: x_perturbed}

        with torch.no_grad():
            next_state = transition_fn(state)
            next_perturbed = transition_fn(state_perturbed)

        next_act = next_perturbed.get(self.activity_key)
        base_act = next_state.get(self.activity_key)
        if next_act is None or base_act is None:
            return 0.0

        delta = next_act - base_act
        Jv = delta / eps

        return Jv.norm(dim=-1).mean().item()
