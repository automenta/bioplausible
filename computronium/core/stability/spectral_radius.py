"""Spectral Radius Estimation: ρ(J_F) for joint transition stability."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from computronium.state import CompositeState

if TYPE_CHECKING:
    from computronium.state import SystemContext


def estimate_spectral_radius(
    transition_fn: Callable[[CompositeState, SystemContext], CompositeState],
    z: CompositeState,
    context: SystemContext,
    num_iterations: int = 20,
    perturbation_scale: float = 1e-4,
    activity_key: str = "x",
) -> float:
    """Estimate spectral radius ρ(J_F) of the joint transition Jacobian.

    Uses the power iteration method on the Jacobian-vector product
    to estimate the dominant eigenvalue magnitude.

    Args:
        transition_fn: Joint transition function F_θ(z; G, S, M).
        z: Base joint state to evaluate Jacobian at.
        context: System context with fixed parameters.
        num_iterations: Number of power iterations.
        perturbation_scale: Scale for finite-difference perturbations.
        activity_key: Key in z.activity to perturb (default: "x" for neural activity).

    Returns:
        Estimated spectral radius ρ(J_F).
    """
    # Get base activity
    x_base = z.activity[activity_key]
    _batch_size, _dim = x_base.shape

    # Initialize random vector for power iteration
    v = torch.randn_like(x_base)
    v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)

    for _ in range(num_iterations):
        # Compute J * v via finite differences
        # Jv ≈ (F(x + εv) - F(x)) / ε
        x_perturbed = x_base + perturbation_scale * v

        # Create perturbed state
        z_perturbed = CompositeState(
            activity={**z.activity, activity_key: x_perturbed},
            plastic=z.plastic,
            substrate=z.substrate,
        )

        # Forward pass
        with torch.no_grad():
            z_next_base = transition_fn(z, context)
            z_next_perturbed = transition_fn(z_perturbed, context)

        # Extract activity difference
        delta = (
            z_next_perturbed.activity[activity_key] - z_next_base.activity[activity_key]
        )
        Jv = delta / perturbation_scale

        # Power iteration update
        v = Jv / (Jv.norm(dim=-1, keepdim=True) + 1e-8)

    # Final estimate: ||J * v||
    x_perturbed = x_base + perturbation_scale * v
    z_perturbed = CompositeState(
        activity={**z.activity, activity_key: x_perturbed},
        plastic=z.plastic,
        substrate=z.substrate,
    )

    with torch.no_grad():
        z_next_base = transition_fn(z, context)
        z_next_perturbed = transition_fn(z_perturbed, context)

    delta = z_next_perturbed.activity[activity_key] - z_next_base.activity[activity_key]
    Jv = delta / perturbation_scale

    # Spectral radius estimate
    rho = Jv.norm(dim=-1).mean().item()

    return rho


@dataclass(slots=True)
class SpectralRadiusEstimator:
    """Configurable spectral radius estimator for joint transitions.

    Supports both full power iteration (accurate) and fast proxy (CI).
    """

    num_iterations: int = 20
    perturbation_scale: float = 1e-4
    activity_key: str = "x"
    fast_mode: bool = False

    def __call__(
        self,
        transition_fn: Callable[[CompositeState, SystemContext], CompositeState],
        z: CompositeState,
        context: SystemContext,
    ) -> float:
        """Estimate spectral radius."""
        if self.fast_mode:
            return self._fast_proxy(transition_fn, z, context)
        return estimate_spectral_radius(
            transition_fn,
            z,
            context,
            num_iterations=self.num_iterations,
            perturbation_scale=self.perturbation_scale,
            activity_key=self.activity_key,
        )

    def _fast_proxy(
        self,
        transition_fn: Callable[[CompositeState, SystemContext], CompositeState],
        z: CompositeState,
        context: SystemContext,
    ) -> float:
        """Fast proxy: single-step norm ratio.

        ρ ≈ ||z_{t+1} - z_t|| / ||z_t - z_{t-1}|| for settling systems.
        Or single perturbation step for general case.
        """
        x = z.activity[self.activity_key]

        # Single finite-difference step
        eps = self.perturbation_scale
        v = torch.randn_like(x)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)

        x_perturbed = x + eps * v
        z_perturbed = CompositeState(
            activity={**z.activity, self.activity_key: x_perturbed},
            plastic=z.plastic,
            substrate=z.substrate,
        )

        with torch.no_grad():
            z_next = transition_fn(z, context)
            z_next_perturbed = transition_fn(z_perturbed, context)

        delta = (
            z_next_perturbed.activity[self.activity_key]
            - z_next.activity[self.activity_key]
        )
        Jv = delta / eps

        return Jv.norm(dim=-1).mean().item()


def estimate_spectral_radius_full_jacobian(
    transition_fn: Callable[[CompositeState, SystemContext], CompositeState],
    z: CompositeState,
    context: SystemContext,
    activity_key: str = "x",
) -> float:
    """Estimate spectral radius by computing full Jacobian (expensive, for validation).

    Computes the exact Jacobian via autograd and returns its spectral norm.
    Only feasible for small systems.

    Args:
        transition_fn: Joint transition function.
        z: Base joint state.
        context: System context.
        activity_key: Activity key to differentiate.

    Returns:
        Spectral norm of the Jacobian.
    """
    x = z.activity[activity_key].clone().requires_grad_(True)

    def forward(x_input: Tensor) -> Tensor:
        z_input = CompositeState(
            activity={**z.activity, activity_key: x_input},
            plastic=z.plastic,
            substrate=z.substrate,
        )
        z_out = transition_fn(z_input, context)
        return z_out.activity[activity_key]

    # Compute Jacobian
    jac = torch.autograd.functional.jacobian(forward, x)

    # jac shape: [batch, out_dim, batch, in_dim] -> take diagonal for single sample
    if jac.dim() == 4:
        # Assume batch=1 for full jacobian
        jac = jac[0, :, 0, :]

    # Spectral norm = largest singular value
    _U, S, _Vh = torch.linalg.svd(jac)
    return S[0].item()
