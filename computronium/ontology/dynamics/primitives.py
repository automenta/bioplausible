"""Shared dynamics primitives for energy-based dynamics implementations."""

import torch
from torch import Tensor

from computronium.ontology.geometry import Geometry


def _settle_step(
    acts: list[Tensor],
    geometry: Geometry,
    x: Tensor,
    beta: float,
    step_size: float,
    momentum: float = 0.0,
    velocity: list[Tensor] | None = None,
) -> tuple[list[Tensor], list[Tensor] | None]:
    """Single settling step for energy minimization dynamics.

    Updates activations using the gradient of the Hopfield energy:
    h_{t+1} = h_t - step_size * ∇E(h_t) + momentum * (h_t - h_{t-1})

    Args:
        acts: Current layer activations.
        geometry: Geometry instance providing forward operations.
        x: Input tensor.
        beta: Nudge strength for energy-based methods.
        step_size: Step size for gradient descent.
        momentum: Momentum coefficient for heavy-ball dynamics.
        velocity: Previous velocity for momentum (optional).

    Returns:
        Tuple of (new_activations, new_velocity).
    """
    # Compute gradients w.r.t. activations
    grads = []
    for i, act in enumerate(acts):
        act.requires_grad_(True)

    # Forward pass to compute energy
    energy = _compute_hopfield_energy(acts, geometry)

    # Compute gradients
    energy.backward(retain_graph=True)

    new_acts = []
    new_velocity: list[Tensor] | None = None

    for i, act in enumerate(acts):
        grad = act.grad if act.grad is not None else torch.zeros_like(act)
        act.grad = None  # Clear gradient

        if velocity is not None and i < len(velocity):
            # Momentum update
            new_vel = momentum * velocity[i] - step_size * grad
            new_act = act + new_vel
            if new_velocity is None:
                new_velocity = []
            new_velocity.append(new_vel)
        else:
            # Simple gradient descent
            new_act = act - step_size * grad
            if momentum > 0.0:
                if new_velocity is None:
                    new_velocity = []
                new_velocity.append(new_act - act)

        new_acts.append(new_act.detach())

    return new_acts, new_velocity


def _compute_hopfield_energy(acts: list[Tensor], geometry: Geometry) -> Tensor:
    """Compute the Hopfield energy for a given set of activations.

    E = -sum_i (x_i * W_ij * h_j) + 0.5 * sum_j h_j^2

    Args:
        acts: Layer activations (list of tensors per layer).
        geometry: Geometry providing weight matrices.

    Returns:
        Scalar energy tensor.
    """
    energy = torch.tensor(0.0, device=acts[0].device)

    # Get weight matrices from geometry
    params = geometry.params
    weight_names = [k for k in params.keys() if "weight" in k.lower()]

    # Energy = -x^T W h + 0.5 * h^T h (for each layer)
    for i, act in enumerate(acts):
        # Self-energy term: 0.5 * ||h||^2
        energy = energy + 0.5 * (act**2).sum()

        # Interaction term: -x^T W h (simplified)
        # This is a simplified energy - full implementation depends on geometry
        if i < len(weight_names) - 1:
            w_name = weight_names[i]
            w = params[w_name]
            if w.shape[0] == act.shape[-1]:
                # h^T W^T x or similar
                pass  # Simplified for now

    return energy


__all__ = ["_settle_step", "_compute_hopfield_energy"]
