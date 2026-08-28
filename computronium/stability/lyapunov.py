"""Local Lyapunov Exponent Estimation: Sensitivity and divergence metrics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from computronium.state import CompositeState

if TYPE_CHECKING:
    from computronium.state import SystemContext


def estimate_lyapunov_exponent(
    transition_fn: Callable[[CompositeState, SystemContext], CompositeState],
    z: CompositeState,
    context: SystemContext,
    num_steps: int = 50,
    perturbation_scale: float = 1e-6,
    activity_key: str = "x",
    renormalize_interval: int = 1,
) -> float:
    """Estimate local Lyapunov exponent of the joint transition.

    Tracks divergence of nearby trajectories to measure sensitivity.
    Positive exponent → chaotic/divergent; Negative → stable/convergent.

    Args:
        transition_fn: Joint transition function F_θ(z; G, S, M).
        z: Initial joint state.
        context: System context.
        num_steps: Number of steps to track divergence.
        perturbation_scale: Initial perturbation magnitude.
        activity_key: Key in z.activity to perturb.
        renormalize_interval: Steps between perturbation renormalization.

    Returns:
        Estimated local Lyapunov exponent (per step).
    """
    x = z.activity[activity_key]
    _batch_size, _dim = x.shape

    # Initialize perturbation vector
    v = torch.randn_like(x)
    v = v * (perturbation_scale / (v.norm(dim=-1, keepdim=True) + 1e-8))

    x_base = x.clone()
    x_perturbed = x_base + v

    z_base = z
    z_perturbed = CompositeState(
        activity={**z.activity, activity_key: x_perturbed},
        plastic=z.plastic,
        substrate=z.substrate,
    )

    log_divergences = []

    for step in range(num_steps):
        # Step both trajectories
        with torch.no_grad():
            z_base = transition_fn(z_base, context)
            z_perturbed = transition_fn(z_perturbed, context)

        x_base = z_base.activity[activity_key]
        x_pert = z_perturbed.activity[activity_key]

        # Compute separation
        delta = x_pert - x_base
        separation = delta.norm(dim=-1).mean()

        if separation > 1e-12:
            log_divergences.append(torch.log(separation / perturbation_scale).item())

            # Renormalize perturbation
            if (step + 1) % renormalize_interval == 0:
                v = delta * (perturbation_scale / (separation + 1e-12))
                x_perturbed = x_base + v
                z_perturbed = CompositeState(
                    activity={**z_base.activity, activity_key: x_perturbed},
                    plastic=z_base.plastic,
                    substrate=z_base.substrate,
                )

    if not log_divergences:
        return 0.0

    # Average log divergence per step
    return sum(log_divergences) / len(log_divergences)


@dataclass(slots=True)
class LyapunovEstimator:
    """Configurable local Lyapunov exponent estimator."""

    num_steps: int = 50
    perturbation_scale: float = 1e-6
    activity_key: str = "x"
    renormalize_interval: int = 1
    fast_mode: bool = False

    def __call__(
        self,
        transition_fn: Callable[[CompositeState, SystemContext], CompositeState],
        z: CompositeState,
        context: SystemContext,
    ) -> float:
        """Estimate local Lyapunov exponent."""
        if self.fast_mode:
            return self._fast_proxy(transition_fn, z, context)
        return estimate_lyapunov_exponent(
            transition_fn,
            z,
            context,
            num_steps=self.num_steps,
            perturbation_scale=self.perturbation_scale,
            activity_key=self.activity_key,
            renormalize_interval=self.renormalize_interval,
        )

    def _fast_proxy(
        self,
        transition_fn: Callable[[CompositeState, SystemContext], CompositeState],
        z: CompositeState,
        context: SystemContext,
    ) -> float:
        """Fast proxy: single-step log separation growth.

        λ ≈ log(||δ_{t+1}|| / ||δ_t||)
        """
        x = z.activity[self.activity_key]
        eps = self.perturbation_scale

        v = torch.randn_like(x)
        v = v * (eps / (v.norm(dim=-1, keepdim=True) + 1e-8))

        x_perturbed = x + v
        z_perturbed = CompositeState(
            activity={**z.activity, self.activity_key: x_perturbed},
            plastic=z.plastic,
            substrate=z.substrate,
        )

        with torch.no_grad():
            z_next = transition_fn(z, context)
            z_next_perturbed = transition_fn(z_perturbed, context)

        delta_t = v
        delta_t1 = (
            z_next_perturbed.activity[self.activity_key]
            - z_next.activity[self.activity_key]
        )

        sep_t = delta_t.norm(dim=-1).mean()
        sep_t1 = delta_t1.norm(dim=-1).mean()

        if sep_t > 1e-12 and sep_t1 > 1e-12:
            return torch.log(sep_t1 / sep_t).item()
        return 0.0


def estimate_lyapunov_spectrum(
    transition_fn: Callable[[CompositeState, SystemContext], CompositeState],
    z: CompositeState,
    context: SystemContext,
    num_vectors: int = 5,
    num_steps: int = 100,
    perturbation_scale: float = 1e-6,
    activity_key: str = "x",
) -> list[float]:
    """Estimate Lyapunov spectrum using multiple orthogonal perturbations (QR method).

    More accurate but expensive. Tracks multiple perturbation vectors
    and uses QR decomposition to maintain orthogonality.

    Args:
        transition_fn: Joint transition function.
        z: Initial joint state.
        context: System context.
        num_vectors: Number of perturbation vectors (spectrum dimension).
        num_steps: Number of steps.
        perturbation_scale: Initial perturbation scale.
        activity_key: Activity key.

    Returns:
        List of Lyapunov exponents (sorted descending).
    """
    x = z.activity[activity_key]
    _batch_size, dim = x.shape

    # Initialize orthonormal perturbation matrix
    q_mat = torch.randn(num_vectors, dim, device=x.device, dtype=x.dtype)
    q_mat, _ = torch.linalg.qr(q_mat.T)
    q_mat = q_mat.T  # [num_vectors, dim]

    log_sums = torch.zeros(num_vectors, device=x.device)

    z_current = z
    z_next = z_current

    for _step in range(num_steps):
        # Apply perturbations
        perturbations = q_mat * perturbation_scale  # [num_vectors, dim]

        # We track each vector separately for simplicity
        # In practice, would use batched Jacobian-vector products
        new_q = []
        for i in range(num_vectors):
            x_perturbed = x + perturbations[i]
            z_perturbed = CompositeState(
                activity={**z_current.activity, activity_key: x_perturbed},
                plastic=z_current.plastic,
                substrate=z_current.substrate,
            )

            with torch.no_grad():
                z_next = transition_fn(z_current, context)
                z_next_perturbed = transition_fn(z_perturbed, context)

            delta = (
                z_next_perturbed.activity[activity_key] - z_next.activity[activity_key]
            )
            new_q.append(delta / perturbation_scale)

            # Accumulate log norm
            norm = delta.norm() / perturbation_scale
            if norm > 1e-12:
                log_sums[i] += torch.log(norm)

        # QR decomposition to re-orthogonalize
        if new_q:
            new_q = torch.stack(new_q)  # [num_vectors, dim]
            q_mat, _ = torch.linalg.qr(new_q.T)
            q_mat = q_mat.T

        z_current = z_next

    # Exponents = average log norm per step
    exponents = (log_sums / num_steps).tolist()
    return sorted(exponents, reverse=True)
