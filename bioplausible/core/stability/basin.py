"""Basin Stability Estimation: Robustness to perturbations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch

from bioplausible.core.joint.state import CompositeState
if TYPE_CHECKING:
    from bioplausible.core.joint.context import SystemContext


def estimate_basin_stability(
    transition_fn: Callable[["CompositeState", "SystemContext"], "CompositeState"],
    z_attractor: "CompositeState",
    context: "SystemContext",
    num_samples: int = 100,
    perturbation_radius: float = 1.0,
    max_steps: int = 200,
    tolerance: float = 1e-3,
    activity_key: str = "x",
    distance_metric: str = "euclidean",
) -> float:
    """Estimate basin stability by sampling perturbations.

    Basin stability = fraction of perturbed initial conditions
    that converge back to the attractor.

    Args:
        transition_fn: Joint transition function.
        z_attractor: Attractor state (e.g., settled fixed point).
        context: System context.
        num_samples: Number of perturbation samples.
        perturbation_radius: Radius of perturbation ball.
        max_steps: Max steps to check convergence.
        tolerance: Convergence tolerance to attractor.
        activity_key: Activity key to measure distance.
        distance_metric: "euclidean" or "cosine".

    Returns:
        Basin stability estimate in [0, 1].
    """
    x_attractor = z_attractor.activity[activity_key]
    batch_size, _dim = x_attractor.shape

    # Get attractor activity for reference
    z_ref = z_attractor
    with torch.no_grad():
        for _ in range(10):  # Ensure attractor is settled
            z_ref = transition_fn(z_ref, context)
    x_attractor = z_ref.activity[activity_key]

    converged = 0

    for _ in range(num_samples):
        # Sample random perturbation on sphere of radius perturbation_radius
        direction = torch.randn_like(x_attractor)
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

        # Scale by random radius in [0, perturbation_radius]
        radius = torch.rand(batch_size, 1, device=x_attractor.device) * perturbation_radius
        perturbation = direction * radius

        # Perturbed initial state
        x_perturbed = x_attractor + perturbation
        z_perturbed = CompositeState(
            activity={**z_attractor.activity, activity_key: x_perturbed},
            plastic=z_attractor.plastic,
            substrate=z_attractor.substrate,
        )

        # Evolve and check convergence
        z_current = z_perturbed
        for _ in range(max_steps):
            with torch.no_grad():
                z_next = transition_fn(z_current, context)

            x_current = z_next.activity[activity_key]
            delta = x_current - x_attractor

            if distance_metric == "euclidean":
                dist = delta.norm(dim=-1).mean()
            else:
                # Cosine distance
                cos_sim = torch.nn.functional.cosine_similarity(x_current, x_attractor, dim=-1)
                dist = (1 - cos_sim).mean()

            if dist < tolerance:
                converged += 1
                break

            z_current = z_next

    return converged / num_samples


@dataclass(slots=True)
class BasinStabilityEstimator:
    """Configurable basin stability estimator."""

    num_samples: int = 100
    perturbation_radius: float = 1.0
    max_steps: int = 200
    tolerance: float = 1e-3
    activity_key: str = "x"
    distance_metric: str = "euclidean"
    fast_mode: bool = False

    def __call__(
        self,
        transition_fn: Callable[["CompositeState", "SystemContext"], "CompositeState"],
        z_attractor: "CompositeState",
        context: "SystemContext",
    ) -> float:
        """Estimate basin stability."""
        if self.fast_mode:
            return self._fast_proxy(transition_fn, z_attractor, context)
        return estimate_basin_stability(
            transition_fn,
            z_attractor,
            context,
            num_samples=self.num_samples,
            perturbation_radius=self.perturbation_radius,
            max_steps=self.max_steps,
            tolerance=self.tolerance,
            activity_key=self.activity_key,
            distance_metric=self.distance_metric,
        )

    def _fast_proxy(
        self,
        transition_fn: Callable[["CompositeState", "SystemContext"], "CompositeState"],
        z_attractor: "CompositeState",
        context: "SystemContext",
    ) -> float:
        """Fast proxy: single perturbation test + local linearization.

        Uses the Jacobian at the attractor to estimate basin size.
        Basin radius ≈ tolerance / ||J - I|| where J is Jacobian at fixed point.
        """
        x = z_attractor.activity[self.activity_key]
        eps = 1e-4

        # Estimate Jacobian at attractor via finite differences
        J_norms = []
        for _ in range(min(5, x.shape[-1])):  # Sample a few directions
            v = torch.randn_like(x)
            v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)

            x_perturbed = x + eps * v
            z_perturbed = CompositeState(
                activity={**z_attractor.activity, self.activity_key: x_perturbed},
                plastic=z_attractor.plastic,
                substrate=z_attractor.substrate,
            )

            with torch.no_grad():
                z_next = transition_fn(z_attractor, context)
                z_next_perturbed = transition_fn(z_perturbed, context)

            delta = z_next_perturbed.activity[self.activity_key] - z_next.activity[self.activity_key]
            Jv = delta / eps
            J_norms.append(Jv.norm(dim=-1).mean().item())

        if not J_norms:
            return 0.5  # Unknown, neutral

        avg_J_norm = sum(J_norms) / len(J_norms)

        # Linearized basin estimate
        # For x_{t+1} = J x_t, basin where ||J^t x_0|| < tolerance
        # Approx: basin_radius ~ tolerance / (1 - rho(J)) for stable systems
        if avg_J_norm < 1.0:
            basin_radius_est = self.tolerance / (1.0 - avg_J_norm)
            # Normalize by perturbation_radius
            return min(1.0, basin_radius_est / self.perturbation_radius)
        else:
            # Unstable fixed point
            return 0.0


def estimate_basin_stability_multistart(
    transition_fn: Callable[["CompositeState", "SystemContext"], "CompositeState"],
    z_attractor: "CompositeState",
    context: "SystemContext",
    num_samples: int = 100,
    perturbation_radii: list[float] | None = None,
    max_steps: int = 200,
    tolerance: float = 1e-3,
    activity_key: str = "x",
) -> dict[float, float]:
    """Estimate basin stability at multiple perturbation radii.

    Returns a radius -> stability mapping for basin profile.
    """
    if perturbation_radii is None:
        perturbation_radii = [0.1, 0.5, 1.0, 2.0, 5.0]

    results = {}
    for radius in perturbation_radii:
        stability = estimate_basin_stability(
            transition_fn,
            z_attractor,
            context,
            num_samples=num_samples,
            perturbation_radius=radius,
            max_steps=max_steps,
            tolerance=tolerance,
            activity_key=activity_key,
        )
        results[radius] = stability

    return results