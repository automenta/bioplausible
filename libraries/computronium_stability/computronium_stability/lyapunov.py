"""Local Lyapunov Exponent Estimation: Sensitivity and divergence metrics.

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


def estimate_lyapunov_exponent(
    transition_fn: TransitionFn,
    state: StepState,
    num_steps: int = 50,
    perturbation_scale: float = 1e-6,
    activity_key: str = "x",
    renormalize_interval: int = 1,
) -> float:
    """Estimate local Lyapunov exponent of the transition.

    Tracks divergence of nearby trajectories to measure sensitivity.
    Positive exponent → chaotic/divergent; Negative → stable/convergent.

    Args:
        transition_fn: State transition function.
        state: Initial state.
        num_steps: Number of steps to track divergence.
        perturbation_scale: Initial perturbation magnitude.
        activity_key: Key in state to perturb.
        renormalize_interval: Steps between perturbation renormalization.

    Returns:
        Estimated local Lyapunov exponent (per step).
    """
    x = state.get(activity_key)
    if x is None or not isinstance(x, Tensor):
        return 0.0

    _batch_size, _dim = x.shape

    # Initialize perturbation vector
    v = torch.randn_like(x)
    v = v * (perturbation_scale / (v.norm(dim=-1, keepdim=True) + 1e-8))

    x_base = x.clone()
    x_perturbed = x_base + v

    state_base = state
    state_perturbed = {**state, activity_key: x_perturbed}

    log_divergences = []

    for step in range(num_steps):
        # Step both trajectories
        with torch.no_grad():
            state_base = transition_fn(state_base)
            state_perturbed = transition_fn(state_perturbed)

        x_base = state_base.get(activity_key)
        x_pert = state_perturbed.get(activity_key)
        if x_base is None or x_pert is None:
            break

        # Compute separation
        delta = x_pert - x_base
        separation = delta.norm(dim=-1).mean()

        if separation > 1e-12:
            log_divergences.append(torch.log(separation / perturbation_scale).item())

            # Renormalize perturbation
            if (step + 1) % renormalize_interval == 0:
                v = delta * (perturbation_scale / (separation + 1e-12))
                x_perturbed = x_base + v
                state_perturbed = {**state_base, activity_key: x_perturbed}

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
        transition_fn: TransitionFn,
        state: StepState,
    ) -> float:
        """Estimate local Lyapunov exponent."""
        if self.fast_mode:
            return self._fast_proxy(transition_fn, state)
        return estimate_lyapunov_exponent(
            transition_fn,
            state,
            num_steps=self.num_steps,
            perturbation_scale=self.perturbation_scale,
            activity_key=self.activity_key,
            renormalize_interval=self.renormalize_interval,
        )

    def _fast_proxy(
        self,
        transition_fn: TransitionFn,
        state: StepState,
    ) -> float:
        """Fast proxy: single-step log separation growth.

        λ ≈ log(||δ_{t+1}|| / ||δ_t||)
        """
        x = state.get(self.activity_key)
        if x is None or not isinstance(x, Tensor):
            return 0.0

        eps = self.perturbation_scale

        v = torch.randn_like(x)
        v = v * (eps / (v.norm(dim=-1, keepdim=True) + 1e-8))

        x_perturbed = x + v
        state_perturbed = {**state, self.activity_key: x_perturbed}

        with torch.no_grad():
            next_state = transition_fn(state)
            next_perturbed = transition_fn(state_perturbed)

        next_act = next_perturbed.get(self.activity_key)
        base_act = next_state.get(self.activity_key)
        if next_act is None or base_act is None:
            return 0.0

        delta_t = v
        delta_t1 = next_act - base_act

        sep_t = delta_t.norm(dim=-1).mean()
        sep_t1 = delta_t1.norm(dim=-1).mean()

        if sep_t > 1e-12 and sep_t1 > 1e-12:
            return torch.log(sep_t1 / sep_t).item()
        return 0.0


def estimate_lyapunov_spectrum(
    transition_fn: TransitionFn,
    state: StepState,
    num_vectors: int = 5,
    num_steps: int = 100,
    perturbation_scale: float = 1e-6,
    activity_key: str = "x",
) -> list[float]:
    """Estimate Lyapunov spectrum using multiple orthogonal perturbations (QR method).

    More accurate but expensive. Tracks multiple perturbation vectors
    and uses QR decomposition to maintain orthogonality.

    Args:
        transition_fn: State transition function.
        state: Initial state.
        num_vectors: Number of perturbation vectors (spectrum dimension).
        num_steps: Number of steps.
        perturbation_scale: Initial perturbation scale.
        activity_key: Activity key.

    Returns:
        List of Lyapunov exponents (sorted descending).
    """
    x = state.get(activity_key)
    if x is None or not isinstance(x, Tensor):
        return []

    _batch_size, dim = x.shape

    # Initialize orthonormal perturbation matrix
    q_mat = torch.randn(num_vectors, dim, device=x.device, dtype=x.dtype)
    q_mat, _ = torch.linalg.qr(q_mat.T)
    q_mat = q_mat.T  # [num_vectors, dim]

    log_sums = torch.zeros(num_vectors, device=x.device)

    state_current = state
    state_next = state_current

    for _step in range(num_steps):
        # Apply perturbations
        perturbations = q_mat * perturbation_scale  # [num_vectors, dim]

        new_q = []
        for i in range(num_vectors):
            x_perturbed = x + perturbations[i]
            state_perturbed = {
                **state_current,
                activity_key: x_perturbed,
            }

            with torch.no_grad():
                state_next = transition_fn(state_current)
                state_next_perturbed = transition_fn(state_perturbed)

            next_act = state_next_perturbed.get(activity_key)
            base_act = state_next.get(activity_key)
            if next_act is None or base_act is None:
                continue

            delta = next_act - base_act
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

        state_current = state_next

    # Exponents = average log norm per step
    exponents = (log_sums / num_steps).tolist()
    return sorted(exponents, reverse=True)