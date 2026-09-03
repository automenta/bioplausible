"""
Constraint strategies for parameter enforcement.

``NoConstraint`` / ``SpectralConstraint`` reuse the generic implementations
from :mod:`computronium.core.optimization.strategies` with the MEP CUDA
power-iteration fast path; ``SettlingSpectralPenalty`` (energy-based) stays
MEP-specific.
"""

import torch
from torch import nn

from computronium.core.optimization.strategies import (
    NoConstraint,
)
from computronium.core.optimization.strategies import (
    SpectralConstraint as _CoreSpectralConstraint,
)

__all__ = [
    "NoConstraint",
    "SettlingSpectralPenalty",
    "SpectralConstraint",
]
try:
    from ...cuda.kernels import spectral_norm_power_iteration_cuda

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


class SpectralConstraint(_CoreSpectralConstraint):
    """Spectral-norm bound via power iteration with the MEP CUDA fast path."""

    def _power_iteration(
        self,
        W: torch.Tensor,
        u: torch.Tensor | None,
        v: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Estimate spectral norm, dispatching to the CUDA kernel when possible."""
        if CUDA_AVAILABLE and W.is_cuda:
            return spectral_norm_power_iteration_cuda(
                W, u, v, niter=self.power_iter, epsilon=self.EPSILON
            )
        return super()._power_iteration(W, u, v)


class SettlingSpectralPenalty:
    """
    Spectral penalty added during settling energy computation.

    Adds a soft penalty to the energy function during settling:

        E_total = E_original + λ * Σ max(0, σ(W) - γ)²
    """

    def __init__(
        self,
        gamma: float = 0.95,
        lambda_penalty: float = 1.0,
    ):
        self.gamma = gamma
        self.lambda_penalty = lambda_penalty

    def compute_penalty(self, model: nn.Module, optimizer_state: dict) -> torch.Tensor:
        """Compute the spectral penalty term for the energy function."""
        penalty = torch.tensor(0.0, device=next(model.parameters()).device)

        for param in model.parameters():
            if param.ndim < 2:
                continue

            state = optimizer_state.get(id(param), {})
            u = state.get("u_spec") if state else None
            v = state.get("v_spec") if state else None

            sigma, u, v = self._power_iteration(param.data, u, v)

            if state:
                state["u_spec"] = u.detach()
                state["v_spec"] = v.detach()

            if sigma > self.gamma:
                diff = sigma - self.gamma
                penalty = penalty + self.lambda_penalty * (diff**2)  # ruff: ignore[non-augmented-assignment]

        return penalty

    def _power_iteration(
        self,
        W: torch.Tensor,
        u: torch.Tensor | None,
        v: torch.Tensor | None,
        niter: int = 3,
        epsilon: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Estimate spectral norm via power iteration."""
        if CUDA_AVAILABLE and W.is_cuda:
            return spectral_norm_power_iteration_cuda(
                W, u, v, niter=niter, epsilon=epsilon
            )

        if W.ndim > 2:
            W = W.view(W.shape[0], -1)

        h, w = W.shape

        if u is None:
            u = torch.randn(h, device=W.device, dtype=W.dtype)
            u = u / (u.norm() + epsilon)  # ruff: ignore[non-augmented-assignment]
        if v is None:
            v = torch.randn(w, device=W.device, dtype=W.dtype)
            v = v / (v.norm() + epsilon)  # ruff: ignore[non-augmented-assignment]

        for _ in range(niter):
            v = W.T @ u
            v = v / (v.norm() + epsilon)  # ruff: ignore[non-augmented-assignment]
            u = W @ v
            u = u / (u.norm() + epsilon)  # ruff: ignore[non-augmented-assignment]

        sigma = (u @ W @ v).abs()
        return sigma, u, v
