"""Generic update-transformation strategies (REFACTOR.md §7).

Plain SGD and Muon (Newton-Schulz orthogonalization). CUDA kernels are
handled opaquely: the GPU fast-path is injected by the MEP package, so the
core stays dependency-free.
"""

from typing import TYPE_CHECKING, cast

import torch
from torch import nn

from .base import UpdateStrategy

if TYPE_CHECKING:
    from collections.abc import Callable


def newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Canonical Muon Newton–Schulz orthogonalization (Jordan et al.).

    Quintic iteration ``X ← aX + (bA + cA²)X`` with A = XXᵀ and the
    (3.4445, −4.7750, 2.0315) coefficient schedule: five steps drive every
    singular value of the Frobenius-normalized matrix into a narrow band
    near 1, so the result is approximately semi-orthogonal regardless of
    the input spectrum — the exact polar factor is NOT required for a
    descent-aligned direction (the naive ``0.5·X(3I − XᵀX)`` iteration
    used previously under-converges from Frobenius normalization and
    measured orthonormality error ~0.85 on Gaussian matrices).

    Returns the update direction as ``float32``.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    # float32 everywhere: bfloat16 is Muon's GPU speed choice but is
    # catastrophically slow on CPU (no wide AMX path) — and fp32 keeps
    # CPU/CUDA numerics in one tolerance regime.
    X = G.float()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(torch.float32)


__all__ = ["MuonUpdate", "PlainUpdate"]


class PlainUpdate(UpdateStrategy):
    """Vanilla SGD update (gradient used directly)."""

    def transform_gradient(
        self,
        param: nn.Parameter,
        gradient: torch.Tensor,
        state: dict,
        group_config: dict,
    ) -> torch.Tensor:
        return gradient


class MuonUpdate(UpdateStrategy):
    """Newton-Schulz orthogonalization (Muon optimizer).

    Iteratively orthogonalizes the gradient:
        X_{k+1} = 0.5 * X_k * (3I - X_k^T X_k)
    producing a well-conditioned (near-orthogonal) update direction.
    """

    def __init__(self, ns_steps: int = 5, newton_schulz: Callable | None = None):
        self.ns_steps = ns_steps
        self._ns = newton_schulz

    def transform_gradient(
        self,
        param: nn.Parameter,
        gradient: torch.Tensor,
        state: dict,
        group_config: dict,
    ) -> torch.Tensor:
        orig_shape = None
        if gradient.ndim > 2:
            orig_shape = gradient.shape
            gradient = gradient.view(gradient.shape[0], -1)
        elif gradient.ndim < 2:
            return gradient

        update = self._newton_schulz(gradient, self.ns_steps)

        if orig_shape is not None:
            update = update.view(orig_shape)

        return update

    def _newton_schulz(
        self, G: torch.Tensor, steps: int, epsilon: float = 1e-4
    ) -> torch.Tensor:
        """Newton-Schulz orthogonalization (CPU fallback)."""
        if self._ns is not None:
            return cast("torch.Tensor", self._ns(G, steps=steps, epsilon=epsilon))

        return newton_schulz5(G, steps)
