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

        r, c = G.shape
        transposed = False
        if r < c:
            G = G.T
            r, c = c, r
            transposed = True

        X = G.clone()
        norm = X.norm().clamp(min=1e-4, max=1e4)
        X = X / norm  # ruff: ignore[non-augmented-assignment]

        identity = torch.eye(c, device=G.device, dtype=G.dtype)
        for _ in range(steps):
            A = X.T @ X
            X = 0.5 * X @ (3 * identity - A)

        if transposed:
            X = X.T

        return cast("torch.Tensor", X)
