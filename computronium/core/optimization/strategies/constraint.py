"""Generic constraint strategies (REFACTOR.md §7).

No-op and spectral-norm-bound constraints. Subclasses can override
:meth:`SpectralConstraint._power_iteration` to inject a CUDA fast path.
"""

import torch
from torch import nn

from .base import ConstraintStrategy

__all__ = ["NoConstraint", "SpectralConstraint"]


class NoConstraint(ConstraintStrategy):
    """No parameter constraints."""

    def enforce(self, param: nn.Parameter, state: dict, group_config: dict) -> None:
        pass


class SpectralConstraint(ConstraintStrategy):
    """Enforce a spectral-norm bound via power iteration.

    If σ(W) > gamma, scales W by gamma/σ to ensure contractive dynamics.
    """

    EPSILON = 1e-6
    POWER_ITER = 3

    def __init__(
        self,
        gamma: float = 0.95,
        power_iter: int = 3,
        timing: str = "post_update",
    ):
        if not (0 < gamma <= 1):
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        if timing not in ("post_update", "during_settling", "both"):  # ruff: ignore[literal-membership]
            raise ValueError(
                "Spectral timing must be 'post_update', 'during_settling', or "
                f"'both', got '{timing}'"
            )

        self.gamma = gamma
        self.power_iter = power_iter
        self.timing = timing

    def enforce(self, param: nn.Parameter, state: dict, group_config: dict) -> None:
        """Enforce the spectral norm bound on 2D+ parameters."""
        if param.ndim < 2:
            return

        u = state.get("u_spec")
        v = state.get("v_spec")

        sigma, u, v = self._power_iteration(param.data, u, v)

        state["u_spec"] = u.detach()
        state["v_spec"] = v.detach()

        if sigma > self.gamma:
            param.data.mul_(self.gamma / sigma)

    def _power_iteration(
        self,
        W: torch.Tensor,
        u: torch.Tensor | None,
        v: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Estimate the spectral norm (CPU reference implementation)."""
        if W.ndim > 2:
            W = W.view(W.shape[0], -1)

        h, w = W.shape
        if u is None:
            u = torch.randn(h, device=W.device, dtype=W.dtype)
            u = u / (u.norm() + self.EPSILON)  # ruff: ignore[non-augmented-assignment]
        if v is None:
            v = torch.randn(w, device=W.device, dtype=W.dtype)
            v = v / (v.norm() + self.EPSILON)  # ruff: ignore[non-augmented-assignment]

        for _ in range(self.power_iter):
            v = W.T @ u
            v = v / (v.norm() + self.EPSILON)  # ruff: ignore[non-augmented-assignment]
            u = W @ v
            u = u / (u.norm() + self.EPSILON)  # ruff: ignore[non-augmented-assignment]

        sigma = (u @ W @ v).abs()
        return sigma, u, v

    def should_apply(self, timing: str) -> bool:
        """Check if the constraint should apply at the given ``timing``."""
        if self.timing == "both":
            return True
        return self.timing == timing
