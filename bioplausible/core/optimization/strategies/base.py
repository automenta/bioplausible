"""Generic strategy protocols (REFACTOR.md §7).

Dependency-free interfaces for composable optimizers. Copied from
``zoo/mep/optimizers/strategies/base`` with the MEP docstrings trimmed;
core modules must never depend on the MEP package.
"""

from typing import Protocol

import torch
from torch import nn

__all__ = [
    "ConstraintStrategy",
    "FeedbackStrategy",
    "GradientStrategy",
    "UpdateStrategy",
]


class GradientStrategy(Protocol):
    """Strategy for computing and accumulating gradients into model params."""

    def compute_gradients(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target: torch.Tensor | None,
        **kwargs: object,
    ) -> None:
        """Compute and accumulate gradients into ``model`` parameters."""
        ...


class UpdateStrategy(Protocol):
    """Strategy for transforming raw gradients into parameter updates."""

    def transform_gradient(
        self,
        param: nn.Parameter,
        gradient: torch.Tensor,
        state: dict,
        group_config: dict,
    ) -> torch.Tensor:
        """Transform ``gradient`` into the update direction for ``param``."""
        ...


class ConstraintStrategy(Protocol):
    """Strategy for enforcing parameter constraints after each update."""

    def enforce(self, param: nn.Parameter, state: dict, group_config: dict) -> None:
        """Enforce a constraint on ``param`` in place."""
        ...


class FeedbackStrategy(Protocol):
    """Strategy for error/residual accumulation across steps."""

    def accumulate(
        self, gradient: torch.Tensor, state: dict, group_config: dict
    ) -> torch.Tensor:
        """Accumulate residual and return the augmented gradient."""
        ...

    def update_buffer(
        self, residual: torch.Tensor, state: dict, group_config: dict
    ) -> None:
        """Update the error buffer with a new residual."""
        ...
