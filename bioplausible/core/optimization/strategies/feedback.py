"""Generic feedback strategies (REFACTOR.md §7)."""

from typing import cast

import torch

from .base import FeedbackStrategy

__all__ = ["ErrorFeedback", "NoFeedback"]


class NoFeedback(FeedbackStrategy):
    """No error accumulation."""

    def accumulate(
        self, gradient: torch.Tensor, state: dict, group_config: dict
    ) -> torch.Tensor:
        return gradient

    def update_buffer(
        self, residual: torch.Tensor, state: dict, group_config: dict
    ) -> None:
        pass


class ErrorFeedback(FeedbackStrategy):
    """Accumulate update residuals with exponential decay.

    g_aug = g + beta * error_buffer
    error_buffer = beta * error_buffer + (g_aug - update)
    """

    def __init__(self, beta: float = 0.9):
        if not (0 <= beta < 1):
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        self.beta = beta

    def accumulate(
        self, gradient: torch.Tensor, state: dict, group_config: dict
    ) -> torch.Tensor:
        """Accumulate residual and return the augmented gradient."""
        if "error_buffer" not in state:
            state["error_buffer"] = torch.zeros_like(gradient)
        buffer = cast("torch.Tensor", state["error_buffer"])
        return gradient + self.beta * buffer

    def update_buffer(
        self, residual: torch.Tensor, state: dict, group_config: dict
    ) -> None:
        """Update the error buffer with a new residual."""
        if "error_buffer" not in state:
            state["error_buffer"] = torch.zeros_like(residual)
        buffer = state["error_buffer"]
        buffer.mul_(self.beta).add_(residual)

        max_norm = group_config.get("max_grad_norm", 10.0) * 2
        buffer.clamp_(-max_norm, max_norm)
