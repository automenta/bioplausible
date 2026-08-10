"""Training mixin — shared training step protocol.

Provides a default ``train_step`` implementation that subclasses can compose
instead of reimplementing the same boilerplate (step counting, loss/accuracy
computation, metrics dict return).
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Protocol, Self

import torch


class _HasTrainStep(Protocol):
    """Protocol for objects that implement _forward_train."""

    @abstractmethod
    def _forward_train(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, dict]: ...

    def compute_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...

    def compute_metrics(self, logits: torch.Tensor, y: torch.Tensor) -> float: ...


@dataclass(eq=False, unsafe_hash=True)
class TrainingMixin:
    """Mixin providing a standard training step implementation.

    Subclasses must implement:
        - ``_forward_train(x, y)`` -> (logits, aux_dict)
        - ``compute_loss(logits, y)`` -> loss tensor
        - ``compute_metrics(logits, y)`` -> accuracy float

    The mixin handles step counting and metric aggregation.
    """

    _step_count: int = 0

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        """Execute one training step.

        Returns:
            Dict with at least ``loss`` and ``accuracy`` keys, plus any
            additional keys from ``_forward_train``'s aux dict.
        """
        self._step_count += 1
        logits, aux = self._forward_train(x, y)
        loss = self.compute_loss(logits, y)
        acc = self.compute_metrics(logits, y)
        return {"loss": loss.item(), "accuracy": acc, **aux}

    def reset_step_count(self: Self) -> None:
        """Reset internal step counter."""
        self._step_count = 0
