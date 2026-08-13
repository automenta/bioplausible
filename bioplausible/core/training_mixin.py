"""Training mixin — shared training step protocol.

Provides a default ``train_step`` implementation that subclasses can compose
instead of reimplementing the same boilerplate (step counting, loss/accuracy
computation, metrics dict return).
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, Self

import torch
from torch import nn

from bioplausible.core.losses import compute_accuracy

type LossFn = Callable[
    [nn.Module, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, dict[str, float] | None],
]


class _HasTrainStep(Protocol):
    """Protocol for objects that implement _forward_train."""

    @abstractmethod
    def _forward_train(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, dict]: ...

    def compute_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...

    def compute_metrics(self, logits: torch.Tensor, y: torch.Tensor) -> float: ...


def _default_loss(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float] | None]:
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    return loss, logits, None


def supervised_step(  # ruff: ignore[too-many-arguments]  # training-step contract: model, optimizer, x, y, + 2 kwargs
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    grad_clip: float | None = None,
    loss_fn: LossFn | None = None,
    extra_keys: dict[str, float] | None = None,
) -> dict[str, float]:
    """Run one standard supervised optimizer step.

    Canonical ``zero_grad → forward → loss → backward → (clip →) step`` shape
    shared by plain-BPTT models (e.g. ``eqprop/_unified.py:EquilibriumMLP``).
    Returns a ``{"loss", "accuracy"}`` metrics dict for the probe/trainer.

    Args:
        model: The module being trained.
        optimizer: Optimizer driving *model*'s parameters.
        x: Input batch.
        y: Target batch.
        grad_clip: Optional global gradient-norm clipping value.
        loss_fn: Optional ``(model, x, y) -> (loss, logits, extras)`` callable
            replacing the default ``CE(model(x), y)``. Enables custom forward
            flows (e.g. composite losses collecting intermediate activations).
            The third element (``extras``) is merged into the returned dict.
        extra_keys: Optional metrics to merge into the returned dict beyond
            ``{"loss", "accuracy"}``.
    """
    optimizer.zero_grad()
    fn = loss_fn or _default_loss
    loss, logits, fn_extras = fn(model, x, y)
    loss.backward()
    if grad_clip:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    acc = compute_accuracy(logits, y)
    result = {"loss": float(loss.item()), "accuracy": acc}
    if extra_keys:
        result.update(extra_keys)
    if fn_extras:
        result.update(fn_extras)
    return result


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
