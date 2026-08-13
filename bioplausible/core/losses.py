"""Shared loss and accuracy utilities.

Consolidates duplicate implementations from ``core/trainer.py`` and
``graph/training.py`` into a single canonical source.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "compute_accuracy",
    "compute_loss",
    "reshape_for_cross_entropy",
]


def reshape_for_cross_entropy(
    logits: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Coerce logits/targets into ``F.cross_entropy`` compatible shapes.

    Models may emit 3-D logits ``[B, L, V]`` (LM autoregressive heads) where
    only the last token's prediction is supervised, and some datasets return
    float ``[B, 1]`` or one-hot targets.  ``F.cross_entropy`` requires logits
    ``[B, C]`` paired with long indices ``[B]``.
    """
    if logits.dim() == 3:
        logits = logits[:, -1, :]
    if y.dim() > 1 and y.size(-1) == 1:
        y = y.squeeze(-1)
    if y.dim() > 1 and y.shape[-1] == logits.shape[-1]:
        y = y.argmax(dim=-1)
    if y.dtype != torch.long:
        y = y.long()
    return logits, y


def compute_loss(
    loss_fn: Callable[..., torch.Tensor] | None,
    logits: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Compute loss using ``loss_fn`` or fallback cross-entropy.

    Handles 3-D logits (LM autoregressive heads), 2-D targets with singleton
    dimension, and non-long targets.
    """
    if loss_fn is not None:
        loss_input = logits
        loss_target = y
        if logits.dim() == 3:
            loss_input = logits[:, -1, :]
        if isinstance(loss_fn, nn.CrossEntropyLoss):
            if y.dim() > 1 and y.size(-1) == 1:
                loss_target = y.squeeze(-1).long()
            elif y.dtype != torch.long:
                loss_target = y.long()
        return loss_fn(loss_input, loss_target)
    logits_ce, y_ce = reshape_for_cross_entropy(logits, y)
    return torch.nn.functional.cross_entropy(logits_ce, y_ce)


def compute_accuracy(logits: torch.Tensor, y: torch.Tensor, scale: int = 1) -> float:
    """Accuracy via argmax, handling one-hot and reshaped targets.

    Returns a 0-1 ratio by default; pass ``scale=100`` for a percentage.
    """
    logits_ce, y_ce = reshape_for_cross_entropy(logits, y)
    with torch.no_grad():
        return (logits_ce.argmax(1) == y_ce).float().mean().item() * scale
