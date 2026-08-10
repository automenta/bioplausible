"""Canonical optimizer factory (REFACTOR.md §2.3).

Single source of truth for ``torch.optim`` creation across the codebase.
Replaces ~40 inline ``torch.optim.Adam``/``AdamW``/``SGD`` construction
sites (deployments, LMs, zoo models, validation tracks) so optimizer
hyperparameters are config-driven rather than hardcoded per caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

__all__ = ["OptimizerConfig", "create_optimizer"]

OptimizerName = Literal["adam", "adamw", "sgd"]


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    """Hyperparameters for a standard ``torch.optim`` optimizer.

    Attributes:
        name: Optimizer family ("adam", "adamw", or "sgd").
        lr: Base learning rate.
        weight_decay: L2 / decoupled weight-decay coefficient.
        momentum: SGD momentum (ignored by Adam/AdamW).
        betas: Adam/AdamW beta coefficients.
        eps: Adam/AdamW numerical stability epsilon.
    """

    name: OptimizerName = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


def create_optimizer(
    model: nn.Module, config: OptimizerConfig
) -> torch.optim.Optimizer:
    """Create a ``torch.optim`` optimizer for *model* from *config*.

    Args:
        model: The module whose parameters the optimizer will update.
        config: Optimizer hyperparameters.

    Returns:
        A configured ``torch.optim.Optimizer`` bound to ``model.parameters()``.

    Raises:
        ValueError: If ``config.name`` is not a supported optimizer family.
    """
    params = model.parameters()
    match config.name:
        case "adam":
            return torch.optim.Adam(
                params,
                lr=config.lr,
                weight_decay=config.weight_decay,
                betas=config.betas,
                eps=config.eps,
            )
        case "adamw":
            return torch.optim.AdamW(
                params,
                lr=config.lr,
                weight_decay=config.weight_decay,
                betas=config.betas,
                eps=config.eps,
            )
        case "sgd":
            return torch.optim.SGD(
                params,
                lr=config.lr,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
            )
        case _:
            raise ValueError(f"Unknown optimizer: {config.name}")
