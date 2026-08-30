"""Canonical optimizer factory (REFACTOR.md §2.3).

Single source of truth for ``torch.optim`` creation across the codebase.
Replaces ~40 inline ``torch.optim.Adam``/``AdamW``/``SGD`` construction
sites (deployments, LMs, zoo models, validation tracks) so optimizer
hyperparameters are config-driven rather than hardcoded per caller.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn

__all__ = ["OptimizerConfig", "create_optimizer"]

OptimizerName = Literal["adam", "adamw", "sgd"]

# Type alias for an iterable acceptable as ``torch.optim`` ``params``:
# either an iterable of parameters / tensors, or an iterable of param-group
# dicts (``{"params": [...], "lr": ...}``).
ParamSpec = Iterable["nn.Parameter | torch.Tensor | dict[str, Any]"]


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
    model_or_params: nn.Module | ParamSpec,
    config: OptimizerConfig,
) -> torch.optim.Optimizer:
    """Create a ``torch.optim`` optimizer from *config*.

    Args:
        model_or_params: Either a ``nn.Module`` (uses ``model.parameters()``)
            or an explicit iterable of parameters / param-group dicts. The
            explicit form supports parameter subsets (e.g.
            ``[p for p in model.parameters() if p.requires_grad]``) and
            multi-group optimizers (e.g.
            ``[{"params": W_in.parameters()}, {"params": W_out.parameters(), "lr": 1e-4}]``).
        config: Optimizer hyperparameters.

    Returns:
        A configured ``torch.optim.Optimizer`` bound to the given parameters.

    Raises:
        ValueError: If ``config.name`` is not a supported optimizer family.
    """
    # Handle native System objects that have parameters() method
    if hasattr(model_or_params, 'parameters') and callable(model_or_params.parameters):
        params = model_or_params.parameters()
    elif isinstance(model_or_params, nn.Module):
        params = model_or_params.parameters()
    else:
        params = model_or_params
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
