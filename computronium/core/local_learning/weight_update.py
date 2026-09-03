"""
Built-in weight update function implementations.

Provides standard weight update rules for the tile algorithm:
- Contrastive Hebbian (Equilibrium Propagation): (free stats - nudged stats) / beta
- Pure local Hebbian: importance * avg(src_free x dst_free)
"""

import torch
from torch import Tensor

from computronium.core.tile.kernels import (
    compute_contrastive_hebbian_update,
    compute_hebbian_update,
)


def contrastive_weight_update(  # dynamics contract signature  # ruff: ignore[too-many-arguments]
    *,
    src_neurons: int,
    dst_neurons: int,
    src_free: Tensor | None,
    dst_free: Tensor | None,
    src_nudged: Tensor | None,
    dst_nudged: Tensor | None,
    learning_rate: float,
    beta: float,
    batch_size: int,
    importance: float,
) -> tuple[Tensor, Tensor]:
    """Contrastive Hebbian: (free stats - nudged stats) / beta (Equilibrium Prop)."""
    if src_free is None or dst_free is None or src_nudged is None or dst_nudged is None:
        return torch.zeros(dst_neurons, src_neurons), torch.zeros(dst_neurons)
    w_up, b_up = compute_contrastive_hebbian_update(
        src_free=src_free,
        dst_free=dst_free,
        src_nudged=src_nudged,
        dst_nudged=dst_nudged,
        learning_rate=learning_rate,
        beta=beta,
        batch_size=batch_size,
    )
    # Kernel returns (src_neurons, dst_neurons); weights are (dst_neurons, src_neurons)
    return importance * w_up.T, importance * b_up


def hebbian_weight_update(  # dynamics contract signature  # ruff: ignore[too-many-arguments]
    *,
    src_neurons: int,
    dst_neurons: int,
    src_free: Tensor | None,
    dst_free: Tensor | None,
    src_nudged: Tensor | None,
    dst_nudged: Tensor | None,
    learning_rate: float,
    beta: float,
    batch_size: int,
    importance: float,
) -> tuple[Tensor, Tensor]:
    """Pure local Hebbian: importance * avg(src_free x dst_free)."""
    if src_free is None or dst_free is None:
        return torch.zeros(dst_neurons, src_neurons), torch.zeros(dst_neurons)
    w_up, b_up = compute_hebbian_update(
        src_act=src_free, dst_err=dst_free, importance=importance, batch_size=batch_size
    )
    return w_up.T, b_up


__all__ = [
    "contrastive_weight_update",
    "hebbian_weight_update",
]
