"""
Built-in feedback function implementations.

Provides standard feedback mechanisms for the tile algorithm:
- Symmetric (transpose) feedback for EP/PC/TB/Hebbian
- No feedback for pure feedforward Hebbian / single-pass settling
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

    from computronium.core.local_learning.protocols import (
        TileGraph,
        TileState,
        WeightLookup,
    )


def symmetric_feedback(tile: TileState, graph: TileGraph, lookup: WeightLookup) -> list:
    """Symmetric (transpose) backward projection; EP/PC/TB/Hebbian default."""

    feedback: list[Tensor] = []
    for dst_id in tile.fwd_neighbors:
        dst = graph.tiles[dst_id]
        if dst.error is None:
            continue
        w = lookup(tile.id, dst_id)
        feedback.append(dst.error @ w)
    return feedback


def no_feedback(tile: TileState, graph: TileGraph, lookup: WeightLookup) -> list:
    """No downstream coupling (pure feedforward Hebbian / single-pass settling)."""
    return []


__all__ = [
    "no_feedback",
    "symmetric_feedback",
]
