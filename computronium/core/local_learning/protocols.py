"""
Dynamics protocols for the tile algorithm (extensibility surface).

These protocols define the three algorithm-specific injection points:
- FeedbackFn: how downstream error reaches a tile
- ActivityUpdateFn: how a tile settles its activity
- WeightUpdateFn: how edge weights change from free/nudged statistics
"""

from collections.abc import Callable
from typing import Protocol

from torch import Tensor

from computronium.core.tile import TileGraph, TileState

# Type alias for weight lookup function
type WeightLookup = Callable[[int, int], Tensor]


class FeedbackFn(Protocol):
    """Downstream error projected back into a tile's state space."""

    def __call__(
        self, tile: TileState, graph: TileGraph, lookup: WeightLookup
    ) -> list[Tensor]: ...


class ActivityUpdateFn(Protocol):
    """Settle a tile: current state + prediction error + feedback -> new activity."""

    def __call__(
        self,
        tile: TileState,
        *,
        feedback: list[Tensor],
        importance: float,
        step_size: float,
        lambda_error: float,
        clamp_min: float,
        clamp_max: float,
        clamp: bool,
    ) -> Tensor: ...


class WeightUpdateFn(Protocol):
    """Per-edge weight/bias deltas from free and nudged activity statistics."""

    def __call__(
        self,
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
    ) -> tuple[Tensor, Tensor]: ...


__all__ = [
    "ActivityUpdateFn",
    "FeedbackFn",
    "WeightLookup",
    "WeightUpdateFn",
]
