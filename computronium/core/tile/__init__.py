"""Generic tile substrate: graph topology, state, and math kernels."""

from computronium.core.tile.kernels import (
    compute_activity_update,
    compute_contrastive_hebbian_update,
    compute_hebbian_update,
    compute_tile_prediction,
)
from computronium.core.tile.topology import TileGraph, TileState

__all__ = [
    "TileGraph",
    "TileState",
    "compute_activity_update",
    "compute_contrastive_hebbian_update",
    "compute_hebbian_update",
    "compute_tile_prediction",
]
