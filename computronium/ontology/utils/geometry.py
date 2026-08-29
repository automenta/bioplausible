"""Shared geometry introspection utilities."""

import torch.nn as nn
from torch import Tensor

from computronium.ontology.geometry import Geometry


def _layer_stack(geometry: Geometry) -> nn.ModuleList | None:
    """Extract the layer ModuleList from a Geometry instance.

    Works across FeedforwardGeometry, RecurrentGeometry, and TileGeometry.
    """
    return getattr(geometry, "_layers", None)


def _recurrent_weight(geometry: Geometry) -> Tensor | None:
    """Extract the recurrent weight from a Geometry instance.

    Works with RecurrentGeometry which stores the recurrent weight.
    """
    return getattr(geometry, "_recurrent_weight", None)
