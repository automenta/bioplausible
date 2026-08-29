"""
Zoo Models Package

All models registered with the unified registry system.
"""

from computronium.core.registry import LocalityLevel, Registry, register_model

# Import native models to trigger registration
from computronium.models import native as native_models  # noqa: F401

from . import (
    backprop,
    deployments,
    eqprop,
    fa,
    forward_only,
    hebbian,
    predictive_coding,
    spiking,
    target_prop,
    tile_fa,
    tile_lm,
    tile_models,
)

__all__: list[str] = [
    "Domain",
    "LocalityLevel",
    "Registry",
    "backprop",
    "deployments",
    "eqprop",
    "fa",
    "forward_only",
    "hebbian",
    "native_models",
    "predictive_coding",
    "register_model",
    "spiking",
    "target_prop",
    "tile_fa",
    "tile_lm",
    "tile_models",
]
