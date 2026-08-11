"""
Zoo Models Package

All models registered with the unified registry system.
"""

from bioplausible.core.registry import Domain, LocalityLevel, Registry, register_model

from . import (
    backprop,
    eqprop,
    fa,
    forward_only,
    hebbian,
    predictive_coding,
    spiking,
    target_prop,
    tile_fa,
)

__all__: list[str] = [
    "Domain",
    "LocalityLevel",
    "Registry",
    "backprop",
    "eqprop",
    "fa",
    "forward_only",
    "hebbian",
    "predictive_coding",
    "register_model",
    "spiking",
    "target_prop",
    "tile_fa",
]
