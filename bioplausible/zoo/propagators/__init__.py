"""
Zoo Propagators Package

Learning rules / propagators registered with the unified registry.
"""

from bioplausible.core.registry import register_propagator

from . import (
    backprop,
    base,
    eqprop,
    fa,
    hebbian,
    mep,
    spiking,
)

__all__: list[str] = [
    "backprop",
    "base",
    "eqprop",
    "fa",
    "hebbian",
    "mep",
    "register_propagator",
    "spiking",
]
