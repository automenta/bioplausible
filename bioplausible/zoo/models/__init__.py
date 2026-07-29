"""
Zoo Models Package

All models registered with the unified registry system.
"""

from bioplausible.core.registry import Domain, LocalityLevel, Registry, register_model

from . import (
    backprop,  # ruff: ignore[unused-import]
    eqprop,  # ruff: ignore[unused-import]
    fa,  # ruff: ignore[unused-import]
    forward_only,  # ruff: ignore[unused-import]
    hebbian,  # ruff: ignore[unused-import]
    predictive_coding,  # ruff: ignore[unused-import]
    spiking,  # ruff: ignore[unused-import]
    target_prop,  # ruff: ignore[unused-import]
)

__all__: list[str] = [
    "Domain",
    "LocalityLevel",
    "Registry",
    "register_model",
]
