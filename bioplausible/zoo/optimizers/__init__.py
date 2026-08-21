"""
Zoo Optimizers Package

Optimizers registered with the unified registry.
"""

from bioplausible.core.registry import LocalityLevel, register_optimizer

from . import ewc, muon, spectral, standard

__all__: list[str] = [
    "LocalityLevel",
    "ewc",
    "muon",
    "register_optimizer",
    "spectral",
    "standard",
]
