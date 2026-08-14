"""
Zoo Propagators Package

MEP (Modulatory Error Propagation) learning rules.
"""

from bioplausible.core.registry import register_propagator

from . import mep

__all__: list[str] = [
    "mep",
    "register_propagator",
]
