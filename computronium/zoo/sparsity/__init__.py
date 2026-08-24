"""
Zoo Sparsity Package

Sparsity methods registered with the unified registry.
"""

from computronium.core.registry import register_sparsity

from . import methods

__all__: list[str] = [
    "methods",
    "register_sparsity",
]
