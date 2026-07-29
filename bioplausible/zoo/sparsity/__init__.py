"""
Zoo Sparsity Package

Sparsity methods registered with the unified registry.
"""

from bioplausible.core.registry import register_sparsity

from . import methods  # ruff: ignore[unused-import]  (triggers registration)

__all__ = [
    "methods",
    "register_sparsity",
]
