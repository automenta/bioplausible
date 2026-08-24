"""
Feedback strategies for error/residual accumulation.

Re-exports the generic implementations from
:mod:`computronium.core.optimization.strategies.feedback` (REFACTOR.md §7).
"""

from computronium.core.optimization.strategies.feedback import (
    ErrorFeedback,
    NoFeedback,
)

__all__ = [
    "ErrorFeedback",
    "NoFeedback",
]
