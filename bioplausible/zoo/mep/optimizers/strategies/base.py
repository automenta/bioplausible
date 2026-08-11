"""
Base strategy protocol definitions.

Re-exported from :mod:`bioplausible.core.optimization.strategies.base`
(REFACTOR.md §7) so MEP stays source-compatible while sharing a single set
of protocols with the generic framework.
"""

from bioplausible.core.optimization.strategies.base import (
    ConstraintStrategy,
    FeedbackStrategy,
    GradientStrategy,
    UpdateStrategy,
)

__all__ = [
    "ConstraintStrategy",
    "FeedbackStrategy",
    "GradientStrategy",
    "UpdateStrategy",
]
