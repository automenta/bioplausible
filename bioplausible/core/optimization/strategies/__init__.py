"""Generic composable-optimizer strategy interfaces (REFACTOR.md §7)."""

from .base import ConstraintStrategy, FeedbackStrategy, GradientStrategy, UpdateStrategy
from .constraint import NoConstraint, SpectralConstraint
from .feedback import ErrorFeedback, NoFeedback
from .gradient import (
    BackpropGradient,
    HebbianGradient,
    PCGradient,
    TargetPropGradient,
)
from .update import MuonUpdate, PlainUpdate

__all__ = [
    "BackpropGradient",
    "ConstraintStrategy",
    "ErrorFeedback",
    "FeedbackStrategy",
    "GradientStrategy",
    "HebbianGradient",
    "MuonUpdate",
    "NoConstraint",
    "NoFeedback",
    "PCGradient",
    "PlainUpdate",
    "SpectralConstraint",
    "TargetPropGradient",
    "UpdateStrategy",
]
