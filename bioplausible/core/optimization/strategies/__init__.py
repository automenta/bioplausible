"""Generic composable-optimizer strategy interfaces (REFACTOR.md §7)."""

from .base import ConstraintStrategy, FeedbackStrategy, GradientStrategy, UpdateStrategy
from .constraint import NoConstraint, SpectralConstraint
from .feedback import ErrorFeedback, NoFeedback
from .gradient import BackpropGradient, FAGradient
from .update import MuonUpdate, PlainUpdate

__all__ = [
    "BackpropGradient",
    "ConstraintStrategy",
    "ErrorFeedback",
    "FAGradient",
    "FeedbackStrategy",
    "GradientStrategy",
    "MuonUpdate",
    "NoConstraint",
    "NoFeedback",
    "PlainUpdate",
    "SpectralConstraint",
    "UpdateStrategy",
]