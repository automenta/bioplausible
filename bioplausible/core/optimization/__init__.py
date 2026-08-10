"""Generic composable optimizer framework (REFACTOR.md §7).

Strategy-pattern optimizer usable by any zoo model without MEP
dependencies. Enables combinatorial permutations (Backprop/Muon/Spectral,
FA + Muon, Hebbian + Plain, EP + Dion, TargetProp + Fisher, ...) through
the strategy registry and frozen configs.
"""

from .config import StrategyConfig, StrategyOptimizerConfig
from .factory import StrategyRegistry, create_strategy_optimizer
from .optimizer import StrategyOptimizer
from .strategies import (
    BackpropGradient,
    ConstraintStrategy,
    ErrorFeedback,
    FAGradient,
    FeedbackStrategy,
    GradientStrategy,
    MuonUpdate,
    NoConstraint,
    NoFeedback,
    PlainUpdate,
    SpectralConstraint,
    UpdateStrategy,
)

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
    "StrategyConfig",
    "StrategyOptimizer",
    "StrategyOptimizerConfig",
    "StrategyRegistry",
    "UpdateStrategy",
    "create_strategy_optimizer",
]