"""
Hyperparameter Optimization Package for Bio-Plausible Learning Research

Now powered by Optuna for multi-objective optimization.
Legacy evolution code has been deprecated.
"""

__version__ = "0.1.0"

# Optuna-based optimization
try:
    from .optuna_bridge import (
        create_optuna_space,
        create_study,
        get_pareto_trials,
        optimize_with_callback,
        trial_to_metrics,
    )

    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# Backward compatibility
from .engine import EvolutionaryOptimizer, OptimizationConfig
from .search_space import SEARCH_SPACES, SearchSpace, get_search_space

__all__ = [
    "create_optuna_space",
    "create_study",
    "get_pareto_trials",
    "optimize_with_callback",
    "trial_to_metrics",
    "EvolutionaryOptimizer",
    "OptimizationConfig",
    "SearchSpace",
    "get_search_space",
    "SEARCH_SPACES",
    "HAS_OPTUNA",
]

