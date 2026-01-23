"""
Hyperparameter Optimization Package for Bio-Plausible Learning Research

Powered by Optuna for multi-objective optimization.
"""

__version__ = "0.1.0"

# Optuna is now required
HAS_OPTUNA = True

# Core Optuna integration
from .optuna_bridge import (
    create_optuna_space,
    create_study,
    get_pareto_trials,
    optimize_with_callback,
    trial_to_metrics,
)

# Search space definitions
from .search_space import SEARCH_SPACES, SearchSpace, get_search_space

__all__ = [
    "create_optuna_space",
    "create_study",
    "get_pareto_trials",
    "optimize_with_callback",
    "trial_to_metrics",
    "SearchSpace",
    "get_search_space",
    "SEARCH_SPACES",
    "HAS_OPTUNA",
]

