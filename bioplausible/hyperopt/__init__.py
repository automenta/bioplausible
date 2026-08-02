"""
Hyperparameter Optimization Package for Bio-Plausible Learning Research

Powered by Optuna for multi-objective optimization.
"""

from .eval_tiers import (
    EVALUATION_TIERS,
    EvaluationConfig,
    PatientLevel,
    estimate_total_time,
    get_evaluation_config,
    print_evaluation_summary,
)
from .optuna_bridge import (
    create_optuna_space,
    create_study,
    get_pareto_trials,
    optimize_with_callback,
    trial_to_metrics,
)
from .search_space import SEARCH_SPACES, SearchSpace, get_search_space

__version__ = "0.1.0"

# Optuna is now required
HAS_OPTUNA = True

# Evaluation tiers for patience-based optimization
# Core Optuna integration
# Search space definitions

__all__ = [
    "EVALUATION_TIERS",
    "HAS_OPTUNA",
    "SEARCH_SPACES",
    "EvaluationConfig",
    "PatientLevel",
    "SearchSpace",
    "create_constrained_optuna_config",  # noqa: F822  # provided lazily via __getattr__
    "create_optuna_space",
    "create_study",
    "estimate_total_time",
    "get_constrained_search_space",  # noqa: F822  # provided lazily via __getattr__
    "get_evaluation_config",
    "get_pareto_trials",
    "get_search_space",
    "optimize_with_callback",
    "print_evaluation_summary",
    "trial_to_metrics",
]


def __getattr__(name: str) -> object:
    """Lazily provide the two re-exported `execution._guards` helpers.

    They were previously imported eagerly at module top, which created a
    circular import with ``bioplausible.execution`` (hyperopt/__init__ →
    execution._guards → execution.task → hyperopt). Nothing imports them from
    this module (``execution.engine`` imports them from `_guards` directly), so
    they are exposed on demand to keep the package import cycle-free.
    """
    if name in ("create_constrained_optuna_config", "get_constrained_search_space"):
        from bioplausible.execution._guards import (  # local import breaks the cycle
            create_constrained_optuna_config,
            get_constrained_search_space,
        )
        value = (
            create_constrained_optuna_config
            if name == "create_constrained_optuna_config"
            else get_constrained_search_space
        )
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return __all__
