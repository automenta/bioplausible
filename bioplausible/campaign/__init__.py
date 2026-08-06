"""Campaign framework — retired (architecture §8, §10).

The FIX2a campaign engine is superseded by the thin experiment layer. The
migrated ``schema`` and ``param_estimator`` modules now live in
``bioplausible.experiment``; this package is a re-export shim so any surviving
import does not dangle. No backwards-compatibility guarantees are provided.
"""

from bioplausible.experiment.param_estimator import (
    InstantiateEstimator,
    ParamEstimateError,
    bound_estimator,
    estimate_param_count,
)
from bioplausible.experiment.schema import (
    Arm,
    Campaign,
    Stage,
    load_campaign,
    validate_yaml,
)

__all__ = [
    "Arm",
    "Campaign",
    "InstantiateEstimator",
    "ParamEstimateError",
    "Stage",
    "bound_estimator",
    "estimate_param_count",
    "load_campaign",
    "validate_yaml",
]
