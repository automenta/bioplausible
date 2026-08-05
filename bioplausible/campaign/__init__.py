"""Campaign framework — the FIX2a YAML-driven experiment engine.

Public API assembled from the framework's core modules. Consumers should import
from ``bioplausible.campaign``; only public names are exported.
"""

from bioplausible.campaign.logger import (
    Epoch,
    ExperimentLogger,
    TrialEnd,
    TrialStart,
)
from bioplausible.campaign.param_estimator import (
    InstantiateEstimator,
    ParamEstimateError,
    estimate_param_count,
)
from bioplausible.campaign.runner import (
    ArmPlan,
    CampaignResult,
    CampaignRunner,
    run_gates,
)
from bioplausible.campaign.schema import (
    Arm,
    Campaign,
    load_campaign,
    validate_yaml,
)
from bioplausible.campaign.search_space import (
    Choice,
    FloatRange,
    IntRange,
    SearchSpace,
    parse_distribution,
)
from bioplausible.campaign.tiers import (
    TierOutcome,
    run_tier0,
    run_tier05,
)

__all__ = [
    "Arm",
    "ArmPlan",
    "Campaign",
    "CampaignResult",
    "CampaignRunner",
    "Choice",
    "Epoch",
    "ExperimentLogger",
    "FloatRange",
    "InstantiateEstimator",
    "IntRange",
    "ParamEstimateError",
    "SearchSpace",
    "TierOutcome",
    "TrialEnd",
    "TrialStart",
    "estimate_param_count",
    "load_campaign",
    "parse_distribution",
    "run_gates",
    "run_tier0",
    "run_tier05",
    "validate_yaml",
]
