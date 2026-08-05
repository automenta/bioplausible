"""Campaign framework — the FIX2a YAML-driven experiment engine.

Public API assembled from the framework's core modules. Consumers should import
from ``bioplausible.campaign``; only public names are exported.
"""

from bioplausible.campaign.executor import (
    CampaignExecutor,
    TrialContext,
    run_campaign,
)
from bioplausible.campaign.logger import (
    Epoch,
    ExperimentLogger,
    TrialEnd,
    TrialStart,
)
from bioplausible.campaign.param_estimator import (
    InstantiateEstimator,
    ParamEstimateError,
    bound_estimator,
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
    ParamDistribution,
    SearchSpace,
    parse_distribution,
)
from bioplausible.campaign.tiers import (
    GateSettings,
    TierOutcome,
    run_tier0,
    run_tier05,
)

__all__ = [
    "Arm",
    "ArmPlan",
    "Campaign",
    "CampaignExecutor",
    "CampaignResult",
    "CampaignRunner",
    "Choice",
    "Epoch",
    "ExperimentLogger",
    "FloatRange",
    "GateSettings",
    "InstantiateEstimator",
    "IntRange",
    "ParamDistribution",
    "ParamEstimateError",
    "SearchSpace",
    "TierOutcome",
    "TrialContext",
    "TrialEnd",
    "TrialStart",
    "bound_estimator",
    "estimate_param_count",
    "load_campaign",
    "parse_distribution",
    "run_campaign",
    "run_gates",
    "run_tier0",
    "run_tier05",
    "validate_yaml",
]
