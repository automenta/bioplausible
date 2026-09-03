"""
Joint Campaign Infrastructure.

Provides persistence, resource accounting, Pareto frontier computation,
kernel caching, and fault tolerance checkpointing for 6-D joint architecture campaigns.
"""

from computronium.core.campaign.campaign_store import (
    SCHEMA_VERSION,
    CampaignState,
    CampaignStore,
    EpisodeRecord,
    SchemaVersionError,
)
from computronium.core.campaign.checkpoint import (
    CheckpointManager,
    JointCheckpoint,
    create_resume_script,
)
from computronium.core.campaign.discovery import (
    BuildKwargs,
    DiscoverySpec,
    LockVerdict,
    verify_attribution_rank,
    verify_fidelity_standing,
    verify_replay,
    verify_winner_replication,
)
from computronium.core.campaign.evaluation import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_GUARD_TAU,
    DEFAULT_INPUT_DIM,
    DEFAULT_NUM_CLASSES,
    GuardKillError,
    IncompatibleCoordinateError,
    UnsupportedCoordinateError,
    activity_transition,
    build_coordinate_system,
    episode_batch,
    evaluate_episode,
)
from computronium.core.campaign.fidelity import (
    AxisCheck,
    CoordinateFidelity,
    DefectFilteredAttribution,
    check_coordinate_fidelity,
    defect_filtered_attribution,
    fidelity_manifest,
)
from computronium.core.campaign.frontier_record import FrontierRecord
from computronium.core.campaign.kernel_cache import (
    JointKernelCache,
    get_kernel_cache,
    set_kernel_cache,
)
from computronium.core.campaign.pareto import ParetoFrontier, pareto_frontier
from computronium.core.campaign.replication import (
    ReplicationReport,
    replication_manifest,
    task_family,
    unreplicated,
    verify_replication,
)
from computronium.core.campaign.report import (
    DiscoveryReport,
    FidelitySummary,
    FrontierRow,
    ReplicationRow,
    StratifiedAttribution,
    TimelineStep,
    build_discovery_report,
    load_campaign_records,
)
from computronium.core.campaign.stack import (
    CampaignRunResult,
    CampaignStack,
    CoordinateSampler,
    EpisodeOutcome,
    grid_sampler,
    space_grid,
)
from computronium.resources import ResourceUsage

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_GUARD_TAU",
    "DEFAULT_INPUT_DIM",
    "DEFAULT_NUM_CLASSES",
    "SCHEMA_VERSION",
    "AxisCheck",
    "BuildKwargs",
    "CampaignRunResult",
    "CampaignStack",
    "CampaignState",
    "CampaignStore",
    "CheckpointManager",
    "CoordinateFidelity",
    "CoordinateSampler",
    "DefectFilteredAttribution",
    "DiscoveryReport",
    "DiscoverySpec",
    "EpisodeOutcome",
    "EpisodeRecord",
    "FidelitySummary",
    "FrontierRecord",
    "FrontierRow",
    "GuardKillError",
    "IncompatibleCoordinateError",
    "JointCheckpoint",
    "JointKernelCache",
    "LockVerdict",
    "ParetoFrontier",
    "ReplicationReport",
    "ReplicationRow",
    "ResourceUsage",
    "SchemaVersionError",
    "StratifiedAttribution",
    "TimelineStep",
    "UnsupportedCoordinateError",
    "activity_transition",
    "build_coordinate_system",
    "build_discovery_report",
    "check_coordinate_fidelity",
    "create_resume_script",
    "defect_filtered_attribution",
    "episode_batch",
    "evaluate_episode",
    "fidelity_manifest",
    "get_kernel_cache",
    "grid_sampler",
    "load_campaign_records",
    "pareto_frontier",
    "replication_manifest",
    "set_kernel_cache",
    "space_grid",
    "task_family",
    "unreplicated",
    "verify_attribution_rank",
    "verify_fidelity_standing",
    "verify_replay",
    "verify_replication",
    "verify_winner_replication",
]
