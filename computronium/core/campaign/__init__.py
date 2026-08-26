"""
Joint Campaign Infrastructure.

Provides persistence, resource accounting, Pareto frontier computation,
kernel caching, and fault tolerance checkpointing for 6-D joint architecture campaigns.
"""

from computronium.core.campaign.campaign_store import (
    CampaignState,
    CampaignStore,
    EpisodeRecord,
)
from computronium.core.campaign.checkpoint import (
    CheckpointManager,
    JointCheckpoint,
    create_resume_script,
)
from computronium.core.campaign.evaluation import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_GUARD_TAU,
    DEFAULT_INPUT_DIM,
    DEFAULT_NUM_CLASSES,
    GuardKillError,
    UnsupportedCoordinateError,
    activity_transition,
    build_coordinate_system,
    episode_batch,
    evaluate_episode,
)
from computronium.core.campaign.frontier_record import FrontierRecord
from computronium.core.campaign.kernel_cache import (
    JointKernelCache,
    get_kernel_cache,
    set_kernel_cache,
)
from computronium.core.campaign.pareto import ParetoFrontier, pareto_frontier
from computronium.core.profiling import ResourceUsage

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_GUARD_TAU",
    "DEFAULT_INPUT_DIM",
    "DEFAULT_NUM_CLASSES",
    "CampaignState",
    "CampaignStore",
    "CheckpointManager",
    "EpisodeRecord",
    "FrontierRecord",
    "GuardKillError",
    "JointCheckpoint",
    "JointKernelCache",
    "ParetoFrontier",
    "ResourceUsage",
    "UnsupportedCoordinateError",
    "activity_transition",
    "build_coordinate_system",
    "create_resume_script",
    "episode_batch",
    "evaluate_episode",
    "get_kernel_cache",
    "pareto_frontier",
    "set_kernel_cache",
]
