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
from computronium.core.campaign.frontier_record import FrontierRecord
from computronium.core.campaign.kernel_cache import (
    JointKernelCache,
    get_kernel_cache,
    set_kernel_cache,
)
from computronium.core.campaign.pareto import ParetoFrontier, pareto_frontier
from computronium.core.profiling import ResourceUsage

__all__ = [
    "CampaignState",
    "CampaignStore",
    "CheckpointManager",
    "EpisodeRecord",
    "FrontierRecord",
    "JointCheckpoint",
    "JointKernelCache",
    "ParetoFrontier",
    "ResourceUsage",
    "create_resume_script",
    "get_kernel_cache",
    "pareto_frontier",
    "set_kernel_cache",
]
