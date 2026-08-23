"""
Joint Campaign Infrastructure.

Provides persistence, resource accounting, Pareto frontier computation,
kernel caching, and fault tolerance checkpointing for 6-D joint architecture campaigns.
"""

from bioplausible.core.campaign.resource_vector import ResourceUsage
from bioplausible.core.campaign.frontier_record import FrontierRecord
from bioplausible.core.campaign.campaign_store import CampaignStore, CampaignState, EpisodeRecord
from bioplausible.core.campaign.pareto import pareto_frontier, ParetoFrontier
from bioplausible.core.campaign.kernel_cache import JointKernelCache, get_kernel_cache, set_kernel_cache
from bioplausible.core.campaign.checkpoint import CheckpointManager, JointCheckpoint, create_resume_script

__all__ = [
    "ResourceUsage",
    "FrontierRecord",
    "CampaignStore",
    "CampaignState",
    "EpisodeRecord",
    "pareto_frontier",
    "ParetoFrontier",
    "JointKernelCache",
    "get_kernel_cache",
    "set_kernel_cache",
    "CheckpointManager",
    "JointCheckpoint",
    "create_resume_script",
]