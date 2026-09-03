"""Generic local-learning infrastructure: task handling, multi-optimizer mixins."""

from computronium.core.local_learning import rules  # trigger propagator registration
from computronium.core.local_learning.activity import (
    ep_activity_update,
    hebbian_activity_update,
    spiking_activity_update,
)
from computronium.core.local_learning.builder import (
    TileAlgorithm,
    TileAlgorithmConfig,
    tile_algorithm,
)
from computronium.core.local_learning.feedback import (
    no_feedback,
    symmetric_feedback,
)
from computronium.core.local_learning.mixins import (
    LocalLearningConfigProtocol,
    MultiOptimizerMixin,
)
from computronium.core.local_learning.protocols import (
    ActivityUpdateFn,
    FeedbackFn,
    WeightLookup,
    WeightUpdateFn,
)
from computronium.core.local_learning.registry import (
    get_tile_algorithm_factory,
    get_tile_algorithm_metadata,
    list_tile_algorithms,
    list_tile_algorithms_with_metadata,
)
from computronium.core.local_learning.task import TaskHandler
from computronium.core.local_learning.weight_update import (
    contrastive_weight_update,
    hebbian_weight_update,
)

__all__ = [
    "ActivityUpdateFn",
    "FeedbackFn",
    "LocalLearningConfigProtocol",
    "MultiOptimizerMixin",
    "TaskHandler",
    "TileAlgorithm",
    "TileAlgorithmConfig",
    "WeightLookup",
    "WeightUpdateFn",
    "contrastive_weight_update",
    "ep_activity_update",
    "get_tile_algorithm_factory",
    "get_tile_algorithm_metadata",
    "hebbian_activity_update",
    "hebbian_weight_update",
    "list_tile_algorithms",
    "list_tile_algorithms_with_metadata",
    "no_feedback",
    "rules",
    "spiking_activity_update",
    "symmetric_feedback",
    "tile_algorithm",
]
