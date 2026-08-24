"""Generic local-learning infrastructure: task handling, multi-optimizer mixins."""

from computronium.core.local_learning import rules  # trigger propagator registration
from computronium.core.local_learning.algorithm import (
    TileAlgorithm,
    TileAlgorithmConfig,
)
from computronium.core.local_learning.mixins import (
    LocalLearningConfigProtocol,
    MultiOptimizerMixin,
)
from computronium.core.local_learning.task import TaskHandler

__all__ = [
    "LocalLearningConfigProtocol",
    "MultiOptimizerMixin",
    "TaskHandler",
    "TileAlgorithm",
    "TileAlgorithmConfig",
    "rules",
]
