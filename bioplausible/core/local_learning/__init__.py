"""Generic local-learning infrastructure: task handling, multi-optimizer mixins."""

from bioplausible.core.local_learning import rules  # trigger propagator registration
from bioplausible.core.local_learning.algorithm import (
    TileAlgorithm,
    TileAlgorithmConfig,
)
from bioplausible.core.local_learning.mixins import (
    LocalLearningConfigProtocol,
    MultiOptimizerMixin,
)
from bioplausible.core.local_learning.task import TaskHandler

__all__ = [
    "LocalLearningConfigProtocol",
    "MultiOptimizerMixin",
    "TaskHandler",
    "TileAlgorithm",
    "TileAlgorithmConfig",
    "rules",
]
