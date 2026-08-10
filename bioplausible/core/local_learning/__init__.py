"""Generic local-learning infrastructure: task handling, multi-optimizer mixins."""

from bioplausible.core.local_learning.mixins import MultiOptimizerMixin
from bioplausible.core.local_learning.task import TaskHandler

__all__ = [
    "MultiOptimizerMixin",
    "TaskHandler",
]
