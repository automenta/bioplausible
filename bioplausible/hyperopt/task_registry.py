"""
Task Registration

Tasks are registered directly into the core Registry under
``ComponentCategory.TASK`` via the ``register_task`` convenience decorator.
"""

from bioplausible.core.registry import register_task
from bioplausible.hyperopt.tasks import LMTask, RLTask, VisionTask

# Register core task types via decorator syntax.
# Instantiation logic (parsing 'mnist_01') lives in the task factory.

__all__ = []
register_task("lm")(LMTask)
register_task("vision")(VisionTask)
register_task("rl")(RLTask)
