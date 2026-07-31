"""
Task Registration

Tasks are registered directly into the core Registry under
``ComponentCategory.TASK`` via the ``register_task`` convenience decorator.

All concrete task classes now live in ``domains/`` (Phase 3.1 merge).
"""

from bioplausible.core.registry import register_task
from bioplausible.domains.lm import LMTask
from bioplausible.domains.rl import RLTask
from bioplausible.domains.vision import VisionTask

__all__ = []
register_task("lm")(LMTask)
register_task("vision")(VisionTask)
register_task("rl")(RLTask)
