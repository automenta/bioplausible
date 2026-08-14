"""Learning-rule optimizer implementations.

Each rule implements the :class:`LearningRuleOptimizer` protocol
(:class:`~bioplausible.core.local_learning.rules.base.LearningRuleOptimizer`)
with a ``step(x, target)`` method that drives forward/backward/propagation.

Moved from ``zoo/propagators/`` as the canonical home for learning rules.
"""

# Import all rules to trigger @register_propagator registration
from . import base, backprop, eqprop, fa, hebbian, spiking
from .composite_adapter import CompositeOptimizerAdapter

__all__ = [
    "base",
    "backprop",
    "CompositeOptimizerAdapter",
    "eqprop",
    "fa",
    "hebbian",
    "spiking",
]
