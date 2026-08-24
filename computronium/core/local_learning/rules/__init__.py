"""Learning-rule optimizer implementations.

Each rule implements the :class:`LearningRuleOptimizer` protocol
(:class:`~computronium.core.local_learning.rules.base.LearningRuleOptimizer`)
with a ``step(x, target)`` method that drives forward/backward/propagation.

Moved from ``zoo/propagators/`` as the canonical home for learning rules.
"""

# Import all rules to trigger @register_propagator registration
from . import backprop, base, eqprop, fa, hebbian, spiking
from .composite_adapter import CompositeOptimizerAdapter

__all__ = [
    "CompositeOptimizerAdapter",
    "backprop",
    "base",
    "eqprop",
    "fa",
    "hebbian",
    "spiking",
]
