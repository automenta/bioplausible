"""
Predictive Coding Network (PCN) propagator.

Classes: PCN
"""

from bioplausible.core.registry import register_propagator

from .base import LearningRuleOptimizer


@register_propagator("predictive_coding")
class PCN(LearningRuleOptimizer):
    """Predictive Coding Network propagator.

    Note: PCN learning is handled by the graph-based implementation
    in bioplausible.graph.training. The standard Zoo propagator
    interface is not fully implemented for PCN yet.
    """

    def step(self, x, target=None):
        msg = (
            "PCN propagator is not yet implemented via the Zoo interface. "
            "Use bioplausible.graph.training.train_pcn for PCN training."
        )
        raise NotImplementedError(msg)
