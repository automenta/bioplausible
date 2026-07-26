"""
Target Propagation propagators.

Classes: TargetProp, DifferenceTargetProp
"""

from bioplausible.core.registry import register_propagator

from .base import LearningRuleOptimizer


@register_propagator("target_prop")
class TargetProp(LearningRuleOptimizer):
    """Target Propagation propagator.

    Note: Target propagation requires a model with learned backward
    connections. See bioplausible.zoo.models.target_prop for
    the model-side implementation.
    """

    def step(self, x, target=None):
        msg = (
            "TargetProp propagator is not yet implemented via the Zoo interface. "
            "Use bioplausible.zoo.models.target_prop which contains "
            "the model-level implementation."
        )
        raise NotImplementedError(msg)


@register_propagator("difference_target_prop")
class DifferenceTargetProp(LearningRuleOptimizer):
    """Difference Target Propagation propagator."""

    def step(self, x, target=None):
        msg = "DifferenceTargetProp propagator is not yet implemented via the Zoo interface."
        raise NotImplementedError(msg)
