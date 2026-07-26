"""
Forward-only propagators.

Classes: FF, PEPITA
"""

from bioplausible.core.registry import register_propagator

from .base import LearningRuleOptimizer


@register_propagator("ff")
class FF(LearningRuleOptimizer):
    """Forward-Forward learning rule.

    Note: A working implementation requires a model that supports
    separate positive/negative phase passes. See ForwardForwardNet
    in bioplausible.zoo.models.forward_only for the model-side implementation.
    """

    def step(self, x, target=None):
        msg = (
            "FF propagator requires a ForwardForwardNet model. "
            "Use bioplausible.zoo.models.forward_only.ForwardForwardNet "
            "which handles the FF update internally."
        )
        raise NotImplementedError(msg)


@register_propagator("pepita")
class PEPITA(LearningRuleOptimizer):
    """PEPITA: forward-only learning with random feedback.

    Note: A working implementation requires a model that supports
    the PEPITA learning rule internally. See PEPITA in
    bioplausible.zoo.models.forward_only for the model-side implementation.
    """

    def step(self, x, target=None):
        msg = (
            "PEPITA propagator requires a PEPITA model. "
            "Use bioplausible.zoo.models.forward_only.PEPITA "
            "which handles the PEPITA update internally."
        )
        raise NotImplementedError(msg)
