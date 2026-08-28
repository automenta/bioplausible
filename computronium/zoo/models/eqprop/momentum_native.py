"""Momentum Equilibrium Propagation native composition.

The native ``momentum_eqprop`` factory (registered as ``momentum_eqprop``) returns a
5-D ``System`` composed via :func:`computronium.models.native.momentum_eqprop_native.create_native_momentum_eqprop`.
"""

from computronium.core.model_status import status_tag
from computronium.core.registry import LocalityLevel, register_model
from computronium.models.native.momentum_eqprop_native import native_momentum_eqprop
from computronium.ontology import System

__all__ = []


# Register native momentum_eqprop factory
@register_model(
    "momentum_eqprop",
    locality_level=LocalityLevel.EQUILIBRIUM,
    bio_plausibility_score=0.9,
    credit_assignment_type="equilibrium",
    requires_backward=False,
    memory_complexity="O(1)",
    family="eqprop",
    typical_lr_range=(0.001, 0.01),
    tags=["eqprop", "momentum", "energy_minimization", status_tag("experimental")],
    extra={"dynamics": "energy_minimization_momentum", "parity_threshold": 0.05},
)
def _native_momentum_eqprop_factory(**kwargs) -> System:
    return native_momentum_eqprop(**kwargs)
