"""Ternary Equilibrium Propagation native composition.

The native ``ternary_eqprop`` factory (registered as ``ternary_eqprop``) returns a
5-D ``System`` composed via :func:`computronium.models.native.ternary_eqprop_native.create_native_ternary_eqprop`.
"""

from computronium.core.model_status import status_tag
from computronium.core.ontology import System
from computronium.core.registry import LocalityLevel, register_model
from computronium.models.native.ternary_eqprop_native import native_ternary_eqprop

__all__ = []


# Register native ternary_eqprop factory
@register_model(
    "ternary_eqprop",
    locality_level=LocalityLevel.EQUILIBRIUM,
    bio_plausibility_score=0.85,
    credit_assignment_type="equilibrium",
    requires_backward=False,
    memory_complexity="O(1)",
    family="eqprop",
    typical_lr_range=(0.001, 0.01),
    tags=["eqprop", "ternary", "quantized", status_tag("experimental")],
    extra={"quantization": "ternary", "parity_threshold": 0.1},
)
def _native_ternary_eqprop_factory(**kwargs) -> System:
    return native_ternary_eqprop(**kwargs)
