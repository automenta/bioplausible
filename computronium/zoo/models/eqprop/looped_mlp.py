"""Equilibrium Propagation native composition.

The native ``eqprop_mlp`` factory (registered as ``eqprop_mlp``) returns a
5-D ``System`` composed via :func:`computronium.models.native.eqprop_native.create_native_eqprop_mlp`.
This replaces the legacy ``LoopedMLP`` facade and ``EquilibriumMLP`` engine.
"""

from computronium.core.model_status import status_tag
from computronium.core.ontology import System
from computronium.core.registry import LocalityLevel, register_model
from computronium.models.native.eqprop_native import native_eqprop_mlp

__all__ = []


# Register native eqprop_mlp factory (bypasses ModelAdapter for 5-D composition)
@register_model(
    "eqprop_mlp",
    locality_level=LocalityLevel.EQUILIBRIUM,
    bio_plausibility_score=0.9,
    credit_assignment_type="equilibrium",
    requires_backward=False,
    memory_complexity="O(1)",
    family="eqprop",
    typical_lr_range=(0.001, 0.01),
    tags=["eqprop", "equilibrium", status_tag("stable")],
    extra={"parity_threshold": 0.05},
)
def _native_eqprop_mlp_factory(**kwargs) -> System:
    return native_eqprop_mlp(**kwargs)
