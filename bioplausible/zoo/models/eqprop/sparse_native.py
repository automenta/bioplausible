"""Sparse Equilibrium Propagation native composition.

The native ``sparse_eqprop`` factory (registered as ``sparse_eqprop``) returns a
5-D ``System`` composed via :func:`bioplausible.models.native.sparse_eqprop_native.create_native_sparse_eqprop`.
"""

from bioplausible.core.model_status import status_tag
from bioplausible.core.ontology import System
from bioplausible.core.registry import LocalityLevel, register_model
from bioplausible.models.native.sparse_eqprop_native import native_sparse_eqprop

__all__ = []


# Register native sparse_eqprop factory
@register_model(
    "sparse_eqprop",
    locality_level=LocalityLevel.EQUILIBRIUM,
    bio_plausibility_score=0.85,
    credit_assignment_type="equilibrium",
    requires_backward=False,
    memory_complexity="O(1)",
    family="eqprop",
    typical_lr_range=(0.001, 0.01),
    tags=["eqprop", "sparse", "dynamic_sparsity", status_tag("experimental")],
    extra={"sparsity": "dynamic", "parity_threshold": 0.1},
)
def _native_sparse_eqprop_factory(**kwargs) -> System:
    return native_sparse_eqprop(**kwargs)
