"""Native models using 5-D Ontology composition."""

from bioplausible.models.native.backprop_native import create_native_backprop_mlp
from bioplausible.models.native.eqprop_native import create_native_eqprop_mlp
from bioplausible.models.native.fa_native import create_native_fa_mlp
from bioplausible.models.native.pepita_native import create_native_pepita_mlp
from bioplausible.models.native.tile_native import (
    create_native_tile_ep,
    create_native_tile_fa,
    create_native_tile_tp,
    create_native_tile_snn,
)
from bioplausible.models.native.research_native import (
    create_native_holomorphic_ep,
    create_native_directed_ep,
    create_native_finite_nudge_ep,
)

__all__ = [
    "create_native_backprop_mlp",
    "create_native_eqprop_mlp",
    "create_native_fa_mlp",
    "create_native_pepita_mlp",
    "create_native_tile_ep",
    "create_native_tile_fa",
    "create_native_tile_tp",
    "create_native_tile_snn",
    "create_native_holomorphic_ep",
    "create_native_directed_ep",
    "create_native_finite_nudge_ep",
]