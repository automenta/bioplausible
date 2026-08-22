"""Native models using 5-D Ontology composition."""

from bioplausible.models.native.backprop_native import create_native_backprop_mlp
from bioplausible.models.native.diffusion_eqprop_native import (
    create_native_diffusion_eqprop,
)
from bioplausible.models.native.eqprop_native import create_native_eqprop_mlp
from bioplausible.models.native.fa_native import create_native_fa_mlp
from bioplausible.models.native.momentum_eqprop_native import (
    create_native_momentum_eqprop,
)
from bioplausible.models.native.pepita_native import create_native_pepita_mlp
from bioplausible.models.native.research_native import (
    create_native_directed_ep,
    create_native_finite_nudge_ep,
    create_native_holomorphic_ep,
)
from bioplausible.models.native.sparse_eqprop_native import create_native_sparse_eqprop
from bioplausible.models.native.ternary_eqprop_native import (
    create_native_ternary_eqprop,
)
from bioplausible.models.native.tile_native import (
    create_native_tile_ep,
    create_native_tile_fa,
    create_native_tile_snn,
    create_native_tile_tp,
)

__all__ = [
    "create_native_backprop_mlp",
    "create_native_diffusion_eqprop",
    "create_native_directed_ep",
    "create_native_eqprop_mlp",
    "create_native_fa_mlp",
    "create_native_finite_nudge_ep",
    "create_native_holomorphic_ep",
    "create_native_momentum_eqprop",
    "create_native_pepita_mlp",
    "create_native_sparse_eqprop",
    "create_native_ternary_eqprop",
    "create_native_tile_ep",
    "create_native_tile_fa",
    "create_native_tile_snn",
    "create_native_tile_tp",
]
