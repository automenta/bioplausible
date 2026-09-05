"""Native models using 5-D Ontology composition."""

from computronium.models.native.backprop_native import (
    create_native_backprop_mlp,
    native_backprop_mlp,
)
from computronium.models.native.deep_hebbian_native import DeepHebbianChain
from computronium.models.native.diffusion_eqprop_native import (
    create_native_diffusion_eqprop,
    native_diffusion_eqprop,
)
from computronium.models.native.eqprop_native import (
    create_native_eqprop_mlp,
    native_eqprop_mlp,
)
from computronium.models.native.fa_native import (
    create_native_fa_mlp,
    native_fa_mlp,
)
from computronium.models.native.momentum_eqprop_native import (
    create_native_momentum_eqprop,
    native_momentum_eqprop,
)
from computronium.models.native.pepita_native import (
    create_native_pepita_mlp,
    native_pepita_mlp,
)
from computronium.models.native.research_native import (
    create_native_directed_ep,
    create_native_finite_nudge_ep,
    create_native_holomorphic_ep,
    native_directed_ep,
    native_finite_nudge_ep,
    native_holomorphic_ep,
)
from computronium.models.native.sparse_eqprop_native import (
    create_native_sparse_eqprop,
    native_sparse_eqprop,
)
from computronium.models.native.ternary_eqprop_native import (
    create_native_ternary_eqprop,
    native_ternary_eqprop,
)
from computronium.models.native.tile_native import (
    create_native_tile_ep,
    create_native_tile_fa,
    create_native_tile_gnn,
    create_native_tile_hebbian,
    create_native_tile_pc,
    create_native_tile_snn,
    create_native_tile_tp,
    native_tile_ep,
    native_tile_fa,
    native_tile_gnn,
    native_tile_hebbian,
    native_tile_pc,
    native_tile_snn,
    native_tile_tp,
)

__all__ = [
    "DeepHebbianChain",
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
    "create_native_tile_gnn",
    "create_native_tile_hebbian",
    "create_native_tile_pc",
    "create_native_tile_snn",
    "create_native_tile_tp",
    "native_backprop_mlp",
    "native_diffusion_eqprop",
    "native_directed_ep",
    "native_eqprop_mlp",
    "native_fa_mlp",
    "native_finite_nudge_ep",
    "native_holomorphic_ep",
    "native_momentum_eqprop",
    "native_pepita_mlp",
    "native_sparse_eqprop",
    "native_ternary_eqprop",
    "native_tile_ep",
    "native_tile_fa",
    "native_tile_gnn",
    "native_tile_hebbian",
    "native_tile_pc",
    "native_tile_snn",
    "native_tile_tp",
]
