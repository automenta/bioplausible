"""TileFA: Feedback Alignment on the generic tile substrate.

Demonstrates the :class:`~computronium.core.local_learning.TileAlgorithm`
injection points: a subclass supplies a ``feedback_fn`` that projects output
errors through *fixed random* backward matrices (never the weight transpose),
which is the defining property of Feedback Alignment. Everything else
(topology, settling, optimizers, contrastive loop) comes from the substrate.
"""

import torch
from torch import nn

from computronium.core.local_learning import TileAlgorithm, TileAlgorithmConfig
from computronium.core.local_learning.algorithm import (
    _contrastive_weight_update,
    _ep_activity_update,
)
from computronium.core.model_status import status_tag
from computronium.core.registry import LocalityLevel, register_model
from computronium.core.tile import TileGraph, TileState

__all__ = ["TileFA"]


@register_model(
    "tile_fa",
    family="tile",
    locality_level=LocalityLevel.LOCAL,
    tags=["fa", "tile", status_tag("experimental")],
)
class TileFA(TileAlgorithm):
    """Feedback Alignment on the tile substrate.

    Constructed via ``TileFA.from_fa(...)``. The local bio-plausible loop
    (:meth:`local_update`) is inherited from the substrate and uses an injected
    FA ``feedback_fn`` that drives settling through fixed random matrices.
    """

    # registry-required classmethod building
    def __init__(self, config: TileAlgorithmConfig, **kwargs) -> None:
        from computronium.core.local_learning import TaskHandler

        handler = kwargs.pop("task_handler", None) or TaskHandler(
            task_type="classification", output_dim=config.output_dim
        )
        super().__init__(
            config,
            task_handler=handler,
            feedback_fn=self._fa_feedback_bound,
            activity_fn=_ep_activity_update,
            weight_fn=_contrastive_weight_update,
            **kwargs,
        )
        self._fa_feedback = nn.ParameterDict()
        self._init_random_feedback()

    def _init_random_feedback(self) -> None:
        """Fixed random backward projection per edge (frozen).

        ``B`` maps a tile's error (src.neurons) down to the previous tile's
        space (dst.neurons): ``src.error @ B.T`` with ``B: (dst.neurons, src.neurons)``.
        """
        for src_id, dst_id in self.graph.edges:
            src = self.graph.tiles[src_id]
            dst = self.graph.tiles[dst_id]
            fb = torch.randn(dst.neurons, src.neurons) * 0.1
            self._fa_feedback[self._weight_key(src_id, dst_id)] = nn.Parameter(
                fb, requires_grad=False
            )

    def _random_feedback_for(self, tile: TileState) -> list[torch.Tensor]:
        """Accumulate fixed-random backward-projected errors from forward tiles."""
        if tile.activity is None:
            return []
        feedback = [None] * len(tile.fwd_neighbors)
        for idx, dst_id in enumerate(tile.fwd_neighbors):
            dst = self.graph.tiles[dst_id]
            if dst.error is None:
                continue
            fb = self._fa_feedback[self._weight_key(tile.id, dst_id)]
            # fb: (dst.neurons, tile.neurons) -> project dst error into tile space
            feedback[idx] = dst.error @ fb
        return [f for f in feedback if f is not None]

    def _fa_feedback_bound(
        self, tile: TileState, graph: TileGraph, lookup: object
    ) -> list[torch.Tensor]:
        return self._random_feedback_for(tile)

    # ──────────────────────────────────────────────
    # Public builders
    # ──────────────────────────────────────────────

    @classmethod
    def from_fa(  # ruff: ignore[too-many-arguments]  # zoo build-classmethod contract
        cls,
        input_dim: int,
        output_dim: int,
        *,
        num_layers: int = 3,
        neurons_per_tile: int = 48,
        tiles_per_layer: int = 4,
        learning_rate: float = 0.001,
        importance_lr: float = 0.01,
        beta: float = 0.1,
        **kwargs,
    ) -> TileFA:
        """Create a TileFA model (FA dynamics on the tile substrate)."""
        config = TileAlgorithmConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            neurons_per_tile=neurons_per_tile,
            tiles_per_layer=tiles_per_layer,
            num_hidden_layers=num_layers,
            algorithm="fa",
            learning_rate=learning_rate,
            importance_lr=importance_lr,
            mode="fa",
            beta=beta,
            extra=kwargs,
        )
        return cls(config)

    @classmethod
    def build(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]  # zoo build contract
        cls,
        spec,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        device: str = "cpu",
        task_type: str = "vision",
        **kwargs,
    ) -> TileFA:
        """Zoo build classmethod."""
        model = cls.from_fa(
            input_dim=input_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            learning_rate=float(kwargs.get("learning_rate", 0.001)),
            importance_lr=float(kwargs.get("importance_lr", 0.01)),
        )
        return model.to(device)
