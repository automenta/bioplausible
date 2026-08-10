"""EquiTile wiring for the shared tile-substrate feature extractors.

The generic extractors, attention layers, and graph utilities now live in
``core.tile.feature_extractors``. This module binds them to ``EquiTile`` via a
tile-model factory and keeps the RL-specific extractor local (its
``get_config`` exposes the inner EquiTile config).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from bioplausible.core.tile.feature_extractors import (
    ConvFeatureExtractor,
    GraphAttentionLayer,
    GraphEquiTileLayer,
    GraphFeatureExtractor,
    TemporalAttentionLayer,
    TemporalEquiTileLayer,
    TemporalFeatureExtractor,
    TemporalPositionalEncoding,
    add_self_loops,
    aggregate_messages,
    create_graph_from_edges,
    scatter_max,
    scatter_mean,
    scatter_sum,
)
from bioplausible.equitile.core import EquiTile
from bioplausible.equitile.core.config import EquiTileConfig

if TYPE_CHECKING:
    from torch import Tensor

    from bioplausible.equitile.deployments.base import RLDeploymentConfig

__all__ = [
    "ConvFeatureExtractor",
    "GraphAttentionLayer",
    "GraphEquiTileLayer",
    "GraphFeatureExtractor",
    "RLFeatureExtractor",
    "TemporalAttentionLayer",
    "TemporalEquiTileLayer",
    "TemporalFeatureExtractor",
    "TemporalPositionalEncoding",
    "add_self_loops",
    "aggregate_messages",
    "create_graph_from_edges",
    "scatter_max",
    "scatter_mean",
    "scatter_sum",
]


def tile_model_factory(
    *, input_dim: int, output_dim: int, **kwargs: object
) -> EquiTile:
    """Bind the generic tile-model factory interface to ``EquiTile``.

    ``kwargs`` carries the tile-substrate architecture fields plus arbitrary
    ``equitile_kwargs`` spillover; the dynamic splat into the frozen
    ``EquiTileConfig`` is intentionally untyped.
    """
    config = EquiTileConfig(**kwargs)  # type: ignore[reportArgumentType]
    return EquiTile(config=config, input_dim=input_dim, output_dim=output_dim)


class RLFeatureExtractor(nn.Module):
    """Default RL feature extractor using EquiTile."""

    def __init__(self, config: RLDeploymentConfig) -> None:
        super().__init__()
        self.config = config

        tile_dim = config.neurons_per_tile * config.tiles_per_layer

        equitile_kwargs = dict(config.equitile_kwargs)
        equitile_kwargs.update({
            "neurons_per_tile": config.neurons_per_tile,
            "num_layers": config.num_layers,
            "tiles_per_layer": config.tiles_per_layer,
            "mode": config.mode,
            "inference_steps": config.inference_steps,
            "learning_rate": config.learning_rate,
            "activation": config.activation,
        })

        self._equitile_config = EquiTileConfig(**equitile_kwargs)
        self.equitile = EquiTile(
            config=self._equitile_config,
            input_dim=config.obs_dim,
            output_dim=tile_dim,
        )

    def forward(self, obs: Tensor) -> Tensor:
        return self.equitile(obs)

    def get_config(self) -> EquiTileConfig:
        """Expose the inner EquiTile config (compatibility with bare EquiTile)."""
        return self._equitile_config
