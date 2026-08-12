"""Tile-substrate wiring for the shared feature extractors.

The generic extractors, attention layers, and graph utilities now live in
``core.tile.feature_extractors``. This module binds them to ``TileAlgorithm``
via a tile-model factory and keeps the RL-specific extractor local.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from bioplausible.core.local_learning import TileAlgorithm, TileAlgorithmConfig
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

if TYPE_CHECKING:
    from torch import Tensor

    from bioplausible.zoo.models.deployments.base import RLDeploymentConfig

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


def tile_model_factory(  # ruff: ignore[too-many-locals] - typed extraction from untyped kwargs requires many locals
    *, input_dim: int, output_dim: int, **kwargs: object
) -> TileAlgorithm:
    """Bind the generic tile-model factory interface to ``TileAlgorithm``.

    ``kwargs`` carries the tile-substrate architecture fields plus arbitrary
    spillover; the dynamic splat into the frozen ``TileAlgorithmConfig`` is
    intentionally untyped. ``num_layers`` (legacy EquiTile semantics: total
    layers incl. input/output) is mapped to ``num_hidden_layers``.
    """
    num_layers_raw = kwargs.pop("num_layers", 2)
    num_layers = (
        int(num_layers_raw) if isinstance(num_layers_raw, (int, float, str)) else 2
    )
    num_hidden_layers = max(0, num_layers - 2)

    algorithm_raw = kwargs.pop("algorithm", "ep")
    algorithm = str(algorithm_raw) if isinstance(algorithm_raw, str) else "ep"

    mode_raw = kwargs.pop("mode", "backprop")
    mode = str(mode_raw) if isinstance(mode_raw, str) else "backprop"

    neurons_raw = kwargs.pop("neurons_per_tile", 48)
    neurons_per_tile = (
        int(neurons_raw) if isinstance(neurons_raw, (int, float, str)) else 48
    )

    tiles_raw = kwargs.pop("tiles_per_layer", 4)
    tiles_per_layer = int(tiles_raw) if isinstance(tiles_raw, (int, float, str)) else 4

    lr_raw = kwargs.pop("learning_rate", 0.001)
    learning_rate = float(lr_raw) if isinstance(lr_raw, (int, float, str)) else 0.001

    imp_lr_raw = kwargs.pop("importance_lr", 0.01)
    importance_lr = (
        float(imp_lr_raw) if isinstance(imp_lr_raw, (int, float, str)) else 0.01
    )

    config = TileAlgorithmConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        neurons_per_tile=neurons_per_tile,
        tiles_per_layer=tiles_per_layer,
        num_hidden_layers=num_hidden_layers,
        algorithm=algorithm,
        mode=mode,
        learning_rate=learning_rate,
        importance_lr=importance_lr,
        extra=kwargs,
    )
    return TileAlgorithm(config)


class RLFeatureExtractor(nn.Module):
    """Default RL feature extractor using the tile substrate."""

    def __init__(self, config: RLDeploymentConfig) -> None:
        super().__init__()
        self.config = config

        tile_dim = config.neurons_per_tile * config.tiles_per_layer
        num_hidden_layers = max(0, config.num_layers - 2)

        substrate_kwargs = dict(config.equitile_kwargs)
        substrate_kwargs.update({
            "neurons_per_tile": config.neurons_per_tile,
            "tiles_per_layer": config.tiles_per_layer,
            "learning_rate": config.learning_rate,
            "importance_lr": config.learning_rate * 0.1,
            "beta": config.beta,
        })

        head_config = TileAlgorithmConfig(
            input_dim=config.obs_dim,
            output_dim=tile_dim,
            num_hidden_layers=num_hidden_layers,
            algorithm="ep",
            mode="backprop",
            extra=substrate_kwargs,
        )
        self.tile_model = TileAlgorithm(head_config)

    def forward(self, obs: Tensor) -> Tensor:
        return self.tile_model(obs)

    def get_config(self) -> TileAlgorithmConfig:
        """Expose the inner TileAlgorithm config."""
        return self.tile_model.get_config()
