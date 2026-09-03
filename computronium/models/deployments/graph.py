"""
TileNet Graph: Graph Neural Networks with TileNet
==================================================

Extends TileNet for graph-structured data:
- GraphTileNet: Graph neural network with tile-based message passing
- Graph attention mechanisms
- Support for node/graph classification
- Integration with networkx and torch_geometric

The shared graph layers (feature extractor, attention, scatter utilities)
now live in the private ``_feature_extractors`` module and are re-exported;
this module adds the graph-specific model (readout/forward over batches).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

from computronium.config.unified import ModelConfig
from computronium.core.model import BioModel
from computronium.models.deployments import _feature_extractors as _fe
from computronium.models.deployments.base import (
    GraphDeploymentConfig,
    build_tile_head,
)

# Re-export shared graph components under their historical names.
GraphAttentionLayer = _fe.GraphAttentionLayer
GraphTileNetLayer = _fe.GraphTileNetLayer
GraphFeatureExtractor = _fe.GraphFeatureExtractor
aggregate_messages = _fe.aggregate_messages
create_graph_from_edges = _fe.create_graph_from_edges
scatter_max = _fe.scatter_max
scatter_mean = _fe.scatter_mean
scatter_sum = _fe.scatter_sum
add_self_loops = _fe.add_self_loops

__all__ = [
    "GraphAttentionLayer",
    "GraphTileNet",
    "GraphTileNetConfig",
    "GraphTileNetLayer",
    "add_self_loops",
    "aggregate_messages",
    "create_graph_from_edges",
    "create_graph_model",
    "create_molecule_model",
    "create_social_graph_model",
    "scatter_max",
    "scatter_mean",
    "scatter_sum",
]
if TYPE_CHECKING:
    from torch import Tensor


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class GraphTileNetConfig(GraphDeploymentConfig):
    """Configuration for Graph TileNet.

    Inherits the shared deployment fields from ``GraphDeploymentConfig`` and
    keeps the same defaults the historical ``GraphTileNetConfig`` exposed.
    """

    # Historical graph defaults differ from the generic base.
    learning_rate: float = 1e-3
    num_layers: int = 3
    neurons_per_tile: int = 32
    tiles_per_layer: int = 4
    attention_heads: int = 4
    mode: Literal["pc", "ep", "backprop"] = "backprop"


# =============================================================================
# Graph TileNet
# =============================================================================


def _credit_assignment_type(algorithm: str) -> str:
    """Map algorithm to credit assignment type."""
    mapping = {
        "ep": "equilibrium",
        "pc": "equilibrium",
        "fa": "target",
        "tp": "target",
        "hebbian": "hebbian",
        "snn": "spiking",
    }
    return mapping.get(algorithm, "equilibrium")


class GraphTileNet(BioModel):
    """Graph TileNet for graph-structured data.

    Combines graph attention with TileNet's tile-based
    message passing for node and graph classification.

    Parameters
    ----------
    config : GraphTileNetConfig, optional
        Configuration
    **kwargs
        Additional configuration parameters
    """

    algorithm_name = "GraphTileNet"

    @classmethod
    def build(
        cls,
        spec,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers,
        device,
        task_type,
        **kwargs,
    ):
        """Build GraphTileNet from factory arguments."""
        config_kwargs = {
            "node_features": input_dim,
            "hidden_dim": hidden_dim,
            "num_classes": output_dim,
            "num_layers": num_layers,
            "task_type": task_type or "classification",
            "learning_rate": kwargs.get("lr", spec.default_lr),
            "neurons_per_tile": kwargs.get("neurons_per_tile", 32),
            "tiles_per_layer": kwargs.get("tiles_per_layer", 4),
            "attention_heads": kwargs.get("attention_heads", 4),
        }

        valid_keys = GraphTileNetConfig.__annotations__.keys()
        for k, v in kwargs.items():
            if k in valid_keys:
                config_kwargs[k] = v

        for k, v in spec.custom_hyperparams.items():
            if k in valid_keys:
                config_kwargs[k] = v

        config = GraphTileNetConfig(**config_kwargs)

        model = cls(config=config)
        return model.to(device)

    def __init__(
        self,
        config: GraphTileNetConfig | None = None,
        **kwargs,
    ) -> None:
        if config is None:
            config = GraphTileNetConfig(**kwargs)

        super().__init__(
            ModelConfig(
                name="graph_tile",
                input_dim=config.node_features,
                output_dim=config.num_classes,
            )
        )

        self.config = config

        # Shared graph feature extractor (input proj + stacked graph EquiTile
        # layers + attention), from the unified deployments module.
        self.feature_extractor = GraphFeatureExtractor(config, _fe.tile_model_factory)

        # Tile-substrate classification head
        self._build_tile_head(config)

        # Readout attention (for attention-based readout)
        if config.readout == "attention":
            self.readout_attention = nn.Linear(config.hidden_dim, 1)

        # State tracking
        self._step_count = 0

    def _build_tile_head(self, config: GraphTileNetConfig) -> None:
        """Build the tile-substrate classification head."""
        feature_dim = config.hidden_dim
        self.head = build_tile_head(config, feature_dim, config.num_classes)

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        batch: Tensor | None = None,
        return_node_embeddings: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Forward pass.

        Parameters
        ----------
        node_features : torch.Tensor
            Node features (num_nodes, node_features)
        edge_index : torch.Tensor
            Edge indices (2, num_edges)
        batch : torch.Tensor, optional
            Batch indices for each node
        return_node_embeddings : bool
            If True, return node embeddings as well

        Returns
        -------
        torch.Tensor or tuple
            Graph predictions, or (predictions, node_embeddings)
        """
        x = self.feature_extractor(node_features, edge_index)

        # Graph readout
        if batch is not None:
            if self.config.readout == "attention":
                attention = torch.sigmoid(self.readout_attention(x))
                graph_features = scatter_mean(x * attention, batch, dim=0)
            elif self.config.readout == "mean":
                graph_features = scatter_mean(x, batch, dim=0)
            elif self.config.readout == "sum":
                graph_features = scatter_sum(x, batch, dim=0)
            elif self.config.readout == "max":
                graph_features = scatter_max(x, batch, dim=0)
            else:
                graph_features = x
        else:
            graph_features = x.mean(dim=0, keepdim=True)

        logits = self.head.forward_logits(graph_features, detach_input=False)

        if return_node_embeddings:
            return logits, x
        return logits

    def train_step(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        labels: Tensor,
        batch: Tensor | None = None,
    ) -> dict[str, float]:
        """Perform one training step."""
        self._step_count += 1

        x = self.feature_extractor(node_features, edge_index)

        # Graph readout
        if batch is not None:
            if self.config.readout == "attention":
                attention = torch.sigmoid(self.readout_attention(x))
                graph_features = scatter_mean(x * attention, batch, dim=0)
            elif self.config.readout == "mean":
                graph_features = scatter_mean(x, batch, dim=0)
            elif self.config.readout == "sum":
                graph_features = scatter_sum(x, batch, dim=0)
            elif self.config.readout == "max":
                graph_features = scatter_max(x, batch, dim=0)
            else:
                graph_features = x
        else:
            graph_features = x.mean(dim=0, keepdim=True)

        return self.head.local_update(graph_features.detach(), labels)

    def predict(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        batch: Tensor | None = None,
    ) -> Tensor:
        """Make predictions."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(node_features, edge_index, batch)
            if isinstance(logits, tuple):
                logits = logits[0]
            return logits.argmax(dim=-1)


# =============================================================================
# Factory Functions
# =============================================================================


def create_graph_model(
    node_features: int,
    num_classes: int,
    hidden_dim: int = 64,
    num_layers: int = 3,
    **kwargs,
) -> GraphTileNet:
    """Create GraphTileNet model."""
    config = GraphTileNetConfig(
        node_features=node_features,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        **kwargs,
    )
    return GraphTileNet(config)


def create_molecule_model(
    atom_features: int = 9,
    num_classes: int = 2,
    **kwargs,
) -> GraphTileNet:
    """Create GraphTileNet for molecular property prediction."""
    return create_graph_model(
        node_features=atom_features,
        num_classes=num_classes,
        hidden_dim=128,
        num_layers=4,
        attention_heads=4,
        **kwargs,
    )


def create_social_graph_model(
    user_features: int = 16,
    num_classes: int = 2,
    **kwargs,
) -> GraphTileNet:
    """Create GraphTileNet for social network analysis."""
    return create_graph_model(
        node_features=user_features,
        num_classes=num_classes,
        hidden_dim=64,
        num_layers=3,
        aggregation="attention",
        **kwargs,
    )


# =============================================================================
# Algorithm-specific Variants (registered separately for discovery)
# =============================================================================


def _register_variant(name: str, algorithm: str, credit_type: str, bio_score: float):
    """Helper to register algorithm-specific GraphTileNet variants."""

    class _GraphTileNetVariant(GraphTileNet):
        algorithm_name = f"GraphTileNet-{algorithm.upper()}"

        def __init__(
            self,
            config: GraphTileNetConfig | None = None,
            **kwargs: object,
        ) -> None:
            if config is None:
                kwargs.setdefault("algorithm", algorithm)
                config = GraphTileNetConfig(**kwargs)
            elif config.algorithm != algorithm:
                config = dataclasses.replace(config, algorithm=algorithm)
            super().__init__(config=config)

    return _GraphTileNetVariant


# Register algorithm-specific variants
_register_variant("graph_tile_fa", "fa", "target", 0.7)
_register_variant("graph_tile_tp", "tp", "target", 0.65)
_register_variant("graph_tile_hebbian", "hebbian", "hebbian", 0.6)
_register_variant("graph_tile_snn", "snn", "spiking", 0.65)
_register_variant("graph_tile_pc", "pc", "equilibrium", 0.75)
