"""
EquiTile Graph: Graph Neural Networks with EquiTile
====================================================

Extends EquiTile for graph-structured data:
- GraphEquiTile: Graph neural network with tile-based message passing
- Graph attention mechanisms
- Support for node/graph classification
- Integration with networkx and torch_geometric

The shared graph layers (feature extractor, attention, scatter utilities)
now live in the private ``_feature_extractors`` module and are re-exported;
this module adds the graph-specific model (readout/forward over batches).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

from bioplausible.config.unified import ModelConfig
from bioplausible.core.model import BioModel
from bioplausible.core.model_status import status_tag
from bioplausible.core.registry import Domain, LocalityLevel, register_model
from bioplausible.core.utils.optimizer import OptimizerConfig, create_optimizer
from bioplausible.equitile.deployments import _feature_extractors as _fe
from bioplausible.equitile.deployments.base import (
    GraphDeploymentConfig,
    build_tile_head,
)

# Re-export shared graph components under their historical names.
GraphAttentionLayer = _fe.GraphAttentionLayer
GraphEquiTileLayer = _fe.GraphEquiTileLayer
GraphFeatureExtractor = _fe.GraphFeatureExtractor
aggregate_messages = _fe.aggregate_messages
create_graph_from_edges = _fe.create_graph_from_edges
scatter_max = _fe.scatter_max
scatter_mean = _fe.scatter_mean
scatter_sum = _fe.scatter_sum
add_self_loops = _fe.add_self_loops

__all__ = [
    "GraphAttentionLayer",
    "GraphEquiTile",
    "GraphEquiTileConfig",
    "GraphEquiTileLayer",
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
class GraphEquiTileConfig(GraphDeploymentConfig):
    """Configuration for Graph EquiTile.

    Inherits the shared deployment fields from ``GraphDeploymentConfig`` and
    keeps the same defaults the historical ``GraphEquiTileConfig`` exposed.
    """

    # Historical graph defaults differ from the generic base.
    learning_rate: float = 1e-3
    num_layers: int = 3
    neurons_per_tile: int = 32
    tiles_per_layer: int = 4
    attention_heads: int = 4
    mode: Literal["pc", "ep", "backprop"] = "backprop"


# =============================================================================
# Graph EquiTile
# =============================================================================


@register_model(
    "graph_equitile",
    domains=[Domain.GRAPH],
    locality_level=LocalityLevel.LOCAL,
    bio_plausibility_score=0.75,
    requires_backward=False,
    credit_assignment_type="hebbian",
    family="equitile",
    tags=[status_tag("experimental")],
)
class GraphEquiTile(BioModel):
    """Graph EquiTile for graph-structured data.

    Combines graph attention with EquiTile's tile-based
    message passing for node and graph classification.

    Parameters
    ----------
    config : GraphEquiTileConfig, optional
        Configuration
    **kwargs
        Additional configuration parameters
    """

    algorithm_name = "GraphEquiTile"

    @classmethod
    def build(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
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
        """Build GraphEquiTile from factory arguments."""
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

        valid_keys = GraphEquiTileConfig.__annotations__.keys()
        for k, v in kwargs.items():
            if k in valid_keys:
                config_kwargs[k] = v

        for k, v in spec.custom_hyperparams.items():
            if k in valid_keys:
                config_kwargs[k] = v

        config = GraphEquiTileConfig(**config_kwargs)

        model = cls(config=config)
        return model.to(device)

    def __init__(
        self,
        config: GraphEquiTileConfig | None = None,
        **kwargs,
    ) -> None:
        if config is None:
            config = GraphEquiTileConfig(**kwargs)

        super().__init__(
            ModelConfig(
                name="graph_equitile",
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

        # Optimizers
        self._optim_feature = create_optimizer(
            self.feature_extractor,
            OptimizerConfig(
                name="adam", lr=config.learning_rate, weight_decay=config.weight_decay
            ),
        )
        self._optim_head = create_optimizer(
            self.head, OptimizerConfig(name="adam", lr=config.learning_rate)
        )

        # Readout attention (for attention-based readout)
        if config.readout == "attention":
            self.readout_attention = nn.Linear(config.hidden_dim, 1)

        # State tracking
        self._step_count = 0

    def _build_tile_head(self, config: GraphEquiTileConfig) -> None:
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

        if self.config.mode == "backprop":
            logits = self.head.forward_logits(graph_features, detach_input=False)
            loss = self.head.compute_loss(logits, labels)

            self._optim_feature.zero_grad()
            self._optim_head.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

            self._optim_feature.step()
            self._optim_head.step()

            return {
                "loss": loss.item(),
                "accuracy": self.head.compute_metrics(logits, labels),
                "mode": self.config.mode,
            }
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
) -> GraphEquiTile:
    """Create GraphEquiTile model."""
    config = GraphEquiTileConfig(
        node_features=node_features,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        **kwargs,
    )
    return GraphEquiTile(config)


def create_molecule_model(
    atom_features: int = 9,
    num_classes: int = 2,
    **kwargs,
) -> GraphEquiTile:
    """Create GraphEquiTile for molecular property prediction."""
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
) -> GraphEquiTile:
    """Create GraphEquiTile for social network analysis."""
    return create_graph_model(
        node_features=user_features,
        num_classes=num_classes,
        hidden_dim=64,
        num_layers=3,
        aggregation="attention",
        **kwargs,
    )
