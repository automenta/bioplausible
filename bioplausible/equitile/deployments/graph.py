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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F
from torch import nn

from bioplausible.core.config import ModelConfig
from bioplausible.core.model import BioModel
from bioplausible.core.model_status import status_tag
from bioplausible.core.registry import Domain, LocalityLevel, register_model
from bioplausible.equitile.deployments import _feature_extractors as _fe

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
class GraphEquiTileConfig:
    """Configuration for Graph EquiTile.

    The graph model trains with standard backprop through its attention + tile
    layers, so it deliberately excludes the PC/EP dynamics fields (``mode``,
    ``inference_steps``, ``step_size``, ``beta``, ``task_type``,
    ``weight_decay``) exposed by the vision/RL deployment configs.
    """

    # Graph settings
    node_features: int = 10
    hidden_dim: int = 64
    num_classes: int = 2

    # Architecture
    num_layers: int = 3
    neurons_per_tile: int = 32
    tiles_per_layer: int = 4
    attention_heads: int = 4

    # Aggregation
    aggregation: Literal["mean", "sum", "max", "attention"] = "mean"
    readout: Literal["mean", "sum", "max", "attention"] = "mean"

    # Learning
    learning_rate: float = 1e-3
    dropout: float = 0.1
    activation: Literal["tanh", "relu", "gelu", "silu"] = "gelu"
    equitile_kwargs: dict[str, object] = field(default_factory=dict)


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

    def __init__(
        self,
        config: GraphEquiTileConfig | None = None,
        **kwargs: object,
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
        self.feature_extractor = GraphFeatureExtractor(config)

        # Output projection
        if config.readout == "attention":
            self.readout_attention = nn.Linear(config.hidden_dim, 1)
        self.output_proj = nn.Linear(config.hidden_dim, config.num_classes)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config.learning_rate,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

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

        logits = self.output_proj(graph_features)

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
        logits = self.forward(node_features, edge_index, batch)

        if labels.dim() == 0 or labels.shape[0] == logits.shape[0]:
            loss = F.cross_entropy(logits, labels)
        else:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        self.optimizer.step()

        with torch.no_grad():
            if labels.dim() == 0 or labels.shape[0] == logits.shape[0]:
                accuracy = (logits.argmax(dim=-1) == labels).float().mean().item()
            else:
                accuracy = (
                    (
                        logits.view(-1, logits.shape[-1]).argmax(dim=-1)
                        == labels.view(-1)
                    )
                    .float()
                    .mean()
                    .item()
                )

        return {
            "loss": loss.item(),
            "accuracy": accuracy,
        }

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
            return logits.argmax(dim=-1)


# =============================================================================
# Factory Functions
# =============================================================================


def create_graph_model(
    node_features: int,
    num_classes: int,
    hidden_dim: int = 64,
    num_layers: int = 3,
    **kwargs: object,
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
    **kwargs: object,
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
    **kwargs: object,
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
