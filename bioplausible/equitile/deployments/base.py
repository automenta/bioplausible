"""Unified base configuration and factory for EquiTile deployments.

Consolidates the common configuration fields and factory patterns shared across
vision.py, timeseries.py, rl.py, and graph.py deployment modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

from bioplausible.core.config import ModelConfig
from bioplausible.core.model import BioModel
from bioplausible.equitile.core.config import EquiTileConfig

if TYPE_CHECKING:
    from torch import Tensor


# =============================================================================
# Base Deployment Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class DeploymentConfig:
    """Base configuration shared by all EquiTile deployments.

    Attributes:
        neurons_per_tile: Number of neurons in each tile.
        tiles_per_layer: Number of tiles per layer.
        num_fc_layers: Number of fully-connected layers in the head.
        learning_rate: Base learning rate.
        dropout: Dropout probability.
        weight_decay: Weight decay coefficient.
        mode: Learning mode (pc, ep, backprop).
        inference_steps: Number of inference/relaxation steps.
        step_size: Step size for relaxation dynamics.
        beta: Beta parameter for EP nudging.
        activation: Activation function.
        task_type: Type of task (classification, regression, etc.).
        equitile_kwargs: Additional kwargs passed to EquiTileConfig.
    """

    neurons_per_tile: int = 64
    tiles_per_layer: int = 4
    num_fc_layers: int = 2
    learning_rate: float = 1e-3
    dropout: float = 0.1
    weight_decay: float = 1e-4
    mode: Literal["pc", "ep", "backprop"] = "pc"
    inference_steps: int = 10
    step_size: float = 0.1
    beta: float = 0.1
    activation: Literal["tanh", "relu", "gelu", "silu"] = "gelu"
    task_type: Literal["classification", "regression", "binary", "multilabel"] = "classification"
    equitile_kwargs: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConvDeploymentConfig(DeploymentConfig):
    """Configuration for convolutional (vision) deployments."""

    input_channels: int = 3
    input_size: int = 32
    num_classes: int = 10
    conv_channels: list[int] = field(default_factory=lambda: [32, 64, 128])
    kernel_sizes: list[int] = field(default_factory=lambda: [3, 3, 3])
    use_pooling: bool = True
    pooling_size: int = 2


@dataclass(frozen=True, slots=True)
class TemporalDeploymentConfig(DeploymentConfig):
    """Configuration for temporal (time series) deployments."""

    seq_len: int = 100
    pred_len: int = 10
    input_dim: int = 10
    output_dim: int = 1
    model_type: Literal["forecasting", "classification", "anomaly_detection"] = "forecasting"
    attention_heads: int = 4
    use_positional_encoding: bool = True
    use_temporal_attention: bool = True
    hidden_dim: int = 64
    num_layers: int = 3


@dataclass(frozen=True, slots=True)
class RLDeploymentConfig(DeploymentConfig):
    """Configuration for RL deployments."""

    obs_dim: int = 8
    action_dim: int = 4
    action_type: Literal["discrete", "continuous"] = "discrete"
    hidden_dim: int = 128
    num_layers: int = 2
    log_std_init: float = 0.0
    log_std_min: float = -20
    log_std_max: float = 2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5


@dataclass(frozen=True, slots=True)
class GraphDeploymentConfig(DeploymentConfig):
    """Configuration for graph deployments."""

    node_features: int = 10
    hidden_dim: int = 64
    num_classes: int = 2
    num_layers: int = 3
    attention_heads: int = 4
    aggregation: Literal["mean", "sum", "max", "attention"] = "mean"
    readout: Literal["mean", "sum", "max", "attention"] = "mean"


# =============================================================================
# Generic Factory Function
# =============================================================================


def create_deployment_model(
    config: DeploymentConfig,
    feature_extractor: nn.Module,
    head_input_dim: int,
    head_output_dim: int,
    **kwargs,
) -> BioModel:
    """Create a deployment model with a feature extractor and EquiTile head.

    This is a generic factory that builds a model combining:
    1. A domain-specific feature extractor (CNN, RNN, GNN, etc.)
    2. An EquiTile head for classification/regression

    Args:
        config: DeploymentConfig subclass instance.
        feature_extractor: Module that extracts features from raw input.
        head_input_dim: Input dimension for the EquiTile head (feature_extractor output).
        head_output_dim: Output dimension for the EquiTile head.
        **kwargs: Additional arguments passed to EquiTileConfig.

    Returns:
        A BioModel with feature_extractor and head attributes.
    """
    # Build EquiTile config for the head
    head_kwargs = config.equitile_kwargs.copy()
    head_kwargs.update({
        "neurons_per_tile": config.neurons_per_tile,
        "tiles_per_layer": config.tiles_per_layer,
        "num_layers": config.num_fc_layers + 2,  # input + fc + output
        "learning_rate": config.learning_rate,
        "dropout": config.dropout,
        "weight_decay": config.weight_decay,
        "mode": config.mode,
        "inference_steps": config.inference_steps,
        "step_size": config.step_size,
        "beta": config.beta,
        "activation": config.activation,
        "task_type": config.task_type,
    })
    head_kwargs.update(kwargs)

    head_config = EquiTileConfig(**head_kwargs)

    # Import here to avoid circular imports
    from bioplausible.equitile.core import EquiTile

    head = EquiTile(
        config=head_config,
        input_dim=head_input_dim,
        output_dim=head_output_dim,
    )

    # Create a composite model
    class DeploymentModel(BioModel):
        def __init__(self) -> None:
            super().__init__(
                ModelConfig(
                    name=config.__class__.__name__.replace("Config", "").lower(),
                    input_dim=head_input_dim,
                    output_dim=head_output_dim,
                )
            )
            self.config = config
            self.feature_extractor = feature_extractor
            self.head = head
            self._step_count = 0

            # Optimizers
            self._optim_feature = torch.optim.Adam(
                self.feature_extractor.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            self._optim_head = torch.optim.Adam(
                self.head.parameters(),
                lr=config.learning_rate,
            )

        def forward(self, x: Tensor, **kwargs) -> Tensor:
            features = self.feature_extractor(x)
            return self.head(features, **kwargs)

        def train_step(self, x: Tensor, y: Tensor) -> dict[str, float]:
            self._step_count += 1
            features = self.feature_extractor(x)
            if config.mode == "backprop":
                logits = self.head(features)
                loss = self.head.task_handler.compute_loss(logits, y)
                self._optim_feature.zero_grad()
                self._optim_head.zero_grad()
                loss.backward()
                self._optim_feature.step()
                self._optim_head.step()
                return {
                    "loss": loss.item(),
                    "accuracy": self.head.compute_metrics(logits, y),
                    "mode": config.mode,
                }
            else:
                return self.head.train_step(features.detach(), y)

    return DeploymentModel()


# =============================================================================
# Convenience Factory Functions
# =============================================================================


def create_vision_deployment(
    config: ConvDeploymentConfig,
    feature_extractor: nn.Module | None = None,
) -> BioModel:
    """Create a vision deployment model.

    Args:
        config: ConvDeploymentConfig instance.
        feature_extractor: Optional custom feature extractor. If None, uses ConvFeatureExtractor.
    """
    if feature_extractor is None:
        feature_extractor = ConvFeatureExtractor(config)

    return create_deployment_model(
        config=config,
        feature_extractor=feature_extractor,
        head_input_dim=feature_extractor.output_size,
        head_output_dim=config.num_classes,
    )


def create_temporal_deployment(
    config: TemporalDeploymentConfig,
    feature_extractor: nn.Module | None = None,
) -> BioModel:
    """Create a time series deployment model."""
    if feature_extractor is None:
        feature_extractor = TemporalFeatureExtractor(config)

    return create_deployment_model(
        config=config,
        feature_extractor=feature_extractor,
        head_input_dim=config.hidden_dim,
        head_output_dim=config.output_dim,
    )


def create_rl_deployment(
    config: RLDeploymentConfig,
    feature_extractor: nn.Module | None = None,
) -> BioModel:
    """Create an RL deployment model."""
    if feature_extractor is None:
        feature_extractor = RLFeatureExtractor(config)

    tile_dim = config.neurons_per_tile * config.tiles_per_layer
    return create_deployment_model(
        config=config,
        feature_extractor=feature_extractor,
        head_input_dim=tile_dim,
        head_output_dim=config.action_dim,  # Actor output
    )


def create_graph_deployment(
    config: GraphDeploymentConfig,
    feature_extractor: nn.Module | None = None,
) -> BioModel:
    """Create a graph deployment model."""
    if feature_extractor is None:
        feature_extractor = GraphFeatureExtractor(config)

    return create_deployment_model(
        config=config,
        feature_extractor=feature_extractor,
        head_input_dim=config.hidden_dim,
        head_output_dim=config.num_classes,
    )


# =============================================================================
# Default Feature Extractors (can be overridden by deployments)
# =============================================================================


class ConvFeatureExtractor(nn.Module):
    """Default convolutional feature extractor for vision."""

    def __init__(self, config: ConvDeploymentConfig) -> None:
        super().__init__()
        self.config = config

        self.conv_stages = nn.ModuleList()
        in_channels = config.input_channels

        for out_channels, kernel_size in zip(config.conv_channels, config.kernel_sizes):
            stages = [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]

            if config.use_pooling:
                stages.append(nn.MaxPool2d(config.pooling_size))

            self.conv_stages.append(nn.Sequential(*stages))
            in_channels = out_channels

        self._output_size = self._compute_output_size(config)

    def _compute_output_size(self, config: ConvDeploymentConfig) -> int:
        size = config.input_size
        channels = config.conv_channels[-1] if config.conv_channels else config.input_channels

        for _ in range(len(config.conv_channels)):
            if config.use_pooling:
                size = size // config.pooling_size

        return channels * size * size

    def forward(self, x: Tensor) -> Tensor:
        for stage in self.conv_stages:
            x = stage(x)
        return x.view(x.size(0), -1)

    @property
    def output_size(self) -> int:
        return self._output_size


class TemporalFeatureExtractor(nn.Module):
    """Default temporal feature extractor for time series."""

    def __init__(self, config: TemporalDeploymentConfig) -> None:
        super().__init__()
        self.config = config

        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)

        if config.use_positional_encoding:
            self.pos_encoding = TemporalPositionalEncoding(
                embed_dim=config.hidden_dim,
                max_len=config.seq_len,
                dropout=config.dropout,
            )
        else:
            self.pos_encoding = None

        self.layers = nn.ModuleList([
            TemporalEquiTileLayer(config) for _ in range(config.num_layers)
        ])

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_proj(x)
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        # Return last time step for forecasting, or pooled for classification
        if config.model_type == "forecasting":
            return x[:, -1, :]
        return x.mean(dim=1)


class TemporalPositionalEncoding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        max_len: int = 500,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TemporalEquiTileLayer(nn.Module):
    def __init__(self, config: TemporalDeploymentConfig) -> None:
        super().__init__()
        self.config = config

        if config.use_temporal_attention:
            self.attention = TemporalAttentionLayer(
                embed_dim=config.hidden_dim,
                num_heads=config.attention_heads,
                dropout=config.dropout,
            )
            self.norm1 = nn.LayerNorm(config.hidden_dim)
        else:
            self.attention = None
            self.norm1 = None

        self.norm2 = nn.LayerNorm(config.hidden_dim)

        layer_kwargs = config.equitile_kwargs.copy()
        layer_kwargs.update({
            "neurons_per_tile": config.neurons_per_tile,
            "num_layers": 2,
            "tiles_per_layer": config.tiles_per_layer,
            "learning_rate": config.learning_rate,
            "dropout": config.dropout,
        })
        from bioplausible.equitile.core.config import EquiTileConfig
        from bioplausible.equitile.core import EquiTile

        equitile_config = EquiTileConfig(**layer_kwargs)
        self.equitile = EquiTile(
            config=equitile_config,
            input_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
        )

        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        if self.attention is not None:
            attn_output = self.attention(x, mask)
            x = x + attn_output
            x = self.norm1(x)

        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(batch_size * seq_len, hidden_dim)
        tile_output = self.equitile(x_flat)
        x = x + tile_output.view(batch_size, seq_len, hidden_dim)

        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.norm2(x)
        return x


class TemporalAttentionLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim**-0.5

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output)


class RLFeatureExtractor(nn.Module):
    """Default RL feature extractor using EquiTile."""

    def __init__(self, config: RLDeploymentConfig) -> None:
        super().__init__()
        self.config = config

        tile_dim = config.neurons_per_tile * config.tiles_per_layer

        from bioplausible.equitile.core.config import EquiTileConfig
        from bioplausible.equitile.core import EquiTile

        equitile_kwargs = config.equitile_kwargs.copy()
        equitile_kwargs.update({
            "neurons_per_tile": config.neurons_per_tile,
            "num_layers": config.num_layers,
            "tiles_per_layer": config.tiles_per_layer,
            "mode": config.mode,
            "inference_steps": config.inference_steps,
            "learning_rate": config.learning_rate,
            "activation": config.activation,
        })
        equitile_config = EquiTileConfig(**equitile_kwargs)

        self.equitile = EquiTile(
            config=equitile_config,
            input_dim=config.obs_dim,
            output_dim=tile_dim,
        )

    def forward(self, obs: Tensor) -> Tensor:
        return self.equitile(obs)


class GraphFeatureExtractor(nn.Module):
    """Default graph feature extractor using EquiTile layers."""

    def __init__(self, config: GraphDeploymentConfig) -> None:
        super().__init__()
        self.config = config

        self.input_proj = nn.Linear(config.node_features, config.hidden_dim)
        self.layers = nn.ModuleList([
            GraphEquiTileLayer(config) for _ in range(config.num_layers)
        ])

    def forward(self, node_features: Tensor, edge_index: Tensor) -> Tensor:
        x = self.input_proj(node_features)
        for layer in self.layers:
            x = layer(x, edge_index)
        return x


class GraphEquiTileLayer(nn.Module):
    def __init__(self, config: GraphDeploymentConfig) -> None:
        super().__init__()
        self.config = config

        self.attention = GraphAttentionLayer(
            in_features=config.hidden_dim,
            out_features=config.hidden_dim,
            num_heads=config.attention_heads,
            dropout=config.dropout,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.hidden_dim)

        layer_kwargs = config.equitile_kwargs.copy()
        layer_kwargs.update({
            "neurons_per_tile": config.neurons_per_tile,
            "num_layers": 2,
            "tiles_per_layer": config.tiles_per_layer,
            "learning_rate": config.learning_rate,
            "dropout": config.dropout,
            "activation": config.activation,
        })
        from bioplausible.equitile.core.config import EquiTileConfig
        from bioplausible.equitile.core import EquiTile

        equitile_config = EquiTileConfig(**layer_kwargs)
        self.equitile = EquiTile(
            config=equitile_config,
            input_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
        )

        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
        )

    def forward(self, node_features: Tensor, edge_index: Tensor) -> Tensor:
        attn_output = self.attention(node_features, edge_index)
        node_features = node_features + self.dropout(attn_output)
        node_features = self.norm(node_features)

        tile_output = self.equitile(node_features)
        node_features = node_features + tile_output

        ffn_output = self.ffn(node_features)
        node_features = node_features + ffn_output
        return node_features


class GraphAttentionLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads

        assert out_features % num_heads == 0, "out_features must be divisible by num_heads"

        self.q_proj = nn.Linear(in_features, out_features)
        self.k_proj = nn.Linear(in_features, out_features)
        self.v_proj = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim**-0.5

    def forward(self, node_features: Tensor, edge_index: Tensor) -> Tensor:
        num_nodes = node_features.shape[0]
        num_edges = edge_index.shape[1]

        q = self.q_proj(node_features).view(num_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(node_features).view(num_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(node_features).view(num_nodes, self.num_heads, self.head_dim)

        src_idx, dst_idx = edge_index[0], edge_index[1]
        q_dst = q[dst_idx]
        k_src = k[src_idx]
        v_src = v[src_idx]

        scores = (q_dst * k_src).sum(dim=-1) * self.scale
        scores = self.dropout(torch.softmax(scores, dim=0))

        messages = scores.unsqueeze(-1) * v_src

        # Aggregate
        out = torch.zeros(num_nodes, self.num_heads, self.head_dim, device=node_features.device)
        out.index_add_(0, dst_idx, messages)

        return out.view(num_nodes, -1)