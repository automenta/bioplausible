"""Generic tile-substrate feature extractors, attention layers, and graph utilities.

EquiTile-free building blocks lifted from ``equitile.deployments._feature_extractors``
during generification (now canonical home: ``zoo.models.deployments._feature_extractors``).
The tile-embedding layers are injected with a
``TileModelFactory`` so they can be bound to any local-learning model without a
``core -> equitile`` dependency.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from torch import Tensor

type TileModelFactory = Callable[..., nn.Module]

__all__ = [
    "ConvFeatureExtractor",
    "GraphAttentionLayer",
    "GraphEquiTileLayer",
    "GraphFeatureExtractor",
    "TemporalAttentionLayer",
    "TemporalEquiTileLayer",
    "TemporalFeatureExtractor",
    "TemporalPositionalEncoding",
    "TileModelFactory",
    "aggregate_messages",
    "create_graph_from_edges",
    "scatter_max",
    "scatter_mean",
    "scatter_sum",
]

# add_self_loops intentionally omitted from __all__: not re-exported upstream.


class _ConvConfig(Protocol):
    input_channels: int
    input_size: int
    conv_channels: list[int]
    kernel_sizes: list[int]
    use_pooling: bool
    pooling_size: int


class _TemporalConfig(Protocol):
    use_temporal_attention: bool
    hidden_dim: int
    attention_heads: int
    dropout: float
    equitile_kwargs: dict
    neurons_per_tile: int
    tiles_per_layer: int
    learning_rate: float
    input_dim: int
    use_positional_encoding: bool
    seq_len: int
    num_layers: int
    model_type: str


class _GraphConfig(Protocol):
    hidden_dim: int
    attention_heads: int
    dropout: float
    equitile_kwargs: dict
    neurons_per_tile: int
    tiles_per_layer: int
    learning_rate: float
    activation: str
    node_features: int
    num_layers: int


# =============================================================================
# Graph scatter utilities
# =============================================================================


def aggregate_messages(
    messages: Tensor,
    edge_index: Tensor,
    num_nodes: int,
    method: str = "mean",
) -> Tensor:
    """Aggregate messages from neighbors."""
    match method:
        case "mean":
            return scatter_mean(messages, edge_index[0], dim=0, dim_size=num_nodes)
        case "sum":
            return scatter_sum(messages, edge_index[0], dim=0, dim_size=num_nodes)
        case "max":
            return scatter_max(messages, edge_index[0], dim=0, dim_size=num_nodes)
        case _:
            raise ValueError(f"Unknown aggregation method: {method}")


def scatter_mean(
    src: Tensor, index: Tensor, dim: int = 0, dim_size: int | None = None
) -> Tensor:
    """Scatter mean aggregation."""
    if dim_size is None:
        dim_size = int(index.max().item()) + 1

    out = src.new_zeros((dim_size,) + src.shape[1:])
    count = src.new_zeros(dim_size)

    out.index_add_(dim, index, src)
    count.index_add_(0, index, src.new_ones(src.shape[0]))

    count = count.clamp(min=1)
    return out / count.unsqueeze(-1)


def scatter_sum(
    src: Tensor, index: Tensor, dim: int = 0, dim_size: int | None = None
) -> Tensor:
    """Scatter sum aggregation."""
    if dim_size is None:
        dim_size = int(index.max().item()) + 1

    out = src.new_zeros((dim_size,) + src.shape[1:])
    out.index_add_(dim, index, src)
    return out


def scatter_max(
    src: Tensor, index: Tensor, dim: int = 0, dim_size: int | None = None
) -> Tensor:
    """Scatter max aggregation."""
    if dim_size is None:
        dim_size = int(index.max().item()) + 1

    out = src.new_full((dim_size,) + src.shape[1:], float("-inf"))
    out.index_reduce_(dim, index, src, reduce="amax")

    out[out == float("-inf")] = 0
    return out


def create_graph_from_edges(
    edge_index: Tensor,
    node_features: Tensor | None = None,
    num_nodes: int | None = None,
) -> tuple[Tensor, Tensor, int]:
    """Create graph data structures from an edge index."""
    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1

    if node_features is None:
        node_features = torch.randn(num_nodes, 10)  # Default features

    return node_features, edge_index, num_nodes


def add_self_loops(
    edge_index: Tensor,
    num_nodes: int | None = None,
) -> Tensor:
    """Add self-loops to an edge index."""
    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1

    self_loop = (
        torch.arange(num_nodes, device=edge_index.device).unsqueeze(0).repeat(2, 1)
    )
    return torch.cat([edge_index, self_loop], dim=1)


# =============================================================================
# Conv feature extractor (vision)
# =============================================================================


class ConvFeatureExtractor(nn.Module):
    """Convolutional feature extractor for vision deployments."""

    def __init__(self, config: _ConvConfig) -> None:
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

    def _compute_output_size(self, config: _ConvConfig) -> int:
        size = config.input_size
        channels = (
            config.conv_channels[-1] if config.conv_channels else config.input_channels
        )

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


# =============================================================================
# Temporal feature extractor (time series)
# =============================================================================


class TemporalFeatureExtractor(nn.Module):
    """Default temporal feature extractor for time series."""

    def __init__(
        self, config: _TemporalConfig, tile_model_factory: TileModelFactory
    ) -> None:
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
            TemporalEquiTileLayer(config, tile_model_factory)
            for _ in range(config.num_layers)
        ])

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_proj(x)
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        if self.config.model_type == "forecasting":
            return x[:, -1, :]
        return x.mean(dim=1)


class TemporalPositionalEncoding(nn.Module):
    """Positional encoding for time series."""

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
            torch.arange(0, embed_dim, 2)
            * (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TemporalAttentionLayer(nn.Module):
    """Temporal attention layer for time series."""

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

        q = (
            self
            .q_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self
            .k_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self
            .v_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = (
            attn_output
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )
        return self.out_proj(attn_output)


class TemporalEquiTileLayer(nn.Module):
    """Time series tile layer with optional attention."""

    def __init__(
        self,
        config: _TemporalConfig,
        tile_model_factory: TileModelFactory,
    ) -> None:
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

        self.tile_model = tile_model_factory(
            input_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
            **tile_model_kwargs(config, num_layers=2),
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
        tile_output = self.tile_model(x_flat)
        x = x + tile_output.view(batch_size, seq_len, hidden_dim)

        ffn_output = self.ffn(x)
        x = x + ffn_output
        x = self.norm2(x)
        return x


class GraphEquiTileLayer(nn.Module):
    """Graph tile layer with tile-based message passing."""

    def __init__(
        self,
        config: _GraphConfig,
        tile_model_factory: TileModelFactory,
    ) -> None:
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

        self.tile_model = tile_model_factory(
            input_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
            **tile_model_kwargs(config, num_layers=2, activation=config.activation),
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

        tile_output = self.tile_model(node_features)
        node_features = node_features + tile_output

        ffn_output = self.ffn(node_features)
        node_features = node_features + ffn_output
        return node_features


# =============================================================================
# Graph feature extractor
# =============================================================================


class GraphFeatureExtractor(nn.Module):
    """Default graph feature extractor using tile layers."""

    def __init__(
        self,
        config: _GraphConfig,
        tile_model_factory: TileModelFactory,
    ) -> None:
        super().__init__()
        self.config = config

        self.input_proj = nn.Linear(config.node_features, config.hidden_dim)
        self.layers = nn.ModuleList([
            GraphEquiTileLayer(config, tile_model_factory)
            for _ in range(config.num_layers)
        ])

    def forward(self, node_features: Tensor, edge_index: Tensor) -> Tensor:
        x = self.input_proj(node_features)
        for layer in self.layers:
            x = layer(x, edge_index)
        return x


class GraphAttentionLayer(nn.Module):
    """Graph attention layer for tile models."""

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

        assert out_features % num_heads == 0, (
            "out_features must be divisible by num_heads"
        )

        self.q_proj = nn.Linear(in_features, out_features)
        self.k_proj = nn.Linear(in_features, out_features)
        self.v_proj = nn.Linear(in_features, out_features)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim**-0.5

    def forward(self, node_features: Tensor, edge_index: Tensor) -> Tensor:
        num_nodes = node_features.shape[0]

        q = self.q_proj(node_features).view(num_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(node_features).view(num_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(node_features).view(num_nodes, self.num_heads, self.head_dim)

        src_idx, dst_idx = edge_index[0], edge_index[1]
        q_dst = q[dst_idx]
        k_src = k[src_idx]
        v_src = v[src_idx]

        scores = (q_dst * k_src).sum(dim=-1) * self.scale  # (num_edges, heads)
        scores = self.dropout(F.softmax(scores, dim=0))

        messages = scores.unsqueeze(-1) * v_src  # (num_edges, heads, head_dim)

        output = aggregate_messages(
            messages.view(messages.shape[0], -1),
            edge_index,
            num_nodes,
            method="sum",
        )

        return output.view(num_nodes, -1)


def tile_model_kwargs(
    config: _TemporalConfig | _GraphConfig,
    *,
    num_layers: int,
    activation: str | None = None,
) -> dict[str, object]:
    """Build the tile-model kwargs from a deployment config (base fields only)."""
    kwargs = dict(config.equitile_kwargs)
    kwargs.update({
        "neurons_per_tile": config.neurons_per_tile,
        "num_layers": num_layers,
        "tiles_per_layer": config.tiles_per_layer,
        "learning_rate": config.learning_rate,
        "dropout": config.dropout,
    })
    if activation is not None:
        kwargs["activation"] = activation
    return kwargs
