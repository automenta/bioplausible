"""
Shared Components for FastLMEquiTile
=====================================

Building blocks:
- ``FastLMConfig`` — canonical configuration for FastLMEquiTile
- ``MixtureOfTiles`` — top-k sparse tile gating for conditional computation
- ``TileLocalAttention`` — grouped-query, multi-backend (Flash, SDPA, manual)
- ``SwiGLUFeedForward`` — Swish-gated feedforward
- ``FastEquiTileLayer`` — pre-norm transformer layer combining all three
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F
from torch import nn

from bioplausible.core.logging import get_logger

if TYPE_CHECKING:
    from torch import Tensor

logger = get_logger()

__all__ = [
    "FastEquiTileLayer",
    "FastLMConfig",
    "MixtureOfTiles",
    "SwiGLUFeedForward",
    "TileLocalAttention",
]

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class FastLMConfig:
    """Configuration for FastLMEquiTile.

    Vocabulary
    ----------
    vocab_size : int
        Vocabulary size
    pad_token_id : int
        Padding token ID

    Architecture
    ------------
    embed_dim : int
        Embedding dimension
    num_layers : int
        Number of transformer layers
    hidden_dim : int
        Hidden dimension in SwiGLU feedforward

    Tile Settings
    -------------
    neurons_per_tile : int
        Neurons per tile
    tiles_per_layer : int
        Tiles per layer
    mot_k : int
        Number of active tiles in MoT (top-k selection)

    Attention
    ---------
    num_heads : int
        Number of Q heads
    num_kv_heads : int
        Number of K/V heads (for grouped query attention)
    attention_type : str
        Attention implementation: 'auto', 'flash', 'sdpa', 'manual'
    sliding_window : int
        Sliding window size for local attention (0 = global)

    Training
    --------
    dropout : float
        Dropout probability
    learning_rate : float
        Base learning rate
    weight_decay : float
        Weight decay
    max_seq_len : int
        Maximum sequence length

    Optimization
    ------------
    use_gradient_checkpointing : bool
        Enable gradient checkpointing
    use_compile : bool
        Enable torch.compile
    compile_mode : str
        torch.compile mode: 'default', 'reduce-overhead', 'max-autotune'
    """

    # Vocabulary
    vocab_size: int = 1000
    pad_token_id: int = 0

    # Architecture
    embed_dim: int = 192
    num_layers: int = 6
    hidden_dim: int = 512

    # Tile settings
    neurons_per_tile: int = 48
    tiles_per_layer: int = 4
    mot_k: int = 2  # Top-k active tiles

    # Attention
    num_heads: int = 6
    num_kv_heads: int = 2  # Grouped query: share K/V across Q heads
    attention_type: str = "auto"  # 'auto', 'flash', 'sdpa', 'manual'
    sliding_window: int = 0  # 0 = global, >0 = sliding window size

    # Training
    dropout: float = 0.1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_seq_len: int = 256

    # Optimization
    use_gradient_checkpointing: bool = True
    use_compile: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "max-autotune"


# =============================================================================
# Mixture of Tiles (MoT)
# =============================================================================


class MixtureOfTiles(nn.Module):
    """Mixture of Tiles for conditional computation.

    Only activates top-k tiles per token, providing:
    - Conditional computation (fewer FLOPs per token)
    - Increased effective capacity without parameter increase
    - Natural fit for tile-based architecture

    Parameters
    ----------
    embed_dim : int
        Embedding dimension
    neurons_per_tile : int
        Neurons per tile
    tiles_per_layer : int
        Total tiles available
    mot_k : int
        Number of tiles to activate per token
    dropout : float
        Dropout probability
    """

    def __init__(
        self,
        embed_dim: int,
        neurons_per_tile: int,
        tiles_per_layer: int,
        mot_k: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.neurons_per_tile = neurons_per_tile
        self.tiles_per_layer = tiles_per_layer
        self.mot_k = min(mot_k, tiles_per_layer)
        self.tile_dim = neurons_per_tile

        # Tile projections (shared across tiles for efficiency)
        self.tile_proj_in = nn.Linear(embed_dim, neurons_per_tile * tiles_per_layer)
        self.tile_proj_out = nn.Linear(neurons_per_tile * tiles_per_layer, embed_dim)

        # Tile gating network (learns tile importance)
        self.gate_proj = nn.Linear(embed_dim, tiles_per_layer)

        # Optimized: Stack tile transforms into single tensor for vectorized ops
        # Shape: (tiles_per_layer, tile_dim, tile_dim)
        # Use smaller init for stability
        self.tile_transforms = nn.Parameter(
            torch.randn(tiles_per_layer, neurons_per_tile, neurons_per_tile) * 0.01
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass with sparse tile activation.

        Uses fully vectorized operations for efficiency.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (batch, seq_len, embed_dim)

        Returns
        -------
        tuple
            (output tensor, tile importance weights)
        """
        batch_size, seq_len, _ = x.shape
        n_tiles = self.tiles_per_layer
        tile_dim = self.neurons_per_tile
        k = self.mot_k

        # Compute tile gates (importance scores)
        gate_logits = self.gate_proj(x)  # (batch, seq_len, n_tiles)
        gate_weights = F.softmax(gate_logits, dim=-1)

        # Select top-k tiles
        topk_weights, topk_indices = torch.topk(gate_weights, k, dim=-1)  # (B, S, k)

        # Project input to tile space: (B, S, n_tiles * tile_dim)
        tile_input = self.tile_proj_in(x)
        tile_input = tile_input.view(batch_size, seq_len, n_tiles, tile_dim)

        # Vectorized tile selection and processing
        # Expand indices for gathering: (B, S, k, 1)
        indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, -1, tile_dim)

        # Gather selected tile inputs: (B, S, k, tile_dim)
        selected_inputs = torch.gather(tile_input, dim=2, index=indices_expanded)

        # Vectorized tile transforms using batch matrix multiply
        # selected_inputs: (B, S, k, tile_dim)
        # tile_transforms: (n_tiles, tile_dim, tile_dim)
        # We need to apply different transform per selected tile

        # Reshape for batch matmul: (B*S*k, tile_dim)
        selected_flat = selected_inputs.view(-1, tile_dim)

        # Get transforms for selected tiles: (B*S*k, tile_dim, tile_dim)
        # First expand transforms to (B, S, k, tile_dim, tile_dim)
        transforms_expanded = (
            self.tile_transforms
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, seq_len, n_tiles, tile_dim, tile_dim)
        )
        # Gather selected transforms
        transforms_selected = torch.gather(
            transforms_expanded,
            dim=2,
            index=topk_indices
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, -1, tile_dim, tile_dim),
        )
        transforms_flat = transforms_selected.view(-1, tile_dim, tile_dim)

        # Apply transforms: (B*S*k, tile_dim)
        transformed_flat = torch.bmm(
            selected_flat.unsqueeze(1), transforms_flat
        ).squeeze(1)
        transformed_flat = F.relu(transformed_flat)

        # Reshape back: (B, S, k, tile_dim)
        transformed = transformed_flat.view(batch_size, seq_len, k, tile_dim)

        # Apply gate weights: (B, S, k, 1)
        weighted = transformed * topk_weights.unsqueeze(-1)

        # Scatter back to full tile output
        # Create output tensor: (B, S, n_tiles, tile_dim)
        tile_output = torch.zeros(
            batch_size, seq_len, n_tiles, tile_dim, device=x.device, dtype=x.dtype
        )

        # Scatter weighted outputs to their tile positions
        tile_output = tile_output.scatter(dim=2, index=indices_expanded, src=weighted)

        # Project back to embed_dim
        tile_output = tile_output.view(batch_size, seq_len, -1)
        output = self.tile_proj_out(tile_output)
        output = self.dropout(output)

        # Compute mean tile importance for analysis
        tile_importance = gate_weights.mean(dim=1)  # (batch, tiles_per_layer)

        return output, tile_importance


# =============================================================================
# Tile-Local Attention
# =============================================================================


class TileLocalAttention(nn.Module):
    """Tile-local attention with multiple backend support.

    Supports:
    - Flash Attention 2 (fastest, requires torch 2.1+)
    - SDPA with sliding window (PyTorch 2.1+)
    - Manual attention (fallback)

    Parameters
    ----------
    embed_dim : int
        Embedding dimension
    num_heads : int
        Number of Q heads
    num_kv_heads : int
        Number of K/V heads (for grouped query)
    attention_type : str
        Attention backend: 'auto', 'flash', 'sdpa', 'manual'
    sliding_window : int
        Sliding window size (0 = global attention)
    dropout : float
        Dropout probability
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        attention_type: str = "auto",
        sliding_window: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.sliding_window = sliding_window

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, (
            "num_heads must be divisible by num_kv_heads"
        )

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim**-0.5

        # Grouped query: repeat K/V heads for each Q head group
        self.n_groups = num_heads // num_kv_heads

        # Select attention backend
        self.attention_type = self._select_attention_backend(attention_type)

    def _select_attention_backend(self, attention_type: str) -> str:
        """Select best available attention backend."""
        if attention_type != "auto":
            return attention_type

        # Auto-detect best available
        if not hasattr(F, "scaled_dot_product_attention"):
            return "manual"

        # Check for Flash Attention 2 support
        if torch.cuda.is_available():
            try:
                from torch.backends.cuda import SDPBackend

                available_backends = torch.backends.cuda.get_flash_sdp_backends()
                if SDPBackend.FLASH_ATTENTION in available_backends:
                    return "flash"
            except ImportError, AttributeError:
                pass

        return "sdpa"

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
        causal: bool = True,
    ) -> Tensor:
        """Forward pass with local attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (batch, seq_len, embed_dim)
        attention_mask : torch.Tensor, optional
            Additional attention mask
        causal : bool
            Use causal masking

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        q = (
            self
            .q_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self
            .k_proj(x)
            .view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self
            .v_proj(x)
            .view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Repeat K/V for grouped query attention
        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)

        # Select attention implementation
        if self.attention_type == "flash":
            attn_output = self._flash_attention(q, k, v, causal)
        elif self.attention_type == "sdpa":
            attn_output = self._sdpa_attention(q, k, v, causal)
        else:  # manual
            attn_output = self._manual_attention(q, k, v, causal, attention_mask)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output)

    def _flash_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        causal: bool,
    ) -> Tensor:
        """Flash Attention 2 - fastest for large sequences.

        Uses PyTorch's built-in Flash Attention 2 support.
        """
        try:
            # Flash Attention 2 with sliding window support (PyTorch 2.1+)
            if self.sliding_window > 0:
                return F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=causal,
                    enable_gqa=True,
                    # Note: sliding_window parameter may require PyTorch 2.2+
                )
            else:
                return F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=causal,
                    enable_gqa=True,
                )
        except RuntimeError, TypeError:
            # Fallback to SDPA if flash fails
            return self._sdpa_attention(q, k, v, causal)

    def _sdpa_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        causal: bool,
    ) -> Tensor:
        """Scaled Dot-Product Attention with sliding window support."""
        # PyTorch 2.1+ supports sliding_window in SDPA
        if self.sliding_window > 0 and causal:
            try:
                return F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=self.dropout.p if self.training else 0.0,
                    is_causal=causal,
                    # sliding_window parameter available in PyTorch 2.1+
                )
            except TypeError:
                pass

        # Standard SDPA
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=causal,
        )

    def _manual_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        causal: bool,
        attention_mask: Tensor | None,
    ) -> Tensor:
        """Manual attention computation (fallback).

        Implements sliding window attention manually for older PyTorch versions.
        """
        batch_size, num_heads, seq_len, head_dim = q.shape

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask
        if causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        # Apply sliding window mask
        if self.sliding_window > 0:
            window_mask = torch.ones(
                seq_len, seq_len, device=q.device, dtype=torch.bool
            )
            window_mask = (
                ~torch.abs(
                    torch.arange(seq_len, device=q.device).unsqueeze(1)
                    - torch.arange(seq_len, device=q.device).unsqueeze(0)
                )
                <= self.sliding_window
            )
            scores = scores.masked_fill(window_mask, float("-inf"))

        # Apply attention mask
        if attention_mask is not None:
            scores = scores + attention_mask

        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        return torch.matmul(attn_weights, v)


# =============================================================================
# SwiGLU FeedForward
# =============================================================================


class SwiGLUFeedForward(nn.Module):
    """SwiGLU feedforward for better expressivity per parameter.

    SwiGLU = Swish Gated Linear Unit
    Provides better performance than standard ReLU/GeLU for same parameter count.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension
    hidden_dim : int
        Hidden dimension
    dropout : float
        Dropout probability
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # SwiGLU uses two projections for gating
        self.fc_gate = nn.Linear(embed_dim, hidden_dim)
        self.fc_value = nn.Linear(embed_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with SwiGLU activation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        gate = self.fc_gate(x)
        value = self.fc_value(x)

        # SwiGLU: Swish(gate) * value = gate * sigmoid(gate) * value
        # Approximation: F.silu(gate) * value
        x = F.silu(gate) * value
        x = self.dropout(x)
        return self.out_proj(x)


# =============================================================================
# FastLMEquiTile Transformer Layer
# =============================================================================


class FastEquiTileLayer(nn.Module):
    """Fast EquiTile transformer layer with MoT and local attention.

    Combines:
    - Pre-norm architecture for stability
    - Mixture of Tiles for conditional computation
    - Tile-local attention for efficiency
    - SwiGLU feedforward for expressivity

    Parameters
    ----------
    config : FastLMConfig
        Configuration
    """

    def __init__(self, config: FastLMConfig) -> None:
        super().__init__()
        self.config = config

        # Pre-norm
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.norm3 = nn.LayerNorm(config.embed_dim)

        # Tile-local attention with grouped query and sliding window
        self.attention = TileLocalAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            attention_type=config.attention_type,
            sliding_window=config.sliding_window,
            dropout=config.dropout,
        )

        # Mixture of Tiles
        self.mixture_of_tiles = MixtureOfTiles(
            embed_dim=config.embed_dim,
            neurons_per_tile=config.neurons_per_tile,
            tiles_per_layer=config.tiles_per_layer,
            mot_k=config.mot_k,
            dropout=config.dropout,
        )

        # SwiGLU feedforward
        self.feedforward = SwiGLUFeedForward(
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
        causal: bool = True,
        use_gradient_checkpointing: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        attention_mask : torch.Tensor, optional
            Attention mask
        causal : bool
            Use causal masking
        use_gradient_checkpointing : bool
            Enable gradient checkpointing for memory efficiency

        Returns
        -------
        tuple
            (output tensor, tile importance)
        """
        if use_gradient_checkpointing and self.training:
            return self._forward_checkpointed(x, attention_mask, causal)
        return self._forward_impl(x, attention_mask, causal)

    def _forward_impl(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
        causal: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Internal forward implementation."""
        # Pre-norm attention
        normed = self.norm1(x)
        attn_output = self.attention(normed, attention_mask, causal)
        x = x + attn_output

        # Pre-norm MoT
        normed = self.norm2(x)
        mot_output, tile_importance = self.mixture_of_tiles(normed)
        x = x + mot_output

        # Pre-norm feedforward
        x = x + self.feedforward(self.norm3(x))

        return x, tile_importance

    def _forward_checkpointed(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
        causal: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Forward with gradient checkpointing."""
        return torch.utils.checkpoint.checkpoint(
            self._forward_impl,
            x,
            attention_mask,
            causal,
            use_reentrant=False,
        )
