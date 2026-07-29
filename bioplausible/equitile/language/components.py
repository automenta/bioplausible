"""Shared components for EquiTile language model variants.

Provides canonical building blocks reused across canonical, optimized,
and fast LM variants:

- PositionalEncoding: Sinusoidal position embeddings
- TileAttention: Multi-head causal attention
- TileFeedForward: GELU-activated FFN
- make_causal_mask: Standalone causal mask builder
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "PositionalEncoding",
    "TileAttention",
    "TileFeedForward",
    "make_causal_mask",
]


def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create a causal (upper-triangular) boolean mask.

    Parameters
    ----------
    seq_len : int
        Sequence length.
    device : torch.device
        Target device.

    Returns
    -------
    torch.Tensor
        Boolean mask of shape (seq_len, seq_len) with ``True``
        in the upper triangle (positions that should be masked).
    """
    return torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
    )


class PositionalEncoding(nn.Module):
    """Positional encoding for sequences.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension.
    max_len : int
        Maximum sequence length.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        embed_dim: int,
        max_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )

        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (batch, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Output with positional encoding.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TileAttention(nn.Module):
    """Attention mechanism for EquiTile language model.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension.
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout probability.
    causal : bool
        Use causal (masked) attention.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (batch, seq_len, embed_dim).
        attention_mask : torch.Tensor, optional
            Attention mask.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
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

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if self.causal:
            mask = make_causal_mask(seq_len, x.device)
            scores = scores.masked_fill(mask, float("-inf"))

        if attention_mask is not None:
            scores = scores + attention_mask

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


class TileFeedForward(nn.Module):
    """Feedforward layer for EquiTile language model.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension.
    hidden_dim : int
        Hidden dimension.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
