import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .utils import spectral_linear

# =============================================================================
# TransformerEqProp - Attention with Equilibrium Dynamics
# =============================================================================

class EqPropAttention(nn.Module):
    """Self-attention that participates in equilibrium dynamics."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, use_sn: bool = True) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.W_q = spectral_linear(hidden_dim, hidden_dim, use_sn=use_sn)
        self.W_k = spectral_linear(hidden_dim, hidden_dim, use_sn=use_sn)
        self.W_v = spectral_linear(hidden_dim, hidden_dim, use_sn=use_sn)
        self.W_o = spectral_linear(hidden_dim, hidden_dim, use_sn=use_sn)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = h.shape

        Q, K, V = self._compute_qkv(h, batch_size, seq_len)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        return self.W_o(self._reshape_output(out, batch_size, seq_len))

    def _compute_qkv(self, h: torch.Tensor, batch_size: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Query, Key, and Value tensors."""
        Q = self.W_q(h).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(h).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(h).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        return Q, K, V

    def _reshape_output(self, out: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        """Reshape attention output back to the original format."""
        return out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)


class TransformerEqProp(nn.Module):
    """
    Transformer with equilibrium dynamics.

    All layers (attention + FFN) iterate together to a joint equilibrium.
    Spectral normalization ensures stable convergence.

    Example:
        >>> model = TransformerEqProp(vocab_size=1000, hidden_dim=256, output_dim=10)
        >>> x = torch.randint(0, 1000, (32, 64))  # [batch, seq_len]
        >>> output = model(x, steps=20)  # [32, 10]
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        max_seq_len: int = 128,
        alpha: float = 0.5,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.alpha = alpha

        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)

        self.attentions = nn.ModuleList([
            EqPropAttention(hidden_dim, num_heads, use_sn=use_spectral_norm)
            for _ in range(num_layers)
        ])

        self.ffns = nn.ModuleList([
            nn.Sequential(
                spectral_linear(hidden_dim, hidden_dim * 2, use_sn=use_spectral_norm),
                nn.ReLU(),
                spectral_linear(hidden_dim * 2, hidden_dim, use_sn=use_spectral_norm)
            ) for _ in range(num_layers)
        ])

        self.norms1 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        self.head = nn.Linear(hidden_dim, output_dim)

    def forward_step(self, h: torch.Tensor, x_emb: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Single equilibrium iteration step for one layer.

        Args:
            h: Current hidden state
            x_emb: Embedded input
            layer_idx: Index of the current layer

        Returns:
            Next hidden state
        """
        h_norm = self.norms1[layer_idx](h)
        h = h + self.attentions[layer_idx](h_norm)

        h_norm = self.norms2[layer_idx](h)
        ffn_out = self.ffns[layer_idx](h_norm)

        h_target = h + ffn_out + x_emb
        # Use torch.lerp for more efficient interpolation
        return torch.lerp(h, torch.tanh(h_target), self.alpha)

    def forward(self, x: torch.Tensor, steps: int = 20) -> torch.Tensor:
        """
        Forward pass: iterate all layers to joint equilibrium.

        Args:
            x: Input tensor [batch, seq_len]
            steps: Number of equilibrium steps

        Returns:
            Output logits [batch, output_dim]
        """
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x_emb = self.token_emb(x) + self.pos_emb(positions)

        h = torch.zeros_like(x_emb)

        for _ in range(steps):
            for i in range(self.num_layers):
                h = self.forward_step(h, x_emb, i)

        return self.head(h.mean(dim=1))
