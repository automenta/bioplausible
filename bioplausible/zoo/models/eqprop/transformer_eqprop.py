"""Equilibrium Propagation model variants."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from bioplausible.acceleration.triton_kernels import TritonEqPropOps

from ....acceleration import compile_settling_loop
from ...utils import spectral_linear
from ..base import EqPropModel


class EqPropAttention(nn.Module):
    """Self-attention that participates in equilibrium dynamics."""

    def __init__(
        self, hidden_dim: int, num_heads: int = 4, use_sn: bool = True
    ) -> None:
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

    def _compute_qkv(
        self, h: torch.Tensor, batch_size: int, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Q = (
            self
            .W_q(h)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self
            .W_k(h)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self
            .W_v(h)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        return Q, K, V

    def _reshape_output(
        self, out: torch.Tensor, batch_size: int, seq_len: int
    ) -> torch.Tensor:
        return (
            out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        )


class TransformerEqProp(EqPropModel):
    """
    Transformer with equilibrium dynamics.

    All layers (attention + FFN) iterate together to a joint equilibrium.
    Spectral normalization ensures stable convergence.

    Example:
        >>> model = TransformerEqProp(vocab_size=1000, hidden_dim=256, output_dim=10)
        >>> x = torch.randint(0, 1000, (32, 64))
        >>> output = model(x, steps=20)
    """

    def __init__(
        self,
        vocab_size: int = None,
        hidden_dim: int = 256,
        output_dim: int = 27,
        num_layers: int = 2,
        num_heads: int = 4,
        max_seq_len: int = 128,
        alpha: float = 0.5,
        use_spectral_norm: bool = True,
        max_steps: int = 20,
        gradient_method: str = "bptt",
        input_dim: int = None,
    ) -> None:
        if vocab_size is None and output_dim is not None:
            vocab_size = output_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.alpha = alpha
        self.use_spectral_norm = use_spectral_norm

        super().__init__(
            input_dim=0,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            max_steps=max_steps,
            use_spectral_norm=use_spectral_norm,
            gradient_method=gradient_method,
        )

    def _build_layers(self):
        self.token_emb = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.pos_emb = nn.Embedding(self.max_seq_len, self.hidden_dim)

        self.attentions = nn.ModuleList([
            EqPropAttention(
                self.hidden_dim, self.num_heads, use_sn=self.use_spectral_norm
            )
            for _ in range(self.num_layers)
        ])

        self.ffns = nn.ModuleList([
            nn.Sequential(
                spectral_linear(
                    self.hidden_dim,
                    self.hidden_dim * 2,
                    use_sn=self.use_spectral_norm,
                ),
                nn.ReLU(),
                spectral_linear(
                    self.hidden_dim * 2,
                    self.hidden_dim,
                    use_sn=self.use_spectral_norm,
                ),
            )
            for _ in range(self.num_layers)
        ])

        self.norms1 = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)
        ])
        self.norms2 = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)
        ])

        self.head = nn.Linear(self.hidden_dim, self.output_dim)

    def _initialize_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        return torch.zeros(
            batch_size, seq_len, self.hidden_dim, device=x.device, dtype=torch.float
        )

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x_emb = self.token_emb(x) + self.pos_emb(positions)
        return x_emb

    def _forward_step_impl(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> torch.Tensor:
        current_h = h
        for i in range(self.num_layers):
            current_h = self._forward_layer(current_h, x_transformed, i)
        return current_h

    @compile_settling_loop
    def forward_step(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> torch.Tensor:
        return self._forward_step_impl(h, x_transformed)

    def _forward_layer(
        self, h: torch.Tensor, x_emb: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        h_norm = self.norms1[layer_idx](h)
        h = h + self.attentions[layer_idx](h_norm)

        h_norm = self.norms2[layer_idx](h)
        ffn_out = self.ffns[layer_idx](h_norm)

        h_target = h + ffn_out + x_emb

        if TritonEqPropOps.is_available() and h.is_cuda:
            return TritonEqPropOps.step(h, h_target, alpha=self.alpha)

        return torch.lerp(h, torch.tanh(h_target), self.alpha)

    def _output_projection(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h.mean(dim=1))
