"""Equilibrium Propagation model variants."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from bioplausible.acceleration.triton_kernels import TritonEqPropOps

from ...utils import spectral_linear


class CausalEqPropAttention(nn.Module):
    """Self-attention with causal masking for autoregressive generation."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, use_sn: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.W_q = spectral_linear(hidden_dim, hidden_dim, use_sn=use_sn)
        self.W_k = spectral_linear(hidden_dim, hidden_dim, use_sn=use_sn)
        self.W_v = spectral_linear(hidden_dim, hidden_dim, use_sn=use_sn)
        self.W_o = spectral_linear(hidden_dim, hidden_dim, use_sn=use_sn)

    def forward(
        self, h: torch.Tensor, causal_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = h.shape

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

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if causal_mask is not None:
            scores = scores.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        return self.W_o(
            out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        )


class CausalTransformerEqProp(nn.Module):
    """
    TransformerEqProp with causal masking for language modeling.

    Key differences from classification TransformerEqProp:
    - Causal attention mask
    - LM head instead of classification head
    - Outputs logits for full sequence (not just final token)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        max_seq_len: int = 512,
        eq_steps: int = 20,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eq_steps = eq_steps
        self.alpha = alpha
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)

        self.attentions = nn.ModuleList([
            CausalEqPropAttention(hidden_dim, num_heads) for _ in range(num_layers)
        ])

        self.ffns = nn.ModuleList([
            nn.Sequential(
                spectral_linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                spectral_linear(hidden_dim * 2, hidden_dim),
            )
            for _ in range(num_layers)
        ])

        self.norms1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.norms2 = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        self.lm_head = nn.Linear(hidden_dim, vocab_size)

        self.register_buffer("causal_mask", None)
        self._create_causal_mask(max_seq_len)

    def _create_causal_mask(self, seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor, steps: int | None = None) -> torch.Tensor:
        steps = steps or self.eq_steps
        batch_size, seq_len = x.shape

        if x.dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
            x = x.long()

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x_emb = self.token_emb(x) + self.pos_emb(positions)

        causal_mask = (
            self.causal_mask[:seq_len, :seq_len]
            if self.causal_mask is not None
            else None
        )

        h = torch.zeros_like(x_emb)

        for _ in range(steps):
            for i in range(self.num_layers):
                h_norm = self.norms1[i](h)
                h = h + self.attentions[i](h_norm, causal_mask=causal_mask)

                h_norm = self.norms2[i](h)
                ffn_out = self.ffns[i](h_norm)

                h_target = h + ffn_out + x_emb

                if TritonEqPropOps.is_available() and h.is_cuda:
                    h = TritonEqPropOps.step(h, h_target, alpha=self.alpha)
                else:
                    h = torch.lerp(h, torch.tanh(h_target), self.alpha)

        logits = self.lm_head(h)

        return logits

    def generate(
        self, prompt: torch.Tensor, max_new_tokens: int = 100, temperature: float = 1.0
    ):
        self.eval()
        generated = prompt.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self(generated)
                next_token_logits = logits[:, -1, :] / temperature

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                generated = torch.cat([generated, next_token], dim=1)

                if generated.size(1) >= self.max_seq_len:
                    break

        return generated
