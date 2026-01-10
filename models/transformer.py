"""
Transformer EqProp (Track 14)

Integrates self-attention into equilibrium dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils.parametrizations import spectral_norm
from .utils import spectral_linear

class EqPropAttention(nn.Module):
    """Self-attention that iterates to equilibrium."""
    
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
        
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = h.shape
        Q = self.W_q(h).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(h).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(h).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        
        return self.W_o(out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim))

class TransformerEqProp(nn.Module):
    """Transformer with equilibrium dynamics."""
    
    def __init__(self, vocab_size: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, num_heads: int = 4,
                 max_seq_len: int = 128, alpha: float = 0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.alpha = alpha
        
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        
        self.attentions = nn.ModuleList([
            EqPropAttention(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.ffns = nn.ModuleList([
             nn.Sequential(
                spectral_linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                spectral_linear(hidden_dim * 2, hidden_dim)
            ) for _ in range(num_layers)
        ])
        
        self.norms1 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
        self.head = nn.Linear(hidden_dim, output_dim)
        
    def forward_step(self, h, x_emb, layer_idx):
        h_norm = self.norms1[layer_idx](h)
        h = h + self.attentions[layer_idx](h_norm)
        
        h_norm = self.norms2[layer_idx](h)
        ffn_out = self.ffns[layer_idx](h_norm)
        
        h_target = h + ffn_out + x_emb
        return (1 - self.alpha) * h + self.alpha * torch.tanh(h_target)
        
    def forward(self, x: torch.Tensor, steps: int = 20) -> torch.Tensor:
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x_emb = self.token_emb(x) + self.pos_emb(positions)
        
        h = torch.zeros_like(x_emb)
        
        for _ in range(steps):
            for i in range(self.num_layers):
                h = self.forward_step(h, x_emb, i)
                
        return self.head(h.mean(dim=1))
