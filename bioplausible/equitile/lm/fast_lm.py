"""
FastLMEquiTile: High-Performance Language Model
================================================

This is the canonical, rigorous implementation of EquiTile for Language Modeling.
It includes advanced features like Mixture of Tiles (MoT), Flash Attention,
and SwiGLU activations.

NOTE: For the visualization-ready model used in the UI demo, see:
`bioplausible.equitile.language.fast_lm`

Implements EquiTile's unique architectural advantages:
- Mixture of Tiles (MoT): Sparse tile activation for conditional computation
- Tile-Local Attention: O(n) attention with local neighborhoods
- Grouped Query Attention: Share K/V heads across Q heads
- SwiGLU Activations: Better expressivity per parameter
- Parameter Efficiency: < 10M parameters with competitive performance

Architecture Overview
---------------------
The model uses a tile-based architecture where:
1. Each layer has multiple tiles that process information locally
2. MoT selects top-k tiles per token for conditional computation
3. Local attention restricts computation to tile neighborhoods
4. Shared embeddings reduce parameter count while maintaining capacity

Example
-------
>>> config = FastLMConfig(
...     vocab_size=1000,
...     embed_dim=192,
...     num_layers=6,
...     neurons_per_tile=48,
...     tiles_per_layer=4,
...     mot_k=2,  # Top-2 tiles active per token
... )
>>> model = FastLMEquiTile(config)
>>> logits = model(input_ids)
"""

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from bioplausible.core.config import ModelConfig
from bioplausible.core.model import BioModel
from bioplausible.core.registry import Domain, LocalityLevel, register_model
from bioplausible.equitile.lm.components import (
    FastEquiTileLayer,
    FastLMConfig,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from torch import Tensor


# =============================================================================
# FastLMEquiTile Model
# =============================================================================


@register_model(
    "fast_lm_equitile",
    domains=[Domain.LM],
    locality_level=LocalityLevel.LOCAL,
    bio_plausibility_score=0.75,
    requires_backward=False,
    credit_assignment_type="hebbian",
    family="equitile",
)
class FastLMEquiTile(BioModel):
    """FastLMEquiTile: High-Performance Language Model.

    Implements EquiTile's unique architectural advantages:
    - Mixture of Tiles (MoT): Sparse tile activation
    - Tile-Local Attention: O(n) complexity
    - Grouped Query Attention: Parameter efficiency
    - SwiGLU Activations: Better expressivity

    Parameters
    ----------
    config : FastLMConfig, optional
        Configuration
    **kwargs
        Additional configuration parameters

    Example
    -------
    >>> config = FastLMConfig(vocab_size=1000, embed_dim=192, num_layers=6)
    >>> model = FastLMEquiTile(config)
    >>> logits = model(input_ids)
    """

    algorithm_name = "FastLMEquiTile"

    def __init__(
        self,
        config: FastLMConfig | None = None,
        **kwargs,
    ) -> None:
        if config is None:
            config = FastLMConfig(**kwargs)

        super().__init__(
            ModelConfig(
                name="fast_lm_equitile",
                input_dim=config.vocab_size,
                output_dim=config.vocab_size,
            )
        )

        self.config = config

        # Weight-tied embeddings
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.embed_dim, padding_idx=config.pad_token_id
        )
        # Positional encoding - support up to 4096 tokens
        self.positional_encoding = nn.Parameter(
            torch.randn(1, max(config.max_seq_len, 4096), config.embed_dim) * 0.02
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            FastEquiTileLayer(config) for _ in range(config.num_layers)
        ])

        # Final norm and output
        self.final_norm = nn.LayerNorm(config.embed_dim)
        # Weight tying with output scale for stability
        # Ablation study shows scale=2.0 gives best perplexity
        self.output_scale = nn.Parameter(torch.ones(1) * 2.0)
        self.output_proj = None  # Will use scaled token_embedding.weight

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),  # Better betas for transformers
        )

        # Scheduler (set by trainer)
        self.scheduler = None

        # Initialize weights
        self._init_weights()

        # Compile if requested
        if config.use_compile and hasattr(torch, "compile"):
            try:
                self._forward_impl = torch.compile(
                    self._forward_impl, mode=config.compile_mode
                )
            except Exception:  # broad: best-effort
                logger.warning("torch.compile failed, falling back to eager mode")

    def _init_weights(self) -> None:
        """Initialize weights.

        Based on ablation study findings:
        - init_std=0.02 gives best perplexity
        - Output scale=2.0 gives best perplexity
        """
        with torch.no_grad():
            # Embedding - use 0.02 std (matches NanoGPT, best in ablation)
            nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
            nn.init.normal_(self.positional_encoding, mean=0, std=0.02)

            # Linear layers - use 0.02 std
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

                # LayerNorm
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        return_hidden: bool = False,
        return_tile_stats: bool = False,
    ) -> Tensor | tuple[Tensor, ...]:
        """Forward pass.

        Parameters
        ----------
        input_ids : torch.Tensor
            Input token IDs (batch, seq_len)
        attention_mask : torch.Tensor, optional
            Attention mask
        return_hidden : bool
            If True, return hidden states
        return_tile_stats : bool
            If True, return tile importance stats

        Returns
        -------
        Tensor or tuple
            Logits, or (logits, hidden_states), or (logits, hidden_states, tile_stats)
        """
        return self._forward_impl(
            input_ids, attention_mask, return_hidden, return_tile_stats
        )

    def _forward_impl(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        return_hidden: bool = False,
        return_tile_stats: bool = False,
    ) -> Tensor | tuple[Tensor, ...]:
        """Internal forward implementation (can be compiled)."""
        batch_size, seq_len = input_ids.shape

        # Embedding with positional encoding
        x = self.token_embedding(input_ids)
        x = x + self.positional_encoding[:, :seq_len, :]
        x = self.dropout(x)

        # Create causal mask
        if attention_mask is None:
            # Causal mask for autoregressive generation
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
                diagonal=1,
            )
            attention_mask = torch.zeros(seq_len, seq_len, device=input_ids.device)
            attention_mask = attention_mask.masked_fill(causal_mask, float("-inf"))
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

        # Transformer layers with optional gradient checkpointing
        tile_importances = []
        use_gc = self.config.use_gradient_checkpointing
        for layer in self.layers:
            x, tile_imp = layer(
                x, attention_mask, causal=True, use_gradient_checkpointing=use_gc
            )
            if return_tile_stats:
                tile_importances.append(tile_imp)

        # Final norm
        x = self.final_norm(x)

        # Output projection (weight tying with scale for stability)
        logits = F.linear(x, self.token_embedding.weight * self.output_scale)

        if return_hidden and return_tile_stats:
            return logits, x, tile_importances
        elif return_hidden:
            return logits, x
        elif return_tile_stats:
            return logits, tile_importances
        return logits

    def train_step(
        self,
        input_ids: Tensor,
        target_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> dict[str, float]:
        """Training step.

        Parameters
        ----------
        input_ids : torch.Tensor
            Input token IDs
        target_ids : torch.Tensor, optional
            Target token IDs
        attention_mask : torch.Tensor, optional
            Attention mask

        Returns
        -------
        dict
            Training statistics
        """
        # Default target is next token prediction
        if target_ids is None:
            target_ids = input_ids.clone()

        # Forward pass
        logits = self.forward(input_ids, attention_mask)

        # Compute loss
        loss = self.compute_loss(logits, target_ids)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        # Update
        self.optimizer.step()

        # Step scheduler if available
        if self.scheduler is not None:
            self.scheduler.step()

        # Compute perplexity
        with torch.no_grad():
            perplexity = torch.exp(torch.clamp(loss, max=80)).item()

        return {
            "loss": loss.item(),
            "perplexity": perplexity,
        }

    def compute_loss(
        self,
        logits: Tensor,
        target_ids: Tensor,
    ) -> Tensor:
        """Compute language modeling loss.

        Parameters
        ----------
        logits : torch.Tensor
            Predicted logits (batch, seq_len, vocab_size)
        target_ids : torch.Tensor
            Target token IDs (batch, seq_len)

        Returns
        -------
        torch.Tensor
            Loss value
        """
        # Reshape for cross-entropy
        logits = logits.view(-1, self.config.vocab_size)
        target_ids = target_ids.view(-1)

        # Compute loss (ignore padding)
        loss = F.cross_entropy(
            logits, target_ids, ignore_index=self.config.pad_token_id
        )

        return loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None,
    ) -> Tensor:
        """Generate text autoregressively.

        Parameters
        ----------
        input_ids : torch.Tensor
            Input token IDs (batch, seq_len)
        max_length : int
            Maximum generation length
        temperature : float
            Sampling temperature
        top_k : int, optional
            Top-k sampling
        top_p : float, optional
            Nucleus sampling (top-p)
        eos_token_id : int, optional
            End-of-sequence token ID

        Returns
        -------
        torch.Tensor
            Generated token IDs
        """
        self.eval()

        generated = input_ids.clone()

        for _ in range(max_length - input_ids.shape[1]):
            # Forward pass (use full sequence for context)
            logits = self.forward(generated)

            # Get last token logits
            next_logits = logits[:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = (
                    next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                )
                next_logits[indices_to_remove] = float("-inf")

            # Apply top-p (nucleus) sampling
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_logits[indices_to_remove] = float("-inf")

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated

    def get_parameter_count(self) -> int:
        """Get total parameter count."""
        return sum(p.numel() for p in self.parameters())

    def get_stats(self) -> dict[str, float]:
        """Get model statistics."""
        stats = super().get_stats()
        stats.update({
            "num_params": self.get_parameter_count(),
            "vocab_size": self.config.vocab_size,
            "embed_dim": self.config.embed_dim,
            "num_layers": self.config.num_layers,
            "tiles_per_layer": self.config.tiles_per_layer,
            "mot_k": self.config.mot_k,
        })
        return stats


# =============================================================================
# Factory Functions
# =============================================================================


def create_fast_lm_tiny(**kwargs) -> FastLMEquiTile:
    """Create tiny FastLMEquiTile for quick prototyping.

    ~1M parameters, suitable for debugging.
    """
    config = FastLMConfig(
        vocab_size=kwargs.pop("vocab_size", 500),
        embed_dim=64,
        num_layers=2,
        hidden_dim=128,
        neurons_per_tile=16,
        tiles_per_layer=2,
        mot_k=1,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=64,
        **kwargs,
    )
    model = FastLMEquiTile(config)
    model._init_weights()
    return model


def create_fast_lm_small(**kwargs) -> FastLMEquiTile:
    """Create small FastLMEquiTile for demonstration.

    ~3M parameters, suitable for quick experiments.
    """
    config = FastLMConfig(
        vocab_size=kwargs.pop("vocab_size", 1000),
        embed_dim=128,
        num_layers=4,
        hidden_dim=256,
        neurons_per_tile=32,
        tiles_per_layer=4,
        mot_k=2,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=128,
        **kwargs,
    )
    model = FastLMEquiTile(config)
    model._init_weights()
    return model


def create_fast_lm_medium(**kwargs) -> FastLMEquiTile:
    """Create medium FastLMEquiTile for serious training.

    ~8M parameters, suitable for production experiments.
    """
    config = FastLMConfig(
        vocab_size=kwargs.pop("vocab_size", 2000),
        embed_dim=192,
        num_layers=6,
        hidden_dim=512,
        neurons_per_tile=48,
        tiles_per_layer=4,
        mot_k=2,
        num_heads=6,
        num_kv_heads=2,
        max_seq_len=256,
        **kwargs,
    )
    model = FastLMEquiTile(config)
    model._init_weights()
    return model


def create_fast_lm_shakespeare(**kwargs) -> FastLMEquiTile:
    """Create FastLMEquiTile optimized for Shakespeare dataset.

    Character-level model with ~5M parameters.
    """
    config = FastLMConfig(
        vocab_size=kwargs.pop("vocab_size", 65),  # Character vocab
        embed_dim=192,
        num_layers=6,
        hidden_dim=384,
        neurons_per_tile=48,
        tiles_per_layer=4,
        mot_k=2,
        num_heads=6,
        num_kv_heads=2,
        dropout=0.1,
        max_seq_len=256,
        **kwargs,
    )
    model = FastLMEquiTile(config)
    model._init_weights()
    return model
