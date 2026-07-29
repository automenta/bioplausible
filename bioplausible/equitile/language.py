"""
EquiTile Language: EquiTile for Language Modeling
==================================================

Extends EquiTile with language modeling capabilities:
- LMEquiTile: Transformer-style EquiTile for sequences
- Embedding layers for token processing
- Causal attention mechanisms
- Support for character/word/subword tokenization

Examples
--------
>>> from bioplausible.equitile.language import LMEquiTile, LMEquiTileConfig
>>> config = LMEquiTileConfig(
...     vocab_size=50257,
...     embed_dim=256,
...     num_heads=4,
...     num_layers=4,
...     max_seq_len=128,
... )
>>> model = LMEquiTile(config)
>>> logits = model(input_ids)
>>> loss = model.compute_loss(logits, target_ids)
"""

import math
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

from bioplausible.core.registry import Domain, LocalityLevel
from bioplausible.equitile.config import EquiTileConfig, LMEquiTileConfig
from bioplausible.equitile.core import EquiTile
from bioplausible.zoo.base import BioModel, ModelConfig, register_model

if TYPE_CHECKING:
    from torch import Tensor


# =============================================================================
# Configuration
# =============================================================================


# =============================================================================
# Positional Encoding
# =============================================================================


class PositionalEncoding(nn.Module):
    """Positional encoding for sequences.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension
    max_len : int
        Maximum sequence length
    dropout : float
        Dropout probability
    """

    def __init__(
        self,
        embed_dim: int,
        max_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create positional encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )

        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (batch, seq_len, embed_dim)

        Returns
        -------
        torch.Tensor
            Output with positional encoding
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# =============================================================================
# Tile Attention
# =============================================================================


class TileAttention(nn.Module):
    """Attention mechanism for EquiTile language model.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension
    num_heads : int
        Number of attention heads
    dropout : float
        Dropout probability
    causal : bool
        Use causal (masked) attention
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
        x: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (batch, seq_len, embed_dim)
        attention_mask : torch.Tensor, optional
            Attention mask

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
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

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply causal mask
        if self.causal:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)
            scores = scores.masked_fill(mask.bool(), float("-inf"))

        # Apply attention mask
        if attention_mask is not None:
            scores = scores + attention_mask

        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = (
            attn_output
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )
        return self.out_proj(attn_output)


# =============================================================================
# Tile FeedForward
# =============================================================================


class TileFeedForward(nn.Module):
    """Feedforward layer for EquiTile language model.

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
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# =============================================================================
# EquiTile Transformer Layer
# =============================================================================


class EquiTileTransformerLayer(nn.Module):
    """Transformer layer with EquiTile integration.

    Parameters
    ----------
    config : LMEquiTileConfig
        Configuration
    """

    def __init__(self, config: LMEquiTileConfig) -> None:
        super().__init__()
        self.config = config

        # Attention
        self.attention = TileAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            causal=True,
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)

        # Feedforward
        self.feedforward = TileFeedForward(
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )

        # EquiTile integration (replaces custom tile logic)
        # Use a minimal config for the layer-wise EquiTile
        # We override num_layers to 2 (Input -> Output) for this block
        layer_equitile_kwargs = config.equitile_kwargs.copy()
        layer_equitile_kwargs.update({
            "neurons_per_tile": config.neurons_per_tile,
            "num_layers": 2,
            "tiles_per_layer": config.tiles_per_layer,
            "learning_rate": config.learning_rate,
            "dropout": config.dropout,
            "weight_decay": config.weight_decay,
            "mode": config.mode,
            "inference_steps": config.inference_steps,
            "step_size": config.step_size,
            "beta": config.beta,
            "activation": config.activation,
        })

        equitile_config = EquiTileConfig(**layer_equitile_kwargs)

        self.equitile = EquiTile(
            config=equitile_config,
            input_dim=config.embed_dim,
            output_dim=config.embed_dim,
        )

    def _process_tiles(self, x: Tensor) -> Tensor:
        """Process input through EquiTile block, handling sequence dimension."""
        b, s, d = x.shape
        # Flatten sequence dimension: (batch, seq, dim) -> (batch * seq, dim)
        x_flat = x.view(b * s, d)

        # Pass through EquiTile
        tile_out = self.equitile(x_flat)

        # Reshape back
        return tile_out.view(b, s, d)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        attention_mask : torch.Tensor, optional
            Attention mask

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        # Self-attention with residual
        attn_output = self.attention(self.norm1(x), attention_mask)
        x = x + attn_output

        # EquiTile processing (replaces feedforward-like tile block)
        tile_out = self._process_tiles(x)

        # Residual connection
        x = x + tile_out

        # Feedforward with residual
        ff_output = self.feedforward(self.norm2(x))
        x = x + ff_output

        return x


# =============================================================================
# Language Model EquiTile
# =============================================================================


@register_model(
    "lm_equitile",
    domains=[Domain.LM],
    locality_level=LocalityLevel.LOCAL,
    bio_plausibility_score=0.78,
    requires_backward=False,
    credit_assignment_type="hebbian",
    family="equitile",
)
class LMEquiTile(BioModel):
    """EquiTile for Language Modeling.

    Combines transformer-style attention with EquiTile's
    tile-based local learning for sequence modeling.

    Parameters
    ----------
    config : LMEquiTileConfig, optional
        Configuration
    **kwargs
        Additional configuration parameters

    Examples
    --------
    >>> config = LMEquiTileConfig(
    ...     vocab_size=50257,
    ...     embed_dim=256,
    ...     num_heads=4,
    ...     num_layers=4,
    ... )
    >>> model = LMEquiTile(config)
    >>> logits = model(input_ids)
    """

    algorithm_name = "LMEquiTile"

    @classmethod
    def build(
        cls,
        spec,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers,
        device,
        task_type,
        **kwargs,
    ):
        """Build LMEquiTile from factory arguments."""
        # For LM, output_dim is vocab_size
        vocab_size = output_dim

        config_kwargs = {
            "vocab_size": vocab_size,
            "embed_dim": kwargs.get("embed_dim", hidden_dim),
            "num_layers": num_layers,
            "neurons_per_tile": kwargs.get("neurons_per_tile", 64),
            "tiles_per_layer": kwargs.get("tiles_per_layer", 4),
            "learning_rate": kwargs.get("lr", spec.default_lr),
        }

        # Pass through valid config keys
        valid_keys = LMEquiTileConfig.__annotations__.keys()
        for k, v in kwargs.items():
            if k in valid_keys:
                config_kwargs[k] = v

        # Also check spec custom_hyperparams
        for k, v in spec.custom_hyperparams.items():
            if k in valid_keys:
                config_kwargs[k] = v

        config = LMEquiTileConfig(**config_kwargs)

        model = cls(config=config)
        return model.to(device)

    def __init__(
        self,
        config: LMEquiTileConfig | None = None,
        **kwargs: Any,
    ) -> None:
        if config is None:
            config = LMEquiTileConfig(**kwargs)

        super().__init__(
            ModelConfig(
                name="lm_equitile",
                input_dim=config.vocab_size,
                output_dim=config.vocab_size,
            )
        )

        self.config = config

        # Embedding
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.embed_dim, padding_idx=config.pad_token_id
        )
        self.positional_encoding = PositionalEncoding(
            embed_dim=config.embed_dim,
            max_len=config.max_seq_len,
            dropout=config.dropout,
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            EquiTileTransformerLayer(config) for _ in range(config.num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(config.embed_dim, config.vocab_size)

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.embed_dim)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        with torch.no_grad():
            # Embedding
            nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)

            # Linear layers
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        return_hidden: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Forward pass.

        Parameters
        ----------
        input_ids : torch.Tensor
            Input token IDs (batch, seq_len)
        attention_mask : torch.Tensor, optional
            Attention mask
        return_hidden : bool
            If True, return hidden states

        Returns
        -------
        torch.Tensor or tuple
            Logits, or (logits, hidden_states)
        """
        # Embedding
        x = self.token_embedding(input_ids)
        x = self.positional_encoding(x)

        # Create attention mask
        if attention_mask is None:
            attention_mask = torch.zeros_like(input_ids, dtype=torch.float)
            attention_mask = attention_mask.masked_fill(
                input_ids == self.config.pad_token_id, float("-inf")
            )
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)

        # Final norm
        x = self.final_norm(x)

        # Output projection
        logits = self.output_proj(x)

        if return_hidden:
            return logits, x
        return logits

    def train_step(
        self,
        input_ids: Tensor,
        target_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> dict[str, float]:
        """Perform one training step.

        Parameters
        ----------
        input_ids : torch.Tensor
            Input token IDs
        target_ids : torch.Tensor, optional
            Target token IDs (defaults to input_ids shifted)
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

        # Compute perplexity
        with torch.no_grad():
            perplexity = torch.exp(torch.clamp(loss, max=80)).item()

        return {
            "loss": loss.item(),
            "perplexity": perplexity,
            "mode": self.config.mode,
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
            Target token IDs (batch, seq_len) or (batch,)

        Returns
        -------
        torch.Tensor
            Loss value
        """
        # Handle shape mismatch (e.g. single-token target from LMTask)
        if logits.dim() == 3:
            # logits: (B, S, V)
            if target_ids.dim() == 1:
                # target: (B,) - Assume last token prediction
                logits = logits[:, -1, :]
            elif target_ids.dim() == 2:
                if target_ids.size(1) != logits.size(1):
                    # target: (B, 1) - Assume last token prediction
                    logits = logits[:, -1, :]
                    target_ids = target_ids.view(-1)
                else:
                    # target: (B, S) - Full sequence
                    logits = logits.view(-1, self.config.vocab_size)
                    target_ids = target_ids.view(-1)

        # Compute loss (ignore padding)
        loss = F.cross_entropy(
            logits, target_ids, ignore_index=self.config.pad_token_id
        )

        return loss

    def generate(
        self,
        input_ids: Tensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Tensor:
        """Generate text autoregressively.

        Parameters
        ----------
        input_ids : torch.Tensor
            Input token IDs
        max_length : int
            Maximum generation length
        temperature : float
            Sampling temperature
        top_k : int, optional
            Top-k sampling

        Returns
        -------
        torch.Tensor
            Generated token IDs
        """
        self.eval()

        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                # Forward pass
                logits = self.forward(generated)

                # Get last token logits
                next_logits = logits[:, -1, :] / temperature

                # Top-k sampling
                if top_k is not None:
                    indices_to_remove = (
                        next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                    )
                    next_logits[indices_to_remove] = float("-inf")

                # Sample
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append
                generated = torch.cat([generated, next_token], dim=1)

                # Check for EOS
                if (next_token == 2).all():  # Assuming EOS token is 2
                    break

        return generated

    def get_hidden_states(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Get hidden states.

        Parameters
        ----------
        input_ids : torch.Tensor
            Input token IDs
        attention_mask : torch.Tensor, optional
            Attention mask

        Returns
        -------
        torch.Tensor
            Hidden states
        """
        _, hidden = self.forward(input_ids, attention_mask, return_hidden=True)
        return hidden


# =============================================================================
# Tokenizer Utilities
# =============================================================================


class SimpleTokenizer:
    """Simple character/word tokenizer for demonstration.

    Parameters
    ----------
    vocab : list of str, optional
        Vocabulary (tokens)
    """

    def __init__(self, vocab: list[str] | None = None) -> None:
        if vocab is None:
            self.vocab = ["<pad>", "<unk>", "<eos>"]
            self.char_to_idx = {
                c: i + 3
                for i, c in enumerate("abcdefghijklmnopqrstuvwxyz0123456789.,!?;: ")
            }
            self.vocab.extend(list(self.char_to_idx.keys()))
        else:
            self.vocab = vocab
            self.char_to_idx = {c: i for i, c in enumerate(vocab)}

        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.vocab_size = len(self.vocab)

    def encode(self, text: str, max_length: int | None = None) -> list[int]:
        """Encode text to token IDs.

        Parameters
        ----------
        text : str
            Input text
        max_length : int, optional
            Maximum length

        Returns
        -------
        list of int
            Token IDs
        """
        ids = [self.char_to_idx.get(c, 1) for c in text.lower()]  # 1 = <unk>

        if max_length:
            ids = ids[:max_length]

        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text.

        Parameters
        ----------
        ids : list of int
            Token IDs

        Returns
        -------
        str
            Decoded text
        """
        return "".join(self.idx_to_char.get(i, "?") for i in ids)

    def batch_encode(
        self,
        texts: list[str],
        max_length: int | None = None,
        padding: bool = True,
    ) -> torch.Tensor:
        """Batch encode texts.

        Parameters
        ----------
        texts : list of str
            Input texts
        max_length : int, optional
            Maximum length
        padding : bool
            Pad to max_length

        Returns
        -------
        torch.Tensor
            Token IDs (batch, seq_len)
        """
        encoded = [self.encode(t, max_length) for t in texts]

        if padding and max_length:
            for ids in encoded:
                while len(ids) < max_length:
                    ids.append(0)  # <pad>

        return torch.tensor(encoded, dtype=torch.long)


# =============================================================================
# Factory Functions
# =============================================================================


def create_lm_model(
    vocab_size: int = 50257,
    embed_dim: int = 256,
    num_heads: int = 4,
    num_layers: int = 4,
    max_seq_len: int = 128,
    **kwargs: Any,
) -> LMEquiTile:
    """Create LMEquiTile model.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size
    embed_dim : int
        Embedding dimension
    num_heads : int
        Number of attention heads
    num_layers : int
        Number of layers
    max_seq_len : int
        Maximum sequence length
    **kwargs
        Additional arguments

    Returns
    -------
    LMEquiTile
        Language model
    """
    config = LMEquiTileConfig(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        **kwargs,
    )
    return LMEquiTile(config)


def create_small_lm(
    vocab_size: int = 1000,
    **kwargs: Any,
) -> LMEquiTile:
    """Create small LMEquiTile for prototyping.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size
    **kwargs
        Additional arguments

    Returns
    -------
    LMEquiTile
        Small language model
    """
    return create_lm_model(
        vocab_size=vocab_size,
        embed_dim=128,
        num_heads=2,
        num_layers=2,
        max_seq_len=64,
        **kwargs,
    )


def create_medium_lm(
    vocab_size: int = 50257,
    **kwargs: Any,
) -> LMEquiTile:
    """Create medium LMEquiTile.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size
    **kwargs
        Additional arguments

    Returns
    -------
    LMEquiTile
        Medium language model
    """
    return create_lm_model(
        vocab_size=vocab_size,
        embed_dim=256,
        num_heads=4,
        num_layers=4,
        max_seq_len=256,
        **kwargs,
    )


def create_large_lm(
    vocab_size: int = 50257,
    **kwargs: Any,
) -> LMEquiTile:
    """Create large LMEquiTile.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size
    **kwargs
        Additional arguments

    Returns
    -------
    LMEquiTile
        Large language model
    """
    return create_lm_model(
        vocab_size=vocab_size,
        embed_dim=512,
        num_heads=8,
        num_layers=8,
        max_seq_len=512,
        **kwargs,
    )


import logging

logger = logging.getLogger(__name__)


def generate_text(
    model: torch.nn.Module,
    char_to_idx: dict[str, int],
    idx_to_char: dict[int, str],
    prompt: str = "",
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
    device: str = "cpu",
) -> str:
    """
    Universal autoregressive text generation for any model.

    Works with:
    - EqProp LM variants (have built-in generate)
    - Research algorithms (need this wrapper)
    - Any model that outputs logits

    Args:
        model: Any torch.nn.Module that outputs logits
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
        prompt: Starting text
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: If set, only sample from top k tokens
        device: Device to run on

    Returns:
        Generated text string
    """
    model.eval()

    # Convert prompt to indices
    if prompt:
        indices = [char_to_idx.get(c, 0) for c in prompt]
    else:
        indices = [0]  # Start with first char

    generated = list(indices)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Prepare input
            if hasattr(model, "config"):
                # Research algorithm - needs one-hot encoding
                vocab_size = len(char_to_idx)
                x = torch.tensor([indices[-1]], device=device)
                x = F.one_hot(x, num_classes=vocab_size).float()
            else:
                # LM variant or standard model
                x = torch.tensor([indices], device=device)

            # Forward pass
            try:
                logits = model(x)

                # Handle different output shapes
                if logits.dim() == 3:
                    # [batch, seq, vocab] - take last token
                    logits = logits[0, -1, :]
                elif logits.dim() == 2:
                    # [batch, vocab] or [seq, vocab]
                    if logits.shape[0] == 1:
                        logits = logits[0]
                    else:
                        logits = logits[-1]

            except Exception:
                logger.warning("Complex dispatch fallback triggered")
                # Fallback: try with different input format
                try:
                    x = torch.tensor([[indices[-1]]], device=device)
                    logits = model(x)
                    if logits.dim() == 3:
                        logits = logits[0, -1, :]
                    elif logits.dim() == 2:
                        logits = logits[0]
                except RuntimeError, ValueError, IndexError:
                    break

            # Apply temperature
            logits = logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[-1]] = -float("Inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).item()

            # Add to generated sequence
            generated.append(next_idx)
            indices.append(next_idx)

    # Decode to text
    text = "".join([idx_to_char.get(idx, "?") for idx in generated])
    return text


def generate_from_dataset(
    model: torch.nn.Module,
    dataset: Any,
    prompt: str = "",
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> str:
    """
    Convenience wrapper that extracts vocab from dataset.

    Args:
        model: Model to generate with
        dataset: CharDataset with vocab_size, char_to_idx, idx_to_char
        prompt: Starting text
        max_new_tokens: Tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering

    Returns:
        Generated text string
    """
    # Extract vocab from dataset
    char_to_idx = dataset.char_to_idx
    idx_to_char = dataset.idx_to_char

    # Determine device
    device = next(model.parameters()).device if list(model.parameters()) else "cpu"

    return generate_text(
        model=model,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        device=device,
    )
