"""TileLM — Language Modeling on the tile substrate.

Thin model class that configures the generic :class:`~bioplausible.core.local_learning.
TileAlgorithm` substrate for token-level language modeling. The substrate acts
as a per-position processor: each token position flows independently through
the same tile graph (shared weights across positions), giving a bio-plausible
backbone whose parameters are shared across the sequence — the standard
weight-sharing inductive bias of Transformers without the global attention.

The first version uses autograd BPTT over the substrate parameters (``mode=
"backprop"``); a bio-plausible local-update variant can be layered on later by
swapping in a contrastive ``train_step``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from bioplausible.core.local_learning import TileAlgorithm, TileAlgorithmConfig
from bioplausible.core.model_status import status_tag
from bioplausible.core.registry import Domain, LocalityLevel, register_model
from bioplausible.core.utils.optimizer import OptimizerConfig, create_optimizer
from bioplausible.utils import count_parameters

__all__ = [
    "TileLM",
    "TileLMExtras",
]


# ──────────────────────────────────────────────────────────────────────────────
# Extras accessor — LM knobs live on the substrate ``config.extra`` dict
# ──────────────────────────────────────────────────────────────────────────────


class TileLMExtras:
    """LM-specific knobs stored on ``config.extra``."""

    __slots__ = (
        "embed_dropout",
        "max_seq_len",
        "output_scale",
        "pad_token_id",
        "vocab_size",
    )

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        pad_token_id: int,
        embed_dropout: float,
        output_scale: float,
    ) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.embed_dropout = embed_dropout
        self.output_scale = output_scale

    @staticmethod
    def from_config(config: TileAlgorithmConfig) -> TileLMExtras:
        extra = config.extra
        return TileLMExtras(
            vocab_size=int(extra.get("vocab_size", 1000)),
            max_seq_len=int(extra.get("max_seq_len", 256)),
            pad_token_id=int(extra.get("pad_token_id", 0)),
            embed_dropout=float(extra.get("embed_dropout", 0.1)),
            output_scale=float(extra.get("output_scale", 2.0)),
        )


# ──────────────────────────────────────────────────────────────────────────────
# TileLM — Language Model on tiles
# ──────────────────────────────────────────────────────────────────────────────


@register_model(
    "tile_lm",
    domains=[Domain.LM],
    family="tile",
    locality_level=LocalityLevel.LOCAL,
    bio_plausibility_score=0.5,
    requires_backward=True,
    credit_assignment_type="gradient",
    tags=["language", "lm", "tile", status_tag("experimental")],
)
class TileLM(TileAlgorithm):
    """Language Model on the tile substrate.

    The :class:`TileAlgorithm` backbone is run in ``mode="backprop"`` (autograd
    BPTT) and configured with ``input_dim = output_dim = embed_dim`` so each
    token embedding is reshaped to ``(batch * seq_len, embed_dim)``, flowed
    through the shared tile graph, and projected back to the vocabulary via a
    weight-tied output head.

    Parameters
    ----------
    config : TileLMConfig
        Substrate config carrying the LM extras (``vocab_size``,
        ``max_seq_len``, ``pad_token_id``, ``embed_dropout``,
        ``output_scale``) in ``config.extra``.

    Examples
    --------
    >>> model = TileLM.from_lm(vocab_size=500, embed_dim=64, num_layers=2)
    >>> logits = model(input_ids)              # (batch, seq_len, vocab)
    >>> stats = model.train_step(input_ids, target_ids)
    """

    _algorithm = "lm"

    def __init__(self, config: TileAlgorithmConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        extra = TileLMExtras.from_config(config)
        self.lm_extra = extra

        self.token_embedding = nn.Embedding(
            extra.vocab_size,
            config.input_dim,
            padding_idx=extra.pad_token_id,
        )
        self.positional_encoding = nn.Parameter(
            torch.randn(1, max(extra.max_seq_len, 4096), config.input_dim) * 0.02
        )
        self.embed_dropout = nn.Dropout(extra.embed_dropout)
        self.output_scale = nn.Parameter(
            torch.ones(1) * extra.output_scale, requires_grad=True
        )
        self._init_lm_weights()

        # Replace the substrate optimizer with an AdamW that covers everything
        # (substrate uses Adam by default; transformers want AdamW + weight_decay).
        self._optim_io = create_optimizer(
            list(self.W_in.parameters())
            + list(self.W_out.parameters())
            + list(self._tile_weights.values())
            + list(self._tile_biases.values())
            + list(self.token_embedding.parameters())
            + [self.positional_encoding]
            + [self.output_scale],
            OptimizerConfig(
                name="adamw",
                lr=config.learning_rate,
                weight_decay=0.1,
                betas=(0.9, 0.95),
            ),
        )

    # ── Init ─────────────────────────────────────────────────────────────────

    def _init_lm_weights(self) -> None:
        with torch.no_grad():
            nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.positional_encoding, mean=0.0, std=0.02)
            nn.init.constant_(
                self.token_embedding.weight[self.lm_extra.pad_token_id], 0.0
            )

    # ── LM forward ──────────────────────────────────────────────────────────

    def forward(self, input_ids: Tensor) -> Tensor:
        """Token IDs → logits over vocab.

        Parameters
        ----------
        input_ids : Tensor
            Integer token IDs of shape ``(batch, seq_len)``.
        """
        return self._forward_logits(input_ids)

    def _embed_tokens(self, input_ids: Tensor) -> Tensor:
        _, seq_len = input_ids.shape
        x = self.token_embedding(input_ids)
        x = x + self.positional_encoding[:, :seq_len, :]
        return self.embed_dropout(x)

    def _substrate_forward(self, hidden: Tensor) -> Tensor:
        # Run each position through the shared substrate backbone.
        # `detach_input=False` so grads flow into the token embeddings when
        # callers chain a feature extractor; the substrate itself still runs
        # autograd BPTT (mode="backprop") over its internal parameters.
        flat = hidden.reshape(-1, self.config.input_dim)
        out = super().forward_logits(flat, detach_input=False)
        return out.reshape(hidden.shape)

    def _forward_logits(self, input_ids: Tensor) -> Tensor:
        hidden = self._substrate_forward(self._embed_tokens(input_ids))
        # Weight-tied output projection with a learned scale for stability.
        return F.linear(hidden, self.token_embedding.weight * self.output_scale)

    def get_hidden_states(self, input_ids: Tensor) -> Tensor:
        """Substrate feature maps before the output head.

        Parameters
        ----------
        input_ids : Tensor
            Integer token IDs of shape ``(batch, seq_len)``.

        Returns
        -------
        Tensor
            Per-position features of shape ``(batch, seq_len, embed_dim)``.
        """
        return self._substrate_forward(self._embed_tokens(input_ids))

    # `forward_logits` / `forward` of the substrate operate on flat (batch,
    # input_dim) tensors; LM callers must go through TileLM.forward instead.
    def forward_logits(self, x: Tensor, *, detach_input: bool = True) -> Tensor:
        """Substrate-shaped forward (kept for head consumers).

        Accepts already-embedded ``(batch, embed_dim)`` tensors for downstream
        feature-extractor code that uses the substrate API directly.
        """
        return super().forward_logits(x, detach_input=detach_input)

    # ── Training ─────────────────────────────────────────────────────────────

    def compute_loss(self, logits: Tensor, target_ids: Tensor) -> Tensor:
        logits = logits.reshape(-1, self.lm_extra.vocab_size)
        target_ids = target_ids.reshape(-1)
        return F.cross_entropy(
            logits, target_ids, ignore_index=self.lm_extra.pad_token_id
        )

    def train_step(
        self,
        input_ids: Tensor,
        target_ids: Tensor | None = None,
    ) -> dict[str, float]:
        """Autograd training step; returns ``{"loss", "perplexity"}``."""
        self.train()
        if target_ids is None:
            target_ids = input_ids.clone()
        logits = self.forward(input_ids)
        loss = self.compute_loss(logits, target_ids)
        self._optim_io.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self._optim_io.step()
        self._step_count += 1
        with torch.no_grad():
            perplexity = torch.exp(torch.clamp(loss, max=80)).item()
        return {"loss": loss.item(), "perplexity": perplexity}

    # ── Generation ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(  # ruff: ignore[too-many-arguments]  # generation sampling contract
        self,
        input_ids: Tensor,
        max_length: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None,
    ) -> Tensor:
        """Greedy/sampled autoregressive generation."""
        self.eval()
        generated = input_ids.clone()
        if max_length <= generated.shape[1]:
            return generated
        while generated.shape[1] < max_length:
            logits = self.forward(
                generated[:, -self.lm_extra.max_seq_len :]
                if self.lm_extra.max_seq_len > 0
                else generated
            )
            next_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                kth = torch.topk(next_logits, top_k, dim=-1)[0][..., -1, None]
                next_logits = next_logits.masked_fill(next_logits < kth, float("-inf"))

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(
                    next_logits, descending=True, dim=-1
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_remove = cumulative_probs > top_p
                sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
                sorted_remove[..., 0] = False
                remove_mask = sorted_remove.scatter(-1, sorted_idx, sorted_remove)
                next_logits = next_logits.masked_fill(remove_mask, float("-inf"))

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        return generated

    # ── Factories ────────────────────────────────────────────────────────────

    def get_parameter_count(self) -> int:
        """Get total parameter count."""
        return count_parameters(self, trainable_only=False)

    @classmethod
    def from_lm(  # ruff: ignore[too-many-arguments]  # zoo build-classmethod contract
        cls,
        vocab_size: int,
        *,
        embed_dim: int = 192,
        num_layers: int = 3,
        neurons_per_tile: int = 48,
        tiles_per_layer: int = 4,
        learning_rate: float = 3e-4,
        importance_lr: float = 0.01,
        beta: float = 0.1,
        max_seq_len: int = 256,
        pad_token_id: int = 0,
        embed_dropout: float = 0.1,
        output_scale: float = 2.0,
        **kwargs,
    ) -> TileLM:
        extra = {
            "vocab_size": vocab_size,
            "max_seq_len": max_seq_len,
            "pad_token_id": pad_token_id,
            "embed_dropout": embed_dropout,
            "output_scale": output_scale,
            **kwargs,
        }
        config = TileAlgorithmConfig(
            input_dim=embed_dim,
            output_dim=embed_dim,
            neurons_per_tile=neurons_per_tile,
            tiles_per_layer=tiles_per_layer,
            num_hidden_layers=num_layers,
            algorithm="ep",
            mode="backprop",
            learning_rate=learning_rate,
            importance_lr=importance_lr,
            beta=beta,
            extra=extra,
        )
        return cls(config)

    @classmethod
    def build(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
        cls,
        spec,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        device: str = "cpu",
        task_type: str = "lm",
        **kwargs,
    ) -> TileLM:
        vocab_size = int(kwargs.pop("vocab_size", output_dim))
        embed_dim = int(kwargs.pop("embed_dim", input_dim))
        return cls.from_lm(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            learning_rate=float(kwargs.get("learning_rate", 3e-4)),
            spec=spec,
            hidden_dim=hidden_dim,
            task_type=task_type,
        ).to(device)
