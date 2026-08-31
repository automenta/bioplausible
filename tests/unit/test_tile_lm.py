"""TileLM smoke — construct, forward, train_step, generate."""

from __future__ import annotations

import torch

from computronium.models.tile_lm import TileLM
from tests.conftest import lm_train_step


def _tiny() -> TileLM:
    return TileLM.from_lm(
        vocab_size=100,
        embed_dim=32,
        num_layers=1,
        neurons_per_tile=16,
        tiles_per_layer=2,
        max_seq_len=16,
    )


def test_tile_lm_forward_shape() -> None:
    torch.manual_seed(0)
    model = _tiny()
    ids = torch.randint(0, 100, (2, 8))
    logits = model.forward(ids)
    assert logits.shape == (2, 8, 100)


def test_tile_lm_train_step() -> None:
    torch.manual_seed(0)
    model = _tiny()
    ids = torch.randint(0, 100, (2, 8))
    target = torch.randint(0, 100, (2, 8))
    stats = lm_train_step(model, ids, target)
    assert "loss" in stats and "perplexity" in stats
    assert stats["loss"] > 0.0


def test_tile_lm_generate() -> None:
    torch.manual_seed(0)
    model = _tiny().eval()
    prompt = torch.randint(0, 100, (2, 2))
    out = model.generate(prompt, max_length=6, top_k=5, top_p=0.9, temperature=0.8)
    assert out.shape == (2, 6)


def test_tile_lm_substrate_logits_unchanged() -> None:
    torch.manual_seed(0)
    model = _tiny()
    flat = torch.randn(3, 32)
    out = model.forward_logits(flat)
    assert out.shape == (3, 32)


def test_tile_lm_build_classmethod() -> None:
    model = TileLM.build(
        spec=None,
        input_dim=32,
        output_dim=100,
        hidden_dim=64,
        num_layers=1,
        vocab_size=100,
        embed_dim=32,
        device="cpu",
    )
    ids = torch.randint(0, 100, (1, 4))
    assert model.forward(ids).shape == (1, 4, 100)
