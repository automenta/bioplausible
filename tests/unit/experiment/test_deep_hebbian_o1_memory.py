"""Deep Hebbian O(1)-in-depth memory fix (EXPERIMENT_PLAN5 §1 regression).

``DeepHebbianChain`` previously returned ``None`` from ``train_step`` and
silently fell back to BPTT (54 MB at 100 layers — worse than backprop). Now it
runs its local Hebbian (Oja) rule layer-by-layer under ``no_grad``: memory is
independent of depth and undercuts a matched-depth backprop model.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch

import bioplausible.zoo  # ruff: ignore[unused-import]  (registration side effect)
from bioplausible.core.trainer import CoreTrainer, TrainerConfig

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU required for memory comparison"
)


def _run(
    model: str, model_kwargs: dict[str, object]
) -> tuple[float, float, Mapping[str, int]]:
    cfg = TrainerConfig(
        model=model,
        model_kwargs=model_kwargs,
        task="digits",
        epochs=1,
        batches_per_epoch=2,
        batch_size=128,
        num_workers=0,
        run_validation=False,
        save_checkpoints=False,
        track_energy=True,
        device="cuda",
    )
    trainer = CoreTrainer(cfg)
    last = trainer.fit()[-1]
    return (
        float(last.peak_memory_mb or 0.0),
        float(last.train_loss or 0.0),
        dict(trainer._training_path_counts),
    )


@requires_gpu
def test_deep_hebbian_uses_local_rule_not_bptt() -> None:
    """100-layer Deep Hebbian must run its own train_step, never BPTT."""
    _, _, paths = _run(
        "hebbian_chain",
        {
            "input_dim": 64,
            "hidden_dim": 128,
            "output_dim": 10,
            "num_layers": 100,
            "use_spectral_norm": True,
        },
    )
    assert paths.get("model_train_step", 0) > 0
    assert paths.get("bptt", 0) == 0


@requires_gpu
def test_deep_hebbian_memory_below_backprop_at_100_layers() -> None:
    """Local Hebbian is O(1)-in-depth: 100-layer memory < backprop at 100 layers."""
    hebb_mem, hebb_loss, _ = _run(
        "hebbian_chain",
        {
            "input_dim": 64,
            "hidden_dim": 128,
            "output_dim": 10,
            "num_layers": 100,
            "use_spectral_norm": True,
        },
    )
    bp_mem, bp_loss, _ = _run(
        "backprop_mlp",
        {"input_dim": 64, "hidden_dim": 128, "output_dim": 10, "num_layers": 100},
    )
    # Loss sanity guard: only meaningful when both are finite.
    assert torch.isfinite(torch.tensor(hebb_loss))
    assert torch.isfinite(torch.tensor(bp_loss))
    assert hebb_mem < bp_mem
