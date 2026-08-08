"""EqProp GPU memory vs Backprop A/B (EXPERIMENT_PLAN5 verification #2).

Thesis-critical targeted comparison: EqProp must undercut Backprop's **peak
GPU memory** at matched architecture. Measured honestly, the current
contrastive/implicit EqProp does **not** yet beat backprop at the scales the
shallow probe can reach — the "Memory is NOT < Backprop" defect from the
plan's fix table is still open. This test is therefore an expected failure
(``xfail strict``) that flips the moment the memory-advantage fix lands, so
the engine never silently reports a local rule as cheaper than it is.
"""

from __future__ import annotations

import pytest
import torch

import bioplausible.zoo  # ruff: ignore[unused-import]  (registration side effect)

from bioplausible.core.trainer import CoreTrainer, TrainerConfig

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU required for memory comparison"
)


def _run(model: str, model_kwargs: dict[str, object]) -> tuple[float, float]:
    cfg = TrainerConfig(
        model=model,
        model_kwargs=model_kwargs,
        task="digits",
        epochs=1,
        batches_per_epoch=10,
        batch_size=64,
        num_workers=0,
        run_validation=False,
        save_checkpoints=False,
        track_energy=True,
        device="cuda",
    )
    last = CoreTrainer(cfg).fit()[-1]
    return float(last.peak_memory_mb or 0.0), float(last.train_loss or 0.0)


@requires_gpu
@pytest.mark.xfail(
    strict=True,
    reason=(
        "PLAN5 fix not landed: contrastive/implicit EqProp does not yet "
        "undercut backprop peak GPU memory at probe scale. Honest negative."
    ),
)
def test_eqprop_contrastive_memory_below_backprop() -> None:
    """Contrastive EqProp peak memory < same-arch backprop peak memory."""
    eq_mem, eq_loss = _run(
        "eqprop_mlp",
        {
            "input_dim": 64,
            "hidden_dim": 64,
            "output_dim": 10,
            "gradient_method": "contrastive",
            "max_steps": 5,
            "use_spectral_norm": True,
        },
    )
    bp_mem, bp_loss = _run(
        "backprop_mlp",
        {"input_dim": 64, "hidden_dim": 64, "output_dim": 10, "num_layers": 2},
    )
    # Only meaningful near matched loss; guard the comparison's premise.
    assert eq_loss / max(bp_loss, 1e-9) < 5.0
    assert eq_mem < bp_mem
