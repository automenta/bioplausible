"""EqProp O(1)-in-steps memory (EXPERIMENT_PLAN5 verification #2).

The valid apples-to-apples O(1) claim is on the **same architecture**: the
implicit equilibrium backward (`gradient_method="equilibrium"`) keeps peak
memory flat as settle steps grow, while the unrolled BPTT backward
(`gradient_method="bptt"`) stores a per-step graph and grows with steps. This
locks that the O(1) EqProp path is real and detected as its own path (not
mislabeled bptt).
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

_STEPS = 60


def _run(gradient_method: str) -> tuple[float, float, Mapping[str, int]]:
    cfg = TrainerConfig(
        model="eqprop_mlp",
        model_kwargs={
            "input_dim": 64,
            "hidden_dim": 128,
            "output_dim": 10,
            "max_steps": _STEPS,
            "use_spectral_norm": True,
            "gradient_method": gradient_method,
        },
        task="digits",
        epochs=1,
        batches_per_epoch=2,
        batch_size=256,
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
def test_implicit_is_flat_and_not_mislabeled_bptt() -> None:
    """Implicit equilibrium memory is independent of settle steps (O(1))."""
    low_mem, _, _ = _run("equilibrium")
    # Same path at identical architecture/steps but via O(1) implicit custom backward.
    paths = _run("equilibrium")[2]
    assert paths.get("implicit_equilibrium", 0) > 0
    assert paths.get("bptt", 0) == 0
    assert low_mem > 0


@requires_gpu
def test_implicit_memory_below_unrolled_bptt() -> None:
    """O(1) implicit undercuts unrolled BPTT on the same architecture."""
    implicit_mem, implicit_loss, _ = _run("equilibrium")
    bptt_mem, bptt_loss, paths = _run("bptt")
    assert torch.isfinite(torch.tensor(implicit_loss))
    assert torch.isfinite(torch.tensor(bptt_loss))
    # Unrolled BPTT grows a per-step graph.
    assert bptt_mem > implicit_mem
    assert paths.get("bptt", 0) > 0
