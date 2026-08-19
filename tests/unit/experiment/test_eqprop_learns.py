"""EqProp contrastive learns on GPU (EXPERIMENT_PLAN5 verification #1).

The eqprop family must be probed with its *own* local rule
(``gradient_method="contrastive"``), which runs the equilibrium loop via the
model's ``train_step`` — never the BPTT fallback. This locks that the
measured bio-rule cost is actually the local rule's cost, and that the rule
decreases loss at 2 epochs.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch

from bioplausible.core.trainer import CoreTrainer, TrainerConfig

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU required for memory comparison"
)


def _cfg() -> TrainerConfig:
    return TrainerConfig(
        model="eqprop_mlp",
        model_kwargs={
            "input_dim": 64,
            "hidden_dim": 32,
            "output_dim": 10,
            "gradient_method": "contrastive",
            "max_steps": 5,
            "use_spectral_norm": True,
        },
        task="digits",
        epochs=2,
        batches_per_epoch=20,
        batch_size=64,
        num_workers=0,
        run_validation=False,
        save_checkpoints=False,
        track_energy=False,
        device="cuda",
    )


@requires_gpu
def test_eqprop_contrastive_decreases_loss() -> None:
    """Contrastive EqProp reduces loss across 2 epochs (not via BPTT)."""
    trainer = CoreTrainer(_cfg())
    history = trainer.fit()
    assert history[-1].train_loss < history[0].train_loss


@requires_gpu
def test_eqprop_contrastive_runs_local_rule_not_bptt() -> None:
    """The probe records the local contrastive path, never a BPTT fallback."""
    trainer = CoreTrainer(_cfg())
    trainer.fit()
    paths: Mapping[str, int] = trainer._training_path_counts
    assert paths.get("model_train_step", 0) > 0
    assert paths.get("bptt", 0) == 0
