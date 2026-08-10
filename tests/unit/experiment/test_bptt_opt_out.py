"""BPTT opt-out default (EXPERIMENT_PLAN5 §1).

``TrainerConfig(allow_bptt_fallback=False)`` for bio families must raise a
loud warning whenever a model silently degrades to the backprop fallback, so
the degradation is never masked (the sweep flags it as a defect via the
training path).
"""

from __future__ import annotations

import logging

from bioplausible.core.trainer import CoreTrainer, TrainerConfig


def _base_cfg(**kw: object) -> TrainerConfig:
    return TrainerConfig(
        model="backprop_mlp",
        task="digits",
        model_kwargs={"input_dim": 64, "hidden_dim": 16, "output_dim": 10},
        epochs=1,
        batches_per_epoch=3,
        run_validation=False,
        save_checkpoints=False,
        **kw,  # type: ignore[arg-type]
    )


def test_fallback_disallowed_raises_loud_warning(caplog) -> None:
    """With the opt-out set, a BPTT fallback is loudly reported."""
    with caplog.at_level(logging.WARNING):
        CoreTrainer(_base_cfg(allow_bptt_fallback=False)).fit()
    assert any("BPTT fallback" in r.message for r in caplog.records)


def test_fallback_allowed_stays_silent(caplog) -> None:
    """Backprop-family models keep the fallback on with no warning."""
    with caplog.at_level(logging.WARNING):
        CoreTrainer(_base_cfg()).fit()
    assert not any("BPTT fallback" in r.message for r in caplog.records)


def test_fallback_disallowed_records_bptt_path() -> None:
    """Even when disallowed, the step runs and its path is recorded (for defect)."""
    trainer = CoreTrainer(_base_cfg(allow_bptt_fallback=False))
    trainer.fit()
    assert trainer._training_path_counts.get("bptt", 0) > 0


def test_fallback_warning_deduped_per_run(caplog) -> None:
    """The BPTT-fallback warning is emitted once per run, not once per batch.

    A probe that degrades to backprop must not flood the log with the same
    warning on every step (the config is 40 batches/epoch here; the warning
    should appear exactly once).
    """
    cfg = dict(
        model="backprop_mlp",
        task="digits",
        model_kwargs={"input_dim": 64, "hidden_dim": 16, "output_dim": 10},
        epochs=2,
        batches_per_epoch=40,
        run_validation=False,
        save_checkpoints=False,
        allow_bptt_fallback=False,
    )
    with caplog.at_level(logging.WARNING):
        CoreTrainer(TrainerConfig(**cfg)).fit()
    count = sum(1 for r in caplog.records if "BPTT fallback" in r.message)
    assert count == 1
