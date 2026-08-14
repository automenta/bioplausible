"""Training-path telemetry (EXPERIMENT_PLAN5 §1).

CoreTrainer must record which credit-assignment path each train step took
(energy | model_train_step | propagator | bptt) and surface it per-epoch and
through the probe driver, so the sweep can flag silent BPTT fallbacks.
"""

from __future__ import annotations

from bioplausible.core.trainer import CoreTrainer, TrainerConfig, TrainingMetrics
from bioplausible.experiment.probe import CoreTrainerDriver


def _base_cfg(**kw: object) -> TrainerConfig:
    if "epochs" in kw:
        epochs = kw.pop("epochs")
    else:
        epochs = 1
    return TrainerConfig(
        model="backprop_mlp",
        task="digits",
        model_kwargs={"input_dim": 64, "hidden_dim": 16, "output_dim": 10},
        epochs=int(epochs),
        batches_per_epoch=3,
        run_validation=False,
        save_checkpoints=False,
        **kw,  # type: ignore[arg-type]
    )


def test_plain_bptt_model_records_bptt_path() -> None:
    """A model with no local rule degrades to the BPTT path — and says so."""
    trainer = CoreTrainer(_base_cfg())
    history = trainer.fit()
    counts = trainer._training_path_counts
    assert counts.get("bptt", 0) > 0
    # Surfaced on the epoch metrics for the probe driver.
    assert history[-1].extra["training_paths"]["bptt"] > 0


def test_propagator_model_records_propagator_path() -> None:
    """A configured learning-rule propagator owns the step (no BPTT)."""
    trainer = CoreTrainer(_base_cfg(propagator="feedback_alignment"))
    trainer.fit()
    counts = trainer._training_path_counts
    assert counts.get("propagator", 0) > 0
    assert counts.get("bptt", 0) == 0


def test_training_path_accumulates_across_epochs() -> None:
    """Path counts accumulate over the whole run, not per epoch."""
    trainer = CoreTrainer(_base_cfg(epochs=2))
    trainer.fit()
    assert trainer._training_path_counts["bptt"] >= 2


def test_probe_surfaces_dominant_training_path(monkeypatch) -> None:
    """The probe driver reports the path so the sweep can gate on it."""
    history = [
        TrainingMetrics(
            epoch=0,
            train_loss=0.5,
            train_acc=0.9,
            extra={"training_paths": {"bptt": 3}},
        )
    ]

    class FakeCoreTrainer:
        def __init__(
            self,
            cfg: object,
            dataset_cache: object = None,
            model_cache: object = None,
        ) -> None:
            self.cfg = cfg

        @staticmethod
        def fit() -> list[TrainingMetrics]:
            return history

    import bioplausible.experiment.probe as probe_module

    monkeypatch.setattr(probe_module, "CoreTrainer", FakeCoreTrainer)
    driver = CoreTrainerDriver(record_results=False)
    out = driver.train(
        model="backprop_mlp",
        task="mnist",
        config={},
        seed=0,
        epochs=1,
        device="cpu",
    )
    assert out["training_path"] == "bptt"
    assert out["training_paths"] == {"bptt": 3}
