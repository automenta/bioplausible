"""Unit tests for the probe driver's compute-config threading (hardening).

Verifies that `CoreTrainerDriver` applies the campaign's `compute` settings
(worker count, tracking toggles) to the `TrainerConfig` it builds for each
probe — no real training; `CoreTrainer` is swapped for a config-capturing fake.
"""

from __future__ import annotations

from bioplausible.core.trainer import TrainingMetrics
from bioplausible.experiment.probe import CoreTrainerDriver


def _fake_history(accuracy: float = 0.9) -> list[TrainingMetrics]:
    return [
        TrainingMetrics(
            epoch=0,
            train_loss=0.1,
            train_accuracy=accuracy,
            val_accuracy=accuracy,
            epoch_time=1.0,
            forward_flops=10,
            backward_flops=20,
            peak_memory_mb=5.0,
        )
    ]


def test_driver_threads_compute_settings_into_trainer_config(monkeypatch):
    """num_workers/tracking from compute flow into every TrainerConfig."""
    captured: dict[str, object] = {}

    class FakeCoreTrainer:
        def __init__(self, cfg) -> None:
            captured["cfg"] = cfg

        @staticmethod
        def fit() -> list[TrainingMetrics]:
            assert captured["cfg"] is not None
            return _fake_history()

    import bioplausible.experiment.probe as probe_module

    monkeypatch.setattr(probe_module, "CoreTrainer", FakeCoreTrainer)

    driver = CoreTrainerDriver(
        num_workers=0,
        batch_size=32,
        track_energy=False,
        track_flops=True,
        track_memory=False,
    )
    out = driver.train(
        model="backprop_mlp",
        task="mnist",
        config={"hidden_dim": 16},
        seed=0,
        epochs=1,
        device="cpu",
    )

    cfg = captured["cfg"]
    assert cfg.num_workers == 0
    assert cfg.batch_size == 32
    assert cfg.track_flops is True
    assert cfg.track_memory is False
    # CoreTrainer gates all profiling on `track_energy`; the driver raises it
    # when ANY metric (here flops) is wanted so the campaign's `compute.track`
    # actually produces values.
    assert cfg.track_energy is True
    assert out["final_acc"] == 0.9
    assert out["epoch_time_s"] == 1.0
    assert out["peak_memory_mb"] == 5.0
    # wall_time_s falls back to the summed epoch time (not CUDA-only).
    assert out["wall_time_s"] == 1.0


def test_driver_all_off_disables_profiling(monkeypatch):
    """With no tracking requested, CoreTrainer's profiler stays off."""
    captured: dict[str, object] = {}

    class FakeCoreTrainer:
        def __init__(self, cfg) -> None:
            captured["cfg"] = cfg

        @staticmethod
        def fit() -> list[TrainingMetrics]:
            return _fake_history()

    import bioplausible.experiment.probe as probe_module

    monkeypatch.setattr(probe_module, "CoreTrainer", FakeCoreTrainer)
    CoreTrainerDriver(track_flops=False, track_memory=False, track_energy=False).train(
        model="backprop_mlp",
        task="mnist",
        config={},
        seed=0,
        epochs=1,
        device="cpu",
    )
    assert captured["cfg"].track_energy is False
