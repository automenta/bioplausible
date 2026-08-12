"""Unit tests for the probe driver's compute-config threading (hardening).

Verifies that `CoreTrainerDriver` applies the campaign's `compute` settings
(worker count, tracking toggles) to the `TrainerConfig` it builds for each
probe — no real training; `CoreTrainer` is swapped for a config-capturing fake.
Also covers the self-diagnosis surfaces (phantom knobs, param count) and the
NaN-divergence guard the same way: with a fake trainer, so the whole file runs
in milliseconds without touching GPU or data loaders.
"""

from __future__ import annotations

from bioplausible.core.trainer import TrainingMetrics
from bioplausible.experiment.probe import CoreTrainerDriver


def _fake_history(accuracy: float = 0.9) -> list[TrainingMetrics]:
    return [
        TrainingMetrics(
            epoch=0,
            train_loss=0.1,
            train_acc=accuracy,
            val_acc=accuracy,
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


def test_driver_surfaces_phantom_knobs(monkeypatch):
    """An unconsumable sampled knob is surfaced as a defect, never hidden.

    backprop_mlp has no ``config=`` and no ``beta`` constructor param, so a
    sampled ``beta`` is phantom — the probe must report it, not silently ignore
    the way the pre-construction-layer code did.
    """
    captured: dict[str, object] = {}

    class FakeCoreTrainer:
        def __init__(self, cfg) -> None:
            captured["cfg"] = cfg

        @staticmethod
        def fit() -> list[TrainingMetrics]:
            return _fake_history()

    import bioplausible.experiment.probe as probe_module

    monkeypatch.setattr(probe_module, "CoreTrainer", FakeCoreTrainer)
    out = CoreTrainerDriver(record_results=False).train(
        model="backprop_mlp",
        task="mnist",
        config={"hidden_dim": 16, "beta": 0.5},
        seed=0,
        epochs=1,
        device="cpu",
    )
    assert "beta" in out["phantom_knobs"]


def test_driver_surfaces_param_count(monkeypatch):
    captured: dict[str, object] = {}

    class FakeCoreTrainer:
        def __init__(self, cfg) -> None:
            captured["cfg"] = cfg

        @staticmethod
        def fit() -> list[TrainingMetrics]:
            return _fake_history()

    import bioplausible.experiment.probe as probe_module

    monkeypatch.setattr(probe_module, "CoreTrainer", FakeCoreTrainer)
    out = CoreTrainerDriver(record_results=False).train(
        model="backprop_mlp",
        task="mnist",
        config={"hidden_dim": 16},
        seed=0,
        epochs=1,
        device="cpu",
    )
    assert out["param_count"] > 0


def test_driver_raises_on_nan_divergence(monkeypatch):
    """The trainer's NumericalInstabilityError becomes a clear diverged error.

    The future NaN guard raises ``NumericalInstabilityError`` from the epoch
    loop; the driver must translate that into a distinct, diagnostic failure
    (not a generic probe error) so the sweep can classify it as a NaN defect.
    """
    from bioplausible.core.exceptions import NumericalInstabilityError

    class FakeCoreTrainer:
        def __init__(self, cfg) -> None:
            pass

        @staticmethod
        def fit():
            raise NumericalInstabilityError("non-finite loss (nan) for model='x'")

    import bioplausible.experiment.probe as probe_module

    monkeypatch.setattr(probe_module, "CoreTrainer", FakeCoreTrainer)
    import pytest

    with pytest.raises(RuntimeError, match="diverged"):
        CoreTrainerDriver(record_results=False).train(
            model="backprop_mlp",
            task="mnist",
            config={"hidden_dim": 16},
            seed=0,
            epochs=1,
            device="cpu",
        )
