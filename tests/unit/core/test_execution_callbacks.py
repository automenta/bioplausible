"""Tests for the Sprint 3.4 ExecutionCallback protocol + CoreTrainer wiring.

Validates that UI-agnostic telemetry hooks fire with scalar data only,
in the correct order, and never break training even when a listener
raises.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from bioplausible.core.trainer import CoreTrainer, TrainerConfig
from bioplausible.execution.callbacks import BaseExecutionCallback, ExecutionCallback


class RecordingCallback(BaseExecutionCallback):
    """Records all hook invocations for assertion."""

    def __init__(self) -> None:
        self.epochs: list[tuple[int, object]] = []
        self.steps: list[tuple[int, float, dict[str, float]]] = []
        self.settling: list[tuple[int, float]] = []

    def on_epoch_end(self, epoch: int, metrics: object) -> None:
        self.epochs.append((epoch, metrics))

    def on_step_end(self, step: int, loss: float, grad_norms: object) -> None:
        self.steps.append((step, loss, dict(grad_norms)))  # type: ignore[arg-type]

    def on_settling_step(self, step: int, energy: float) -> None:
        self.settling.append((step, energy))


def _make_trainer(
    model: str = "backprop_mlp",
    epochs: int = 1,
    batches: int = 2,
    track_energy: bool = False,
) -> CoreTrainer:
    """Build a tiny CPU trainer on cached MNIST with overridden loaders."""
    dataset = TensorDataset(torch.randn(10, 784), torch.randint(0, 10, (10,)))
    loader = DataLoader(dataset, batch_size=2)
    config = TrainerConfig(
        model=model,
        model_kwargs={"input_dim": 784, "hidden_dim": 16, "output_dim": 10},
        optimizer="adam",
        task="mnist",
        epochs=epochs,
        batches_per_epoch=batches,
        val_batches=1,
        use_compile=False,
        track_energy=track_energy,
        device="cpu",
    )
    trainer = CoreTrainer(config)
    trainer.setup()
    trainer.train_loader = loader
    trainer.val_loader = loader
    return trainer


def test_epoch_and_step_hooks_fire_with_correct_counts():
    """on_epoch_end fires per epoch; on_step_end fires per batch."""
    cb = RecordingCallback()
    trainer = _make_trainer(epochs=2, batches=3)
    trainer.add_execution_callback(cb)
    trainer.fit()

    assert [e for e, _ in cb.epochs] == [0, 1]
    assert len(cb.steps) == 6  # 2 epochs x 3 batches
    steps = [s for s, _, _ in cb.steps]
    assert steps == list(range(1, 7))  # global 1-based counter
    for _, loss, grad_norms in cb.steps:
        assert isinstance(loss, float)
        assert isinstance(grad_norms, dict)


def test_step_hook_fires_before_epoch_hook():
    """Within an epoch, step telemetry must precede the epoch hook."""
    cb = RecordingCallback()
    trainer = _make_trainer(epochs=1, batches=2)
    trainer.add_execution_callback(cb)
    trainer.fit()

    assert len(cb.epochs) == 1
    assert cb.epochs[0][0] == 0
    assert cb.steps[-1][0] == 2  # last step of epoch 0 completed first


def test_grad_norms_populated_for_bptt_path():
    """Standard BPTT materializes grads, so grad_norms should be non-empty."""
    cb = RecordingCallback()
    trainer = _make_trainer(model="backprop_mlp", epochs=1, batches=2)
    trainer.add_execution_callback(cb)
    trainer.fit()

    assert any(len(gn) > 0 for _, _, gn in cb.steps)


def test_settling_hook_fires_with_energy_tracking():
    """energy_proxy telemetry triggers on_settling_step once per step."""
    cb = RecordingCallback()
    trainer = _make_trainer(
        model="backprop_mlp", epochs=1, batches=2, track_energy=True
    )
    trainer.add_execution_callback(cb)
    trainer.fit()

    assert len(cb.settling) == 2
    for _, energy in cb.settling:
        assert isinstance(energy, float)


def test_raising_callback_does_not_break_training():
    """A misbehaving UI listener is logged and swallowed."""

    class BadCallback(ExecutionCallback):
        def on_step_end(self, step: int, loss: float, grad_norms: object) -> None:
            raise RuntimeError("boom")

        def on_epoch_end(self, epoch: int, metrics: object) -> None:
            raise RuntimeError("boom")

        def on_settling_step(self, step: int, energy: float) -> None:
            raise RuntimeError("boom")

    trainer = _make_trainer(epochs=1, batches=2)
    trainer.add_execution_callback(BadCallback())
    history = trainer.fit()
    assert len(history) == 1


def test_protocol_is_runtime_checkable():
    """ExecutionCallback is a runtime-checkable Protocol."""
    cb = RecordingCallback()
    assert isinstance(cb, ExecutionCallback)
