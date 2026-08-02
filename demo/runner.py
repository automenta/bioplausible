"""Headless training runner for the demo.

Wraps :class:`CoreTrainer` and the Sprint 3.4 ``ExecutionCallback`` protocol so
the NiceGUI UI stays a pure consumer of telemetry events — no UI object ever
touches the training loop. Designed to run in a worker thread/event loop so the
browser never blocks, matching the "engine stays UI-agnostic" architecture rule.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from threading import Lock

from bioplausible.core.trainer import CoreTrainer, TrainerConfig
from bioplausible.execution.callbacks import BaseExecutionCallback


@dataclass
class DemoPanel:
    """One side of the two-panel side-by-side comparison (Config A / Config B)."""

    trainer_config: TrainerConfig
    epochs: int = 10
    losses: list[float] = field(default_factory=list)
    accuracies: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    energies: list[float] = field(default_factory=list)
    running: bool = False
    finished: bool = False
    error: str | None = None


class _DemoCallback(BaseExecutionCallback):
    """Collect telemetry from CoreTrainer into a DemoPanel (thread-safe)."""

    def __init__(self, panel: DemoPanel) -> None:
        self._panel = panel
        self._lock = Lock()

    def on_epoch_end(self, epoch: int, metrics: object) -> None:
        acc = getattr(metrics, "accuracy", None)
        loss = getattr(metrics, "loss", None)
        with self._lock:
            acc_val = float(acc) if acc is not None else float("nan")
            self._panel.accuracies.append(acc_val)
            if loss is not None:
                self._panel.losses.append(float(loss))

    def on_step_end(self, step: int, loss: float, grad_norms: object) -> None:
        with self._lock:
            self._panel.losses.append(float(loss))

    def on_settling_step(self, step: int, energy: float) -> None:
        with self._lock:
            self._panel.energies.append(float(energy))


def default_trainer_config(
    model: str = "backprop_mlp",
    task: str = "mnist",
    epochs: int = 10,
    lr: float = 0.001,
    hidden_dim: int = 256,
    optimizer: str = "adam",
) -> TrainerConfig:
    """Build a sane default TrainerConfig for the demo."""
    return TrainerConfig(
        model=model,
        model_kwargs={
            "input_dim": 784,
            "hidden_dim": hidden_dim,
            "output_dim": 10,
        },
        optimizer=optimizer,
        optimizer_kwargs={"lr": lr},
        task=task,
        epochs=epochs,
    )


def run_headless(panel: DemoPanel) -> None:
    """Synchronously train a panel to completion (call from a thread)."""
    try:
        panel.running = True
        panel.finished = False
        trainer = CoreTrainer(panel.trainer_config)
        trainer.add_execution_callback(_DemoCallback(panel))
        trainer.fit()
    except Exception as e:  # noqa: BLE001 - surface any error to the UI
        panel.error = str(e)
    finally:
        panel.running = False
        panel.finished = True


async def run_async(panel: DemoPanel) -> None:
    """Train a panel in a worker thread so the event loop stays responsive."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, run_headless, panel)


def elapsed(last: float) -> float:
    """Return seconds since ``last`` (helper for UI pacing)."""
    return time.perf_counter() - last
