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

import torch

# Ensure the full component registry is populated. `import bioplausible` is
# lazy (Sprint 0.5), so model registration no longer happens as a side effect
# of importing the top-level package. The demo's Registry lookups (and any
# CoreTrainer instantiations by registered model name) need the zoo imported
# explicitly and up-front for deterministic behavior (the zoo owns the substrate
# deployment models that used to live in the separate equitile package).
import bioplausible.zoo  # ruff: ignore[unused-import]  (registers zoo models/propagators)
from bioplausible.core.registry import ComponentCategory, Registry
from bioplausible.core.trainer import CoreTrainer, TrainerConfig
from bioplausible.domains.registry import resolve_task
from bioplausible.execution.callbacks import BaseExecutionCallback
from bioplausible.utils import seed_everything


@dataclass
class DemoPanel:
    """One side of the two-panel side-by-side comparison (Config A / Config B)."""

    trainer_config: TrainerConfig
    epochs: int = 10
    losses: list[float] = field(default_factory=list)
    accuracies: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    energies: list[float] = field(default_factory=list)
    weight_history: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    running: bool = False
    finished: bool = False
    error: str | None = None
    # Fixed-seed reproducibility for this panel. When set, `run_headless` seeds
    # the global RNG before training so the two-panel comparison (and the
    # `biopl-parity` CLI cross-check in Sprint 3.7) are bitwise-consistent.
    seed: int | None = None


class _WeightProbe:
    """Online-decimated capture of training-time weight snapshots.

    Records a per-layer series of weight matrices (CPU) so the UI can animate
    how weights evolve (Sprint 3.5). Memory is bounded to ``max_snaps`` frames
    per layer by doubling the capture stride whenever history overflows, so
    even a 10k-step run stores at most ~max_snaps snapshots per layer.
    """

    def __init__(self, max_snaps: int = 120) -> None:
        self.max_snaps = max(max_snaps, 4)
        self.stride = 1
        self._count = 0
        self.history: dict[str, list[torch.Tensor]] = {}

    def capture(self, model: torch.nn.Module) -> None:
        self._count += 1
        if self._count % self.stride != 0:
            return
        for name, param in model.named_parameters():
            if "weight" not in name:
                continue
            with torch.no_grad():
                self.history.setdefault(name, []).append(param.detach().float().cpu())
        if any(len(v) > self.max_snaps for v in self.history.values()):
            self._compact()

    def _compact(self) -> None:
        self.stride *= 2
        for key in list(self.history):
            self.history[key] = self.history[key][::2]


class _DemoCallback(BaseExecutionCallback):
    """Collect telemetry from CoreTrainer into a DemoPanel (thread-safe)."""

    def __init__(self, panel: DemoPanel, model: torch.nn.Module | None) -> None:
        self._panel = panel
        self._model = model
        self._lock = Lock()
        self._probe = _WeightProbe()

    def on_epoch_end(self, epoch: int, metrics: object) -> None:
        # TrainingMetrics exposes train_acc/val_acc (no bare `accuracy`); accept
        # both plus a bare `loss`/`train_loss`.
        acc = getattr(metrics, "train_acc", None) or getattr(metrics, "val_acc", None)
        loss = getattr(metrics, "loss", None) or getattr(metrics, "train_loss", None)
        with self._lock:
            acc_val = float(acc) if acc is not None else float("nan")
            self._panel.accuracies.append(acc_val)
            if loss is not None:
                self._panel.losses.append(float(loss))

    def on_step_end(self, step: int, loss: float, grad_norms: object) -> None:
        with self._lock:
            self._panel.losses.append(float(loss))
            if self._model is not None:
                self._probe.capture(self._model)
                self._panel.weight_history = self._probe.history

    def on_settling_step(self, step: int, energy: float) -> None:
        with self._lock:
            self._panel.energies.append(float(energy))


# Models the demo can train through the generic CoreTrainer path on the
# supported tasks. Started as backprop+eqprop; TileNet families now included.
TRAINABLE_MODELS: tuple[str, ...] = (
    "backprop_mlp",
    "eqprop_mlp",
    "tile_pc",
    "pepita",
    "forward_forward",
    "standard_fa",
    "conv_tile",
    "graph_tile",
    "rl_tile",
    "timeseries_tile",
    "tile_lm",
)


def model_metadata(model: str) -> dict[str, object]:
    """Return calibrated Sprint 2.5 registry metadata for a demo model name.

    Looks the model up in the ``MODEL`` category registry and returns the
    compact dict the UI surfaces as tooltips (bio_plausibility_score,
    locality_level, family, requires_backward). Unknown names degrade to
    ``{}`` so the UI never crashes on a stale model list.
    """
    try:
        meta = Registry.get_metadata(ComponentCategory.MODEL, model)
    except ValueError, KeyError:
        return {}
    return {
        "bio_plausibility_score": meta.bio_plausibility_score,
        "locality_level": meta.locality_level.value,
        "family": meta.family,
        "requires_backward": meta.requires_backward,
        # Absolute accuracy-gap ceiling (0-1) mirroring the hyperparam YAML
        # `parity_threshold`; absent for backprop-like baselines.
        "parity_threshold": meta.extra.get("parity_threshold", 0.05),
    }


# Per-model default hidden_dim. The shared 256 default is wasteful/slow for
# tile-based families where `neurons_per_tile`/`num_tiles` track hidden_dim
# (TileNet builds a graph proportional to it), and for tiny forward-only
# nets (PEPITA/FF) that are best explored at small scale. Kept central so the
# demo stays snappy while still allowing the widget tree to override.
_DEFAULT_HIDDEN_DIM: dict[str, int] = {
    "backprop_mlp": 128,
    "eqprop_mlp": 128,
    "tile_pc": 32,
    "pepita": 32,
    "forward_forward": 32,
    "standard_fa": 128,
    "conv_tile": 32,
    "graph_tile": 32,
    "rl_tile": 32,
    "timeseries_tile": 32,
    "tile_lm": 192,
}


def default_hidden_dim(model: str) -> int:
    """Return the per-model default hidden dimension (fallback 128)."""
    return _DEFAULT_HIDDEN_DIM.get(model, 128)


def default_trainer_config(
    model: str = "backprop_mlp",
    task: str = "mnist",
    epochs: int = 10,
    lr: float = 0.001,
    hidden_dim: int | None = None,
    optimizer: str = "adam",
) -> TrainerConfig:
    """Build a sane default TrainerConfig for the demo.

    ``hidden_dim`` defaults per model (see ``_DEFAULT_HIDDEN_DIM``) so e.g. the
    flagship EquiTile config starts small (32) instead of the generic 256.
    """
    spec = resolve_task(task)
    input_dim, output_dim = spec.input_dim, spec.output_dim
    if hidden_dim is None:
        hidden_dim = default_hidden_dim(model)
    return TrainerConfig(
        model=model,
        model_kwargs={
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
        },
        optimizer=optimizer,
        optimizer_kwargs={"lr": lr},
        task=task,
        epochs=epochs,
    )


def prepare_trainer_config(
    prev: TrainerConfig | None,
    model: str,
    task: str,
    epochs: int,
    lr: float,
) -> TrainerConfig:
    """Return the config to train, preserving live widget-tree knob edits.

    When ``prev`` targets the same ``model``/``task``, its object is returned
    mutated (epochs/lr refreshed) so Sprint 3.2 slider/number edits to the
    expanded ``model_kwargs``/``optimizer_kwargs`` knobs actually feed the run.
    A model/task change rebuilds from defaults (the widget tree needs re-render
    on a new config object — a documented UI limitation).
    """
    if prev is not None and prev.model == model and prev.task == task:
        prev.epochs = int(epochs)
        prev.optimizer_kwargs["lr"] = float(lr)
        return prev
    return default_trainer_config(
        model=model, task=task, epochs=int(epochs), lr=float(lr)
    )


def run_headless(panel: DemoPanel) -> None:
    """Synchronously train a panel to completion (call from a thread)."""
    try:
        panel.running = True
        panel.finished = False
        if panel.seed is not None:
            seed_everything(panel.seed)
        trainer = CoreTrainer(panel.trainer_config)
        trainer.setup()  # materialize model so the callback can probe weights
        trainer.add_execution_callback(_DemoCallback(panel, trainer.model))
        trainer.fit()
    except Exception as e:
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
