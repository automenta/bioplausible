"""Headless training runner for the demo.

Wraps :class:`SystemTrainer` and the Sprint 3.4 ``ExecutionCallback`` protocol so
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

# Ensure the full component registry is populated. `import computronium` is
# lazy (Sprint 0.5), so model registration no longer happens as a side effect
# of importing the top-level package. The demo's Registry lookups (and any
# SystemTrainer instantiations by factory functions) need the zoo imported
# explicitly and up-front for deterministic behavior (the zoo owns the substrate
# deployment models that used to live in the separate equitile package).
from computronium.core.registry import ComponentCategory, Registry
from computronium.core.system_trainer import SystemTrainer, SystemTrainerConfig
from computronium import (
    create_backprop_mlp,
    create_eqprop_mlp,
    create_fa_mlp,
    create_ff_mlp,
    create_pepita_mlp,
    create_tp_mlp,
    create_pc_mlp,
    create_hebbian_mlp,
    create_snn_mlp,
    create_routing_mlp,
    create_fast_weight_mlp,
)
from computronium.domains.registry import resolve_task
from computronium.execution.callbacks import BaseExecutionCallback
from computronium.utils import seed_everything


@dataclass
class DemoPanel:
    """One side of the two-panel side-by-side comparison (Config A / Config B)."""

    trainer_config: SystemTrainerConfig
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
    # `comp parity` CLI cross-check in Sprint 3.7) are bitwise-consistent.
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
    """Collect telemetry from SystemTrainer into a DemoPanel (thread-safe)."""

    def __init__(self, panel: DemoPanel, model: torch.nn.Module | None) -> None:
        self._panel = panel
        self._model = model
        self._lock = Lock()
        self._probe = _WeightProbe()

    def on_epoch_end(self, epoch: int, metrics: dict) -> None:
        # SystemTrainer returns a dict with train_acc/val_acc, loss
        acc = metrics.get("val_acc") or metrics.get("train_acc")
        loss = metrics.get("loss") or metrics.get("train_loss")
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


# Models the demo can train through the generic SystemTrainer path on the
# supported tasks. Uses 5-D ontology factory functions.
TRAINABLE_MODELS: tuple[str, ...] = (
    "backprop_mlp",
    "eqprop_mlp",
    "fa_mlp",
    "ff_mlp",
    "pepita_mlp",
    "tp_mlp",
    "pc_mlp",
    "hebbian_mlp",
    "snn_mlp",
    "tile_mlp",
    "routing_mlp",
    "fast_weight_mlp",
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
    except (ValueError, KeyError):
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
    "fa_mlp": 128,
    "ff_mlp": 32,
    "pepita_mlp": 32,
    "tp_mlp": 128,
    "pc_mlp": 128,
    "hebbian_mlp": 128,
    "snn_mlp": 128,
    "tile_mlp": 128,
    "routing_mlp": 128,
    "fast_weight_mlp": 128,
}


def default_hidden_dim(model: str) -> int:
    """Return the per-model default hidden dimension (fallback 128)."""
    return _DEFAULT_HIDDEN_DIM.get(model, 128)


def create_system(model: str, task: str, hidden_dim: int | None, device: str) -> object:
    """Create a System using the appropriate factory function for the model."""
    spec = resolve_task(task)
    input_dim = spec.input_dim
    output_dim = spec.output_dim
    if isinstance(input_dim, (tuple, list)):
        import math
        input_dim = math.prod(input_dim)
    if hidden_dim is None:
        hidden_dim = default_hidden_dim(model)

    # Map model names to factory functions
    if model == "backprop_mlp":
        return create_backprop_mlp(
            input_dim=input_dim,
            hidden_dims=(hidden_dim, hidden_dim),
            output_dim=output_dim,
            lr=0.001,
            device=device,
        )
    elif model == "eqprop_mlp":
        return create_eqprop_mlp(
            input_dim=input_dim,
            hidden_dims=(hidden_dim, hidden_dim),
            output_dim=output_dim,
            beta=0.1,
            inference_steps=20,
            lr=0.001,
            device=device,
        )
    elif model == "fa_mlp":
        return create_fa_mlp(
            input_dim=input_dim,
            hidden_dims=(hidden_dim, hidden_dim),
            output_dim=output_dim,
            lr=0.001,
            device=device,
        )
    elif model == "ff_mlp":
        return create_ff_mlp(
            input_dim=input_dim,
            hidden_dims=(hidden_dim, hidden_dim),
            output_dim=output_dim,
            layer_lr=0.03,
            classifier_lr=0.01,
            threshold=2.0,
            num_layers=2,
            device=device,
        )
    elif model == "pepita_mlp":
        return create_pepita_mlp(
            input_dim=input_dim,
            hidden_dims=(hidden_dim, hidden_dim),
            output_dim=output_dim,
            lr=0.001,
            device=device,
        )
    elif model == "tp_mlp":
        return create_tp_mlp(
            input_dim=input_dim,
            hidden_dims=(hidden_dim, hidden_dim),
            output_dim=output_dim,
            lr=0.001,
            device=device,
        )
    elif model == "pc_mlp":
        return create_pc_mlp(
            input_dim=input_dim,
            hidden_dims=(hidden_dim, hidden_dim),
            output_dim=output_dim,
            lr=0.001,
            device=device,
        )
    elif model == "hebbian_mlp":
        return create_hebbian_mlp(
            input_dim=input_dim,
            hidden_dims=(hidden_dim, hidden_dim),
            output_dim=output_dim,
            lr=0.001,
            device=device,
        )
    elif model == "snn_mlp":
        return create_snn_mlp(
            input_dim=input_dim,
            hidden_dims=(hidden_dim, hidden_dim),
            output_dim=output_dim,
            lr=0.001,
            device=device,
        )
    elif model == "tile_mlp":
        return create_tile_mlp(
            input_dim=input_dim,
            hidden_dims=(hidden_dim, hidden_dim),
            output_dim=output_dim,
            lr=0.001,
            device=device,
        )
    elif model == "routing_mlp":
        return create_routing_mlp(
            input_dim=input_dim,
            hidden_dims=(hidden_dim, hidden_dim),
            output_dim=output_dim,
            lr=0.001,
            device=device,
        )
    elif model == "fast_weight_mlp":
        return create_fast_weight_mlp(
            input_dim=input_dim,
            hidden_dims=(hidden_dim, hidden_dim),
            output_dim=output_dim,
            lr=0.001,
            device=device,
        )
    else:
        raise ValueError(f"Unknown model: {model}")


def default_trainer_config(
    model: str = "backprop_mlp",
    task: str = "mnist",
    epochs: int = 10,
    lr: float = 0.001,
    hidden_dim: int | None = None,
    optimizer: str = "adam",
) -> SystemTrainerConfig:
    """Build a sane default SystemTrainerConfig for the demo.

    ``hidden_dim`` defaults per model (see ``_DEFAULT_HIDDEN_DIM``) so e.g. the
    flagship PEPITA/FF config starts small (32) instead of the generic 256.
    """
    if hidden_dim is None:
        hidden_dim = default_hidden_dim(model)
    return SystemTrainerConfig(
        max_epochs=epochs,
        batch_size=64,
        val_batch_size=None,
        device="auto",
        grad_clip=1.0,
        track_energy=True,
        track_flops=True,
        track_memory=True,
        log_every_n_steps=10,
        seed=42,
        deterministic=False,
    )


def prepare_trainer_config(
    prev: SystemTrainerConfig | None,
    model: str,
    task: str,
    epochs: int,
    lr: float,
) -> SystemTrainerConfig:
    """Return the config to train, preserving live widget-tree knob edits.

    When ``prev`` targets the same ``model``/``task``, its object is returned
    mutated (epochs/lr refreshed) so Sprint 3.2 slider/number edits to the
    expanded knobs actually feed the run. A model/task change rebuilds from
    defaults (the widget tree needs re-render on a new config object -- a
    documented UI limitation).
    """
    if prev is not None:
        prev.max_epochs = int(epochs)
        return prev
    return default_trainer_config(model=model, task=task, epochs=int(epochs), lr=float(lr))


def run_headless(panel: DemoPanel) -> None:
    """Synchronously train a panel to completion (call from a thread)."""
    try:
        panel.running = True
        panel.finished = False
        if panel.seed is not None:
            seed_everything(panel.seed)

        # Create the system using the factory function
        # The panel.trainer_config contains the training config
        # We need to extract model info from the panel - for now use a default
        # The actual model is determined by the UI selection
        # This is a simplified version - the UI should pass the model name
        import sys
        model_name = getattr(panel, "_model_name", "backprop_mlp")
        task_name = getattr(panel, "_task_name", "mnist")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        hidden_dim = getattr(panel, "_hidden_dim", None)

        system = create_system(model_name, task_name, hidden_dim, device)
        task = resolve_task(task_name)
        task.setup()

        # Create data loaders
        from torch.utils.data import DataLoader

        class _FlattenLoader:
            def __init__(self, loader: DataLoader):
                self.loader = loader

            def __iter__(self):
                for x, y in self.loader:
                    if x.dim() > 2:
                        x = x.view(x.size(0), -1)
                    yield x, y

            def __len__(self) -> int:
                return len(self.loader)

        train_loader = _FlattenLoader(task.get_dataloader("train"))
        val_loader = _FlattenLoader(task.get_dataloader("val"))

        trainer = SystemTrainer(
            system=system,
            config=panel.trainer_config,
            train_data=train_loader,
            val_data=val_loader,
        )
        trainer.add_execution_callback(_DemoCallback(panel, trainer.system.geometry))
        with trainer:
            trainer.fit()
    except Exception as e:
        panel.error = str(e)
    finally:
        panel.running = False
        panel.finished = True


async def run_async(panel: DemoPanel) -> None:
    """Train a panel in a worker thread so the event loop stays responsive."""
    loop = asyncio.get_running_loop()
    # Use a dedicated thread pool executor that we can shut down
    executor = panel._executor if hasattr(panel, "_executor") else None
    try:
        await loop.run_in_executor(executor, run_headless, panel)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)


def elapsed(last: float) -> float:
    """Return seconds since ``last`` (helper for UI pacing)."""
    return time.perf_counter() - last
