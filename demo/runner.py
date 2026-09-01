"""Headless training runner for the demo.

Wraps :class:`SystemTrainer` so the NiceGUI UI stays a pure consumer of
telemetry — no UI object ever touches the training loop. Designed to run in a
worker thread/event loop so the browser never blocks.
"""

from __future__ import annotations

import asyncio
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from computronium import (
    create_backprop_mlp,
    create_eqprop_mlp,
    create_fa_mlp,
    create_fast_weight_mlp,
    create_ff_mlp,
    create_hebbian_mlp,
    create_pc_mlp,
    create_pepita_mlp,
    create_routing_mlp,
    create_snn_mlp,
    create_tile_mlp,
    create_tp_mlp,
)

# Ensure the full component registry is populated. `import computronium` is
# lazy, so model registration happens on first Registry access. Trigger it
# explicitly and up-front so the demo's Registry lookups (and any
# SystemTrainer instantiations by factory functions) behave deterministically.
from computronium.core.registry import ComponentCategory, Registry
from computronium.core.system_trainer import SystemTrainer, SystemTrainerConfig
from computronium.domains.registry import resolve_task
from computronium.utils import seed_everything

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class DemoPanel:
    """One side of the two-panel side-by-side comparison (Config A / Config B)."""

    trainer_config: SystemTrainerConfig
    epochs: int = 10
    model_name: str = "backprop_mlp"
    task_name: str = "mnist"
    hidden_dim: int | None = None
    lr: float = 0.001
    # Pre-composed System (ontology mode); overrides create_system.
    system: object | None = None
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


# Models the demo can train through the generic SystemTrainer path on the
# supported tasks. Uses 5-D ontology factory functions.
_FACTORIES: dict[str, Callable[..., object]] = {
    "backprop_mlp": create_backprop_mlp,
    "eqprop_mlp": create_eqprop_mlp,
    "fa_mlp": create_fa_mlp,
    "ff_mlp": create_ff_mlp,
    "pepita_mlp": create_pepita_mlp,
    "tp_mlp": create_tp_mlp,
    "pc_mlp": create_pc_mlp,
    "hebbian_mlp": create_hebbian_mlp,
    "snn_mlp": create_snn_mlp,
    "tile_mlp": create_tile_mlp,
    "routing_mlp": create_routing_mlp,
    "fast_weight_mlp": create_fast_weight_mlp,
}

# Factory kwargs beyond the shared (input_dim, hidden_dims, output_dim,
# lr, device) signature. Families absent from the table take `lr=0.001`.
_FACTORY_KWARGS: dict[str, dict[str, float | int]] = {
    "eqprop_mlp": {"beta": 0.1, "inference_steps": 20},
    "ff_mlp": {
        "layer_lr": 0.03,
        "classifier_lr": 0.01,
        "threshold": 2.0,
        "num_layers": 2,
    },
}

TRAINABLE_MODELS: tuple[str, ...] = tuple(_FACTORIES)


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


def create_system(
    model: str, task: str, hidden_dim: int | None, device: str, lr: float = 0.001
) -> object:
    """Create a System via the model's native factory function."""
    factory = _FACTORIES.get(model)
    if factory is None:
        raise ValueError(f"Unknown model: {model}")
    spec = resolve_task(task)
    input_dim = spec.input_dim
    if isinstance(input_dim, (tuple, list)):
        input_dim = math.prod(input_dim)
    if hidden_dim is None:
        hidden_dim = default_hidden_dim(model)

    kwargs: dict[str, float | int] = _FACTORY_KWARGS.get(model) or {"lr": lr}
    return factory(
        input_dim=input_dim,
        hidden_dims=(hidden_dim, hidden_dim),
        output_dim=spec.output_dim,
        device=device,
        **kwargs,
    )


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

        device = "cuda" if torch.cuda.is_available() else "cpu"
        system = panel.system or create_system(
            panel.model_name, panel.task_name, panel.hidden_dim, device, lr=panel.lr
        )
        from computronium.domains.factory import create_task

        task = create_task(panel.task_name, device=device, quick_mode=True)
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
        with trainer:
            for _ in range(panel.trainer_config.max_epochs):
                metrics = trainer.train_epoch()
                panel.losses.append(float(metrics["train_loss"]))
                acc = metrics.get("val_acc") or metrics.get("train_acc")
                panel.accuracies.append(float(acc))
                panel.energies.append(float(metrics.get("train_energy", 0.0)))
                for name, param in system.geometry.named_parameters():
                    if "weight" in name:
                        panel.weight_history.setdefault(name, []).append(
                            param.detach().float().cpu()
                        )
    except Exception as e:
        panel.error = str(e)
    finally:
        panel.running = False
        panel.finished = True


async def run_async(panel: DemoPanel) -> None:
    """Train a panel in a worker thread so the event loop stays responsive."""
    loop = asyncio.get_running_loop()
    # Use a dedicated thread pool executor that we can shut down
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        await loop.run_in_executor(executor, run_headless, panel)
    finally:
        executor.shutdown(wait=True)


def elapsed(last: float) -> float:
    """Return seconds since ``last`` (helper for UI pacing)."""
    return time.perf_counter() - last
