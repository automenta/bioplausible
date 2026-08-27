"""``biopl-parity`` — Backprop-baseline parity CLI.

Trains two configurations via ``SystemTrainer`` under one global seed and reports
the final accuracy gap (percentage points).

Parity is ``(final_acc_B - final_acc_A) * 100`` where each ``final_acc`` is the
last per-epoch accuracy (preferring ``train_acc``, falling back to ``val_acc``).

Usage::

    uv run biopl parity --config-a backprop --config-b eqprop --task mnist
    uv run biopl parity --config-a fa --config-b backprop --seed 7 --json
"""

import argparse
import json
import logging
from typing import Protocol

from torch.utils.data import DataLoader

# Import zoo models to trigger registration
from computronium.core.logging import get_logger
from computronium.core.system_trainer import (
    SystemTrainer,
    SystemTrainerConfig,
    create_backprop_system,
    create_eqprop_system,
    create_fa_system,
)
from computronium.domains.factory import create_task
from computronium.domains.registry import SUPPORTED_TASKS, resolve_task
from computronium.utils import seed_everything

logger = get_logger()

_DEFAULT_TASK = "mnist"

# Map CLI config names to factory functions
_CONFIG_FACTORIES = {
    "backprop": create_backprop_system,
    "eqprop": create_eqprop_system,
    "fa": create_fa_system,
}


class _DataProvider(Protocol):
    """Protocol for data providers (DataLoader, etc.)."""

    def __iter__(self): ...
    def __len__(self) -> int: ...


def _per_epoch_accuracy(history: list[dict[str, float]]) -> list[float]:
    """Extract per-epoch accuracy using the demo's train-first rule."""
    out: list[float] = []
    for metrics in history:
        train = metrics.get("train_acc")
        val = metrics.get("val_acc")
        match train is not None, val is not None:
            case True, _:
                acc = float(train)
            case _, True:
                acc = float(val)
            case _:
                acc = float("nan")
        out.append(acc)
    return out


def _get_factory(name: str):
    """Get system factory by name."""
    if name not in _CONFIG_FACTORIES:
        available = ", ".join(_CONFIG_FACTORIES.keys())
        raise ValueError(f"Unknown config '{name}'. Available: {available}")
    return _CONFIG_FACTORIES[name]


class _FlattenLoader:
    """Wrapper that flattens input tensors from a DataLoader."""

    def __init__(self, loader: DataLoader):
        self.loader = loader

    def __iter__(self):
        for x, y in self.loader:
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            yield x, y

    def __len__(self) -> int:
        return len(self.loader)


def _make_dataloaders(
    task_name: str, batch_size: int, device: str
) -> tuple[_FlattenLoader, _FlattenLoader]:
    """Create train and validation DataLoaders for a task with flattening."""
    task = create_task(task_name, device=device, quick_mode=True)
    task.setup()
    train_loader = _FlattenLoader(task.get_dataloader("train"))
    val_loader = _FlattenLoader(task.get_dataloader("val"))
    return train_loader, val_loader


def run_parity(
    config_a: str,
    config_b: str,
    task: str,
    epochs: int,
    lr: float,
    hidden_dim: int,
    seed: int,
    device: str = "cpu",
) -> dict[str, object]:
    """Train both configs under one seed; return the parity report dict."""
    if task not in SUPPORTED_TASKS:
        raise ValueError(
            f"task '{task}' not supported (need one of {sorted(SUPPORTED_TASKS)})"
        )
    spec = resolve_task(task)

    factory_a = _get_factory(config_a)
    factory_b = _get_factory(config_b)

    accs: dict[str, float] = {}
    for name, factory in ((config_a, factory_a), (config_b, factory_b)):
        seed_everything(seed, device)

        system = factory(
            input_dim=spec.input_dim,
            hidden_dim=hidden_dim,
            output_dim=spec.output_dim,
            lr=lr,
        )

        train_loader, val_loader = _make_dataloaders(task, 64, device)

        trainer_config = SystemTrainerConfig(
            max_epochs=epochs,
            batch_size=64,
            device=device,
            seed=seed,
        )

        trainer = SystemTrainer(
            system=system,
            config=trainer_config,
            train_data=train_loader,
            val_data=val_loader,
        )

        history = trainer.fit()
        per_epoch = _per_epoch_accuracy(history)
        accs[name] = per_epoch[-1] if per_epoch else float("nan")

    gap_pp = round((accs[config_b] - accs[config_a]) * 100, 3)
    return {
        "config_a": config_a,
        "config_b": config_b,
        "task": task,
        "epochs": epochs,
        "lr": lr,
        "hidden_dim": hidden_dim,
        "seed": seed,
        "accuracy_a": round(accs[config_a], 4),
        "accuracy_b": round(accs[config_b], 4),
        "gap_pp": gap_pp,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config-a", default="backprop", help="First config (baseline)"
    )
    parser.add_argument("--config-b", default="eqprop", help="Second config (compare)")
    parser.add_argument("--task", default=_DEFAULT_TASK)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden", dest="hidden_dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    try:
        report = run_parity(
            args.config_a,
            args.config_b,
            args.task,
            args.epochs,
            args.lr,
            args.hidden_dim,
            args.seed,
            args.device,
        )
    except Exception as e:
        logger.error("biopl-parity failed: %s", e)
        return 2

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        logger.info(
            "%s=%.4f vs %s=%.4f -> gap %.1f pp",
            args.config_a,
            report["accuracy_a"],
            args.config_b,
            report["accuracy_b"],
            report["gap_pp"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
