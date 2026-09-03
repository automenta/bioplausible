"""EqProp competitive verification on MNIST (TODO4 §7.2).

Trains the canonical EqProp ontology coordinate (``MODEL_CONFIGS["eqprop"]``
via ``create_eqprop_system``) for a full 20-epoch schedule with per-epoch LR
decay and val-based early stopping (late-drift fix), and reports test
accuracy against the >80% target.

Usage:
    python -m computronium.experiments.joint.eqprop_mnist [--epochs 20] [--quick]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from torch.utils.data import DataLoader, Subset

from computronium.core.system_trainer import (
    SystemTrainer,
    SystemTrainerConfig,
    create_eqprop_system,
)
from computronium.data.vision import get_vision_dataset
from computronium.experiments.eqprop_vision_parity import MODEL_CONFIGS
from computronium.utils import seed_everything

logger = logging.getLogger(__name__)

_ACCURACY_TARGET = 0.80


@dataclass(frozen=True, slots=True)
class EqPropMnistConfig:
    """Locked-in 7.2 configuration (defaults mirror ``MODEL_CONFIGS['eqprop']``)."""

    hidden_dim: int = 512
    num_layers: int = 3
    beta: float = 0.1
    settle_steps: int = 20
    step_size: float = 0.1
    lr: float = 0.001
    grad_clip: float = 1.0
    momentum: float = 0.0
    batch_size: int = 128
    epochs: int = 20
    lr_gamma: float = 0.9
    lr_decay_start: int = 3
    early_stop_patience: int = 4
    seed: int = 42
    device: str = "auto"
    max_train_samples: int | None = None
    output_dir: str = "results/eqprop_mnist"

    @classmethod
    def from_parity_config(cls, **overrides: object) -> EqPropMnistConfig:
        """Build from the canonical ``MODEL_CONFIGS['eqprop']`` entry."""
        cfg = MODEL_CONFIGS["eqprop"]
        defaults: dict[str, object] = {
            "hidden_dim": cfg["hidden_dim"],
            "num_layers": cfg["num_layers"],
            "beta": cfg["beta"],
            "settle_steps": cfg["inference_steps"],
            "step_size": cfg["step_size"],
        }
        return cls(**{**defaults, **overrides})  # type: ignore[arg-type]


def _loaders(
    config: EqPropMnistConfig,
) -> tuple[DataLoader, DataLoader]:
    train_ds = get_vision_dataset("mnist", flatten=True, train=True)
    test_ds = get_vision_dataset("mnist", flatten=True, train=False)
    if config.max_train_samples is not None:
        train_ds = Subset(train_ds, range(config.max_train_samples))
    return (
        DataLoader(train_ds, batch_size=config.batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=config.batch_size),
    )


def _decay_lr(system: object, gamma: float, epoch: int, start: int) -> None:
    """Multiplicative per-epoch LR decay on the system's Euclidean update.

    Counters late-phase drift: a constant step size keeps random-walking past
    the accuracy peak while energy keeps falling (objective misalignment).
    """
    if gamma >= 1.0 or epoch < start:
        return
    update = getattr(system, "update", None)
    config = getattr(update, "config", None)
    step_size = getattr(config, "step_size", None)
    if update is None or config is None or not isinstance(step_size, int | float):
        return
    update.config = replace(config, step_size=float(step_size) * gamma)


def train_eqprop_mnist(config: EqPropMnistConfig) -> dict[str, object]:  # ruff: ignore[too-many-locals]
    """Run the full training schedule and return the result record."""
    seed_everything(config.seed)

    system = create_eqprop_system(
        input_dim=784,
        hidden_dim=config.hidden_dim,
        output_dim=10,
        num_layers=config.num_layers,
        beta=config.beta,
        settle_steps=config.settle_steps,
        lr=config.lr,
        update_momentum=config.momentum,
    )
    train_loader, test_loader = _loaders(config)

    history: list[dict[str, float]] = []
    aborted: str | None = None
    early_stopped: str | None = None
    best_acc = 0.0
    best_epoch = -1
    epochs_since_best = 0
    start = time.time()

    with SystemTrainer(
        system=system,
        config=SystemTrainerConfig(
            max_epochs=config.epochs,
            batch_size=config.batch_size,
            device=config.device,
            grad_clip=config.grad_clip,
            seed=config.seed,
        ),
        train_data=train_loader,
        val_data=test_loader,
    ) as trainer:
        for _ in range(config.epochs):
            metrics = trainer.train_epoch()
            history.append(metrics)
            val_acc = metrics.get("val_acc", 0.0)
            epoch = int(metrics.get("epoch", -1))
            if val_acc > best_acc:
                best_acc, best_epoch, epochs_since_best = val_acc, epoch, 0
            else:
                epochs_since_best += 1
            logger.info(
                "epoch %d: train_loss=%.4f val_acc=%.4f (best %.4f @%d)",
                epoch,
                metrics.get("train_loss", float("nan")),
                val_acc,
                best_acc,
                best_epoch,
            )
            if not math.isfinite(metrics.get("train_loss", float("nan"))):
                aborted = f"non-finite loss at epoch {epoch}"
                break
            patience = config.early_stop_patience
            if patience > 0 and epochs_since_best >= patience:
                early_stopped = (
                    f"no val improvement for {patience} epochs (best ep{best_epoch})"
                )
                break
            _decay_lr(system, config.lr_gamma, epoch, config.lr_decay_start)

    elapsed = time.time() - start
    final_val_acc = history[-1].get("val_acc", 0.0) if history else 0.0
    return {
        "config": asdict(config),
        "param_count": sum(p.numel() for p in system.geometry.params.values()),
        "device": str(trainer.device),
        "elapsed_s": elapsed,
        "best_val_acc": best_acc,
        "best_epoch": best_epoch,
        "final_val_acc": final_val_acc,
        "target_met": bool(history) and final_val_acc >= _ACCURACY_TARGET,
        "aborted": aborted,
        "early_stopped": early_stopped,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto | cuda | cpu")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Cap training set size (smoke runs)",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke profile: 1 epoch on a small sample cap",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s"
    )

    overrides: dict[str, object] = {}
    if args.quick:
        overrides |= {"epochs": 1, "max_train_samples": 2048}
    for name, value in (
        ("epochs", args.epochs),
        ("batch_size", args.batch_size),
        ("lr", args.lr),
        ("seed", args.seed),
        ("device", args.device),
        ("max_train_samples", args.max_train_samples),
        ("output_dir", args.output_dir),
    ):
        if value is not None:
            overrides[name] = value

    config = EqPropMnistConfig.from_parity_config(**overrides)
    result = train_eqprop_mnist(config)

    output_path = Path(str(config.output_dir))
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "results.json").open("w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info(
        "EqProp MNIST complete: best_val_acc=%.4f @ep%s final=%.4f target_met=%s "
        "aborted=%s early_stopped=%s (%.1fs)",
        result["best_val_acc"],
        result["best_epoch"],
        result["final_val_acc"],
        result["target_met"],
        result["aborted"],
        result["early_stopped"],
        result["elapsed_s"],
    )


if __name__ == "__main__":
    main()
