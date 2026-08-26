"""EqProp competitive verification on MNIST (TODO4 §7.2).

Trains the canonical EqProp ontology coordinate (``MODEL_CONFIGS["eqprop"]``
via ``create_eqprop_system``) for a full 20-epoch schedule and reports
test accuracy against the >80% target.

Usage:
    python -m computronium.experiments.joint.eqprop_mnist [--epochs 20] [--quick]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
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


def train_eqprop_mnist(config: EqPropMnistConfig) -> dict[str, object]:
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
            logger.info(
                "epoch %d: train_loss=%.4f val_acc=%.4f",
                metrics.get("epoch", -1),
                metrics.get("train_loss", float("nan")),
                metrics.get("val_acc", 0.0),
            )
            if not math.isfinite(metrics.get("train_loss", float("nan"))):
                aborted = f"non-finite loss at epoch {metrics.get('epoch')}"
                break

    elapsed = time.time() - start
    best_acc = max((h.get("val_acc", 0.0) for h in history), default=0.0)
    return {
        "config": asdict(config),
        "param_count": sum(p.numel() for p in system.geometry.params.values()),
        "device": str(trainer.device),
        "elapsed_s": elapsed,
        "best_val_acc": best_acc,
        "final_val_acc": history[-1].get("val_acc", 0.0) if history else 0.0,
        "target_met": bool(history)
        and history[-1].get("val_acc", 0.0) >= _ACCURACY_TARGET,
        "aborted": aborted,
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
        "EqProp MNIST complete: best_val_acc=%.4f target_met=%s aborted=%s (%.1fs)",
        result["best_val_acc"],
        result["target_met"],
        result["aborted"],
        result["elapsed_s"],
    )


if __name__ == "__main__":
    main()
