"""``biopl-parity`` — Sprint 3.7 backprop-baseline parity CLI.

Trains two configurations via ``CoreTrainer`` under one global seed and reports
the final accuracy gap (percentage points), matching the demo's
``charts.parity_gap`` definition so the UI and the CLI cross-check.

Parity is ``(final_acc_B - final_acc_A) * 100`` where each ``final_acc`` is the
last per-epoch accuracy (preferring ``train_accuracy``, falling back to
``val_accuracy`` — the same rule the demo's telemetry callback uses).

Usage::

    uv run biopl-parity --config-a equitile --config-b backprop_mlp --task mnist
    uv run biopl-parity --config-a pepita --config-b backprop_mlp --seed 7 --json
"""

import argparse
import json
import logging

# Register the models the parity CLI trains against. With Sprint 0.5 lazy
# imports, importing the top-level package is no longer a registration side
# effect; both the zoo (pepita/FF/FA/eqprop_mlp/backprop_mlp) and the separate
# equitile package must be imported explicitly.
import bioplausible.equitile  # ruff: ignore[unused-import]
import bioplausible.zoo  # ruff: ignore[unused-import]
from bioplausible.core.trainer import CoreTrainer, TrainerConfig
from bioplausible.utils import set_global_seed

logger = logging.getLogger(__name__)

# input_dim/output_dim per task — mirrors demo/runner.py `_TASK_DIMS`.
_TASK_DIMS: dict[str, tuple[int, int]] = {
    "xor": (2, 2),
    "spiral": (2, 2),
    "circles": (2, 2),
    "digits": (64, 10),
    "mnist": (784, 10),
    "cifar10": (3072, 10),
    "tiny_shakespeare": (16, 16),
}

_DEFAULT_TASK = "mnist"


def _per_epoch_accuracy(history: list[object]) -> list[float]:
    """Extract per-epoch accuracy using the demo's train-first rule."""
    out: list[float] = []
    for metrics in history:
        train = getattr(metrics, "train_accuracy", None)
        val = getattr(metrics, "val_accuracy", None)
        if train is not None:
            acc = float(train)
        elif val is not None:
            acc = float(val)
        else:
            acc = float("nan")
        out.append(acc)
    return out


def run_parity(
    model_a: str,
    model_b: str,
    task: str,
    epochs: int,
    lr: float,
    hidden_dim: int,
    seed: int,
    device: str = "cpu",
) -> dict[str, object]:
    """Train both configs under one seed; return the parity report dict."""
    if task not in _TASK_DIMS:
        raise ValueError(
            f"task '{task}' not supported by CoreTrainer parity (need one of "
            f"{sorted(_TASK_DIMS)})"
        )
    input_dim, output_dim = _TASK_DIMS[task]
    model_kwargs = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
    }

    accs: dict[str, float] = {}
    for name in (model_a, model_b):
        set_global_seed(seed, device)
        cfg = TrainerConfig(
            model=name,
            model_kwargs=dict(model_kwargs),
            task=task,
            epochs=epochs,
            optimizer_kwargs={"lr": lr},
        )
        history = CoreTrainer(cfg).fit()
        per_epoch = _per_epoch_accuracy(history)
        accs[name] = per_epoch[-1] if per_epoch else float("nan")

    gap_pp = round((accs[model_b] - accs[model_a]) * 100, 3)
    return {
        "config_a": model_a,
        "config_b": model_b,
        "task": task,
        "epochs": epochs,
        "lr": lr,
        "hidden_dim": hidden_dim,
        "seed": seed,
        "accuracy_a": round(accs[model_a], 4),
        "accuracy_b": round(accs[model_b], 4),
        "gap_pp": gap_pp,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config-a", default="equitile")
    parser.add_argument("--config-b", default="backprop_mlp")
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
    except Exception as e:  # broad: report any parity failure
        logger.error("biopl-parity failed: %s", e)
        return 2

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        logger.info(
            "%s=%s vs %s=%s → gap %.1f pp",
            args.config_a,
            report["accuracy_a"],
            args.config_b,
            report["accuracy_b"],
            report["gap_pp"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
