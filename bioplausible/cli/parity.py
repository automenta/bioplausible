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
from bioplausible.core.logging import get_logger
from bioplausible.core.trainer import CoreTrainer, TrainerConfig
from bioplausible.domains.registry import SUPPORTED_TASKS, resolve_task
from bioplausible.utils import seed_everything

logger = get_logger()

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


def run_parity(  # ruff: ignore[too-many-arguments,too-many-positional-arguments]  (parity CLI signature is the public report contract)
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
    if task not in SUPPORTED_TASKS:
        raise ValueError(  # ruff: ignore[raise-vanilla-args]  # descriptive message is the public API
            f"task '{task}' not supported by CoreTrainer parity (need one of "
            f"{sorted(SUPPORTED_TASKS)})"
        )
    spec = resolve_task(task)
    model_kwargs = {
        "input_dim": spec.input_dim,
        "hidden_dim": hidden_dim,
        "output_dim": spec.output_dim,
    }

    accs: dict[str, float] = {}
    for name in (model_a, model_b):
        seed_everything(seed, device)
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


def _run_campaign_stage(
    config: str,
    stage_name: str,
    report_path: str,
    device: str,
) -> list[object]:
    """Drive one campaign stage through the experiment staircase (4.1.1).

    Loads the campaign, isolates the named stage, and runs just that rung via
    :class:`StaircaseRunner`, appending every probe to the Report JSONL.
    """
    from bioplausible.experiment.probe import CoreTrainerDriver
    from bioplausible.experiment.producer import HyperoptGridProducer
    from bioplausible.experiment.report import Report
    from bioplausible.experiment.schema import load_campaign
    from bioplausible.experiment.staircase import StaircaseRunner

    campaign = load_campaign(config)
    stage = next(s for s in campaign.stages if s.name == stage_name)
    single = campaign.model_copy(update={"stages": [stage]})
    wants_energy = bool(stage.energy)
    track = single.compute.track
    report = Report(report_path)
    runner = StaircaseRunner(
        single,
        report,
        CoreTrainerDriver(
            num_workers=single.compute.num_workers,
            track_energy=wants_energy,
            track_flops=track.flops,
            track_memory=track.memory,
        ),
        HyperoptGridProducer(seed=campaign.reproducibility.seed),
        compute=single.compute,
    )
    if runner.compute is not None and device:
        runner.compute.device = device
    return list(runner.run())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config-a", default="tile_pc")
    parser.add_argument("--config-b", default="backprop_mlp")
    parser.add_argument("--task", default=_DEFAULT_TASK)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden", dest="hidden_dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--campaign",
        default=None,
        help="Campaign YAML to drive a parity stage through the experiment layer",
    )
    parser.add_argument(
        "--stage", default="parity", help="Stage name to run from --campaign"
    )
    parser.add_argument("--report", default=None, help="Report JSONL path")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    if args.campaign:
        report_path = args.report or f"{args.campaign}.report.jsonl"
        try:
            outcomes = _run_campaign_stage(
                args.campaign, args.stage, report_path, args.device
            )
        except Exception as e:  # broad: report any parity campaign failure
            logger.error("biopl-parity campaign failed: %s", e)  # ruff: ignore[error-instead-of-exception]  # user-facing CLI: a traceback is noise
            return 2
        if args.json:
            print(
                json.dumps(
                    {
                        "campaign": args.campaign,
                        "stage": args.stage,
                        "report": str(report_path),
                        "n_probes": sum(len(o.metrics.results) for o in outcomes),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            logger.info(
                "%s: %d probes recorded to %s",
                args.stage,
                sum(len(o.metrics.results) for o in outcomes),
                report_path,
            )
        return 0
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
        logger.error("biopl-parity failed: %s", e)  # ruff: ignore[error-instead-of-exception]  # user-facing CLI: a traceback is noise
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
