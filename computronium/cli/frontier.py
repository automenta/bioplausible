"""``biopl-frontier`` — plan §9/§11 Pareto-frontier report over a probe JSONL.

Reads an experiment report (the app hash append-only probe JSONL from
:class:`~computronium.experiment.report.Report`) and reports, per model:

* the **Pareto frontier** over ``(accuracy, total_flops, memory, time)``
  (plan §11), and
* the **cost_of_plausibility** of each bio rule vs a named backprop reference
  at matched accuracy.

This turns the experiment layer's raw probe tuples into the single composite
number that decides "deploy or not" for the autonomous pipeline (§8).

Usage::

    uv run biopl-frontier --report parity_mnist_trio.report.jsonl --backprop backprop_mlp
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from computronium.core.logging import get_logger
from computronium.hyperopt.frontier import (
    RulePoint,
    cost_of_plausibility,
    pareto_frontier,
)

logger = get_logger()

__all__ = ["load_report_points", "run_frontier_report"]

_VIABLE_THRESHOLD: float = 1.5
_CURIOSITY_THRESHOLD: float = 5.0


def load_report_points(report_path: str) -> list[RulePoint]:
    """Load ``RulePoint``s (ok probes only) from a probe JSONL.

    Args:
        report_path: Path to the append-only probe JSONL.

    Returns:
        One :class:`RulePoint` per successful probe.
    """
    points: list[RulePoint] = []
    for line in Path(report_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("status") != "ok":
            continue
        model = str(record.get("model", ""))
        config = record.get("config") or {}
        if not isinstance(config, dict):
            config = {}
        total_flops = int(record.get("forward_flops", 0) or 0) + int(
            record.get("backward_flops", 0) or 0
        )
        points.append(
            RulePoint(
                rule=model,
                accuracy=float(record.get("final_acc", 0.0)),
                total_flops=total_flops,
                peak_memory_mb=float(record.get("peak_memory_mb", 0.0)),
                wall_time_s=float(record.get("wall_time_s", 0.0)),
                config=tuple(sorted(config.items())),
            )
        )
    return points


def run_frontier_report(report_path: str, backprop: str) -> dict[str, object]:
    """Compute per-model frontiers and cost-of-plausibility for a report.

    Args:
        report_path: Probe JSONL path.
        backprop: Model name to treat as the ideal-backprop reference.

    Returns:
        Dict of ``models`` (per-model frontier stats) and
        ``cost_of_plausibility`` for each bio model.
    """
    points = load_report_points(report_path)
    if not points:
        return {"report": report_path, "n_probes": 0, "models": {}}

    by_model: dict[str, list[RulePoint]] = {}
    for p in points:
        by_model.setdefault(p.rule, []).append(p)

    models: dict[str, object] = {}
    for name, model_points in sorted(by_model.items()):
        front = pareto_frontier(model_points)
        best = sorted(model_points, key=lambda p: p.accuracy, reverse=True)[0]
        models[name] = {
            "n_probes": len(model_points),
            "n_frontier": len(front),
            "best_accuracy": best.accuracy,
            "best_total_flops": best.total_flops,
            "best_peak_memory_mb": best.peak_memory_mb,
            "best_wall_time_s": best.wall_time_s,
            "frontier": [
                {
                    "accuracy": p.accuracy,
                    "total_flops": p.total_flops,
                    "peak_memory_mb": p.peak_memory_mb,
                    "wall_time_s": p.wall_time_s,
                    "config": dict(p.config),
                }
                for p in front
            ],
        }

    backprop_points = by_model.get(backprop, [])
    costs: dict[str, float] = {}
    for name, model_points in by_model.items():
        if name == backprop:
            continue
        costs[name] = cost_of_plausibility(model_points, backprop_points)

    return {
        "report": report_path,
        "n_probes": len(points),
        "backprop": backprop,
        "models": models,
        "cost_of_plausibility": costs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", required=True, help="Probe JSONL path")
    parser.add_argument("--backprop", default="backprop_mlp")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    report = run_frontier_report(args.report, args.backprop)
    if not report["models"]:
        logger.error("No ok probes found in %s", args.report)
        return 1

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    logger.info("report: %s (%d probes)", args.report, report["n_probes"])
    for name, stats in sorted(report["models"].items()):
        logger.info(
            "model=%s acc=%.4f flops=%d mem=%.1fMB time=%.1fs frontier=%d/%d",
            name,
            stats["best_accuracy"],
            stats["best_total_flops"],
            stats["best_peak_memory_mb"],
            stats["best_wall_time_s"],
            stats["n_frontier"],
            stats["n_probes"],
        )
    for name, cost in sorted(report["cost_of_plausibility"].items()):
        match cost <= _VIABLE_THRESHOLD, cost >= _CURIOSITY_THRESHOLD:
            case True, _:
                label = "viable"
            case _, True:
                label = "curiosity"
            case _:
                label = "neutral"
        logger.info("cost_of_plausibility %s = %.2f (%s)", name, cost, label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
