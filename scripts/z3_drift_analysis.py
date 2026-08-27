"""Offline mechanistic read of controller drift across Z3 adaptation phases.

First-pass analysis over PERSISTED artifacts only (no rerun):

- v3 proportion run (`benchmark_results/z3_proportion/`): order-cell
  outcome structure, pre-adaptation prior vs outcome, speed-window log
  ratios;
- R4/R5 repair-round artifacts (`benchmark_results/z3_r4_probe/`,
  `benchmark_results/z3_meta_repair/round5.json`): flat-at-chance curve
  signatures distinguishing exploration failure from optimization failure.

Known gap (recorded in findings): neither the v2 nor the v3 run persisted
per-step accuracy curves or gate/operator histograms, so the bandit's
operator distribution DURING the parity phase is not recoverable offline.
`_run_adaptation` now records a gate history per task, making the full
mechanistic read available from the next run at zero extra compute.

Writes `benchmark_results/z3_drift_analysis/findings.json`.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

V3_RESULTS = Path("benchmark_results/z3_proportion/z3_proportion_results.json")
R4_RESULTS = Path("benchmark_results/z3_r4_probe/round4.json")
R5_RESULTS = Path("benchmark_results/z3_meta_repair/round5.json")
TASKS = ("parity", "last_symbol", "threshold")
ACCURACY_FLOOR = 0.95
_CHANCE_THRESHOLD = 0.6


def _order_cell(order: list[str]) -> str:
    """Compact signature: tasks before parity, joined."""
    i = order.index("parity")
    return "+".join(t[:4] for t in order[:i]) or "first"


def _v3_order_structure(seeds: list[dict]) -> dict:
    """Per (arm, cell) solve/fail counts restricted to parity."""
    cells: dict[str, dict[str, dict[str, list[int]]]] = {}
    for row in seeds:
        cell = _order_cell(row["task_order"])
        for arm, accs in (
            ("z3", row["accuracies"]),
            ("random", row["random_accuracies"]),
        ):
            arm_cells = cells.setdefault(arm, {}).setdefault(
                cell, {"solve": [], "fail": []}
            )
            key = "solve" if accs["parity"] >= ACCURACY_FLOOR else "fail"
            arm_cells[key].append(row["seed"])
    return {
        arm: {
            cell: {
                "solved_seeds": sorted(v["solve"]),
                "failed_seeds": sorted(v["fail"]),
            }
            for cell, v in sorted(per_cell.items())
        }
        for arm, per_cell in cells.items()
    }


def _v3_pre_adapt_vs_outcome(seeds: list[dict]) -> dict:
    """Pre-adaptation prior and outcome per task, pooled across arms/seeds."""
    rows = []
    for row in seeds:
        for arm, accs in (
            ("z3", row["accuracies"]),
            ("random", row["random_accuracies"]),
        ):
            for task in TASKS:
                rows.append({
                    "arm": arm,
                    "seed": row["seed"],
                    "task": task,
                    "pre_adapt": row["pre_adapt_accuracy"][task],
                    "final": accs[task],
                })
    per_task = {}

    def _mean_pre_adapt(rows: list[dict]) -> float | None:
        return sum(r["pre_adapt"] for r in rows) / len(rows) if rows else None

    for task in TASKS:
        subset = [r for r in rows if r["task"] == task]
        solved = [r for r in subset if r["final"] >= ACCURACY_FLOOR]
        failed = [r for r in subset if r["final"] < ACCURACY_FLOOR]
        per_task[task] = {
            "solves": len(solved),
            "fails": len(failed),
            "mean_pre_adapt_when_solved": _mean_pre_adapt(solved),
            "mean_pre_adapt_when_failed": _mean_pre_adapt(failed),
        }
    return per_task


def _v3_speed_null(seeds: list[dict]) -> dict:
    """Per-task speed log ratios z3-vs-finetune at each registered window."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for window in ("w20", "w50", "w100"):
        per_task: dict[str, list[float]] = {t: [] for t in TASKS}
        for row in seeds:
            for task, ratio in row["speed_log_ratios"][window].items():
                per_task[task].append(ratio)
        out[window] = {
            task: {
                "mean": sum(v) / len(v),
                "min": min(v),
                "max": max(v),
            }
            for task, v in per_task.items()
        }
    return out


def _flat_chance_signature(curve: list[float], *, tail: int = 40) -> float:
    """Mean of the trailing segment — near 0.5 flags flat-at-chance curves."""
    tail_vals = curve[-tail:]
    return sum(tail_vals) / len(tail_vals)


def _round_curve_signatures(path: Path) -> dict:
    """Flat-at-chance tails per config/task from repair-round artifacts."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for result in data.get("results", []):
        curves_by_seed = result.get("seeds") or result.get("curves_by_seed") or []
        per_task: dict[str, list[float]] = {}
        for seed_entry in curves_by_seed:
            curves = seed_entry.get("curves", {})
            for task, curve in (curves.get("meta") or {}).items():
                per_task.setdefault(task, []).append(_flat_chance_signature(curve))
        out[result["config"]] = {
            task: {
                "tail_mean": sum(v) / len(v),
                "n_seeds": len(v),
                "exploration_failure": sum(v) / len(v) < _CHANCE_THRESHOLD,
            }
            for task, v in sorted(per_task.items())
        }
    return out


def run() -> dict:
    v3 = json.loads(V3_RESULTS.read_text(encoding="utf-8"))
    seeds = v3["seeds"]
    findings = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": [str(V3_RESULTS), str(R4_RESULTS), str(R5_RESULTS)],
        "data_gap": (
            "v2/v3 artifacts persist final accuracies, pre-adapt priors and "
            "windowed speed summaries only - no per-step curves or gate "
            "histograms; operator distributions during each adaptation phase "
            "are not recoverable offline. Gate-history recording is wired into "
            "_run_adaptation for subsequent runs."
        ),
        "parity_order_cells": _v3_order_structure(seeds),
        "pre_adapt_vs_outcome": _v3_pre_adapt_vs_outcome(seeds),
        "speed_log_ratios": _v3_speed_null(seeds),
        "r4_curve_signatures": _round_curve_signatures(R4_RESULTS),
        "r5_curve_signatures": _round_curve_signatures(R5_RESULTS),
        "fisher_endpoint": v3["primary_fisher_exact"],
    }

    failed_pre = findings["pre_adapt_vs_outcome"]["parity"]
    logger.info(
        "parity: %d solves / %d fails; pre-adapt prior %.3f (solved) vs %.3f (failed)",
        failed_pre["solves"],
        failed_pre["fails"],
        failed_pre["mean_pre_adapt_when_solved"] or float("nan"),
        failed_pre["mean_pre_adapt_when_failed"] or float("nan"),
    )
    for arm, cells in findings["parity_order_cells"].items():
        for cell, outcome in cells.items():
            logger.info(
                "%s cell[%s]: solved=%s failed=%s",
                arm,
                cell,
                outcome["solved_seeds"],
                outcome["failed_seeds"],
            )
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/z3_drift_analysis"),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    findings = run()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "findings.json"
    out.write_text(json.dumps(findings, indent=2))
    logger.info("→ %s", out)


if __name__ == "__main__":
    main()
