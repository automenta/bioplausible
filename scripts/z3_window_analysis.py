"""Offline re-analysis of Z3 adaptation curves at alternative window sizes.

The registered 100-step window floors every arm at ≥100 steps, which cannot
resolve speed differences between arms that converge near the floor. This
tool recomputes steps-to-criterion from saved per-step accuracy curves at
candidate window sizes. Used for the 2026-08-26 E-1 instrument redesign
(DECISIONS.md): the speed endpoint proved unconfirmable at every tested
window, motivating the capability re-registration.

Reads ``benchmark_results/z3_meta_repair/round{N}.json`` (repair-round shape,
curves under ``seeds[].curves``) or ``run_z3_suite`` result JSONs.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from computronium.experiments.joint.z3_fixed_weights import _windowed_criterion_step

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

ARMS = ("meta", "random", "finetune")
TASKS = ("parity", "last_symbol", "threshold")


def _iter_seed_rows(path: Path) -> Iterator[tuple[str, int, dict]]:
    """Yield (label, seed_index, curves-by-arm) across known artifact shapes."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):  # run_z3_suite: [{coordinate, seeds: [...]}]
        for coord in payload:
            for i, row in enumerate(coord["seeds"]):
                yield str(coord.get("coordinate", "?")), i, _suite_curves(row)
        return
    for result in payload.get("results", []):  # repair rounds
        for i, row in enumerate(result["seeds"]):
            label = f"{result['config']}"
            curves = row.get("curves")
            if curves is None:
                continue
            yield (
                label,
                i,
                {
                    arm: {t: c for t, c in curves.get(arm, {}).items() if c}
                    for arm in ARMS
                },
            )


def _suite_curves(row: dict) -> dict[str, dict[str, list[float]]]:
    """Extract curves from a run_z3_suite seed row."""
    out: dict[str, dict[str, list[float]]] = {
        "meta": {t: r["accuracy_curve"] for t, r in row["tasks"].items()}
    }
    baselines = row.get("baselines")
    if baselines:
        out["random"] = {
            t: r.get("accuracy_curve")
            for t, r in baselines["random_psi"]["tasks"].items()
        }
        out["finetune"] = baselines["finetune_forgetting"].get("accuracy_curves")
    return out


def analyze(paths: list[Path], windows: list[int], threshold: float) -> dict:
    """Criterion steps per source/arm/task at each candidate window."""
    report: dict = {"threshold": threshold, "windows": windows, "sources": {}}
    for path in paths:
        for label, seed_i, arms in _iter_seed_rows(path):
            src = report["sources"].setdefault(f"{path.name}:{label}", {})
            out = src.setdefault(f"seed{seed_i}", {})
            for arm, tasks in arms.items():
                out[arm] = {
                    task: {
                        w: _windowed_criterion_step(
                            curve, window=w, threshold=threshold
                        )
                        for w in windows
                    }
                    for task, curve in tasks.items()
                }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--windows", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--threshold", type=float, default=0.98)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(json.dumps(analyze(args.artifacts, args.windows, args.threshold), indent=2))


if __name__ == "__main__":
    main()
