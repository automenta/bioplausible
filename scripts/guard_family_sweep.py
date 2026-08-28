"""Guard τ-recalibration sweep over real settling families (PR-5 confirmatory).

For every composed settling coordinate (substrates x settling dynamics at
the Z3 family plasticity/credit/update), measures:

- ``windowed_growth`` at the deployed default τ (record-only, no kill) —
  confirms the PR-5 ROC point stays lossless now that the substrate kill
  set is empty;
- fast-proxy vs full-Jacobian spectral-radius disagreement
  (:func:`quantify_proxy_disagreement`) — the family-specific bias number
  PR-5 deferred ("fast_proxy INFEASIBLE on non-normal systems");
- probe overhead relative to a real transition step.

Writes ``benchmark_results/stability_guard_calibration/family_sweep.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from computronium.core.campaign.evaluation import (
    DEFAULT_GUARD_TAU,
    activity_transition,
    build_coordinate_system,
    episode_batch,
)
from computronium.core.stability.guard import (
    ProbeSpec,
    StabilityGuard,
    measure_guard_overhead,
    quantify_proxy_disagreement,
)
from computronium.state import CompositeState

logger = logging.getLogger(__name__)

SUBSTRATES = (
    "digital",
    "analog",
    "memristive",
    "neuromorphic",
    "sparse",
    "ternary",
    "optical",
    "quantum",
)
SETTLING_DYNAMICS = ("energy_minimization", "predictive_settling")
FAMILY_SUFFIX = "feedforward/{dyn}/rule_state/thermodynamic_contrast/euclidean"


def _sweep_coordinate(coordinate: str, episode: int = 0) -> dict:
    joint = build_coordinate_system(coordinate)
    x, y = episode_batch(episode)
    metrics = joint.train_step(x, y)
    z = CompositeState(activity={"x": x}, plastic={}, substrate={})
    transition = activity_transition(joint)

    guard = StabilityGuard(threshold=float("inf"), statistic="windowed_growth")
    started = time.perf_counter()
    growth = guard.probe(transition, z, joint.context)
    windowed_seconds = time.perf_counter() - started

    disagreement = quantify_proxy_disagreement(
        transition, z, joint.context, probes=ProbeSpec(n_probes=20, seed=episode)
    )
    overhead = measure_guard_overhead(transition, z, joint.context, guard, n_steps=20)

    return {
        "coordinate": coordinate,
        "substrate": coordinate.split("/")[0],
        "dynamics": coordinate.split("/")[2],
        "train_metrics": {k: float(v) for k, v in metrics.items()},
        "windowed_growth": growth,
        "guard_would_kill_at_default_tau": bool(growth > DEFAULT_GUARD_TAU),
        "default_tau": DEFAULT_GUARD_TAU,
        "proxy_disagreement": {
            "mean_relative_error": disagreement.mean_relative_error,
            "median_relative_error": disagreement.median_relative_error,
            "p95_relative_error": disagreement.p95_relative_error,
            "pearson_correlation": disagreement.pearson_correlation,
            "median_absolute_error": disagreement.median_absolute_error,
            "mean_absolute_error": disagreement.mean_absolute_error,
            "median_reference_norm": disagreement.median_reference_norm,
        },
        "overhead_probe_per_step": overhead,
        "probe_seconds": windowed_seconds,
    }


def run() -> dict:
    coordinates = [
        f"{sub}/{FAMILY_SUFFIX.format(dyn=dyn)}"
        for sub in SUBSTRATES
        for dyn in SETTLING_DYNAMICS
    ]
    rows: list[dict] = []
    skipped: list[dict] = []
    for coordinate in coordinates:
        try:
            row = _sweep_coordinate(coordinate)
        except Exception as exc:  # sweep must survive one bad family
            logger.warning("skip %s: %s", coordinate, exc)
            skipped.append({"coordinate": coordinate, "error": str(exc)})
            continue
        rows.append(row)
        logger.info(
            "%s: growth=%.4f kill=%s proxy_med_err=%.2f corr=%.2f",
            coordinate,
            row["windowed_growth"],
            row["guard_would_kill_at_default_tau"],
            row["proxy_disagreement"]["median_relative_error"],
            row["proxy_disagreement"]["pearson_correlation"],
        )

    growths = [r["windowed_growth"] for r in rows]
    summary = {
        "n_coordinates": len(rows),
        "n_skipped": len(skipped),
        "max_windowed_growth": max(growths) if growths else None,
        "false_kill_rate_at_default_tau": (
            sum(r["guard_would_kill_at_default_tau"] for r in rows) / len(rows)
            if rows
            else None
        ),
        "worst_proxy_median_relative_error": max(
            (r["proxy_disagreement"]["median_relative_error"] for r in rows),
            default=None,
        ),
        "tau_lossless": all(not r["guard_would_kill_at_default_tau"] for r in rows),
    }
    return {"summary": summary, "coordinates": rows, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/stability_guard_calibration"),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    payload = run()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "family_sweep.json"
    out.write_text(json.dumps(payload, indent=2))
    logger.info("summary=%s", json.dumps(payload["summary"]))
    logger.info("→ %s", out)


if __name__ == "__main__":
    main()
