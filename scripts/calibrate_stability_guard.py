"""PR-5 stability-guard calibration driver.

Generates known-good/known-bad runs from a non-normal linear family
(Ginibre ensemble), labels them by unrolled divergence, then ROC-calibrates
guard kill thresholds and quantifies proxy-vs-full-Jacobian disagreement.
Writes `benchmark_results/stability_guard_calibration/calibration.json`.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from computronium.core.joint.context import SystemContext
from computronium.core.joint.state import CompositeState
from computronium.core.joint.transition import PlasticityConfig
from computronium.core.stability import (
    StabilityGuard,
    calibrate_threshold,
    measure_guard_overhead,
    quantify_proxy_disagreement,
)
from computronium.core.stability.guard import ProbeSpec

if TYPE_CHECKING:
    from computronium.core.stability.guard import StatisticKind

logger = logging.getLogger(__name__)

GAINS = (0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4)
SEEDS_PER_GAIN = 4
UNROLL_STEPS = 200
EXPLOSION_FACTOR = 1e3
STATISTIC_KINDS: tuple[StatisticKind, ...] = ("fast_proxy", "windowed_growth")


def _synthetic_context(dim: int) -> SystemContext:
    from computronium.core.joint.state import StateRegistry, StateVariable
    from computronium.core.ontology import (
        CreditAssignmentConfig,
        DigitalSubstrate,
        GeometryConfig,
        ParameterUpdateConfig,
        RecurrentGeometry,
        StateDynamicsConfig,
        SubstrateConfig,
    )

    geometry_config = GeometryConfig.feedforward(
        input_dim=dim, output_dim=2, hidden_dims=(dim,)
    )
    geometry = RecurrentGeometry(geometry_config)
    registry = StateRegistry()
    for name in geometry.params:
        registry.register(StateVariable(name=name, persistent=True))
    registry.register(StateVariable(name="x", persistent=True))

    return SystemContext(
        theta=geometry.params,
        geometry=geometry,
        substrate=DigitalSubstrate(),
        substrate_config=SubstrateConfig.digital(),
        geometry_config=geometry_config,
        dynamics_config=StateDynamicsConfig.instantaneous(),
        credit_config=CreditAssignmentConfig.thermodynamic_contrast(),
        update_config=ParameterUpdateConfig.euclidean(step_size=0.01),
        plasticity_config=PlasticityConfig.null(),
        registry=registry,
    )


def _make_run(gain: float, seed: int, dim: int, batch: int):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(dim, dim, generator=generator) / np.sqrt(dim) * gain
    state = CompositeState(
        activity={"x": torch.randn(batch, dim, generator=generator)},
        plastic={},
        substrate={},
    )

    def transition(z: CompositeState, _context: SystemContext) -> CompositeState:
        return CompositeState(
            activity={"x": z.activity["x"] @ weight.T},
            plastic=z.plastic,
            substrate=z.substrate,
        )

    return transition, state


def _diverges(transition, state, context) -> bool:
    initial_norm = state.activity["x"].norm().item()
    z = state
    for _ in range(UNROLL_STEPS):
        z = transition(z, context)
        norm = z.activity["x"].norm().item()
        if not np.isfinite(norm) or norm > EXPLOSION_FACTOR * initial_norm:
            return True
    return False


def _probe_statistic(transition, state, context, statistic: StatisticKind) -> float:
    guard = StabilityGuard(threshold=float("inf"), statistic=statistic)
    return guard.probe(transition, state, context)


def _collect_labeled_stats(
    dim: int, batch: int, context: SystemContext
) -> tuple[dict[StatisticKind, list[float]], dict[StatisticKind, list[float]]]:
    good: dict[StatisticKind, list[float]] = {kind: [] for kind in STATISTIC_KINDS}
    bad: dict[StatisticKind, list[float]] = {kind: [] for kind in STATISTIC_KINDS}
    started = time.perf_counter()

    for gain in GAINS:
        for seed in range(SEEDS_PER_GAIN):
            transition, state = _make_run(gain, seed, dim, batch)
            diverged = _diverges(transition, state, context)
            bucket = bad if diverged else good
            for kind in STATISTIC_KINDS:
                bucket[kind].append(_probe_statistic(transition, state, context, kind))

    logger.info(
        "labeled %d runs (%d good / %d bad) in %.1fs",
        len(good["fast_proxy"]) + len(bad["fast_proxy"]),
        len(good["fast_proxy"]),
        len(bad["fast_proxy"]),
        time.perf_counter() - started,
    )
    return good, bad


def _calibrate_all(good, bad) -> dict[str, object]:
    calibrations: dict[str, object] = {}
    for kind in STATISTIC_KINDS:
        report = calibrate_threshold(good[kind], bad[kind])
        calibrations[kind] = asdict(report) if report else None
        if report is None:
            logger.error("%s: no feasible threshold; classes not separated", kind)
        else:
            logger.info(
                "%s: threshold=%.4f false_kill=%.3f kill_rate=%.3f",
                kind,
                report.threshold,
                report.false_kill_rate,
                report.kill_rate,
            )
    return calibrations


def _summarize(stats: dict[StatisticKind, list[float]], key: str) -> dict[str, object]:
    return {
        kind: {
            "n": len(values),
            key: float(np.max(values)) if values else None,
            "mean": float(np.mean(values)) if values else None,
        }
        for kind, values in stats.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument(
        "--output-dir", default="benchmark_results/stability_guard_calibration"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    context = _synthetic_context(args.dim)
    good, bad = _collect_labeled_stats(args.dim, args.batch, context)
    calibration = _calibrate_all(good, bad)

    mid_gain = GAINS[len(GAINS) // 2]
    transition, state = _make_run(mid_gain, SEEDS_PER_GAIN, args.dim, args.batch)
    disagreement = quantify_proxy_disagreement(
        transition, state, context, probes=ProbeSpec(n_probes=20)
    )
    overheads = {
        kind: measure_guard_overhead(
            transition,
            state,
            context,
            guard=StabilityGuard(threshold=float("inf"), statistic=kind),
        )
        for kind in STATISTIC_KINDS
    }
    logger.info(
        "proxy disagreement median_rel_err=%.3f corr=%.3f overhead=%s",
        disagreement.median_relative_error,
        disagreement.pearson_correlation,
        {k: round(v, 2) for k, v in overheads.items()},
    )

    artifact = {
        "family": {
            "type": "ginibre_linear",
            "dim": args.dim,
            "batch": args.batch,
            "gains": list(GAINS),
            "seeds_per_gain": SEEDS_PER_GAIN,
        },
        "label_rule": f"norm > {EXPLOSION_FACTOR:.0e}x over {UNROLL_STEPS} steps",
        "good_stats_summary": _summarize(good, "max"),
        "bad_stats_summary": _summarize(bad, "min"),
        "calibration": calibration,
        "disagreement": asdict(disagreement),
        "overhead_ratio": overheads,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "calibration.json"
    output_path.write_text(json.dumps(artifact, indent=2))
    logger.info("artifact written to %s", output_path)


if __name__ == "__main__":
    main()
