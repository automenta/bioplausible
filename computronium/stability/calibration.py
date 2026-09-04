"""PR-5 demo-harvest calibration for the stability guard.

Known-good statistics are harvested from the demo-suite coordinate family
(the campaign-builder-expressible coordinates the demonstration suite pins);
known-bad statistics from a Ginibre linear ensemble whose runs are labeled
by unrolled divergence (norm explosion). The ROC calibration certifies both
the max-margin operating point and the deployed ``DEFAULT_TAU``; proxy-vs-
full-Jacobian disagreement and probe overhead complete the PR-5 acceptance
triple (<5% false-kill, >95% kill rate, <10% overhead).

Measured findings this calibration records (tiny and demo scale):
- ``windowed_growth`` reads ≈ 1.0 on every known-good arm — bounded
  activations (saturating geometry + imp-60 zero-pad feedback) — and fires
  only on genuinely explosive maps; it is the deployed kill statistic.
- ``fast_proxy`` is calibration-only: its one-step Jacobian-vector gain
  under-estimates σ_max on non-normal maps and is inflated by substrate
  noise on memristive/neuromorphic arms (the family-sweep "INFEASIBLE on
  non-normal systems" deferral, quantified here on real coordinates).
- Per-probe cost is a multiple of a training step (2-13x measured), so the
  <10% overhead bar is met through the calibrated probe interval.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor

from computronium.stability.guard import (
    DEFAULT_TAU,
    CalibrationReport,
    DisagreementReport,
    ProbeSpec,
    StabilityGuard,
    StatisticKind,
    calibrate_threshold,
    measure_guard_overhead,
    quantify_proxy_disagreement,
)
from computronium.state import CompositeState

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from computronium.state import SystemContext

    Transition = Callable[[CompositeState, "SystemContext | None"], CompositeState]

STATISTIC_KINDS: tuple[StatisticKind, ...] = ("fast_proxy", "windowed_growth")

# The demo-suite coordinate family, expressed in campaign-builder syntax:
# D1/D2 (recurrent settling + credit arms), D6 (substrate arms), D7 (spike
# settle), D3-family P-axis arms, and the quickstart instantaneous arm.
DEMO_GOOD_COORDINATES: tuple[str, ...] = (
    "digital/recurrent/energy_minimization/null/thermodynamic_contrast/euclidean",
    "digital/recurrent/energy_minimization/null/gradient/euclidean",
    "digital/recurrent/energy_minimization/null/random_projections/euclidean",
    "memristive/recurrent/energy_minimization/null/thermodynamic_contrast/euclidean",
    "neuromorphic/recurrent/energy_minimization/null/thermodynamic_contrast/euclidean",
    "digital/feedforward/spike_integration/null/gradient/euclidean",
    "digital/feedforward/instantaneous/null/gradient/euclidean",
    "digital/recurrent/instantaneous/routing/gradient/euclidean",
    "digital/recurrent/instantaneous/fast_weights/gradient/euclidean",
)

# Representative coordinates for the proxy-disagreement quantification:
# the digital baseline and the two noisy substrates (D6 arms).
DISAGREEMENT_COORDINATES: tuple[str, ...] = (
    "digital/recurrent/energy_minimization/null/thermodynamic_contrast/euclidean",
    "memristive/recurrent/energy_minimization/null/thermodynamic_contrast/euclidean",
    "neuromorphic/recurrent/energy_minimization/null/thermodynamic_contrast/euclidean",
)

GINIBRE_GAINS: tuple[float, ...] = (0.95, 1.0, 1.05, 1.1, 1.2, 1.4)
GINIBRE_SEEDS_PER_GAIN = 3
GINIBRE_DIM = 32
GINIBRE_BATCH = 4
UNROLL_STEPS = 200
EXPLOSION_FACTOR = 1e3
OVERHEAD_BUDGET = 0.10
HARVEST_SEED = 0
_DISAGREEMENT_BATCH = 16


@dataclass(frozen=True, slots=True)
class PR5Calibration:
    """The PR-5 acceptance triple over one demo-harvest run.

    Attributes:
        good: Harvested known-good statistics per statistic kind.
        bad: Harvested known-bad (divergence-labeled) statistics per kind.
        calibration: ROC report per kind (``None`` = infeasible classes).
        deployed_tau: False-kill / kill-rate at ``DEFAULT_TAU`` per kind.
        overhead_ratio: Probe-cost / transition-step-cost per kind.
        probe_interval: Episodes between probes meeting the overhead budget.
        disagreement: Proxy-vs-full-Jacobian reports keyed by coordinate.
        family: Harvest metadata (coordinates, dims, label rule, budgets).
    """

    good: dict[StatisticKind, list[float]]
    bad: dict[StatisticKind, list[float]]
    calibration: dict[StatisticKind, CalibrationReport | None]
    deployed_tau: dict[StatisticKind, tuple[float, float]]
    overhead_ratio: dict[StatisticKind, float]
    probe_interval: dict[StatisticKind, int]
    disagreement: dict[str, DisagreementReport]
    family: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable artifact shape."""
        return {
            "good_summary": {kind: _summarize(v) for kind, v in self.good.items()},
            "bad_summary": {kind: _summarize(v) for kind, v in self.bad.items()},
            "calibration": {
                kind: asdict(report) if report is not None else None
                for kind, report in self.calibration.items()
            },
            "deployed_tau": {
                kind: {
                    "tau": DEFAULT_TAU,
                    "false_kill_rate": false_kill,
                    "kill_rate": kill_rate,
                }
                for kind, (false_kill, kill_rate) in self.deployed_tau.items()
            },
            "overhead_ratio": dict(self.overhead_ratio),
            "probe_interval": dict(self.probe_interval),
            "disagreement": {
                coordinate: asdict(report)
                for coordinate, report in self.disagreement.items()
            },
            "family": dict(self.family),
        }


def _summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _activity_norm(z: CompositeState) -> float:
    x = z.activity.get("x")
    return float(torch.linalg.vector_norm(x)) if isinstance(x, Tensor) else 0.0


def ginibre_run(
    gain: float,
    seed: int,
    dim: int = GINIBRE_DIM,
    batch: int = GINIBRE_BATCH,
) -> tuple[Transition, CompositeState]:
    """One closed-form linear run: ``gain/sqrt(dim)``-scaled Ginibre weight."""
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(dim, dim, generator=generator) * (gain / dim**0.5)
    state = CompositeState(
        activity={"x": torch.randn(batch, dim, generator=generator)},
        plastic={},
        substrate={},
    )

    def transition(z: CompositeState, _context: SystemContext | None) -> CompositeState:
        x = z.activity["x"]
        return CompositeState(
            activity={"x": x @ weight.T if isinstance(x, Tensor) else x},
            plastic=z.plastic,
            substrate=z.substrate,
        )

    return transition, state


def unrolled_divergence(
    transition: Transition,
    z: CompositeState,
    context: SystemContext | None,
    *,
    steps: int = UNROLL_STEPS,
    factor: float = EXPLOSION_FACTOR,
) -> bool:
    """Label rule: activity norm explodes past ``factor`` x initial or NaNs."""
    base = _activity_norm(z)
    current = z
    for _ in range(steps):
        current = transition(current, context)
        norm = _activity_norm(current)
        if not math.isfinite(norm) or norm > factor * base:
            return True
    return False


def harvest_good_statistics(
    *,
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    batch_size: int,
    episodes: Sequence[int],
    window: int = 10,
    coordinates: tuple[str, ...] = DEMO_GOOD_COORDINATES,
    seed: int = HARVEST_SEED,
) -> dict[StatisticKind, list[float]]:
    """Guard statistics over the known-good demo-suite coordinate arms."""
    from computronium.core.campaign.evaluation import (
        activity_transition,
        build_coordinate_system,
        episode_batch,
    )

    stats: dict[StatisticKind, list[float]] = {kind: [] for kind in STATISTIC_KINDS}
    for index, coordinate in enumerate(coordinates):
        with torch.random.fork_rng():
            torch.manual_seed(seed + index)
            joint = build_coordinate_system(
                coordinate,
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=hidden_dims,
            )
        transition = activity_transition(joint)
        context = joint.context
        for episode in episodes:
            x, _ = episode_batch(episode, input_dim=input_dim, batch_size=batch_size)
            z = CompositeState(activity={"x": x}, plastic={}, substrate={})
            for kind in STATISTIC_KINDS:
                guard = StabilityGuard(
                    threshold=float("inf"), statistic=kind, window=window
                )
                stats[kind].append(guard.probe(transition, z, context))
    return stats


def harvest_bad_statistics(
    *,
    dim: int = GINIBRE_DIM,
    batch: int = GINIBRE_BATCH,
    gains: tuple[float, ...] = GINIBRE_GAINS,
    seeds_per_gain: int = GINIBRE_SEEDS_PER_GAIN,
    window: int = 10,
) -> dict[StatisticKind, list[float]]:
    """Guard statistics over verified-divergent Ginibre runs.

    Non-diverging (marginal) runs enter neither set: they are not known-good
    coordinates and not verified-unstable.
    """
    stats: dict[StatisticKind, list[float]] = {kind: [] for kind in STATISTIC_KINDS}
    for gain in gains:
        for seed in range(seeds_per_gain):
            transition, state = ginibre_run(gain, seed, dim, batch)
            if not unrolled_divergence(transition, state, None):
                continue
            # The closed-form transition ignores its context; the probe's
            # declared signature still requires a (typed-null) context.
            context = cast("SystemContext", None)
            for kind in STATISTIC_KINDS:
                guard = StabilityGuard(
                    threshold=float("inf"), statistic=kind, window=window
                )
                stats[kind].append(guard.probe(transition, state, context))
    return stats


def _rates_at_tau(
    good: Sequence[float], bad: Sequence[float], tau: float
) -> tuple[float, float]:
    good_arr = torch.tensor(good, dtype=torch.float64)
    bad_arr = torch.tensor(bad, dtype=torch.float64)
    false_kill = float((good_arr > tau).float().mean()) if good_arr.numel() else 0.0
    kill = float((bad_arr > tau).float().mean()) if bad_arr.numel() else 0.0
    return false_kill, kill


def probe_interval_for_overhead(ratio: float, budget: float = OVERHEAD_BUDGET) -> int:
    """Episodes between guard probes so amortized cost stays within budget."""
    if ratio <= 0.0:
        return 1
    return max(1, math.ceil(ratio / budget))


def _overhead_and_interval(
    transition: object,
    z: CompositeState,
    context: SystemContext,
    window: int,
    budget: float,
    n_steps: int,
) -> tuple[dict[StatisticKind, float], dict[StatisticKind, int]]:
    overhead: dict[StatisticKind, float] = {}
    interval: dict[StatisticKind, int] = {}
    for kind in STATISTIC_KINDS:
        guard = StabilityGuard(threshold=float("inf"), statistic=kind, window=window)
        ratio = measure_guard_overhead(transition, z, context, guard, n_steps=n_steps)  # type: ignore[arg-type]
        overhead[kind] = ratio
        interval[kind] = probe_interval_for_overhead(ratio, budget)
    return overhead, interval


def _quantify_disagreement(
    coordinates: tuple[str, ...],
    *,
    input_dim: int,
    hidden_dims: tuple[int, ...],
    probes: ProbeSpec,
    seed: int,
) -> dict[str, DisagreementReport]:
    from computronium.core.campaign.evaluation import (
        activity_transition,
        build_coordinate_system,
        episode_batch,
    )

    reports: dict[str, DisagreementReport] = {}
    for index, coordinate in enumerate(coordinates):
        with torch.random.fork_rng():
            torch.manual_seed(seed + index)
            joint = build_coordinate_system(
                coordinate,
                input_dim=input_dim,
                output_dim=input_dim,
                hidden_dims=hidden_dims,
            )
        x, _ = episode_batch(0, input_dim=input_dim, batch_size=_DISAGREEMENT_BATCH)
        z = CompositeState(activity={"x": x}, plastic={}, substrate={})
        reports[coordinate] = quantify_proxy_disagreement(
            activity_transition(joint), z, joint.context, probes=probes
        )
    return reports


def calibrate_demo_harvest(  # ruff: ignore[too-many-arguments]
    *,
    input_dim: int = 784,
    hidden_dims: tuple[int, ...] = (32,),
    output_dim: int = 10,
    batch_size: int = 64,
    episodes: Sequence[int] = (0, 1, 2, 3),
    window: int = 10,
    max_false_kill: float = 0.05,
    min_kill_rate: float = 0.95,
    overhead_budget: float = OVERHEAD_BUDGET,
    disagreement_input_dim: int = 8,
    disagreement_hidden_dims: tuple[int, ...] = (16,),
    disagreement_probes: ProbeSpec | None = None,
    include_demo_cost_probe: bool = True,
    ginibre_dim: int = GINIBRE_DIM,
    ginibre_batch: int = GINIBRE_BATCH,
    ginibre_gains: tuple[float, ...] = GINIBRE_GAINS,
    ginibre_seeds: int = GINIBRE_SEEDS_PER_GAIN,
    seed: int = HARVEST_SEED,
) -> PR5Calibration:
    """Run the full PR-5 calibration over the demo-harvest.

    Args:
        input_dim: Known-good arm input width (demo-suite pin: 784).
        hidden_dims: Known-good arm hidden widths (demo-suite pin: (32,)).
        output_dim: Known-good arm output width (demo-suite pin: 10).
        batch_size: Probe batch (demo-suite pin: 64).
        episodes: Probe-state episode indices (deterministic synthetic stream).
        window: Windowed-growth window (campaign pin: 10).
        max_false_kill: Pre-registered false-kill ceiling (PR-5: 0.05).
        min_kill_rate: Pre-registered kill-rate floor (PR-5: 0.95).
        overhead_budget: Amortized probe-cost budget (PR-5: 0.10).
        disagreement_input_dim: Input width for the disagreement study (tiny:
            the full-Jacobian reference is cost-feasible only at tiny dims).
        disagreement_hidden_dims: Hidden widths for the disagreement study.
        disagreement_probes: Probe count/seed for the disagreement study.
        include_demo_cost_probe: When True, adds a one-probe full-Jacobian
            cost datapoint at the harvest dims (≈20 s at demo scale) — the
            cost-infeasibility evidence for in-loop disagreement tracking.
        ginibre_dim: Known-bad family width.
        ginibre_batch: Known-bad family batch.
        ginibre_gains: Known-bad family gain sweep.
        ginibre_seeds: Known-bad runs per gain.
        seed: Root seed for coordinate construction (forked per build).

    Returns:
        The frozen calibration record; ``to_dict`` renders the artifact.
    """
    from computronium.core.campaign.evaluation import (
        activity_transition,
        build_coordinate_system,
        episode_batch,
    )

    good = harvest_good_statistics(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        batch_size=batch_size,
        episodes=episodes,
        window=window,
        seed=seed,
    )
    bad = harvest_bad_statistics(
        dim=ginibre_dim,
        batch=ginibre_batch,
        gains=ginibre_gains,
        seeds_per_gain=ginibre_seeds,
        window=window,
    )
    calibration: dict[StatisticKind, CalibrationReport | None] = {
        kind: calibrate_threshold(good[kind], bad[kind], max_false_kill, min_kill_rate)
        for kind in STATISTIC_KINDS
    }
    deployed: dict[StatisticKind, tuple[float, float]] = {
        kind: _rates_at_tau(good[kind], bad[kind], DEFAULT_TAU)
        for kind in STATISTIC_KINDS
    }

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        overhead_joint = build_coordinate_system(
            DEMO_GOOD_COORDINATES[0],
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
        )
    overhead_x, _ = episode_batch(0, input_dim=input_dim, batch_size=batch_size)
    overhead_z = CompositeState(activity={"x": overhead_x}, plastic={}, substrate={})
    overhead, interval = _overhead_and_interval(
        activity_transition(overhead_joint),
        overhead_z,
        overhead_joint.context,
        window,
        overhead_budget,
        n_steps=5,
    )

    disagreement = _quantify_disagreement(
        DISAGREEMENT_COORDINATES,
        input_dim=disagreement_input_dim,
        hidden_dims=disagreement_hidden_dims,
        probes=disagreement_probes or ProbeSpec(n_probes=10, seed=seed),
        seed=seed,
    )
    if include_demo_cost_probe:
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            cost_joint = build_coordinate_system(
                DEMO_GOOD_COORDINATES[0],
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=hidden_dims,
            )
        cost_x, _ = episode_batch(0, input_dim=input_dim, batch_size=batch_size)
        cost_z = CompositeState(activity={"x": cost_x}, plastic={}, substrate={})
        disagreement[f"demo_dims_cost/{DEMO_GOOD_COORDINATES[0]}"] = (
            quantify_proxy_disagreement(
                activity_transition(cost_joint),
                cost_z,
                cost_joint.context,
                probes=ProbeSpec(n_probes=1, seed=seed),
            )
        )

    family: dict[str, object] = {
        "good_coordinates": list(DEMO_GOOD_COORDINATES),
        "input_dim": input_dim,
        "hidden_dims": list(hidden_dims),
        "output_dim": output_dim,
        "batch_size": batch_size,
        "episodes": list(episodes),
        "window": window,
        "bad_family": {
            "type": "ginibre_linear",
            "dim": ginibre_dim,
            "batch": ginibre_batch,
            "gains": list(ginibre_gains),
            "seeds_per_gain": ginibre_seeds,
            "label_rule": (
                f"norm > {EXPLOSION_FACTOR:.0e}x initial over {UNROLL_STEPS} steps"
            ),
        },
        "overhead_budget": overhead_budget,
        "max_false_kill": max_false_kill,
        "min_kill_rate": min_kill_rate,
        "seed": seed,
    }
    return PR5Calibration(
        good=good,
        bad=bad,
        calibration=calibration,
        deployed_tau=deployed,
        overhead_ratio=overhead,
        probe_interval=interval,
        disagreement=disagreement,
        family=family,
    )
