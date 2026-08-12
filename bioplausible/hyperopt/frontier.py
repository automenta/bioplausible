"""Pareto frontier over multi-resource objectives and the ``cost_of_plausibility``.

Experiment plan §9/§11: a rule's competitiveness on a task is not single-number
accuracy but its **Pareto frontier** over ``(accuracy, total_flops, memory, time)``,
and the composite ``cost_of_plausibility`` that reports how many more
FLOPs x memory x time a bio rule needs vs the ideal backprop at matched
accuracy. These are the decision inputs for the autonomous pipeline (§8).

The upstream ``hyperopt.metrics`` module scores trials on a fixed 4D objective
(accuracy / perplexity / time / params). This module is the resource-frontier
view the plan asks for: it operates on per-rule experiment tuples that already
carry ``total_flops`` / ``peak_memory_mb`` / ``wall_time_s`` (emitted by the
experiment layer via :class:`~bioplausible.experiment.probe.ProbeResult`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bioplausible.hyperopt.metrics import non_dominated_indices

__all__ = [
    "RulePoint",
    "cost_of_plausibility",
    "pareto_frontier",
]

_ACCURACY_EPS: float = 1e-3


@dataclass(frozen=True, slots=True)
class RulePoint:
    """A single measured operating point of a learning rule on a task.

    Mirrors the resource tuple the experiment layer emits per probe:
    accuracy plus the three resource axes, with ``rule``/``config`` retained
    so a Pareto point can be traced back to a deployable configuration.
    """

    rule: str
    accuracy: float
    total_flops: float
    peak_memory_mb: float
    wall_time_s: float
    config: tuple[tuple[str, object], ...] = ()

    @property
    def config_dict(self) -> dict[str, object]:
        """Return the config as a plain dict."""
        return dict(self.config)


def pareto_frontier(points: list[RulePoint]) -> list[RulePoint]:
    """Return the Pareto-optimal subset of ``points`` (plan §11).

    A point is on the frontier if no other point achieves at least as high
    accuracy with no more FLOPs / memory / time (and strictly better on one).
    Dominance is delegated to the shared
    :func:`~bioplausible.hyperopt.metrics.non_dominated_indices` primitive.

    Args:
        points: Measured operating points for a single rule on a task.

    Returns:
        The non-dominated points preserving input order.
    """
    if not points:
        return points
    values = [
        (p.accuracy, p.total_flops, p.peak_memory_mb, p.wall_time_s) for p in points
    ]
    keep = set(
        non_dominated_indices(
            values,
            maximize=(True, False, False, False),
            tol=(_ACCURACY_EPS, 0.0, 0.0, 0.0),
        )
    )
    return [p for i, p in enumerate(points) if i in keep]


def cost_of_plausibility(
    bio_points: list[RulePoint],
    backprop_points: list[RulePoint],
) -> float:
    """Geometric-mean FLOPs x memory x time ratio of a bio rule vs backprop (§11).

    Computes, for every bio point, the best (minimum) geometric mean of
    ``(flops_ratio, mem_ratio, time_ratio)`` against the backprop point with
    the closest accuracy, then returns the minimum over the bio frontier.

    Args:
        bio_points: Operating points of the bio rule on the task.
        backprop_points: Operating points of ideal backprop on the task.

    Returns:
        The minimal geometric-mean cost ratio. ``<= 1.5`` ⇒ deployment-viable;
        ``>= 5`` ⇒ curiosity. ``inf`` if no reference backprop point exists
        within the accuracy window.
    """
    if not bio_points or not backprop_points:
        return float("inf")

    bp = np.array([
        (p.accuracy, p.total_flops, p.peak_memory_mb, p.wall_time_s)
        for p in backprop_points
    ])
    best_ratio = float("inf")

    for p in bio_points:
        delta = np.abs(bp[:, 0] - p.accuracy)
        i = int(np.argmin(delta))
        ref_acc, ref_flops, ref_mem, ref_time = bp[i]

        if abs(ref_acc - p.accuracy) > _ACCURACY_EPS * 10 and p.accuracy > ref_acc:
            continue

        if ref_flops <= 0 or ref_mem <= 0 or ref_time <= 0:
            continue

        geo_mean = (
            (p.total_flops / ref_flops)
            * (p.peak_memory_mb / ref_mem)
            * (p.wall_time_s / ref_time)
        ) ** (1.0 / 3.0)
        best_ratio = min(best_ratio, geo_mean)

    return best_ratio
