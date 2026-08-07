"""Compare a bio rule's frontier against the ideal-backprop frontier (§4D.5-7).

Plan §4D: the only fair comparison is between a bio rule's own Pareto frontier
(at its own Bayesian optimum) and the **ideal backprop** frontier for the same
task. This module, given a bio frontier and the cached ideal-backprop frontier,
emits:

* the :func:`~bioplausible.hyperopt.frontier.cost_of_plausibility` composite
  (§11), and
* a per-operating-point breakdown: for every bio point, the nearest backprop
  point, the accuracy delta, and the FLOPs/memory/time ratios — the raw data
  that answers "is the bio rule better than backprop at *any* operating
  point?", "at what resource budget does it dominate?", and "what is the
  cost of bio-plausibility at matched accuracy?".

The comparator is pure (no training); it consumes :class:`RulePoint` lists.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bioplausible.hyperopt.frontier import (
    RulePoint,
    cost_of_plausibility,
    pareto_frontier,
)

__all__ = [
    "FrontierComparison",
    "OperatingPointMatch",
    "compare_frontiers",
    "resource_ratios",
]

_MIN_RATIO_TOL: float = 0.0  # ratios are exact floats; no tolerance needed


@dataclass(frozen=True, slots=True)
class OperatingPointMatch:
    """One bio point matched to its nearest backprop reference point."""

    bio: RulePoint
    matched_backprop: RulePoint
    accuracy_delta: float  # bio - backprop (positive means bio is more accurate)
    flops_ratio: float  # bio/backprop; <1 means bio uses fewer FLOPs
    memory_ratio: float
    time_ratio: float

    def dominates(self) -> bool:
        """True if bio beats backprop on accuracy AND all three resources."""
        return (
            self.accuracy_delta > 0
            and self.flops_ratio <= 1
            and self.memory_ratio <= 1
            and self.time_ratio <= 1
        )


@dataclass(frozen=True, slots=True)
class FrontierComparison:
    """Result of comparing one bio frontier against the ideal-backprop frontier."""

    rule: str
    backprop: str
    task: str
    matches: tuple[OperatingPointMatch, ...]
    cost_of_plausibility: float  # §11 composite
    n_dominating_points: int  # bio points that beat bp on all four axes

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible dict."""
        return {
            "rule": self.rule,
            "backprop": self.backprop,
            "task": self.task,
            "cost_of_plausibility": self.cost_of_plausibility,
            "n_dominating_points": self.n_dominating_points,
            "matches": [
                {
                    "bio_accuracy": m.bio.accuracy,
                    "bp_accuracy": m.matched_backprop.accuracy,
                    "accuracy_delta": m.accuracy_delta,
                    "flops_ratio": round(m.flops_ratio, 3),
                    "memory_ratio": round(m.memory_ratio, 3),
                    "time_ratio": round(m.time_ratio, 3),
                    "dominates": m.dominates(),
                    "bio_config": dict(m.bio.config),
                }
                for m in self.matches
            ],
        }


def resource_ratios(a: RulePoint, b: RulePoint) -> tuple[float, float, float]:
    """Return ``(flops_ratio, memory_ratio, time_ratio)`` for ``a`` vs ``b``.

    ``b`` is the reference: each ratio is ``a / b``, so a value below 1 means
    ``a`` uses fewer resources.
    """
    return (
        _safe_ratio(a.total_flops, b.total_flops),
        _safe_ratio(a.peak_memory_mb, b.peak_memory_mb),
        _safe_ratio(a.wall_time_s, b.wall_time_s),
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= _MIN_RATIO_TOL:
        return float("inf") if numerator > _MIN_RATIO_TOL else 0.0
    return numerator / denominator


def compare_frontiers(
    bio_frontier: list[RulePoint],
    backprop_frontier: list[RulePoint],
    *,
    rule: str,
    backprop: str = "backprop_mlp",
    task: str = "",
) -> FrontierComparison:
    """Compare a bio frontier against the ideal-backprop frontier.

    Each bio point is matched to the backprop point with the closest accuracy,
    then the resource ratios and accuracy delta are recorded. The §11 composite
    ``cost_of_plausibility`` is computed over the full bio frontier.

    Args:
        bio_frontier: The bio rule's Pareto frontier (may be unfiltered; the
            non-dominated subset is used).
        backprop_frontier: The ideal-backprop frontier (from
            :class:`~bioplausible.hyperopt.ideal_backprop.IdealBackpropFinder`).
        rule: Name of the bio rule.
        backprop: Name of the backprop reference model.
        task: Task name (only used in the result record).

    Returns:
        A :class:`FrontierComparison`.
    """
    bio = pareto_frontier(bio_frontier) or bio_frontier
    if not bio or not backprop_frontier:
        return FrontierComparison(
            rule=rule,
            backprop=backprop,
            task=task,
            matches=(),
            cost_of_plausibility=float("inf"),
            n_dominating_points=0,
        )

    bp = np.array([
        (p.accuracy, p.total_flops, p.peak_memory_mb, p.wall_time_s)
        for p in backprop_frontier
    ])

    matches: list[OperatingPointMatch] = []
    n_dominating = 0
    for p in bio:
        delta = np.abs(bp[:, 0] - p.accuracy)
        i = int(np.argmin(delta))
        ref = backprop_frontier[i]
        flops_r, mem_r, time_r = resource_ratios(p, ref)
        match = OperatingPointMatch(
            bio=p,
            matched_backprop=ref,
            accuracy_delta=p.accuracy - ref.accuracy,
            flops_ratio=flops_r,
            memory_ratio=mem_r,
            time_ratio=time_r,
        )
        if match.dominates():
            n_dominating += 1
        matches.append(match)

    cost = cost_of_plausibility(bio, backprop_frontier)
    return FrontierComparison(
        rule=rule,
        backprop=backprop,
        task=task,
        matches=tuple(matches),
        cost_of_plausibility=cost,
        n_dominating_points=n_dominating,
    )
