"""Counterfactual attribution over campaign evaluation records.

Data-grounded counterpart to the LLM-based
``computronium.autoscientist.counterfactual`` generator: instead of asking a
model "what if axis X changed", attribute observed metric deltas to axis
swaps. Two records are a *minimal counterfactual pair* when they share a task
and differ in exactly one 6-D axis; the metric delta is then attributable to
that single axis change. Aggregated over pairs, this yields per-axis
counterfactual effects and a ``what_if`` prediction for hypothetical swaps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from computronium.core.campaign.frontier_record import FrontierRecord

AXES: tuple[str, ...] = (
    "substrate",
    "geometry",
    "dynamics",
    "plasticity",
    "credit",
    "update",
)

_METRIC_ALIASES: dict[str, str] = {"accuracy": "task_accuracy", "loss": "task_loss"}


class UnknownAxisError(ValueError):
    """Axis name is not one of the six ontology axes."""


class InvalidCoordinateError(ValueError):
    """Coordinate string does not have exactly six axis components."""


@dataclass(frozen=True, slots=True)
class CounterfactualPair:
    """Minimal one-axis difference between two evaluation records."""

    axis: str
    from_value: str
    to_value: str
    task_name: str
    from_coordinate: str
    to_coordinate: str
    delta: float  # metric(to) - metric(from)


@dataclass(frozen=True, slots=True)
class AxisAttribution:
    """Aggregated counterfactual effect of one axis value transition."""

    axis: str
    from_value: str
    to_value: str
    mean_delta: float
    n_pairs: int

    def to_dict(self) -> dict[str, object]:
        return {
            "axis": self.axis,
            "from": self.from_value,
            "to": self.to_value,
            "mean_delta": self.mean_delta,
            "n_pairs": self.n_pairs,
        }


def _metric_value(record: FrontierRecord, metric: str) -> float:
    if metric == "task_loss":
        return record.task_loss
    if metric == "task_accuracy":
        return record.task_accuracy
    if metric == "stability_score":
        return record.stability_score()
    if metric == "efficiency_score":
        return record.efficiency_score()
    return float(record.metadata.get(metric, math.nan))


def _axis_values(coordinate: str) -> dict[str, str] | None:
    parts = coordinate.split("/")
    if len(parts) != len(AXES):
        return None
    return dict(zip(AXES, parts))


def counterfactual_pairs(
    records: Sequence[FrontierRecord],
    *,
    metric: str = "task_accuracy",
) -> list[CounterfactualPair]:
    """Find all minimal one-axis-difference pairs and their metric deltas.

    Args:
        records: Campaign evaluation records (6-D coordinates).
        metric: Metric to diff; one of ``task_accuracy``, ``task_loss``,
            ``stability_score``, ``efficiency_score``, or a metadata key.

    Returns:
        Pairs ordered by absolute delta (largest attribution first).
    """
    metric = _METRIC_ALIASES.get(metric, metric)
    parsed: list[tuple[FrontierRecord, dict[str, str]]] = []
    for record in records:
        axes = _axis_values(record.coordinate)
        if axes is not None:
            parsed.append((record, axes))

    pairs: list[CounterfactualPair] = []
    for i, (a, axes_a) in enumerate(parsed):
        for b, axes_b in parsed[i + 1 :]:
            if a.task_name != b.task_name:
                continue
            differing = [axis for axis in AXES if axes_a[axis] != axes_b[axis]]
            if len(differing) != 1:
                continue
            axis = differing[0]
            delta = _metric_value(b, metric) - _metric_value(a, metric)
            if math.isnan(delta):
                continue
            pairs.append(
                CounterfactualPair(
                    axis=axis,
                    from_value=axes_a[axis],
                    to_value=axes_b[axis],
                    task_name=a.task_name,
                    from_coordinate=a.coordinate,
                    to_coordinate=b.coordinate,
                    delta=delta,
                )
            )
    return sorted(pairs, key=lambda p: abs(p.delta), reverse=True)


def attribute_axis_effects(
    records: Sequence[FrontierRecord],
    *,
    metric: str = "task_accuracy",
) -> list[AxisAttribution]:
    """Aggregate minimal counterfactual pairs into per-axis effects.

    Returns:
        Attributions sorted by absolute mean delta (most influential first).
    """
    sums: dict[tuple[str, str, str], tuple[float, int]] = {}
    for pair in counterfactual_pairs(records, metric=metric):
        key = (pair.axis, pair.from_value, pair.to_value)
        total, count = sums.get(key, (0.0, 0))
        sums[key] = (total + pair.delta, count + 1)

    attributions = [
        AxisAttribution(
            axis=axis,
            from_value=from_value,
            to_value=to_value,
            mean_delta=total / count,
            n_pairs=count,
        )
        for (axis, from_value, to_value), (total, count) in sums.items()
    ]
    return sorted(attributions, key=lambda a: abs(a.mean_delta), reverse=True)


def what_if(
    records: Sequence[FrontierRecord],
    coordinate: str,
    axis: str,
    new_value: str,
    *,
    metric: str = "task_accuracy",
) -> float | None:
    """Predict the metric change of swapping one axis of a coordinate.

    Uses observed minimal counterfactual pairs matching the swap direction;
    returns ``None`` when no data supports the prediction.
    """
    if axis not in AXES:
        raise UnknownAxisError(axis)
    axes = _axis_values(coordinate)
    if axes is None:
        raise InvalidCoordinateError(coordinate)
    current = axes[axis]
    metric = _METRIC_ALIASES.get(metric, metric)

    forward = backward = 0.0
    forward_n = backward_n = 0
    for pair in counterfactual_pairs(records, metric=metric):
        if pair.axis != axis:
            continue
        if pair.from_value == current and pair.to_value == new_value:
            forward += pair.delta
            forward_n += 1
        elif pair.from_value == new_value and pair.to_value == current:
            backward -= pair.delta
            backward_n += 1

    if forward_n + backward_n == 0:
        return None
    return (forward + backward) / (forward_n + backward_n)
