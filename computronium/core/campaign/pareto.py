"""
Pareto Frontier Computation for Joint Campaigns.

Provides multi-objective Pareto frontier computation over
(accuracy, stability, efficiency, resources) for 6-D coordinate evaluations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from computronium.core.campaign.frontier_record import FrontierRecord


@dataclass(frozen=True, slots=True)
class ParetoFrontier:
    """Pareto frontier result with metadata."""

    frontier: list[FrontierRecord]  # Records on the Pareto frontier
    dominated: list[FrontierRecord]  # Records not on frontier
    objectives: tuple[str, ...]  # Objective names used
    hypervolume: float  # Hypervolume indicator (if reference point provided)

    def __len__(self) -> int:
        return len(self.frontier)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "frontier": [r.to_dict() for r in self.frontier],
            "dominated": [r.to_dict() for r in self.dominated],
            "objectives": self.objectives,
            "hypervolume": self.hypervolume,
        }


_OBJECTIVE_EXTRACTORS: dict[str, Callable[[FrontierRecord], float]] = {
    "task_accuracy": lambda r: r.task_accuracy,
    "task_loss": lambda r: -r.task_loss,  # Minimize loss = maximize -loss
    "stability_score": lambda r: r.stability_score(),
    "efficiency_score": lambda r: r.efficiency_score(),
    "rho_jacobian": lambda r: -r.rho_jacobian,  # Minimize rho
    "lyapunov_local": lambda r: -abs(r.lyapunov_local),  # Minimize |lyapunov|
    "settling_time": lambda r: -r.settling_time,  # Minimize settling time
    "basin_stability": lambda r: r.basin_stability,
    "compute": lambda r: -r.resources.compute,  # Minimize compute
    "memory": lambda r: -r.resources.memory,  # Minimize memory
    "latency": lambda r: -r.resources.latency,  # Minimize latency
    "energy": lambda r: -r.resources.energy,  # Minimize energy
    "plastic_state_capacity": lambda r: -r.resources.plastic_state_capacity,
}


def objective_vector(
    record: FrontierRecord, objectives: tuple[str, ...]
) -> tuple[float, ...]:
    """Normalized (maximize-form) objective values for one record.

    Cost objectives named in ``objectives`` are pre-negated, so callers pass
    ``maximize=True`` for every entry. Unknown names fall back to record
    metadata (0.0 when absent).
    """
    values = []
    for objective in objectives:
        extractor = _OBJECTIVE_EXTRACTORS.get(objective)
        if extractor is not None:
            values.append(extractor(record))
        else:
            values.append(float(record.metadata.get(objective, 0.0)))
    return tuple(values)


def _dominates(
    a: tuple[float, ...], b: tuple[float, ...], maximize: tuple[bool, ...]
) -> bool:
    """True when a is at least as good everywhere and strictly better once."""
    at_least_one_better = False
    for av, bv, wants_max in zip(a, b, maximize, strict=True):
        if wants_max:
            if av < bv:
                return False
            if av > bv:
                at_least_one_better = True
        else:
            if av > bv:
                return False
            if av < bv:
                at_least_one_better = True
    return at_least_one_better


def _dominated_mask(
    obj_values: list[tuple[float, ...]], maximize: tuple[bool, ...]
) -> list[bool]:
    """Pairwise dominance mask: entry i is True when some j dominates i."""
    n = len(obj_values)
    mask = [False] * n
    for i in range(n):
        if mask[i]:
            continue
        for j in range(n):
            if j != i and _dominates(obj_values[j], obj_values[i], maximize):
                mask[i] = True
                break
    return mask


def pareto_frontier(
    records: list[FrontierRecord],
    objectives: tuple[str, ...] = (
        "task_accuracy",
        "stability_score",
        "efficiency_score",
    ),
    maximize: tuple[bool, ...] = (True, True, True),
    reference_point: tuple[float, ...] | None = None,
) -> ParetoFrontier:
    """
    Compute Pareto frontier over multiple objectives.

    Args:
        records: List of FrontierRecord objects
        objectives: Names of objectives to optimize
        maximize: Whether to maximize each objective
        reference_point: Reference point for hypervolume computation

    Returns:
        ParetoFrontier with frontier and dominated records
    """
    if not records:
        return ParetoFrontier(
            frontier=[],
            dominated=[],
            objectives=objectives,
            hypervolume=0.0,
        )

    obj_values = [objective_vector(r, objectives) for r in records]
    is_dominated = _dominated_mask(obj_values, maximize)
    n = len(records)

    frontier = [records[i] for i in range(n) if not is_dominated[i]]
    dominated = [records[i] for i in range(n) if is_dominated[i]]

    # Compute hypervolume if reference point provided
    hypervolume = 0.0
    if reference_point and frontier:
        hypervolume = _compute_hypervolume(
            [obj_values[i] for i in range(n) if not is_dominated[i]],
            reference_point,
            maximize,
        )

    return ParetoFrontier(
        frontier=frontier,
        dominated=dominated,
        objectives=objectives,
        hypervolume=hypervolume,
    )


def _compute_hypervolume(
    points: list[tuple[float, ...]],
    reference: tuple[float, ...],
    maximize: tuple[bool, ...],
) -> float:
    """Compute hypervolume indicator for a set of points.

    Uses a simple Monte Carlo approximation for >3 objectives.
    For 2-3 objectives, uses exact computation.
    """
    if len(points) == 0:
        return 0.0

    dim = len(points[0])
    if dim != len(reference):
        raise ValueError("Reference point dimension must match points")

    # For 2D: exact computation via sorting
    if dim == 2:
        return _hypervolume_2d(points, reference, maximize)

    # For 3D: exact via recursive slicing
    if dim == 3:
        return _hypervolume_3d(points, reference, maximize)

    # For higher dimensions: Monte Carlo approximation
    return _hypervolume_monte_carlo(points, reference, maximize)


def _hypervolume_2d(
    points: list[tuple[float, ...]],
    reference: tuple[float, ...],
    maximize: tuple[bool, ...],
) -> float:
    """Exact 2D hypervolume."""
    # Normalize so we're always maximizing
    normalized = []
    for p in points:
        norm_p = []
        for i, val in enumerate(p):
            if maximize[i]:
                norm_p.append(val)
            else:
                norm_p.append(-val)
        normalized.append(tuple(norm_p))

    norm_ref = []
    for i, val in enumerate(reference):
        if maximize[i]:
            norm_ref.append(val)
        else:
            norm_ref.append(-val)

    # Points not strictly above the reference contribute no volume
    # (matches the 3D filter).
    normalized = [p for p in normalized if p[0] > norm_ref[0] and p[1] > norm_ref[1]]
    if not normalized:
        return 0.0

    # Sort by first objective descending; strip i covers
    # (next_x, p_i.x] at height (p_i.y - ref.y).
    normalized.sort(key=lambda x: x[0], reverse=True)

    hv = 0.0
    for i, p in enumerate(normalized):
        next_x = normalized[i + 1][0] if i + 1 < len(normalized) else norm_ref[0]
        width = p[0] - next_x
        height = p[1] - norm_ref[1]
        hv += width * height

    return hv


def _hypervolume_3d(
    points: list[tuple[float, ...]],
    reference: tuple[float, ...],
    maximize: tuple[bool, ...],
) -> float:
    """Exact 3D hypervolume via recursive slicing (WFG algorithm)."""
    # Normalize
    normalized = []
    for p in points:
        norm_p = []
        for i, val in enumerate(p):
            if maximize[i]:
                norm_p.append(val)
            else:
                norm_p.append(-val)
        normalized.append(tuple(norm_p))

    norm_ref = []
    for i, val in enumerate(reference):
        if maximize[i]:
            norm_ref.append(val)
        else:
            norm_ref.append(-val)

    # Filter points that dominate reference
    normalized = [p for p in normalized if all(p[i] > norm_ref[i] for i in range(3))]
    if not normalized:
        return 0.0

    # Sort by first objective
    normalized.sort(key=lambda x: x[0], reverse=True)

    hv = 0.0
    for i, p in enumerate(normalized):
        # 2D slice for remaining objectives
        slice_points = [(p[1], p[2])]
        for q in normalized[i + 1 :]:
            if q[0] < p[0]:  # Only points with smaller first objective
                slice_points.append((q[1], q[2]))

        slice_ref = (norm_ref[1], norm_ref[2])
        slice_hv = _hypervolume_2d(slice_points, slice_ref, (True, True))
        width = p[0] - (
            normalized[i + 1][0] if i + 1 < len(normalized) else norm_ref[0]
        )
        hv += width * slice_hv

    return hv


def _hypervolume_monte_carlo(
    points: list[tuple[float, ...]],
    reference: tuple[float, ...],
    maximize: tuple[bool, ...],
    n_samples: int = 100000,
) -> float:
    """Monte Carlo hypervolume approximation for >3 objectives."""
    import random

    # Normalize
    normalized = []
    for p in points:
        norm_p = []
        for i, val in enumerate(p):
            if maximize[i]:
                norm_p.append(val)
            else:
                norm_p.append(-val)
        normalized.append(tuple(norm_p))

    norm_ref = []
    for i, val in enumerate(reference):
        if maximize[i]:
            norm_ref.append(val)
        else:
            norm_ref.append(-val)

    # Find bounds
    dim = len(norm_ref)
    mins = [min(p[i] for p in normalized) for i in range(dim)]
    maxs = [max(p[i] for p in normalized) for i in range(dim)]

    # Expand bounds slightly
    bounds = [
        (mins[i] - 0.1 * (maxs[i] - mins[i]), maxs[i] + 0.1 * (maxs[i] - mins[i]))
        for i in range(dim)
    ]

    # Volume of bounding box. A constant objective (zero span) is vacuous
    # for dominance — every point and sample share the value — so it
    # contributes unit width instead of collapsing the volume to zero.
    box_volume = 1.0
    for i in range(dim):
        box_volume *= (bounds[i][1] - bounds[i][0]) or 1.0

    # Monte Carlo sampling
    count = 0
    for _ in range(n_samples):
        sample = tuple(random.uniform(b[0], b[1]) for b in bounds)
        # Check if sample is dominated by any frontier point
        for p in normalized:
            if all(sample[i] <= p[i] for i in range(dim)):
                count += 1
                break

    return (count / n_samples) * box_volume


def rank_by_pareto(
    records: list[FrontierRecord],
    objectives: tuple[str, ...] = (
        "task_accuracy",
        "stability_score",
        "efficiency_score",
    ),
    maximize: tuple[bool, ...] = (True, True, True),
) -> list[tuple[FrontierRecord, int]]:
    """
    Rank records by Pareto layers (non-dominated sorting).

    Returns list of (record, layer) where layer=0 is Pareto frontier.
    """
    if not records:
        return []

    layers: list[list[FrontierRecord]] = []
    remaining = records.copy()

    while remaining:
        pf = pareto_frontier(remaining, objectives, maximize)
        layers.append(pf.frontier)
        remaining = pf.dominated

    result = []
    for layer_idx, layer in enumerate(layers):
        for record in layer:
            result.append((record, layer_idx))

    return result
