"""Replication gate for campaign claims.

A campaign claim is *replicated* only when it is backed by enough independent
repetitions: at least ``min_seeds`` distinct seeds AND at least
``min_families`` distinct task families per coordinate. Unreplicated
coordinates are flagged so runners can auto-extend them instead of promoting
single-run results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from computronium.core.campaign.frontier_record import FrontierRecord

DEFAULT_MIN_SEEDS = 5
DEFAULT_MIN_FAMILIES = 2


class EmptyReplicationScopeError(ValueError):
    """Replication requested without records or an explicit coordinate."""


# Canonical task-name -> family mapping (unlisted names form their own family).
TASK_FAMILIES: dict[str, str] = {
    "mnist": "vision",
    "fashion_mnist": "vision",
    "kmnist": "vision",
    "usps": "vision",
    "cifar10": "vision",
    "cifar100": "vision",
    "svhn": "vision",
    "digits": "vision",
    "xor": "synthetic",
    "spiral": "synthetic",
    "circles": "synthetic",
    "tiny_shakespeare": "lm",
    "char_ngram": "lm",
    "cartpole": "rl",
    "pendulum": "rl",
    "acrobot": "rl",
    "cora": "graph",
    "citeseer": "graph",
    "pubmed": "graph",
    "breast_cancer": "tabular",
    "iris": "tabular",
    "wine": "tabular",
    "synthetic_forecast": "time_series",
    "lorenz": "scientific",
}


def task_family(task_name: str) -> str:
    """Resolve a task name to its evaluation-domain family."""
    return TASK_FAMILIES.get(task_name, task_name)


@dataclass(frozen=True, slots=True)
class ReplicationReport:
    """Replication status of one coordinate's evaluation records."""

    coordinate: str
    seeds: tuple[int, ...]
    task_families: tuple[str, ...]
    min_seeds: int
    min_families: int

    @property
    def replicated(self) -> bool:
        return (
            len(self.seeds) >= self.min_seeds
            and len(self.task_families) >= self.min_families
        )

    def unmet(self) -> tuple[str, ...]:
        """Human-readable unmet replication requirements."""
        missing: list[str] = []
        if len(self.seeds) < self.min_seeds:
            missing.append(
                f"seeds: {len(self.seeds)}/{self.min_seeds} "
                f"(need {self.min_seeds - len(self.seeds)} more)"
            )
        if len(self.task_families) < self.min_families:
            missing.append(
                f"task families: {len(self.task_families)}/{self.min_families} "
                f"(need {self.min_families - len(self.task_families)} more)"
            )
        return tuple(missing)


def verify_replication(
    records: Sequence[FrontierRecord],
    *,
    coordinate: str | None = None,
    min_seeds: int = DEFAULT_MIN_SEEDS,
    min_families: int = DEFAULT_MIN_FAMILIES,
) -> ReplicationReport:
    """Check that records replicate a coordinate across seeds and task families.

    Args:
        records: Frontier records for one coordinate (all filtered to
            ``coordinate`` when it is provided).
        coordinate: Coordinate label for the report; inferred from records
            when omitted.
        min_seeds: Required number of distinct seeds.
        min_families: Required number of distinct task families.

    Returns:
        ReplicationReport with the observed seeds/families and requirement
        compliance.
    """
    if coordinate is None:
        if not records:
            raise EmptyReplicationScopeError
        coordinate = records[0].coordinate
    relevant = [r for r in records if r.coordinate == coordinate]
    seeds = tuple(sorted({r.seed for r in relevant}))
    families = tuple(
        sorted({task_family(r.task_name) for r in relevant})
    )
    return ReplicationReport(
        coordinate=coordinate,
        seeds=seeds,
        task_families=families,
        min_seeds=min_seeds,
        min_families=min_families,
    )


def replication_manifest(
    records: Sequence[FrontierRecord],
    *,
    min_seeds: int = DEFAULT_MIN_SEEDS,
    min_families: int = DEFAULT_MIN_FAMILIES,
) -> dict[str, ReplicationReport]:
    """Verify replication for every coordinate present in the records."""
    coordinates = {r.coordinate for r in records}
    return {
        coordinate: verify_replication(
            records,
            coordinate=coordinate,
            min_seeds=min_seeds,
            min_families=min_families,
        )
        for coordinate in sorted(coordinates)
    }


def unreplicated(
    records: Sequence[FrontierRecord],
    *,
    min_seeds: int = DEFAULT_MIN_SEEDS,
    min_families: int = DEFAULT_MIN_FAMILIES,
) -> tuple[ReplicationReport, ...]:
    """Coordinates that fail the replication gate, worst shortfall first."""
    manifest = replication_manifest(
        records, min_seeds=min_seeds, min_families=min_families
    )
    return tuple(
        sorted(
            (report for report in manifest.values() if not report.replicated),
            key=lambda r: (len(r.unmet()), -len(r.seeds)),
        )
    )
