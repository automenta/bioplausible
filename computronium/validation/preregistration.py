"""Pre-registration & paired-comparison kit (RESEARCH3 prerequisite PR-4).

Provides the minimal contract every empirical claim must satisfy before it
enters a paper or a campaign gate:

- ``MIN_SEEDS`` policy constant with :func:`require_min_seeds` enforcement.
- :class:`ThresholdRegistration`: the pre-registered threshold template —
  declare metric, superiority margin, error rate, and seed budget *before*
  running; serializable to/from JSON for repo-checked registrations.
- :func:`paired_comparison`: paired-difference harness (bootstrap CI on the
  mean difference, sign-flip permutation p, Cohen's dz) evaluated against a
  registration.

Pure NumPy; reuses the bootstrap/permutation primitives in
``computronium.validation.statistics``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

from computronium.validation.statistics import (
    bootstrap_percentile_ci,
    cohens_dz,
    permutation_test_p,
)

__all__ = [
    "DEFAULT_ALPHA",
    "MIN_SEEDS",
    "PairedComparison",
    "ThresholdRegistration",
    "paired_comparison",
    "require_min_seeds",
]

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

MIN_SEEDS = 5
DEFAULT_ALPHA = 0.05


def require_min_seeds(n_seeds: int, min_seeds: int = MIN_SEEDS) -> None:
    """Raise unless the seed budget meets the pre-registration floor."""
    if n_seeds < min_seeds:
        raise ValueError(
            f"seed budget {n_seeds} below pre-registration floor {min_seeds}"
        )


@dataclass(frozen=True, slots=True)
class ThresholdRegistration:
    """A pre-registered decision threshold for one empirical claim.

    Attributes:
        claim: Statement under test (e.g. "EqProp >= 80% MNIST accuracy").
        metric: Metric name the threshold applies to.
        threshold: Minimum material effect (metric units, treatment minus
            control).
        alpha: Family-wise error rate for the confirmation test.
        min_seeds: Minimum per-arm seed count.
        created: ISO date the registration was committed to the repo.
    """

    claim: str
    metric: str
    threshold: float
    alpha: float = DEFAULT_ALPHA
    min_seeds: int = MIN_SEEDS
    created: str = ""

    def to_dict(self) -> dict[str, str | float | int]:
        """Serialize for JSON registration files."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ThresholdRegistration:
        """Deserialize from a registration file."""
        return cls(
            claim=str(data["claim"]),
            metric=str(data["metric"]),
            threshold=float(data["threshold"]),  # type: ignore[arg-type]
            alpha=float(data.get("alpha", DEFAULT_ALPHA)),  # type: ignore[arg-type]
            min_seeds=int(data.get("min_seeds", MIN_SEEDS)),  # type: ignore[arg-type]
            created=str(data.get("created", "")),
        )

    @classmethod
    def load(cls, path: Path) -> ThresholdRegistration:
        """Load a registration from a JSON file in the repo."""
        with path.open(encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """Paired-difference result between matched treatment/control seeds."""

    n: int
    mean_diff: float
    ci_lower: float
    ci_upper: float
    p_value: float
    cohens_dz: float

    def passes(self, registration: ThresholdRegistration) -> bool:
        """Confirm against a registration: CI excludes the margin at alpha."""
        return self.ci_lower > registration.threshold and self.p_value < (
            registration.alpha
        )


def paired_comparison(
    treatment: list[float],
    control: list[float],
    *,
    n_boot: int = 10_000,
    n_permutations: int = 10_000,
    seed: int | None = None,
) -> PairedComparison:
    """Run the paired-comparison harness over matched seed results.

    Args:
        treatment: Per-seed metric values for the treatment arm.
        control: Per-seed metric values for the control arm (index-matched).
        n_boot: Bootstrap resamples for the CI on the mean difference.
        n_permutations: Sign-flip permutations for the exact p-value.
        seed: RNG seed for resampling reproducibility.

    Returns:
        A :class:`PairedComparison`; evaluate it with ``passes(registration)``.
    """
    require_min_seeds(min(len(treatment), len(control)))
    t = np.asarray(treatment, dtype=float)
    c = np.asarray(control, dtype=float)
    diffs = t - c

    ci_lo, ci_hi = bootstrap_percentile_ci(
        diffs.tolist(), stat=np.mean, n_boot=n_boot, seed=seed
    )
    p_value = permutation_test_p(
        diffs.tolist(),
        [0.0] * diffs.size,
        n_perm=n_permutations,
    )
    # Degenerate identical-arm case has zero-variance diffs (dz undefined);
    # report 0.0 so the harness degrades to "no effect" instead of crashing.
    dz = cohens_dz(diffs.tolist()) if np.std(diffs, ddof=1) > 0 else 0.0

    return PairedComparison(
        n=diffs.size,
        mean_diff=float(np.mean(diffs)),
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        p_value=float(p_value),
        cohens_dz=float(dz),
    )
