"""Power preregistration schema + label gate (R8.4) and embedded positive
controls (R8.5).

Every commission declares its claim scope, expected effect size, variance
estimate, n/group, alpha, and stratification structure *before* running
(imp-55). MDE@80% is derived from the declared n; a commission that sits
below the power floor is labeled ``pilot`` / ``plumbing`` /
``instrument_check`` — never claim-grade. A claim-grade commission must
also carry an embedded planted-effect control arm (imp-52 extension): the
control's records are verified against its planted expectation after the
run, and a failed or missing control quarantines the campaign.

Construct validity is part of the gate (imp-54): an
``accumulated_learning`` claim scope requires the stationary task stream —
honest metrics on the legacy per-episode stream still cannot support
accumulation claims. A ``retention`` claim scope (R9.1) requires the
segmented task stream — a structured task-sequence A→B whose within-segment
teachers are stationary and whose across-segment shift makes forgetting
measurable; neither the per-episode nor the single-teacher stream can.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from computronium.validation.preregistration import MIN_SEEDS
from computronium.validation.statistics import power_for_two_sample

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from computronium.core.campaign.frontier_record import FrontierRecord

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_TARGET_POWER",
    "ClaimLabel",
    "ClaimScope",
    "ControlVerdict",
    "EmbeddedControl",
    "PowerPreregistration",
    "at_chance_band",
    "min_detectable_effect",
    "n_for_target_power",
    "verify_embedded_control",
]

DEFAULT_ALPHA = 0.05
DEFAULT_TARGET_POWER = 0.80
_MIN_OBS_PER_GROUP = 2  # the two-sample t-test's floor

# imp-59: the at-chance control band is a statistical instrument — width must
# scale with sqrt(N) of the control arm's scored samples or small pilots
# self-quarantine on sampling noise. 6 binomial sigmas at chance plus a
# floor for init-to-init variation (the registered stationary pilot's
# ±0.05 at 1920 samples sits at ~6.6 sigma).
_CONTROL_BAND_SIGMAS = 6.0
_CONTROL_BAND_FLOOR = 0.05


def at_chance_band(chance: float, n_scored_samples: int) -> float:
    """Half-width of the at-chance control band for ``n`` scored samples.

    The band is a statistical instrument (imp-59): 6 binomial sigmas at
    chance, floored at 0.05 for init-to-init variation, so a frozen arm
    cannot be quarantined by sampling noise alone.
    """
    sigma = math.sqrt(chance * (1.0 - chance) / max(1, n_scored_samples))
    return max(_CONTROL_BAND_FLOOR, _CONTROL_BAND_SIGMAS * sigma)


ClaimLabel = Literal["claim_grade", "pilot", "plumbing", "instrument_check"]
ClaimScope = Literal[
    "per_episode_adaptation",
    "accumulated_learning",
    "resource_efficiency",
    "stability",
    "m_axis_plasticity",
    "retention",
    "credit_at_depth",
]

_NON_CLAIM_RUNGS: frozenset[str] = frozenset({
    "pilot",
    "plumbing",
    "instrument_check",
})


def min_detectable_effect(
    n_per_group: int,
    alpha: float = DEFAULT_ALPHA,
    target_power: float = DEFAULT_TARGET_POWER,
) -> float:
    """Cohen's d detectable at ``target_power`` for equal groups of size n.

    Bisection on the two-sample power curve; the power function is monotone
    in d, so 60 halvings of [0, 10] converge far below any reportable
    effect-size resolution.
    """

    def power_minus_target(d: float) -> float:
        return power_for_two_sample(d, n_per_group, alpha=alpha) - target_power

    if n_per_group < _MIN_OBS_PER_GROUP:
        return float("inf")
    lo, hi = 0.0, 10.0
    if power_minus_target(hi) < 0:
        return float("inf")
    for _ in range(60):
        mid = (lo + hi) / 2
        if power_minus_target(mid) > 0:
            hi = mid
        else:
            lo = mid
    return hi


def n_for_target_power(
    d: float,
    alpha: float = DEFAULT_ALPHA,
    target_power: float = DEFAULT_TARGET_POWER,
) -> int:
    """Smallest equal-group n achieving ``target_power`` for effect ``d``.

    Inverse of ``min_detectable_effect`` — the planning number a pilot's
    observed effect feeds into the registered design (R8.4). Returns the
    bisection ceiling (power monotone in n); ``inf``-safe via a 2^16 cap.
    """
    if d <= 0:
        return 2**16
    lo, hi = 2, 2**16
    if power_for_two_sample(d, hi, alpha=alpha) < target_power:
        return hi
    while lo < hi:
        mid = (lo + hi) // 2
        if power_for_two_sample(d, mid, alpha=alpha) >= target_power:
            hi = mid
        else:
            lo = mid + 1
    return lo


@dataclass(frozen=True, slots=True)
class EmbeddedControl:
    """A planted-effect control arm declared inside a commission (R8.5).

    Attributes:
        arm: Human-readable arm label (e.g. ``"null_frozen_lr0"``).
        coordinate: Coordinate string whose records are the control arm. The
            planted expectation is that this arm CANNOT learn — a control
            that moves is an instrument defect and quarantines the campaign.
        chance: Chance-level accuracy for the task (1 / num_classes).
        tolerance: Mean-accuracy band half-width around ``chance``.
    """

    arm: str
    coordinate: str
    chance: float
    tolerance: float = 0.05

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> EmbeddedControl:
        return cls(
            arm=str(data["arm"]),
            coordinate=str(data["coordinate"]),
            chance=float(data["chance"]),  # type: ignore[arg-type]
            tolerance=float(data.get("tolerance", 0.05)),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ControlVerdict:
    """Outcome of the post-run embedded-control check."""

    arm: str
    verdict: Literal["passed", "failed", "missing"]
    observed_mean: float | None
    detail: str

    @property
    def quarantines(self) -> bool:
        """A failed or missing control quarantines the campaign (R8.5)."""
        return self.verdict != "passed"


@dataclass(frozen=True, slots=True)
class PowerPreregistration:
    """A commission's power preregistration (imp-55; TODO9 R8.4).

    Attributes:
        claim: Statement under test.
        metric: Metric the effect size refers to (claim-grade per the
            imp-46 provenance census — e.g. ``task_accuracy``).
        claim_scope: Which effect type the design can support (claim-scope
            rule, TODO9 R8; ``retention`` joins in R9.1).
        task_stream: ``stationary`` (accumulation-capable, R8.3 Option A),
            ``segmented`` (structured task-sequence stream, R9.1), or
            ``per_episode`` (legacy imp-54 stream).
        expected_effect: Minimum effect size (Cohen's d) the claim needs.
        variance_estimate: Pooled SD of ``metric`` (pilot-derived for R8.4).
        n_per_group: Planned observations per arm.
        alpha: Two-sided significance level.
        target_power: Power the design is gated at.
        stratification: Structure the comparison must respect.
        embedded_control: The planted-effect control arm (R8.5); required
            for claim-grade.
        declared_rung: Optional self-declared rung that CAPS the label below
            claim-grade (``pilot``/``plumbing``/``instrument_check``) even
            when the power gates pass. Claim-grade is derived: a commission
            simply declares no cap.
        created: ISO date the registration was committed.
    """

    claim: str
    metric: str
    claim_scope: ClaimScope
    task_stream: Literal["stationary", "per_episode", "segmented"]
    expected_effect: float
    variance_estimate: float
    n_per_group: int
    alpha: float = DEFAULT_ALPHA
    target_power: float = DEFAULT_TARGET_POWER
    stratification: tuple[str, ...] = ("coordinate", "seed")
    embedded_control: EmbeddedControl | None = None
    declared_rung: ClaimLabel | None = None
    created: str = ""

    @property
    def mde_cohens_d(self) -> float:
        """Detectable effect size (d units) at the declared n and power."""
        return min_detectable_effect(
            self.n_per_group, alpha=self.alpha, target_power=self.target_power
        )

    @property
    def mde_metric(self) -> float:
        """Detectable effect in metric units (``mde_cohens_d`` x pooled SD)."""
        return self.mde_cohens_d * self.variance_estimate

    def unmet_requirements(self) -> tuple[str, ...]:
        """Named reasons this commission cannot be claim-grade."""
        unmet: list[str] = []
        if self.expected_effect <= 0:
            unmet.append("expected_effect must be positive")
        if self.variance_estimate <= 0:
            unmet.append("variance_estimate must be positive")
        if self.n_per_group < MIN_SEEDS:
            unmet.append(f"n_per_group {self.n_per_group} below floor {MIN_SEEDS}")
        if self.mde_cohens_d > self.expected_effect:
            unmet.append(
                f"power floor: MDE@{self.target_power:.0%} d={self.mde_cohens_d:.3f} "
                f"exceeds expected effect d={self.expected_effect:.3f}"
            )
        if self.embedded_control is None:
            unmet.append("no embedded positive control declared (R8.5)")
        if (
            self.claim_scope == "accumulated_learning"
            and self.task_stream != "stationary"
        ):
            unmet.append(
                "accumulated_learning scope requires the stationary stream (imp-54)"
            )
        if self.claim_scope == "retention" and self.task_stream != "segmented":
            unmet.append(
                "retention scope requires the segmented task stream (structured "
                "A→B sequence; per-episode and single-teacher streams cannot "
                "measure forgetting)"
            )
        if self.claim_scope == "credit_at_depth" and self.task_stream != "stationary":
            unmet.append(
                "credit_at_depth scope requires the stationary stream (a fixed "
                "ground-truth function; the per-episode stream redraws targets "
                "every episode, confounding depth with non-stationarity)"
            )
        return tuple(unmet)

    def label(self) -> ClaimLabel:
        """Derived claim label; a declared rung caps the commission below
        claim-grade regardless of the power math (a pilot stays a pilot)."""
        if self.declared_rung in _NON_CLAIM_RUNGS:
            return self.declared_rung
        if not self.unmet_requirements():
            return "claim_grade"
        return "pilot"

    def require_claim_grade(self) -> None:
        """Raise unless the commission passes every claim-grade gate."""
        unmet = self.unmet_requirements()
        if unmet:
            reasons = "; ".join(unmet)
            raise ValueError(  # ruff: ignore[raise-vanilla-args] - the gate failure must name every unmet requirement
                f"commission is not claim-grade: {reasons}"
            )

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = asdict(self)
        data["stratification"] = list(self.stratification)
        data["mde_cohens_d"] = self.mde_cohens_d
        data["mde_metric"] = self.mde_metric
        data["label"] = self.label()
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> PowerPreregistration:
        control = data.get("embedded_control")
        declared = data.get("declared_rung")
        return cls(
            claim=str(data["claim"]),
            metric=str(data["metric"]),
            claim_scope=data["claim_scope"],  # type: ignore[arg-type]
            task_stream=data["task_stream"],  # type: ignore[arg-type]
            expected_effect=float(data["expected_effect"]),  # type: ignore[arg-type]
            variance_estimate=float(data["variance_estimate"]),  # type: ignore[arg-type]
            n_per_group=int(data["n_per_group"]),  # type: ignore[arg-type]
            alpha=float(data.get("alpha", DEFAULT_ALPHA)),  # type: ignore[arg-type]
            target_power=float(
                data.get("target_power", DEFAULT_TARGET_POWER)  # type: ignore[arg-type]
            ),
            stratification=tuple(
                data.get("stratification", ("coordinate", "seed"))  # type: ignore[arg-type]
            ),
            embedded_control=(
                EmbeddedControl.from_dict(control)
                if isinstance(control, Mapping)
                else None
            ),
            declared_rung=str(declared) if declared else None,  # type: ignore[arg-type]
            created=str(data.get("created", "")),
        )

    def save(self, path: Path) -> Path:
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> PowerPreregistration:
        with path.open(encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def verify_embedded_control(
    records: Sequence[FrontierRecord], control: EmbeddedControl
) -> ControlVerdict:
    """Check the planted control arm against its at-chance expectation.

    The control arm's mean ``task_accuracy`` (claim-grade, target-free per
    the imp-46 census) must sit within ``chance ± tolerance`` — a control
    that learns means the instrument manufactures effects, quarantining
    every delta the campaign produced.
    """
    arm_records = [r for r in records if r.coordinate == control.coordinate]
    if not arm_records:
        return ControlVerdict(
            arm=control.arm,
            verdict="missing",
            observed_mean=None,
            detail=f"no records for control coordinate {control.coordinate!r}",
        )
    observed = float(np.mean([r.task_accuracy for r in arm_records]))
    lo, hi = control.chance - control.tolerance, control.chance + control.tolerance
    if lo <= observed <= hi:
        return ControlVerdict(
            arm=control.arm,
            verdict="passed",
            observed_mean=observed,
            detail=f"mean acc {observed:.4f} within chance band [{lo:.4f}, {hi:.4f}]",
        )
    return ControlVerdict(
        arm=control.arm,
        verdict="failed",
        observed_mean=observed,
        detail=(
            f"planted control moved: mean acc {observed:.4f} outside chance "
            f"band [{lo:.4f}, {hi:.4f}] — campaign quarantined"
        ),
    )
