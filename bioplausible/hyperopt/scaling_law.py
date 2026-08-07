"""Fitted scaling-law conditionals for the autonomous pipeline (§8).

Plan §8: experiments must emit *fitted conditionals with uncertainty*, not
point estimates. The two the plan names concretely are

* ``accuracy ~ log(FLOPs)`` — the resource-to-performance law, and
* ``memory ~ L x B x D`` — the memory model.

This module fits the first (the one that drives the "resource allocation"
decision: *"To reach accuracy A on task T with rule R, need F FLOPs"*) and
returns the fitted parameters **with uncertainty** (standard errors and a
confidence interval on the predicted FLOPs for a target accuracy), so the
pipeline can decide with a quantified risk rather than a bare number.

Like the rest of the hyperopt frontier stack, it is pure (no training) and
operates on :class:`~bioplausible.hyperopt.frontier.RulePoint` lists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

if TYPE_CHECKING:
    from bioplausible.hyperopt.frontier import RulePoint

__all__ = [
    "AccuracyScalingLaw",
    "fit_accuracy_scaling",
    "predict_flops_for_accuracy",
]

_MIN_POINTS: int = 3
_CONF_LEVEL: float = 0.95


def _linear(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Accuracy as a linear function of log(FLOPs + 1): ``a * log(x + 1) + b``."""
    return a * np.log(x + 1.0) + b


@dataclass(frozen=True, slots=True)
class AccuracyScalingLaw:
    """Fitted ``accuracy ~ a * log(FLOPs) + b`` conditionals with uncertainty.

    ``slope`` and ``intercept`` are the fitted parameters; ``slope_se`` and
    ``intercept_se`` their standard errors from the covariance matrix. ``r2``
    reports how much of the accuracy variance the log-FLOPs effect explains.
    ``n`` is the number of points that produced the fit.
    """

    rule: str
    task: str
    slope: float
    slope_se: float
    intercept: float
    intercept_se: float
    r2: float
    n: int

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible dict."""
        return {
            "rule": self.rule,
            "task": self.task,
            "slope": self.slope,
            "slope_se": self.slope_se,
            "intercept": self.intercept,
            "intercept_se": self.intercept_se,
            "r2": self.r2,
            "n": self.n,
        }


def fit_accuracy_scaling(
    points: list[RulePoint], *, rule: str, task: str = ""
) -> AccuracyScalingLaw | None:
    """Fit ``accuracy ~ a * log(FLOPs) + b`` for a rule's measured points.

    Args:
        points: Measured operating points (may be all raw probes or the
            frontier; all are usable for the fit).
        rule: Rule name (recorded in the resulting law).
        task: Task name (recorded in the resulting law).

    Returns:
        The fitted :class:`AccuracyScalingLaw` with uncertainty, or ``None`` if
        fewer than :data:`_MIN_POINTS` valid points exist or the fit fails
        (insufficient spread / degenerate data).
    """
    flops = np.array([p.total_flops for p in points], dtype=float)
    acc = np.array([p.accuracy for p in points], dtype=float)
    valid = flops > 0
    flops, acc = flops[valid], acc[valid]
    n = flops.size
    if n < _MIN_POINTS:
        return None

    try:
        popt, pcov = curve_fit(_linear, flops, acc, p0=[0.05, 0.5], maxfev=5000)
    except RuntimeError, ValueError:
        return None

    slope, intercept = popt
    perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.array([np.nan, np.nan])
    fitted = _linear(flops, slope, intercept)
    resid = acc - fitted
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((acc - np.mean(acc)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return AccuracyScalingLaw(
        rule=rule,
        task=task,
        slope=float(slope),
        slope_se=float(perr[0]),
        intercept=float(intercept),
        intercept_se=float(perr[1]),
        r2=r2,
        n=n,
    )


def predict_flops_for_accuracy(
    law: AccuracyScalingLaw,
    target_accuracy: float,
    *,
    confidence: float = _CONF_LEVEL,
) -> tuple[float, float, float]:
    """Predict the FLOPs needed to reach ``target_accuracy``, with a CI.

    Inverts the fitted ``accuracy = a * log(F + 1) + b`` law and propagates the
    parameter uncertainty into a confidence interval on the predicted FLOPs.
    Also reports the *deterministic* (mean) point so callers can choose.

    Args:
        law: A fitted :class:`AccuracyScalingLaw`.
        target_accuracy: The accuracy the pipeline must reach.
        confidence: Confidence level for the interval (default 0.95).

    Returns:
        ``(mean_flops, ci_low, ci_high)``. If the target is *below* the
        attainable range (i.e. ``F <= 1`` / outside the log domain), returns
        ``(nan, nan, nan)`` to signal "not attainable with this law".
    """
    if law.slope <= 0:
        return (float("nan"), float("nan"), float("nan"))

    # accuracy = a * log(F + 1) + b  =>  log(F + 1) = (target - b) / a
    log_f_plus_1 = (target_accuracy - law.intercept) / law.slope
    mean_flops = float(np.exp(log_f_plus_1) - 1.0)

    # Uncertainty on log(F + 1) via delta method:
    # var(log(F+1)) = var(intercept)/slope^2 + (log(F+1))^2 * var(slope)/slope^2
    se_logf = np.sqrt(
        (law.intercept_se**2 + log_f_plus_1**2 * law.slope_se**2)
        / max(law.slope**2, 1e-12)
    )
    z = stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    ci_low = float(np.exp(log_f_plus_1 - z * se_logf) - 1.0)
    ci_high = float(np.exp(log_f_plus_1 + z * se_logf) - 1.0)

    if not all(np.isfinite(x) for x in (mean_flops, ci_low, ci_high)):
        return (float("nan"), float("nan"), float("nan"))
    return mean_flops, ci_low, ci_high
