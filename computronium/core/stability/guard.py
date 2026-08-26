"""Calibrated stability guard for unattended campaigns (PR-5).

Kill-switch logic driven by ROC-calibrated thresholds over the fast
spectral-radius proxy, with disagreement quantification against the exact
full-Jacobian estimate.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from computronium.core.stability.spectral_radius import (
    SpectralRadiusEstimator,
    estimate_spectral_radius_full_jacobian,
)

if TYPE_CHECKING:
    from computronium.core.joint.context import SystemContext
    from computronium.core.joint.state import CompositeState

TransitionFn = Callable[["CompositeState", "SystemContext"], "CompositeState"]
StatisticKind = Literal["fast_proxy", "windowed_growth"]


@dataclass(frozen=True, slots=True)
class GuardDecision:
    """Outcome of one guard probe."""

    statistic: float
    threshold: float
    kill: bool


@dataclass(frozen=True, slots=True)
class StabilityGuard:
    """Threshold guard on stability statistics of the joint transition.

    Two statistic modes:
    - ``fast_proxy``: one-step Jacobian-vector gain (`SpectralRadiusEstimator`
      in fast mode); cheap but blind to non-normal transients.
    - ``windowed_growth``: peak activity growth over a settling window;
      tracks asymptotic divergence directly and separates good/unstable runs
      that the one-step proxy conflates.

    Kills a run when the statistic exceeds the calibrated threshold.
    """

    threshold: float
    estimator: SpectralRadiusEstimator = field(
        default_factory=lambda: SpectralRadiusEstimator(fast_mode=True)
    )
    statistic: StatisticKind = "fast_proxy"
    window: int = 10

    def probe(
        self,
        transition_fn: TransitionFn,
        z: CompositeState,
        context: SystemContext,
    ) -> float:
        """Compute the guard statistic at the given state."""
        match self.statistic:
            case "fast_proxy":
                return self.estimator(transition_fn, z, context)
            case "windowed_growth":
                return self._windowed_growth(transition_fn, z, context)

    def _windowed_growth(
        self,
        transition_fn: TransitionFn,
        z: CompositeState,
        context: SystemContext,
    ) -> float:
        key = self.estimator.activity_key
        base_norm = torch.linalg.vector_norm(z.activity[key]) + 1e-12
        peak = 1.0
        with torch.no_grad():
            current = z
            for _ in range(self.window):
                nxt = transition_fn(current, context)
                growth = float(torch.linalg.vector_norm(nxt.activity[key])) / float(
                    base_norm
                )
                peak = max(peak, growth)
                current = nxt
        return peak

    def decide(self, statistic: float) -> GuardDecision:
        """Classify a statistic against the calibrated threshold."""
        return GuardDecision(
            statistic=statistic,
            threshold=self.threshold,
            kill=statistic > self.threshold,
        )

    def __call__(
        self,
        transition_fn: TransitionFn,
        z: CompositeState,
        context: SystemContext,
    ) -> GuardDecision:
        return self.decide(self.probe(transition_fn, z, context))


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """ROC operating point selected by `calibrate_threshold`."""

    threshold: float
    false_kill_rate: float
    kill_rate: float
    n_good: int
    n_bad: int
    roc_points: tuple[tuple[float, float], ...]


def _rates_at(good: np.ndarray, bad: np.ndarray, tau: float) -> tuple[float, float]:
    false_kill = float((good > tau).mean()) if good.size else 0.0
    kill = float((bad > tau).mean()) if bad.size else 0.0
    return false_kill, kill


def calibrate_threshold(
    good_stats: Sequence[float],
    bad_stats: Sequence[float],
    max_false_kill: float = 0.05,
    min_kill_rate: float = 0.95,
) -> CalibrationReport | None:
    """Select a kill threshold meeting both ROC constraints.

    Args:
        good_stats: Guard statistics from known-good runs.
        bad_stats: Guard statistics from known-unstable runs.
        max_false_kill: Upper bound on false-kill rate on good runs.
        min_kill_rate: Lower bound on kill rate on unstable runs.

    Returns:
        Report for the max-margin feasible threshold, or None when no
        candidate satisfies both constraints.
    """
    if not good_stats or not bad_stats:
        return None

    good = np.asarray(good_stats, dtype=np.float64)
    bad = np.asarray(bad_stats, dtype=np.float64)
    candidates = np.unique(np.concatenate([good, bad]))
    lows = np.concatenate([
        [candidates[0] - 1.0],
        (candidates[:-1] + candidates[1:]) / 2,
    ])
    highs = np.concatenate([
        (candidates[:-1] + candidates[1:]) / 2,
        [candidates[-1] + 1.0],
    ])

    feasible: list[tuple[float, float, float]] = []
    roc_points: list[tuple[float, float]] = []
    for low, high in zip(lows, highs, strict=True):
        fkr, kr = _rates_at(good, bad, high)
        roc_points.append((fkr, kr))
        if fkr <= max_false_kill and kr >= min_kill_rate:
            margin = min(high - good.max(), bad.min() - high)
            feasible.append((high, fkr, kr, margin))  # type: ignore[arg-type]

    if not feasible:
        return None

    best = max(feasible, key=lambda t: (t[3], t[2], -t[1]))
    threshold, fkr, kr = best[0], best[1], best[2]
    return CalibrationReport(
        threshold=float(threshold),
        false_kill_rate=fkr,
        kill_rate=kr,
        n_good=len(good),
        n_bad=len(bad),
        roc_points=tuple(roc_points),
    )


@dataclass(frozen=True, slots=True)
class DisagreementReport:
    """Fast-proxy vs full-Jacobian accuracy and cost accounting.

    Relative errors are denominator-dominated wherever the reference
    spectral radius sits near zero (optical/quantum families quote
    ~1800-4400x ratios there); the absolute-error fields are the honest
    companion statistic for those regimes.
    """

    n_probes: int
    mean_relative_error: float
    median_relative_error: float
    p95_relative_error: float
    pearson_correlation: float
    mean_absolute_error: float
    median_absolute_error: float
    median_reference_norm: float
    proxy_seconds: float
    full_jacobian_seconds: float


@dataclass(frozen=True, slots=True)
class ProbeSpec:
    """Randomized probe generation around a base state."""

    n_probes: int = 20
    noise_scale: float = 1e-2
    seed: int = 0


def _collect_estimates(
    transition_fn: TransitionFn,
    z: CompositeState,
    context: SystemContext,
    estimator: SpectralRadiusEstimator,
    probes: ProbeSpec,
) -> tuple[list[float], list[float], float, float]:
    generator = torch.Generator(device="cpu").manual_seed(probes.seed)
    x_base = z.activity["x"]
    proxy_vals: list[float] = []
    full_vals: list[float] = []
    timings = [0.0, 0.0]

    for _ in range(probes.n_probes):
        noise = torch.randn(x_base.shape, generator=generator) * probes.noise_scale
        x_perturbed = x_base + noise.to(device=x_base.device)
        z_probe = type(z)(
            activity={**z.activity, "x": x_perturbed},
            plastic=z.plastic,
            substrate=z.substrate,
        )

        estimates = (
            lambda: estimator(transition_fn, z_probe, context),
            lambda: estimate_spectral_radius_full_jacobian(
                transition_fn, z_probe, context
            ),
        )
        for index, estimate in enumerate(estimates):
            start = time.perf_counter()
            value = estimate()
            timings[index] += time.perf_counter() - start
            (proxy_vals if index == 0 else full_vals).append(value)

    return proxy_vals, full_vals, timings[0], timings[1]


def quantify_proxy_disagreement(
    transition_fn: TransitionFn,
    z: CompositeState,
    context: SystemContext,
    estimator: SpectralRadiusEstimator | None = None,
    probes: ProbeSpec = ProbeSpec(),
) -> DisagreementReport:
    """Measure fast-proxy error against the exact Jacobian spectral norm.

    Args:
        transition_fn: Joint transition to probe.
        z: Base state; probes are perturbed copies of it.
        context: Fixed system context.
        estimator: Proxy configuration (defaults to fast mode).
        probes: Number/scale/seeding of independent probe states.

    Returns:
        Error distribution and cumulative timing per method.
    """
    estimator = estimator or SpectralRadiusEstimator(fast_mode=True)
    proxy_vals, full_vals, proxy_seconds, full_seconds = _collect_estimates(
        transition_fn, z, context, estimator, probes
    )

    proxy = np.asarray(proxy_vals)
    full = np.asarray(full_vals)
    absolute = np.abs(proxy - full)
    relative = absolute / (np.abs(full) + 1e-12)
    variance_present = proxy.std() > 0 and full.std() > 0
    correlation = float(np.corrcoef(proxy, full)[0, 1]) if variance_present else 0.0

    return DisagreementReport(
        n_probes=probes.n_probes,
        mean_relative_error=float(relative.mean()),
        median_relative_error=float(np.median(relative)),
        p95_relative_error=float(np.percentile(relative, 95)),
        pearson_correlation=correlation,
        mean_absolute_error=float(absolute.mean()),
        median_absolute_error=float(np.median(absolute)),
        median_reference_norm=float(np.median(np.abs(full))),
        proxy_seconds=proxy_seconds,
        full_jacobian_seconds=full_seconds,
    )


def measure_guard_overhead(
    transition_fn: TransitionFn,
    z: CompositeState,
    context: SystemContext,
    guard: StabilityGuard,
    n_steps: int = 50,
) -> float:
    """Ratio of mean probe cost to mean transition-step cost."""
    start = time.perf_counter()
    for _ in range(n_steps):
        transition_fn(z, context)
    step_seconds = (time.perf_counter() - start) / n_steps

    start = time.perf_counter()
    for _ in range(n_steps):
        guard.probe(transition_fn, z, context)
    probe_seconds = (time.perf_counter() - start) / n_steps

    return probe_seconds / step_seconds if step_seconds > 0 else float("inf")
