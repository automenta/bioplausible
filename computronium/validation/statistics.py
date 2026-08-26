"""Pure inference statistics for the experiment reporter (architecture §7.1).

Implements the small, dependency-free statistics surface the reporter and the
nightly gate need: bootstrap confidence intervals (percentile + bias-corrected
and accelerated BCa), two-sample effect sizes (Cohen's d, Cliff's δ),
Benjamini-Hochberg FDR control, and a two-sample power estimate.

Every function is pure and NumPy-only so the module stays trivially testable
and runs anywhere (including the overnight smoke rail). Hypothesis-based
property tests and golden values live in ``tests/unit/validation/``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from fractions import Fraction

import numpy as np

__all__ = [
    "benjamini_hochberg",
    "bootstrap_bca_ci",
    "bootstrap_ci",
    "bootstrap_percentile_ci",
    "cliffs_delta",
    "cohens_d",
    "cohens_dz",
    "fisher_exact_p_one_sided",
    "permutation_test_p",
    "power_for_two_sample",
]

Statistic = Callable[[np.ndarray], float]


def bootstrap_percentile_ci(
    data: Sequence[float],
    stat: Statistic = np.mean,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> tuple[float, float]:
    """Bootstrap percentile confidence interval for ``stat``.

    Resamples ``data`` with replacement ``n_boot`` times and reports the
    ``alpha/2`` and ``1 - alpha/2`` quantiles of the bootstrap distribution.

    Args:
        data: Observed sample.
        stat: Statistic to bootstrap (default mean).
        n_boot: Number of resamples.
        alpha: Two-sided error rate (0.05 -> 95% CI).
        seed: Optional RNG seed for reproducibility.

    Returns:
        ``(lower, upper)`` bootstrap percentile interval bounds.
    """
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        raise ValueError(  # ruff: ignore[raise-vanilla-args]  # descriptive message is the public API
            "cannot bootstrap an empty sample"
        )
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        resample = arr[rng.integers(0, arr.size, size=arr.size)]
        boot[i] = stat(resample)
    lo = np.quantile(boot, alpha / 2)
    hi = np.quantile(boot, 1 - alpha / 2)
    return float(lo), float(hi)


def bootstrap_bca_ci(  # ruff: ignore[too-many-locals]  (BCa keeps z0, acceleration, and two BCa quantile adjusters local)
    data: Sequence[float],
    stat: Statistic = np.mean,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> tuple[float, float]:
    """Bias-corrected and accelerated (BCa) bootstrap confidence interval.

    BCa corrects the percentile interval for both median bias (via the bias
    correction ``z0``) and skew (via the acceleration ``a`` estimated from
    leave-one-out jackknife). Preferred over the plain percentile interval for
    skewed statistics such as ratios and time measurements.

    Args:
        data: Observed sample.
        stat: Statistic to bootstrap.
        n_boot: Number of resamples.
        alpha: Two-sided error rate.
        seed: Optional RNG seed for reproducibility.

    Returns:
        ``(lower, upper)`` BCa interval bounds.
    """
    arr = np.asarray(data, dtype=float)
    n = arr.size
    if n == 0:
        raise ValueError(  # ruff: ignore[raise-vanilla-args]  # descriptive message is the public API
            "cannot bootstrap an empty sample"
        )
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        resample = arr[rng.integers(0, n, size=n)]
        boot[i] = stat(resample)

    observed = stat(arr)
    theta_hat = np.asarray(observed, dtype=float)
    if not np.isfinite(theta_hat) or boot.size == 0:
        raise ValueError(  # ruff: ignore[raise-vanilla-args]  # descriptive message is the public API
            f"statistic returned non-finite value {observed!r}"
        )

    from scipy.stats import norm

    z0 = norm.ppf((np.sum(boot < observed) + np.sum(boot == observed) / 2) / n_boot)
    z0 = float(np.clip(z0, -10, 10))

    jack = np.empty(n, dtype=float)
    for i in range(n):
        leave_out = np.delete(arr, i)
        jack[i] = stat(leave_out)
    mean_jack = np.mean(jack)
    num = np.sum((mean_jack - jack) ** 3)
    den = np.sum((mean_jack - jack) ** 2)
    a = 0.0 if den == 0 else float(num / (6 * den**1.5))
    a = float(np.clip(a, -0.5, 0.5))

    z_alpha = float(norm.ppf(alpha / 2))
    z_1_minus = -z_alpha

    def _adjust(z0z: float) -> float:
        num = z0 + z0z
        den = 1 - a * (z0 + z0z)
        return float(norm.cdf(z0 + num / den))

    lo_q = _adjust(z_alpha)
    hi_q = _adjust(z_1_minus)
    return float(np.quantile(boot, lo_q)), float(np.quantile(boot, hi_q))


def bootstrap_ci(  # ruff: ignore[too-many-arguments]  (stat, n_boot, alpha, method, seed for one dispatch entrypoint)
    data: Sequence[float],
    stat: Statistic = np.mean,
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    method: str = "percentile",
    seed: int | None = None,
) -> tuple[float, float]:
    """Dispatch to the requested bootstrap interval method.

    Args:
        data: Observed sample.
        stat: Statistic to bootstrap.
        n_boot: Number of resamples.
        alpha: Two-sided error rate.
        method: ``"percentile"`` or ``"bca"``.
        seed: Optional RNG seed.

    Returns:
        ``(lower, upper)`` interval bounds.

    Raises:
        ValueError: For an unknown ``method``.
    """
    if method == "percentile":
        return bootstrap_percentile_ci(
            data, stat, n_boot=n_boot, alpha=alpha, seed=seed
        )
    if method == "bca":
        return bootstrap_bca_ci(data, stat, n_boot=n_boot, alpha=alpha, seed=seed)
    raise ValueError(  # ruff: ignore[raise-vanilla-args]  # descriptive message is the public API
        f"unknown bootstrap method {method!r} (use 'percentile'|'bca')"
    )


def cohens_d(group_a: Sequence[float], group_b: Sequence[float]) -> float:
    """Cohen's d (pooled standard deviation) between two samples.

    Positive values mean ``group_a`` is larger on average. Uses the pooled
    SD with the usual Bessel correction (n - 1) per group, matching scipy
    conventions for the two-sample t-test.

    Args:
        group_a: First sample.
        group_b: Second sample.

    Returns:
        Cohen's d effect size.

    Raises:
        ValueError: When either group is empty or has zero variance.
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    if a.size < 2 or b.size < 2:  # ruff: ignore[magic-value-comparison]  (t-test needs >=2 obs/sample)
        raise ValueError(  # ruff: ignore[raise-vanilla-args]  # descriptive message is the public API
            "Cohen's d requires at least 2 observations per group"
        )
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled = (var_a + var_b) / 2
    if pooled == 0:
        raise ValueError(  # ruff: ignore[raise-vanilla-args]  # descriptive message is the public API
            "Cohen's d undefined: both samples have zero variance"
        )
    mean_diff = float(np.mean(a) - np.mean(b))
    return mean_diff / float(np.sqrt(pooled))


def cohens_dz(diffs: Sequence[float]) -> float:
    """One-sample Cohen's dz for paired differences.

    ``dz = mean(diffs) / std(diffs, ddof=1)`` — the effect size matched to
    paired/sign-flip tests, unlike the pooled two-sample :func:`cohens_d`.

    Args:
        diffs: Per-pair differences (treatment minus control).

    Returns:
        Cohen's dz effect size.

    Raises:
        ValueError: When fewer than 2 differences are given or all are
            identical (dz undefined).
    """
    d = np.asarray(diffs, dtype=float)
    if d.size < 2:  # ruff: ignore[magic-value-comparison]  (t-test needs >=2 obs/sample)
        raise ValueError(  # ruff: ignore[raise-vanilla-args]  # descriptive message is the public API
            "Cohen's dz requires at least 2 differences"
        )
    sd = np.std(d, ddof=1)
    if sd == 0:
        raise ValueError("Cohen's dz undefined: differences have zero variance")
    return float(np.mean(d)) / float(sd)


def cliffs_delta(group_a: Sequence[float], group_b: Sequence[float]) -> float:
    """Cliff's delta (dominance measure) between two samples.

    ``δ = P(a > b) - P(a < b)``, bounded in ``[-1, 1]``. Non-parametric, so it
    is robust to outliers and skewed distributions where Cohen's d is fragile.

    Args:
        group_a: First sample.
        group_b: Second sample.

    Returns:
        Cliff's delta in ``[-1, 1]``.

    Raises:
        ValueError: When either group is empty.
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    if a.size == 0 or b.size == 0:
        raise ValueError(  # ruff: ignore[raise-vanilla-args]  # descriptive message is the public API
            "Cliff's delta requires two non-empty samples"
        )
    wins = sum(1 for x in a for y in b if x > y)
    losses = sum(1 for x in a for y in b if x < y)
    return float((wins - losses) / (a.size * b.size))


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Apply Benjamini-Hochberg FDR control to a list of p-values.

    Returns the q-values (adjusted p-values) in the same order as the input.
    A comparison is significant at FDR ``q`` when ``q_value <= q``.

    Args:
        p_values: Raw p-values (one per comparison).

    Returns:
        Adjusted q-values, same order as input.

    Raises:
        ValueError: For negative or out-of-range p-values.
    """
    ps = np.asarray(p_values, dtype=float)
    if np.any((ps < 0) | (ps > 1)):
        raise ValueError(  # ruff: ignore[raise-vanilla-args]  # descriptive message is the public API
            "p-values must lie in [0, 1]"
        )
    n = ps.size
    if n == 0:
        return []
    order = np.argsort(ps, kind="stable")
    ranked = np.arange(1, n + 1)
    adjusted = ps[order] * n / ranked
    # Enforce monotonicity from the largest p-value down.
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])
    q = np.empty_like(adjusted)
    q[order] = np.minimum(adjusted, 1.0)
    return q.tolist()


def power_for_two_sample(
    d: float,
    n_per_group: int,
    alpha: float = 0.05,
) -> float:
    """Statistical power of a two-sample t-test for effect size ``d``.

    Closed-form approximation for the (equal-n, equal-variance) two-sample
    t-test: the test statistic is non-central t with ``2n - 2`` degrees of
    freedom and non-centrality ``d * sqrt(n / 2)``.

    Args:
        d: Population effect size (Cohen's d).
        n_per_group: Observations in each group.
        alpha: Two-sided significance level.

    Returns:
        Power (probability of rejecting the null) in ``[0, 1]``.
    """
    from scipy.stats import nct, t

    if n_per_group < 2:  # ruff: ignore[magic-value-comparison]  (t-test needs >=2 obs/group)
        raise ValueError(  # ruff: ignore[raise-vanilla-args]  # descriptive message is the public API
            "power requires at least 2 observations per group"
        )
    df = 2 * n_per_group - 2
    ncp = d * np.sqrt(n_per_group / 2)
    crit = float(t.ppf(1 - alpha / 2, df))
    # scipy's nct.cdf is numerically unstable for large |ncp| (returns NaN).
    # Use the survival function of the reflected statistic for the lower tail:
    #   P(T_ncp < -crit) = P(-T_ncp > crit) = sf(crit; df, -ncp)
    lower = float(nct.sf(crit, df, -ncp))
    upper = float(nct.sf(crit, df, ncp))
    power = float(np.clip(lower + upper, 0.0, 1.0))
    return power


def permutation_test_p(
    group_a: Sequence[float],
    group_b: Sequence[float],
    *,
    n_perm: int = 10_000,
    seed: int = 0,
) -> float:
    """Two-sample permutation p-value for the difference in means.

    Repeatedly relabels the pooled observations and recomputes ``|Δmean|`,
    returning the fraction of permutations whose absolute difference is at
    least as extreme as the observed one. This is the parity report's
    ``bootstrap_p`` field (Plan 8 §C2): distribution-free and robust to small
    cell sizes, with the one-sided/two-sided compromise already baked in via
    the absolute value.

    Args:
        group_a: First sample.
        group_b: Second sample.
        n_perm: Number of relabel permutations (``0`` ⇒ exhaustive via a
            Fisher-Yates shuffle of every resolvable index).
        seed: RNG seed for permutation reproducibility.

    Returns:
        Two-sided permutation p-value in ``[0, 1]``.

    Raises:
        ValueError: If either sample has fewer than one observation.
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    if a.size < 1 or b.size < 1:
        raise ValueError("permutation_test_p requires >=1 observation per group")
    observed = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    rng = np.random.default_rng(seed)
    n_a = a.size
    ge = 0
    for _ in range(max(n_perm, 1)):
        perm = rng.permutation(pooled.size)
        delta = abs(pooled[perm[:n_a]].mean() - pooled[perm[n_a:]].mean())
        if delta >= observed:
            ge += 1
    # Add-one smoothing so a small sample can never report ``0.0`` (which would
    # over-claim certainty): the lowest credible p under ``n_perm`` draws is
    # ``1 / (n_perm + 1)``.
    return (ge + 1) / (max(n_perm, 1) + 1)


def fisher_exact_p_one_sided(
    failures_treatment: int, failures_control: int, arm_size: int
) -> float:
    """Exact one-sided Fisher p-value that the control fails more often.

    Conditional on the observed total failure count, the control's failure
    count follows a hypergeometric distribution; the p-value is the
    probability of drawing at least ``failures_control`` failures into the
    control arm. Exact rational arithmetic (no scipy dependency).

    Args:
        failures_treatment: Seeds in the treatment arm failing the event.
        failures_control: Seeds in the control arm failing the event.
        arm_size: Per-arm seed count.

    Returns:
        P(control failures >= observed | margins), in [0, 1].
    """
    if not 0 <= failures_treatment <= arm_size or not 0 <= failures_control <= arm_size:
        raise ValueError(  # noqa: TRY003 - caller passes seed counts directly
            "failure counts must lie within [0, arm_size]"
        )
    total_failures = failures_treatment + failures_control
    total = 2 * arm_size
    p = Fraction(0)
    for k in range(failures_control, min(total_failures, arm_size) + 1):
        if total_failures - k > arm_size:
            continue
        p += Fraction(
            math.comb(total_failures, k)
            * math.comb(total - total_failures, arm_size - k),
            math.comb(total, arm_size),
        )
    return float(p)
