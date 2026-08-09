"""Unit tests for :mod:`bioplausible.validation.statistics`.

Covers golden values on synthetic data (exact analytical expectations) and
hypothesis-based properties for the bootstrap/effect-size/control functions.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bioplausible.validation.statistics import (
    benjamini_hochberg,
    bootstrap_bca_ci,
    bootstrap_ci,
    bootstrap_percentile_ci,
    cliffs_delta,
    cohens_d,
    permutation_test_p,
    power_for_two_sample,
)

SAMPLE = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]


# ---------------------------------------------------------------------------
# Cohen's d — golden values
# ---------------------------------------------------------------------------


def test_cohens_d_identical_samples_is_zero():
    assert cohens_d(SAMPLE, SAMPLE) == pytest.approx(0.0)


def test_cohens_d_golden_value():
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]
    # Pooled SD = sqrt((1+1)/2) = 1; mean diff = -3.
    assert cohens_d(a, b) == pytest.approx(-3.0)
    assert cohens_d(b, a) == pytest.approx(3.0)


def test_cohens_d_matches_scipy():
    rng = np.random.default_rng(7)
    a = rng.normal(0, 1, 50)
    b = rng.normal(1, 1, 50)
    expected = (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    assert cohens_d(a.tolist(), b.tolist()) == pytest.approx(float(expected))


def test_cohens_d_empty_or_constant_raises():
    with pytest.raises(ValueError):
        cohens_d([], [1.0, 2.0])
    with pytest.raises(ValueError):
        cohens_d([1.0, 1.0], [1.0, 1.0])


# ---------------------------------------------------------------------------
# Cliff's delta — golden values
# ---------------------------------------------------------------------------


def test_cliffs_delta_total_dominance():
    # Every element of a > every element of b -> delta = 1.
    assert cliffs_delta([5, 6, 7], [1, 2, 3]) == pytest.approx(1.0)
    assert cliffs_delta([1, 2, 3], [5, 6, 7]) == pytest.approx(-1.0)


def test_cliffs_delta_equal_samples_zero():
    assert cliffs_delta(SAMPLE, SAMPLE) == pytest.approx(0.0)


def test_cliffs_delta_golden_value():
    a = [1.0, 2.0, 4.0]
    b = [1.0, 3.0, 5.0]
    # wins: (2,1)=1 (4,1),(4,3)=2 -> 3 ; losses: (1,3)=1 (1,5)=1 (2,3)=1
    # (2,5)=1 (4,5)=1 -> 5 ; ties: (1,1)=1 -> delta = (3-5)/9.
    assert cliffs_delta(a, b) == pytest.approx(-2 / 9)


def test_cliffs_delta_empty_raises():
    with pytest.raises(ValueError):
        cliffs_delta([], [1.0, 2.0])


# ---------------------------------------------------------------------------
# Bootstrap intervals
# ---------------------------------------------------------------------------


def test_percentile_ci_contains_mean():
    lo, hi = bootstrap_percentile_ci(SAMPLE, np.mean, n_boot=2000, seed=0)
    assert lo <= np.mean(SAMPLE) <= hi


def test_percentile_ci_narrow_with_small_spread():
    data = [5.0, 5.1, 4.9, 5.0, 5.0, 5.05, 4.95, 5.0]
    lo, hi = bootstrap_percentile_ci(data, np.mean, n_boot=2000, seed=1)
    assert (hi - lo) < 0.5


def test_bca_ci_contains_statistic():
    rng = np.random.default_rng(3)
    data = rng.exponential(scale=1.0, size=200)  # skewed: BCa != percentile
    lo, hi = bootstrap_bca_ci(data, np.mean, n_boot=2000, seed=2)
    assert lo <= float(np.mean(data)) <= hi


def test_bootstrap_ci_dispatch_and_error():
    lo, hi = bootstrap_ci(SAMPLE, method="percentile", n_boot=500, seed=0)
    assert lo < hi
    with pytest.raises(ValueError):
        bootstrap_ci(SAMPLE, method="nope")


def test_bootstrap_empty_raises():
    with pytest.raises(ValueError):
        bootstrap_percentile_ci([], np.mean)


# ---------------------------------------------------------------------------
# Benjamini-Hochberg
# ---------------------------------------------------------------------------


def test_bh_returns_same_order_as_input():
    ps = [0.01, 0.04, 0.03, 0.9]
    qs = benjamini_hochberg(ps)
    assert len(qs) == len(ps)
    # Smallest p-value must get the smallest q.
    assert qs[0] == min(qs)


def test_bh_all_insignificant_stay_high():
    qs = benjamini_hochberg([0.5, 0.6, 0.7])
    assert all(q > 0.5 for q in qs)


def test_bh_rejects_bad_p_values():
    with pytest.raises(ValueError):
        benjamini_hochberg([1.5])
    with pytest.raises(ValueError):
        benjamini_hochberg([-0.1])


def test_bh_empty():
    assert benjamini_hochberg([]) == []


def test_bh_golden_values():
    # Hand-computed for the textbook example p = [0.01, 0.02, 0.05, 0.2].
    qs = benjamini_hochberg([0.01, 0.02, 0.05, 0.2])
    expected = [0.04, 0.04, 0.06667, 0.2]
    assert qs == pytest.approx(expected, abs=1e-4)


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------


def test_power_increases_with_effect_size():
    small = power_for_two_sample(0.2, n_per_group=30)
    large = power_for_two_sample(0.8, n_per_group=30)
    assert large > small


def test_power_increases_with_sample_size():
    assert power_for_two_sample(0.5, 20) < power_for_two_sample(0.5, 100)


def test_power_zero_effect_is_alpha():
    # d=0 -> power should equal the type-I error rate alpha.
    assert power_for_two_sample(0.0, 30, alpha=0.05) == pytest.approx(0.05, abs=0.01)


def test_power_rejects_small_samples():
    with pytest.raises(ValueError):
        power_for_two_sample(0.5, 1)


# ---------------------------------------------------------------------------
# Permutation-test p-value (bootstrap_p in the parity report)
# ---------------------------------------------------------------------------


def test_permutation_test_p_null_is_high():
    """Two samples drawn from the same distribution yield a high p-value."""
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 30)
    b = rng.normal(0, 1, 30)
    p = permutation_test_p(a.tolist(), b.tolist(), n_perm=2_000, seed=0)
    assert 0.05 < p <= 1.0, p


def test_permutation_test_p_strong_alternative_is_low():
    """Two well-separated samples yield a small p-value."""
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 30)
    b = rng.normal(3, 1, 30)  # ~3 SD shift → near-certain rejection
    p = permutation_test_p(a.tolist(), b.tolist(), n_perm=2_000, seed=0)
    assert p < 0.05, p


def test_permutation_test_p_identical_samples_is_one():
    """Identical samples: every permutation reproduces the observed mean diff."""
    p = permutation_test_p([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], n_perm=200, seed=0)
    assert p == pytest.approx(1.0, abs=0.01)


def test_permutation_test_p_never_returns_zero():
    """Add-one smoothing: a small sample cannot over-claim p = 0."""
    p = permutation_test_p([1.0, 1.0, 1.0], [10.0, 10.0, 10.0], n_perm=10, seed=0)
    assert p > 0.0
    # Lowest credible p under n_perm draws is 1/(n_perm+1).
    assert p >= 1 / 11 - 1e-12


def test_permutation_test_p_empty_raises():
    with pytest.raises(ValueError):
        permutation_test_p([], [1.0])
    with pytest.raises(ValueError):
        permutation_test_p([1.0], [])


def test_permutation_test_p_in_unit_interval():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 2, 8)
    b = rng.normal(1, 2, 8)
    p = permutation_test_p(a.tolist(), b.tolist(), n_perm=500, seed=0)
    assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# Hypothesis properties
# ---------------------------------------------------------------------------

finite_floats = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False)


@given(st.lists(finite_floats, min_size=4, max_size=40))
@settings(max_examples=50, deadline=5000)
def test_hypothesis_cohens_d_sign_matches_mean_diff(values):
    a = list(values[: len(values) // 2])
    b = list(values[len(values) // 2 :])
    try:
        d = cohens_d(a, b)
    except ValueError:
        return  # zero-variance branch
    assert (d > 0) == (np.mean(a) > np.mean(b)) or d == 0


@given(st.lists(finite_floats, min_size=3, max_size=30))
@settings(max_examples=50, deadline=5000)
def test_hypothesis_cliffs_delta_bounded(values):
    a = list(values[: len(values) // 2])
    b = list(values[len(values) // 2 :])
    try:
        delta = cliffs_delta(a, b)
    except ValueError:
        return
    assert -1.0 <= delta <= 1.0
    assert cliffs_delta(a, a) == 0.0


@given(
    st.lists(
        st.floats(min_value=0.0, max_value=1.0, exclude_max=False),
        min_size=2,
        max_size=12,
    )
)
@settings(max_examples=50, deadline=5000)
def test_hypothesis_bh_monotone_and_bounded(ps):
    ps = [max(1e-9, p) for p in ps]
    qs = benjamini_hochberg(ps)
    assert all(0.0 < q <= 1.0 for q in qs)
    # Adjusted values must be >= raw p-values (BH inflates); allow float rounding.
    assert all(q >= p - 1e-12 for q, p in zip(qs, ps, strict=True))


@given(
    st.lists(finite_floats, min_size=8, max_size=60),
    st.integers(min_value=2, max_value=6),
)
@settings(max_examples=50, deadline=5000)
def test_hypothesis_bootstrap_intervals_contain_estimate(values, seed):
    data = list(values)
    lo, hi = bootstrap_percentile_ci(data, np.mean, n_boot=500, seed=seed)
    mean = float(np.mean(data))
    # Bootstrap CI should be centered near the sample mean.
    assert lo <= mean + 1e-6
    assert hi >= mean - 1e-6


@given(st.floats(min_value=0.0, max_value=3.0), st.integers(min_value=2, max_value=50))
@settings(max_examples=50, deadline=5000)
def test_hypothesis_power_bounded(d, n):
    p = power_for_two_sample(d, n_per_group=n)
    assert 0.0 <= p <= 1.0
