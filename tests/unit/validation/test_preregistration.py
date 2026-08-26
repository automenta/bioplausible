"""PR-4 pre-registration kit: ThresholdRegistration + paired_comparison."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from computronium.validation.preregistration import (
    MIN_SEEDS,
    PairedComparison,
    ThresholdRegistration,
    paired_comparison,
    require_min_seeds,
)
from computronium.validation.statistics import cohens_dz

_ALPHA = 0.05

_REG = ThresholdRegistration(
    claim="treatment beats control by >=0.2",
    metric="acc",
    threshold=0.2,
)


def test_registration_round_trip() -> None:
    assert ThresholdRegistration.from_dict(_REG.to_dict()) == _REG


def test_min_seeds_enforced() -> None:
    require_min_seeds(MIN_SEEDS)
    with pytest.raises(ValueError, match="below pre-registration floor"):
        require_min_seeds(MIN_SEEDS - 1)


def test_passes_logic() -> None:
    confirm = PairedComparison(
        n=MIN_SEEDS,
        mean_diff=0.5,
        ci_lower=0.3,
        ci_upper=0.7,
        p_value=0.01,
        cohens_dz=1.2,
    )
    reject_margin = PairedComparison(
        n=MIN_SEEDS,
        mean_diff=0.5,
        ci_lower=0.1,
        ci_upper=0.7,
        p_value=0.01,
        cohens_dz=1.2,
    )
    reject_alpha = PairedComparison(
        n=MIN_SEEDS,
        mean_diff=0.5,
        ci_lower=0.3,
        ci_upper=0.7,
        p_value=0.10,
        cohens_dz=1.2,
    )
    assert confirm.passes(_REG)
    assert not reject_margin.passes(_REG)
    assert not reject_alpha.passes(_REG)


def test_registration_example_file_loads() -> None:
    reg = ThresholdRegistration.load(
        Path("configs/preregistrations/eqprop_mnist_80pct.json")
    )
    assert reg.threshold == pytest.approx(0.8) and reg.min_seeds >= MIN_SEEDS
    json.dumps(reg.to_dict())


def test_paired_comparison_confirms_strong_effect() -> None:
    control = [0.0, 1.0, 2.0, 3.0, 4.0]
    treatment = [c + 10.0 + 0.1 * (i % 2) for i, c in enumerate(control)]
    result = paired_comparison(treatment, control, seed=0)
    assert result.cohens_dz == pytest.approx(
        cohens_dz([t - c for t, c in zip(treatment, control)])
    )
    assert result.passes(_REG)


def test_paired_comparison_rejects_null_effect() -> None:
    control = [0.0, 1.0, 2.0, 3.0, 4.0]
    treatment = [c + 0.01 * (i % 2) for i, c in enumerate(control)]
    result = paired_comparison(treatment, control, seed=0)
    assert not result.passes(_REG)


def test_paired_comparison_identical_arms_degrades() -> None:
    result = paired_comparison([1.0] * MIN_SEEDS, [1.0] * MIN_SEEDS)
    assert result.mean_diff == pytest.approx(0.0)
    assert result.cohens_dz == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)
    assert not result.passes(_REG)


def test_paired_comparison_rejects_small_budget() -> None:
    with pytest.raises(ValueError, match="below pre-registration floor"):
        paired_comparison([1.0] * (MIN_SEEDS - 1), [0.0] * (MIN_SEEDS - 1))


@settings(max_examples=25, deadline=None)
@given(
    base=st.lists(
        st.floats(-100, 100, allow_nan=False, allow_infinity=False),
        min_size=MIN_SEEDS,
        max_size=12,
    ),
    shift=st.floats(-50, 50, allow_nan=False, allow_infinity=False),
    noise_scale=st.floats(0.01, 20),
    seed=st.integers(0, 2**31 - 1),
)
def test_paired_comparison_synthetic_properties(
    base: list[float], shift: float, noise_scale: float, seed: int
) -> None:
    """Synthetic paired data: exact mean diff, sane CI/p/dz wiring."""
    import numpy as np

    rng = np.random.default_rng(seed)
    control = list(base)
    treatment = [x + shift + float(rng.normal(0.0, noise_scale)) for x in control]

    result = paired_comparison(
        treatment, control, n_boot=256, n_permutations=256, seed=seed
    )

    assert result.n == len(base)
    assert abs(result.mean_diff - shift) <= 10 * noise_scale
    assert result.ci_lower <= result.mean_diff <= result.ci_upper
    assert 0.0 <= result.p_value <= 1.0
    assert math.isfinite(result.cohens_dz)


def test_fisher_exact_known_values() -> None:
    from computronium.validation.statistics import fisher_exact_p_one_sided

    # 0 treatment vs 6/10 control failures (v2 full-run autopsy rates):
    # all 6 failures land in the control arm.
    p = fisher_exact_p_one_sided(0, 6, 10)
    assert p == pytest.approx(1001 / 184756)
    # Rejection region at alpha=0.05 given 0 treatment failures is >=4.
    assert fisher_exact_p_one_sided(0, 4, 10) < _ALPHA
    assert fisher_exact_p_one_sided(0, 3, 10) > _ALPHA
    # Monotone in the observed effect; a symmetric split cannot reject.
    assert fisher_exact_p_one_sided(0, 10, 10) <= fisher_exact_p_one_sided(0, 5, 10)
    assert fisher_exact_p_one_sided(5, 5, 10) > _ALPHA


def test_fisher_exact_validates_inputs() -> None:
    from computronium.validation.statistics import fisher_exact_p_one_sided

    with pytest.raises(ValueError):
        fisher_exact_p_one_sided(-1, 3, 10)
    with pytest.raises(ValueError):
        fisher_exact_p_one_sided(0, 11, 10)
