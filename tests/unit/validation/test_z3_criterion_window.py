"""Registered 100-step-window criterion for the Z3 benchmark (PR-4/E-1)."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from computronium.experiments.joint.z3_fixed_weights import (
    _CRITERION_ACCURACY,
    _WINDOW_STEPS,
    _windowed_criterion_step,
)


def _oracle(curve: list[float], window: int, threshold: float) -> int | None:
    for step in range(window, len(curve) + 1):
        if sum(curve[step - window : step]) / window >= threshold:
            return step
    return None


@given(
    curve=st.lists(st.floats(0.0, 1.0, width=16), min_size=0, max_size=400),
    window=st.integers(1, 50),
)
def test_matches_bruteforce_oracle(curve: list[float], window: int) -> None:
    assert _windowed_criterion_step(curve, window=window) == _oracle(
        curve, window, _CRITERION_ACCURACY
    )


@pytest.mark.parametrize(
    ("curve", "expected"),
    [
        ([0.5] * 99, None),
        ([0.0] * 150 + [1.0] * _WINDOW_STEPS, 248),
        ([1.0] + [0.0] * (_WINDOW_STEPS - 2) + [1.0] + [0.0], None),
        ([0.979] * _WINDOW_STEPS, None),
        ([0.98] * _WINDOW_STEPS, _WINDOW_STEPS),
    ],
)
def test_window_boundaries(curve: list[float], expected: int | None) -> None:
    assert _windowed_criterion_step(curve) == expected


def test_transient_spike_does_not_trigger() -> None:
    prefix_len = 120
    curve = [0.0] * prefix_len
    curve[59] = 1.0
    assert _windowed_criterion_step(curve) is None
    # First trailing window holding 98 of the ones ends at prefix + 98.
    full = curve + [1.0] * _WINDOW_STEPS
    assert _windowed_criterion_step(full) == prefix_len + 98
