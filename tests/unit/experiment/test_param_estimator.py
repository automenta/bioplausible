"""Unit tests for static parameter estimation (FIX2a §4.2, §13 step 3)."""

from __future__ import annotations

import pytest

import bioplausible.zoo  # ruff: ignore[unused-import]  # triggers model registration
from bioplausible.core.registry import ComponentCategory, Registry
from bioplausible.experiment.param_estimator import (
    ParamEstimateError,
    build_model_kwargs,
    estimate_param_count,
)


def test_backprop_mlp_manual_count():
    # forward-only FFLayers are weightless, so num_layers=1 leaves only the
    # Linear(64 -> 10) classifier: 64*10 + 10 = 650.
    expected = 64 * 10 + 10
    count = estimate_param_count(
        "backprop_mlp", {"hidden_dim": 64, "num_layers": 1}, input_dim=64, output_dim=10
    )
    assert count == expected


def test_backprop_mlp_num_layers_2():
    count = estimate_param_count(
        "backprop_mlp", {"hidden_dim": 16, "num_layers": 2}, input_dim=64, output_dim=10
    )
    expected = 64 * 16 + 16 + 16 * 10 + 10
    assert count == expected


def test_neural_cube_derives_cube_size_from_hidden():
    count = estimate_param_count(
        "neural_cube", {"hidden_dim": 64}, input_dim=64, output_dim=10
    )
    # cube_size = max(3, round(64**(1/3))) = 4 -> 4^3 = 64 neurons
    assert count > 0
    model_cls = Registry.get(ComponentCategory.MODEL, "neural_cube")
    kwargs = build_model_kwargs(
        model_cls,
        {"hidden_dim": 64},
        input_dim=64,
        output_dim=10,
        model_name="neural_cube",
    )
    assert kwargs["cube_size"] == 4


def test_unknown_model_raises():
    with pytest.raises(Exception):
        estimate_param_count(
            "does_not_exist", {"hidden_dim": 64}, input_dim=64, output_dim=10
        )


def test_signature_filter_skips_unaccepted_kwargs():
    # backprop_mlp does not accept `beta`; estimation must not raise.
    count = estimate_param_count(
        "backprop_mlp",
        {"hidden_dim": 64, "num_layers": 1, "beta": 0.5},
        input_dim=64,
        output_dim=10,
    )
    assert count > 0


def test_all_mlp_campaign_models_estimate():
    models = [
        "backprop_mlp",
        "eqprop_mlp",
        "neural_cube",
        "deep_hebbian",
        "three_factor_hebbian",
        "standard_fa",
        "diff_target_prop",
        "pepita",
        "forward_forward",
    ]
    for model in models:
        count = estimate_param_count(
            model, {"hidden_dim": 64, "num_layers": 1}, input_dim=64, output_dim=10
        )
        assert count > 0, model


def test_param_estimate_raises_when_construction_fails():
    # A non-integer hidden_dim reaches nn.Linear and raises before counting.
    with pytest.raises(ParamEstimateError):
        estimate_param_count(
            "backprop_mlp", {"hidden_dim": "bad"}, input_dim=64, output_dim=10
        )
