"""Unit tests for the declarative search space (FIX2a §4.1)."""

from __future__ import annotations

import optuna
import pytest

from bioplausible.campaign.search_space import (
    Choice,
    FloatRange,
    IntRange,
    SearchSpace,
    parse_distribution,
)


def test_parse_two_number_list_is_int_range():
    dist = parse_distribution([16, 32])
    assert isinstance(dist, IntRange)
    assert dist.low == 16
    assert dist.high == 32


def test_parse_float_pair_is_float_range():
    dist = parse_distribution([0.1, 0.9])
    assert isinstance(dist, FloatRange)
    assert not dist.log


def test_parse_explicit_log_scale():
    dist = parse_distribution([1e-4, 1e-2, "log"])
    assert isinstance(dist, FloatRange)
    assert dist.log


def test_parse_three_plus_values_is_choice():
    dist = parse_distribution([10, 20, 50])
    assert isinstance(dist, Choice)
    assert dist.values == (10, 20, 50)


def test_parse_invalid_shape_raises():
    with pytest.raises(TypeError):
        parse_distribution("lr")


def test_for_model_merges_overrides():
    space = SearchSpace(
        base={"lr": FloatRange(1e-4, 1e-2), "hidden_dim": Choice((64, 128))},
        overrides={"eqprop_mlp": {"beta": FloatRange(0.1, 0.5)}},
        defaults={"gradient_method": "equilibrium"},
    )
    merged = space.for_model("eqprop_mlp")
    assert set(merged) == {"lr", "hidden_dim", "beta"}
    assert merged["beta"].low == 0.1
    assert set(space.for_model("backprop_mlp")) == {"lr", "hidden_dim"}


def test_sample_injects_defaults_and_constants():
    space = SearchSpace(
        base={"lr": FloatRange(1e-4, 1e-2)},
        defaults={"optimizer": "adam"},
        constants={"eqprop_mlp": {"gradient_method": "equilibrium"}},
    )
    trial = optuna.create_study().ask()
    config = space.sample(trial, "eqprop_mlp")
    assert config["optimizer"] == "adam"
    assert config["gradient_method"] == "equilibrium"
    assert "lr" in config


def test_sample_feasible_rejects_over_budget_config():
    space = SearchSpace(
        base={"hidden_dim": Choice((64, 128)), "num_layers": Choice((1, 4))},
        constraints=("estimate(config) <= 1000",),
    )
    trial = optuna.create_study().ask()

    def estimator(config: dict[str, object]) -> int:
        return config["hidden_dim"] * config["num_layers"] * 10

    result = space.sample_feasible(trial, "m", estimator=estimator, max_params=1000)
    assert result is None or estimator(result) <= 1000


def test_constraint_violation_returns_none():
    space = SearchSpace(
        base={"hidden_dim": Choice((64,))},
        constraints=("estimate(config) <= 10",),
    )
    trial = optuna.create_study().ask()

    def estimator(_config: dict[str, object]) -> int:
        return 100

    assert space.sample_feasible(trial, "m", estimator=estimator) is None
