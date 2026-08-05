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


def test_float_range_describe():
    assert FloatRange(1e-4, 1e-2, log=True).describe() == "float[0.0001, 0.01] log"
    assert FloatRange(0.1, 0.9).describe() == "float[0.1, 0.9] linear"


def test_int_range_sample_and_describe():
    dist = IntRange(1, 5)
    trial = optuna.create_study().ask()
    value = dist.sample(trial, "n")
    assert 1 <= value <= 5
    assert dist.describe() == "int[1, 5]"


def test_choice_describe():
    assert Choice((1, 2, 3)).describe() == "choice[1, 2, 3]"


def test_render_describes_distributions():
    space = SearchSpace(
        base={"lr": FloatRange(1e-4, 1e-2), "hidden_dim": Choice((64, 128))},
    )
    rendered = space.render("any_model")
    assert "float[" in rendered["lr"]
    assert "choice" in rendered["hidden_dim"]


def test_sample_feasible_returns_config_when_feasible():
    space = SearchSpace(base={"hidden_dim": Choice((64,))})
    trial = optuna.create_study().ask()

    def estimator(_config: dict[str, object]) -> int:
        return 10

    result = space.sample_feasible(trial, "m", estimator=estimator, max_params=100)
    assert result is not None
    assert result["hidden_dim"] == 64


def test_sample_feasible_rejects_over_budget_without_constraints():
    space = SearchSpace(base={"hidden_dim": Choice((64,))})
    trial = optuna.create_study().ask()

    def estimator(_config: dict[str, object]) -> int:
        return 500

    assert (
        space.sample_feasible(trial, "m", estimator=estimator, max_params=100) is None
    )


def test_constraints_hold_without_estimator_returns_true():
    space = SearchSpace(
        base={"hidden_dim": Choice((64,))},
        constraints=("estimate(config) <= 10",),
    )
    trial = optuna.create_study().ask()
    # No estimator provided -> constraints treated as holding.
    result = space.sample_feasible(trial, "m")
    assert result is not None


def test_constraint_evaluation_raises_on_bad_expression():
    space = SearchSpace(
        base={"hidden_dim": Choice((64,))},
        constraints=("undefined_name > 10",),
    )
    trial = optuna.create_study().ask()

    def estimator(_config: dict[str, object]) -> int:
        return 10

    with pytest.raises(ValueError, match="Constraint"):
        space.sample_feasible(trial, "m", estimator=estimator)


def test_parse_distribution_passes_through_existing_distribution():
    dist = FloatRange(0.1, 0.9)
    assert parse_distribution(dist) is dist


def test_parse_linear_scale_explicit():
    dist = parse_distribution([0.1, 0.9, "linear"])
    assert isinstance(dist, FloatRange)
    assert not dist.log


def test_parse_int_scale_explicit():
    dist = parse_distribution([1, 10, "int"])
    assert isinstance(dist, IntRange)
    assert dist.low == 1
    assert dist.high == 10
