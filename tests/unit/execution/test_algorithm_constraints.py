"""Tests for algorithm constraints module — validates get_model_spec import fix."""

from bioplausible.execution.algorithm_constraints import (
    ALGORITHM_FAMILY_CONSTRAINTS,
    get_constrained_search_space,
)


def test_get_constrained_search_space_known_model():
    """Known model name returns correct constraint family."""
    space = get_constrained_search_space("eqprop_mlp")
    assert space is not None
    assert isinstance(space, dict)
    assert "lr" in space
    assert "hidden_dim" in space


def test_get_constrained_search_space_unknown_model():
    """Unknown model name falls back to baseline constraints."""
    space = get_constrained_search_space("__nonexistent_model__")
    assert space is not None
    assert space["lr"] == ALGORITHM_FAMILY_CONSTRAINTS["baseline"]["lr"]


def test_get_constrained_search_space_fallback_family():
    """Model with known family gets its constraint family."""
    space = get_constrained_search_space("forward_forward")
    assert "lr" in space
    assert "contrastive_steps" not in space  # fa-specific, not expected
