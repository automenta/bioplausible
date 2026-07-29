"""Hypothesis property-based tests for zoo/base.py pure functions."""

from hypothesis import given
from hypothesis import strategies as st

from bioplausible.zoo.base import compute_hidden_dims


@given(
    hidden_dim=st.integers(min_value=1, max_value=512),
    num_layers=st.integers(min_value=0, max_value=20),
    max_layers=st.integers(min_value=1, max_value=10),
)
def test_compute_hidden_dims_length(hidden_dim, num_layers, max_layers):
    """Result length == min(num_layers, max_layers) when hidden_dim is set."""
    result = compute_hidden_dims(hidden_dim, num_layers, max_layers)
    expected_len = min(num_layers, max_layers)
    assert len(result) == expected_len


@given(
    num_layers=st.integers(min_value=0, max_value=20),
    max_layers=st.integers(min_value=1, max_value=10),
)
def test_compute_hidden_dims_none(num_layers, max_layers):
    """Returns [] when hidden_dim is None."""
    result = compute_hidden_dims(None, num_layers, max_layers)
    assert result == []


@given(
    hidden_dim=st.integers(min_value=1, max_value=512),
    num_layers=st.integers(min_value=1, max_value=5),
    max_layers=st.integers(min_value=1, max_value=10),
)
def test_compute_hidden_dims_all_equal(hidden_dim, num_layers, max_layers):
    """All elements equal hidden_dim."""
    result = compute_hidden_dims(hidden_dim, num_layers, max_layers)
    assert all(v == hidden_dim for v in result)


@given(
    hidden_dim=st.integers(min_value=1, max_value=512),
    num_layers=st.integers(min_value=0, max_value=0),
    max_layers=st.integers(min_value=1, max_value=10),
)
def test_compute_hidden_dims_zero_layers(hidden_dim, num_layers, max_layers):
    """Returns [] when num_layers is 0."""
    result = compute_hidden_dims(hidden_dim, 0, max_layers)
    assert result == []
