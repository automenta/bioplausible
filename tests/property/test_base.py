"""Hypothesis property-based tests for zoo/base.py pure functions."""

from hypothesis import assume, given
from hypothesis import strategies as st

from bioplausible.config.unified import (
    ModelConfig,
    compute_hidden_dims,
    resolve_hidden_dims,
)


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
    result = compute_hidden_dims(hidden_dim, num_layers, max_layers)
    assert result == []


# ---------- resolve_hidden_dims (Sprint 5.2) ----------


@given(
    dims=st.lists(st.integers(min_value=1, max_value=128), min_size=1, max_size=8),
    fallback=st.one_of(st.none(), st.integers(min_value=1, max_value=128)),
)
def test_resolve_hidden_dims_config_win_is_exact(dims, fallback):
    """A non-empty config.hidden_dims is returned verbatim (idempotent)."""
    cfg = ModelConfig(name="m", input_dim=8, output_dim=2, hidden_dims=list(dims))
    assert resolve_hidden_dims(cfg, fallback) == list(dims)


@given(fallback=st.one_of(st.none(), st.integers(min_value=1, max_value=128)))
def test_resolve_hidden_dims_nonempty_config_ignores_fallback(fallback):
    """hidden_dim fallback is never used when config.hidden_dims is set."""
    cfg = ModelConfig(name="m", input_dim=8, output_dim=2, hidden_dims=[4, 8])
    assert resolve_hidden_dims(cfg, fallback) == [4, 8]


@given(hidden_dim=st.integers(min_value=1, max_value=512))
def test_resolve_hidden_dims_none_config_fallback(hidden_dim):
    """No config -> [hidden_dim] singleton when hidden_dim set."""
    assert resolve_hidden_dims(None, hidden_dim) == [hidden_dim]


@given(fallback=st.integers(min_value=1, max_value=512))
def test_resolve_hidden_dims_empty_config_uses_fallback(fallback):
    """Empty config.hidden_dims falls back to the singleton fallback."""
    cfg = ModelConfig(name="m", input_dim=8, output_dim=2, hidden_dims=[])
    assert resolve_hidden_dims(cfg, fallback) == [fallback]


def test_resolve_hidden_dims_all_empty():
    """Both empty -> []."""
    assert resolve_hidden_dims(None, None) == []


@given(dims=st.lists(st.integers(min_value=1, max_value=128), max_size=8))
def test_resolve_hidden_dims_fixed_point(dims):
    """Resolving already-resolved dims is a fixed point (idempotence)."""
    cfg = ModelConfig(name="m", input_dim=8, output_dim=2, hidden_dims=list(dims))
    first = resolve_hidden_dims(cfg, None)
    cfg2 = ModelConfig(name="m", input_dim=8, output_dim=2, hidden_dims=first)
    assert resolve_hidden_dims(cfg2, None) == first


@given(
    hidden_dim=st.one_of(st.none(), st.integers(min_value=1, max_value=256)),
    n1=st.integers(min_value=0, max_value=20),
    n2=st.integers(min_value=0, max_value=20),
)
def test_compute_hidden_dims_monotonic_in_layers(hidden_dim, n1, n2):
    """Result length is monotone non-decreasing in num_layers.

    Only meaningful when hidden_dim is set (hidden_dim=None yields [] always).
    """
    assume(hidden_dim is not None)
    len1 = len(compute_hidden_dims(hidden_dim, n1, max_layers=64))
    len2 = len(compute_hidden_dims(hidden_dim, n2, max_layers=64))
    assert (len1 <= len2) == (n1 <= n2)
