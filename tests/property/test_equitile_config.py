"""Hypothesis property-based tests for EquiTileConfig.validate()
(Sprint 5.6).

Laws:
  - Any in-bounds configuration constructs without raising and preserves
    its field values (valid configs don't raise).
  - Violating any guarded bound raises ValueError at construction
    (__post_init__ -> validate()).
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from bioplausible.equitile.core.config import EquiTileConfig

mode_strat = st.sampled_from(["pc", "ep", "backprop"])


@st.composite
def valid_config_strat(draw):
    """Generate a fully in-bounds EquiTileConfig."""
    return EquiTileConfig(
        neurons_per_tile=draw(st.integers(min_value=1, max_value=256)),
        num_layers=draw(st.integers(min_value=1, max_value=64)),
        tiles_per_layer=draw(st.integers(min_value=1, max_value=32)),
        learning_rate=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        importance_lr=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        weight_decay=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        dropout=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        sparsity_threshold=draw(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
        ),
        importance_decay=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        inference_steps=draw(st.integers(min_value=0, max_value=100)),
        mode=draw(mode_strat),
    )


@given(cfg=valid_config_strat())
def test_valid_config_does_not_raise(cfg):
    """In-bounds configurations construct (validate) without raising."""
    assert cfg.neurons_per_tile >= 1
    assert cfg.num_layers >= 1
    assert cfg.tiles_per_layer >= 1


def test_neurons_per_tile_zero_raises():
    """neurons_per_tile = 0 raises ValueError."""
    with pytest.raises(ValueError):
        EquiTileConfig(neurons_per_tile=0)


@given(num_layers=st.integers(min_value=-32, max_value=0))
def test_num_layers_nonpositive_raises(num_layers):
    """num_layers <= 0 raises ValueError."""
    with pytest.raises(ValueError):
        EquiTileConfig(num_layers=num_layers)


@given(tiles=st.integers(min_value=-32, max_value=0))
def test_tiles_per_layer_nonpositive_raises(tiles):
    """tiles_per_layer <= 0 raises ValueError."""
    with pytest.raises(ValueError):
        EquiTileConfig(tiles_per_layer=tiles)


def test_negative_learning_rate_raises():
    """learning_rate < 0 raises ValueError."""
    with pytest.raises(ValueError):
        EquiTileConfig(learning_rate=-0.1)


@given(dropout=st.floats(min_value=1.0001, max_value=5.0, allow_nan=False))
def test_dropout_above_one_raises(dropout):
    """dropout > 1 raises ValueError."""
    with pytest.raises(ValueError):
        EquiTileConfig(dropout=dropout)


@given(sparsity=st.floats(min_value=1.0001, max_value=5.0, allow_nan=False))
def test_sparsity_threshold_above_one_raises(sparsity):
    """sparsity_threshold > 1 raises ValueError."""
    with pytest.raises(ValueError):
        EquiTileConfig(sparsity_threshold=sparsity)


@given(decay=st.floats(min_value=-1.0, max_value=-0.0001, allow_nan=False))
def test_importance_decay_negative_raises(decay):
    """importance_decay < 0 raises ValueError."""
    with pytest.raises(ValueError):
        EquiTileConfig(importance_decay=decay)


@given(steps=st.integers(min_value=-64, max_value=-1))
def test_negative_inference_steps_raises(steps):
    """inference_steps < 0 raises ValueError."""
    with pytest.raises(ValueError):
        EquiTileConfig(inference_steps=steps)


@pytest.mark.parametrize("mode", ["pc", "ep", "backprop"])
def test_valid_mode_constructs(mode):
    """Each documented mode constructs without raising."""
    EquiTileConfig(mode=mode)


def test_invalid_mode_raises():
    """An unknown mode raises ValueError."""
    with pytest.raises(ValueError):
        EquiTileConfig(mode="bogus")
