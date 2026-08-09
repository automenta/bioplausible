"""Tests for EqProp recurrent weight initialization (Plan 8 Track A3)."""

import torch
from bioplausible.core.config import ModelConfig
from bioplausible.zoo.models.eqprop import StandardEqProp


def _make_config(**overrides) -> ModelConfig:
    """Create a minimal ModelConfig with sensible defaults."""
    defaults = dict(
        name="test",
        input_dim=10,
        output_dim=5,
        hidden_dims=[20, 15],  # 2 hidden layers
        max_steps=3,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _get_wrec_weights(model: StandardEqProp) -> list[torch.Tensor]:
    """Extract W_rec weight tensors."""
    weights = []
    for layer in model.W_rec:
        actual = layer
        if hasattr(layer, "parametrizations") and hasattr(
            layer.parametrizations, "weight"
        ):
            actual = layer.parametrizations.weight.original
        # actual is already the Parameter (nn.Parameter)
        if isinstance(actual, torch.nn.Parameter):
            weights.append(actual)
        elif hasattr(actual, "weight"):
            weights.append(actual.weight)
        else:
            weights.append(actual)
    return weights


def test_wrec_init_zero_mode_yields_zeros():
    """w_rec_init='zero' produces all-zero W_rec weights."""
    config = _make_config(extra={"w_rec_init": "zero"})
    model = StandardEqProp(config=config)

    wrec_weights = _get_wrec_weights(model)
    for w in wrec_weights:
        assert torch.allclose(w, torch.zeros_like(w), atol=1e-6), (
            f"Expected zero weights, got max abs: {w.abs().max().item()}"
        )


def test_wrec_init_xavier_mode_yields_nonzeros():
    """w_rec_init='xavier' produces non-zero W_rec weights."""
    config = _make_config(extra={"w_rec_init": "xavier", "w_rec_gain": 0.1})
    model = StandardEqProp(config=config)

    wrec_weights = _get_wrec_weights(model)
    for w in wrec_weights:
        # Xavier init with gain=0.1 should produce small but non-zero weights
        assert not torch.allclose(w, torch.zeros_like(w), atol=1e-6), (
            "Expected non-zero weights with xavier init"
        )
        # Check they're in a reasonable range
        assert w.abs().max().item() > 0.001
        assert w.abs().max().item() < 1.0


def test_wrec_gain_affects_magnitude():
    """w_rec_gain controls the magnitude of xavier initialization."""
    config_small = _make_config(extra={"w_rec_init": "xavier", "w_rec_gain": 0.01})
    config_large = _make_config(extra={"w_rec_init": "xavier", "w_rec_gain": 1.0})

    model_small = StandardEqProp(config=config_small)
    model_large = StandardEqProp(config=config_large)

    wrec_small = _get_wrec_weights(model_small)
    wrec_large = _get_wrec_weights(model_large)

    for w_s, w_l in zip(wrec_small, wrec_large):
        std_small = w_s.std().item()
        std_large = w_l.std().item()
        # Larger gain should produce larger weights (roughly proportional)
        assert std_large > std_small * 10, (
            f"Expected larger gain to produce larger weights: "
            f"std_small={std_small:.6f}, std_large={std_large:.6f}"
        )


def test_wrec_init_knob_visible_in_constructor():
    """The w_rec_init knob is visible to the construction/search-space audit."""
    # This test ensures the config accepts the knobs and they reach the model
    config = _make_config(extra={"w_rec_init": "xavier", "w_rec_gain": 0.5})
    model = StandardEqProp(config=config)

    # The model should have stored these values
    assert hasattr(model, "w_rec_init")
    assert model.w_rec_init == "xavier"
    assert hasattr(model, "w_rec_gain")
    assert model.w_rec_gain == 0.5


def test_default_wrec_init_is_zero():
    """Default w_rec_init is 'zero' (backwards compatible)."""
    config = _make_config()
    config.extra.pop("w_rec_init", None)
    config.extra.pop("w_rec_gain", None)
    model = StandardEqProp(config=config)

    assert model.w_rec_init == "zero"
    assert model.w_rec_gain == 0.1  # default gain

    wrec_weights = _get_wrec_weights(model)
    for w in wrec_weights:
        assert torch.allclose(w, torch.zeros_like(w), atol=1e-6)
