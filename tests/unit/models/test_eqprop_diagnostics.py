"""Tests for EqProp contrastive diagnostics (Plan 8 Track A1)."""

import torch

from bioplausible.config.unified import ModelConfig
from bioplausible.zoo.models.eqprop import DirectedEP, StandardEqProp


def _make_config(**overrides) -> ModelConfig:
    """Create a minimal ModelConfig with sensible defaults."""
    defaults = dict(
        name="test",
        input_dim=10,
        output_dim=5,
        hidden_dims=[20, 15],  # 2 hidden layers
        max_steps=3,
        extra={"contrastive_diagnostics": True},
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def test_eqprop_diagnostics_keys_exist():
    """Diagnostic keys exist when enabled."""
    model = StandardEqProp(config=_make_config())
    x = torch.randn(4, 10)
    y = torch.randint(0, 5, (4,))
    result = model.train_step(x, y)

    assert "layer_diagnostics" in result
    assert "global_diagnostics" in result
    assert isinstance(result["layer_diagnostics"], list)
    assert len(result["layer_diagnostics"]) == 3  # 2 hidden + output layer

    layer_diag = result["layer_diagnostics"][0]
    expected_keys = {
        "layer",
        "pre_state_delta_norm",
        "post_state_delta_norm",
        "weight_grad_norm",
        "bias_grad_norm",
        "update_scale",
    }
    assert set(layer_diag.keys()) == expected_keys


def test_eqprop_diagnostics_values_finite():
    """Diagnostic values are finite for a small model."""
    model = StandardEqProp(config=_make_config())
    x = torch.randn(4, 10)
    y = torch.randint(0, 5, (4,))
    result = model.train_step(x, y)

    for layer_diag in result["layer_diagnostics"]:
        assert isinstance(layer_diag["layer"], int)
        for key, val in layer_diag.items():
            if key == "layer":
                continue
            assert isinstance(val, float), f"{key} should be float"
            assert torch.isfinite(torch.tensor(val)), f"{key} should be finite: {val}"

    global_diag = result["global_diagnostics"]
    for key, val in global_diag.items():
        assert isinstance(val, (float, int)), f"{key} should be numeric"
        if isinstance(val, float):
            assert torch.isfinite(torch.tensor(val)), f"{key} should be finite: {val}"


def test_eqprop_output_layer_delta_nonzero_when_beta_positive():
    """Output-layer delta is non-zero when beta > 0."""
    config = _make_config(beta=0.5)
    model = StandardEqProp(config=config)
    x = torch.randn(4, 10)
    y = torch.randint(0, 5, (4,))
    result = model.train_step(x, y)

    global_diag = result["global_diagnostics"]
    # With beta > 0, there should be a non-zero output state delta
    assert global_diag["output_state_delta_norm"] > 0
    assert global_diag["beta"] == 0.5


def test_eqprop_diagnostics_disabled_by_default():
    """Diagnostics are not included when not explicitly enabled."""
    config = _make_config()
    config.extra.pop("contrastive_diagnostics", None)
    model = StandardEqProp(config=config)
    x = torch.randn(4, 10)
    y = torch.randint(0, 5, (4,))
    result = model.train_step(x, y)

    assert "layer_diagnostics" not in result
    assert "global_diagnostics" not in result
    assert "loss" in result
    assert "accuracy" in result


def test_directed_ep_diagnostics_include_feedback():
    """DirectedEP diagnostics work with feedback layers."""
    config = _make_config()
    model = DirectedEP(config=config)
    x = torch.randn(4, 10)
    y = torch.randint(0, 5, (4,))
    result = model.train_step(x, y)

    assert "layer_diagnostics" in result
    assert len(result["layer_diagnostics"]) == 3  # 2 hidden + output layer


def test_eqprop_diagnostics_via_config_extra():
    """Diagnostics can be enabled via config.extra."""
    config = _make_config(extra={"contrastive_diagnostics": True})
    model = StandardEqProp(config=config)
    x = torch.randn(4, 10)
    y = torch.randint(0, 5, (4,))
    result = model.train_step(x, y)

    assert "layer_diagnostics" in result
    assert "global_diagnostics" in result
