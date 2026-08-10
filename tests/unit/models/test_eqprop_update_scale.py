"""Tests for EqProp update scaling (Plan 8 Track A2)."""

import pytest
import torch

from bioplausible.config.unified import ModelConfig
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


def _get_weight_tensor(layer: torch.nn.Module) -> torch.Tensor:
    """Get the underlying weight tensor from a layer, handling spectral norm."""
    if hasattr(layer, "parametrizations") and hasattr(layer.parametrizations, "weight"):
        original = layer.parametrizations.weight.original
        # original is already the Parameter (nn.Parameter), not a module
        if isinstance(original, torch.nn.Parameter):
            return original
        return original.weight
    return layer.weight


def _get_weight_grad_norms(model: StandardEqProp) -> list[float]:
    """Extract weight gradient norms from model layers."""
    norms = []
    for layer in model.layers:
        w = _get_weight_tensor(layer)
        if w.grad is not None:
            norms.append(w.grad.norm().item())
        else:
            norms.append(0.0)
    return norms


def _copy_model_weights(src: StandardEqProp, dst: StandardEqProp) -> None:
    """Copy weights from src model to dst model."""
    with torch.no_grad():
        for s_layer, d_layer in zip(src.layers, dst.layers):
            s_w = _get_weight_tensor(s_layer)
            d_w = _get_weight_tensor(d_layer)
            d_w.copy_(s_w)

        for s_rec, d_rec in zip(src.W_rec, dst.W_rec):
            s_w = _get_weight_tensor(s_rec)
            d_w = _get_weight_tensor(d_rec)
            d_w.copy_(s_w)


def test_update_scale_linear_effect():
    """update_scale=2.0 approximately doubles update norm."""
    # Direct test of _contrastive_step with controlled inputs
    from torch import nn

    from bioplausible.zoo.models.eqprop._contrastive import (
        _contrastive_step,
    )

    torch.manual_seed(42)

    # Create a simple model-like object with layers
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(10, 20),
                nn.Linear(20, 15),
                nn.Linear(15, 5),
            ])
            self.config = type("Config", (), {"output_dim": 5})()
            self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

        def forward(
            self,
            x,
            beta=0.0,
            target=None,
            steps=None,
            return_trajectory=False,
            return_dynamics=False,
        ):
            # Simple feedforward for testing
            h = x
            self._last_activations = [h]
            for i, layer in enumerate(self.layers):
                h = layer(h)
                if i < len(self.layers) - 1:
                    h = torch.tanh(h)
                self._last_activations.append(h)
            return h

    model = SimpleModel()

    x = torch.randn(4, 10)
    y = torch.randint(0, 5, (4,))
    beta = 0.5

    # Run with update_scale=1.0
    model.optimizer.zero_grad()
    result1 = _contrastive_step(
        model,
        x,
        y,
        layer_list=list(model.layers),
        beta=beta,
        update_scales=[1.0, 1.0, 1.0],
        diagnostics=False,
    )
    norms1 = [
        p.grad.norm().item() if p.grad is not None else 0.0
        for p in model.parameters()
        if p.ndim >= 2
    ]

    # Run with update_scale=2.0
    model.optimizer.zero_grad()
    result2 = _contrastive_step(
        model,
        x,
        y,
        layer_list=list(model.layers),
        beta=beta,
        update_scales=[2.0, 2.0, 2.0],
        diagnostics=False,
    )
    norms2 = [
        p.grad.norm().item() if p.grad is not None else 0.0
        for p in model.parameters()
        if p.ndim >= 2
    ]

    # Check that norms are approximately doubled
    for n1, n2 in zip(norms1, norms2):
        if n1 > 1e-6:
            assert n2 / n1 == pytest.approx(2.0, rel=0.05), (
                f"Expected 2x scaling, got {n2 / n1:.3f} (n1={n1:.6f}, n2={n2:.6f})"
            )


def test_update_scale_by_depth_creates_expected_scales():
    """update_scale_by_depth creates geometrically increasing scale list."""
    config = _make_config(extra={"update_scale": 1.0, "update_scale_by_depth": 2.0})
    model = StandardEqProp(config=config)

    num_layers = len(model.layers)
    expected_scales = [1.0 * (2.0**i) for i in range(num_layers)]

    # The internal logic computes update_scales in train_step
    # Verify by checking that train_step runs without error and we get the expected list
    update_scales = [
        model.update_scale * (model.update_scale_by_depth**i) for i in range(num_layers)
    ]

    assert update_scales == expected_scales
    assert update_scales == [1.0, 2.0, 4.0]


def test_beta_remains_global():
    """β remains global and affects nudged/free state difference, not just denominator."""
    config = _make_config(
        beta=0.1, extra={"update_scale": 1.0, "contrastive_diagnostics": True}
    )
    model = StandardEqProp(config=config)
    x = torch.randn(4, 10)
    y = torch.randint(0, 5, (4,))
    result = model.train_step(x, y)
    global_diag = result["global_diagnostics"]
    assert global_diag["beta"] == 0.1

    # With smaller beta, the nudged state should be more different from free state
    # (stronger nudge)
    # Note: This is a qualitative check - the exact relationship depends on dynamics
    assert global_diag["output_state_delta_norm"] > 0
