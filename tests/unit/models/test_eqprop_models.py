"""Tests for EqProp model train_step implementations.

Covers models in bioplausible.zoo.models.eqprop that define train_step.
Tests are model-specific due to varying constructor signatures.
"""

import torch
from torch import nn

from bioplausible.core.config import ModelConfig
from bioplausible.zoo.models.eqprop.deep_ep import DirectedEP
from bioplausible.zoo.models.eqprop.eqprop_diffusion import EqPropDiffusion
from bioplausible.zoo.models.eqprop.finite_nudge_ep import FiniteNudgeEP
from bioplausible.zoo.models.eqprop.holomorphic_ep import HolomorphicEP
from bioplausible.zoo.models.eqprop.mom_eq import MomentumEquilibrium
from bioplausible.zoo.models.eqprop.sparse_eq import SparseEquilibrium
from bioplausible.zoo.models.eqprop.standard_eqprop import StandardEqProp


def _check_train_step(
    model: nn.Module, input_dim: int = 10, output_dim: int = 5
) -> dict:
    """Verify train_step returns a valid dict with expected keys."""
    x = torch.randn(4, input_dim)
    y = torch.randint(0, output_dim, (4,))
    result = model.train_step(x, y)
    assert isinstance(result, dict), (
        f"train_step should return dict, got {type(result)}"
    )
    assert "loss" in result, f"train_step result missing 'loss': {result.keys()}"
    assert isinstance(result["loss"], float), (
        f"loss should be float, got {type(result['loss'])}"
    )
    return result


def _check_forward(model: nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:
    """Verify forward returns a tensor with valid shape."""
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    assert isinstance(output, torch.Tensor), (
        f"forward should return tensor, got {type(output)}"
    )
    return output


def _make_config(**overrides) -> ModelConfig:
    """Create a minimal ModelConfig with sensible defaults."""
    defaults = dict(
        name="test",
        input_dim=10,
        output_dim=5,
        hidden_dims=[20],
        equilibrium_steps=3,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


# --- StandardEqProp ---


def test_standard_eqprop_train_step():
    """StandardEqProp.train_step returns proper dict."""
    model = StandardEqProp(config=_make_config())
    result = _check_train_step(model)
    assert result["loss"] >= 0


# --- DirectedEP (deep_ep) ---


def test_directed_ep_train_step():
    """DirectedEP.train_step returns proper dict."""
    model = DirectedEP(config=_make_config())
    result = _check_train_step(model)
    assert result["loss"] >= 0


# --- HolomorphicEP ---


def test_holomorphic_ep_train_step():
    """HolomorphicEP.train_step returns proper dict."""
    model = HolomorphicEP(config=_make_config())
    result = _check_train_step(model)
    assert result["loss"] >= 0


# --- FiniteNudgeEP ---


def test_finite_nudge_ep_train_step():
    """FiniteNudgeEP.train_step returns proper dict."""
    model = FiniteNudgeEP(config=_make_config())
    result = _check_train_step(model)
    assert result["loss"] >= 0


# --- SparseEquilibrium (config-based, BioModel) ---


def test_sparse_equilibrium_build():
    """SparseEquilibrium can be built with config."""
    model = SparseEquilibrium(config=_make_config())
    x = torch.randn(4, 10)
    out = _check_forward(model, x)
    assert out.shape == (4, 5)


def test_sparse_equilibrium_forward_extra_steps():
    """SparseEquilibrium.forward accepts steps parameter."""
    model = SparseEquilibrium(config=_make_config())
    x = torch.randn(4, 10)
    out = model(x, steps=5)
    assert out.shape == (4, 5)


# --- MomentumEquilibrium (config-based, BioModel) ---


def test_momentum_equilibrium_build():
    """MomentumEquilibrium can be built with config."""
    model = MomentumEquilibrium(config=_make_config())
    x = torch.randn(4, 10)
    out = _check_forward(model, x)
    assert out.shape == (4, 5)


# --- EqPropDiffusion (own train_step, positional args) ---


def test_eqprop_diffusion_build():
    """EqPropDiffusion can be built with positional args."""
    model = EqPropDiffusion(img_channels=1, hidden_channels=16)
    x = torch.randn(4, 1, 28, 28)
    t = torch.randint(0, 10, (4,))
    out = model(x, t)
    assert out.shape == (4, 1, 28, 28)


def test_eqprop_diffusion_train_step():
    """EqPropDiffusion.train_step returns dict with loss."""
    model = EqPropDiffusion(img_channels=1, hidden_channels=16)
    x = torch.randn(4, 1, 28, 28)
    result = model.train_step(x)
    assert isinstance(result, dict)
    assert "loss" in result
