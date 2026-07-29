"""Forward-pass smoke tests for EqProp models with positional-arg constructors.

These models do not define their own train_step — they inherit from
EqPropModel or nn.Module. Each test verifies construction + forward pass.
"""

import torch

from bioplausible.zoo.models.eqprop.causal_transformer_eqprop import (
    CausalTransformerEqProp,
)
from bioplausible.zoo.models.eqprop.conv_eqprop import ConvEqProp
from bioplausible.zoo.models.eqprop.homeostatic import HomeostaticEqProp
from bioplausible.zoo.models.eqprop.lazy_eqprop import LazyEqProp
from bioplausible.zoo.models.eqprop.modern_conv_eqprop import ModernConvEqProp
from bioplausible.zoo.models.eqprop.neural_cube import NeuralCube
from bioplausible.zoo.models.eqprop.temporal_resonance import TemporalResonanceEqProp
from bioplausible.zoo.models.eqprop.ternary import TernaryEqProp
from bioplausible.zoo.models.eqprop.transformer_eqprop import TransformerEqProp


def _img_input() -> torch.Tensor:
    return torch.randn(2, 1, 28, 28)


def _flat_input() -> torch.Tensor:
    return torch.randn(2, 10)


def _seq_input() -> torch.Tensor:
    return torch.randint(0, 50, (2, 16))


# --- LazyEqProp ---


def test_lazy_eqprop_forward():
    m = LazyEqProp(input_dim=10, hidden_dim=20, output_dim=5)
    out = m(_flat_input())
    assert out.shape == (2, 5)


# --- ConvEqProp ---


def test_conv_eqprop_forward():
    m = ConvEqProp(input_channels=1, hidden_channels=16, output_dim=10)
    out = m(_img_input())
    assert out.shape == (2, 10)


# --- ModernConvEqProp ---


def test_modern_conv_eqprop_forward():
    m = ModernConvEqProp(hidden_channels=16, input_dim=784, output_dim=10)
    out = m(_img_input())
    assert out.shape == (2, 10)


# --- TransformerEqProp ---


def test_transformer_eqprop_forward():
    m = TransformerEqProp(
        vocab_size=50, hidden_dim=32, output_dim=27, num_layers=2, num_heads=2
    )
    out = m(_seq_input())
    assert out.shape == (2, 27)


# --- CausalTransformerEqProp ---


def test_causal_transformer_eqprop_forward():
    m = CausalTransformerEqProp(vocab_size=50, hidden_dim=32, num_layers=2, num_heads=2)
    out = m(_seq_input())
    assert out.shape == (2, 16, 50)


# --- HomeostaticEqProp ---


def test_homeostatic_eqprop_forward():
    m = HomeostaticEqProp(input_dim=10, hidden_dim=20, output_dim=5)
    out = m(_flat_input())
    assert out.shape == (2, 5)


def test_homeostatic_eqprop_forward_no_homeostasis():
    m = HomeostaticEqProp(input_dim=10, hidden_dim=20, output_dim=5)
    out = m(_flat_input(), apply_homeostasis=False)
    assert out.shape == (2, 5)


# --- TemporalResonanceEqProp ---


def test_temporal_resonance_forward():
    m = TemporalResonanceEqProp(input_dim=10, hidden_dim=20, output_dim=5)
    out = m(_flat_input())
    assert out.shape == (2, 5)


# --- NeuralCube ---


def test_neural_cube_forward():
    m = NeuralCube(input_dim=10, output_dim=5)
    out = m(_flat_input())
    assert out.shape == (2, 5)


def test_neural_cube_forward_with_steps():
    m = NeuralCube(input_dim=10, output_dim=5)
    out = m(_flat_input(), steps=5)
    assert out.shape == (2, 5)


# --- TernaryEqProp ---


def test_ternary_eqprop_forward():
    m = TernaryEqProp(input_dim=10, hidden_dim=20, output_dim=5)
    out = m(_flat_input())
    assert out.shape == (2, 5)
