"""Tests for zoo utility functions (spectral_norm wrappers, Lipschitz estimation).

Verifies spectral_linear, spectral_conv2d, estimate_lipschitz, and
related helpers produce the expected types and behaviors.
"""

import torch
from torch import nn

from bioplausible.zoo.utils import (
    _get_layer_weight,
    _has_spectral_norm,
    _reshape_weight_for_power_iteration,
    estimate_lipschitz,
    spectral_conv2d,
    spectral_linear,
)


class TestSpectralLinear:
    """spectral_linear factory."""

    def test_creates_linear(self):
        layer = spectral_linear(10, 5, use_sn=False)
        assert isinstance(layer, nn.Linear)
        assert layer.in_features == 10
        assert layer.out_features == 5

    def test_with_spectral_norm(self):
        layer = spectral_linear(10, 5, use_sn=True)
        assert isinstance(layer, nn.Linear)
        assert _has_spectral_norm(layer)

    def test_without_spectral_norm(self):
        layer = spectral_linear(10, 5, use_sn=False)
        assert not _has_spectral_norm(layer)

    def test_bias_default_true(self):
        layer = spectral_linear(10, 5)
        assert layer.bias is not None

    def test_bias_false(self):
        layer = spectral_linear(10, 5, bias=False)
        assert layer.bias is None


class TestSpectralConv2d:
    """spectral_conv2d factory."""

    def test_creates_conv2d(self):
        layer = spectral_conv2d(3, 16, kernel_size=3, use_sn=False)
        assert isinstance(layer, nn.Conv2d)
        assert layer.in_channels == 3
        assert layer.out_channels == 16

    def test_with_spectral_norm(self):
        layer = spectral_conv2d(3, 16, kernel_size=3, use_sn=True)
        assert _has_spectral_norm(layer)

    def test_without_spectral_norm(self):
        layer = spectral_conv2d(3, 16, kernel_size=3, use_sn=False)
        assert not _has_spectral_norm(layer)

    def test_stride_default(self):
        layer = spectral_conv2d(3, 16, kernel_size=3)
        assert layer.stride == (1, 1)

    def test_custom_stride(self):
        layer = spectral_conv2d(3, 16, kernel_size=3, stride=2)
        assert layer.stride == (2, 2)


class TestGetLayerWeight:
    """_get_layer_weight helper."""

    def test_linear_weight(self):
        layer = nn.Linear(10, 5)
        w = _get_layer_weight(layer)
        assert w is not None
        assert w.shape == (5, 10)

    def test_conv2d_weight(self):
        layer = nn.Conv2d(3, 16, 3)
        w = _get_layer_weight(layer)
        assert w is not None

    def test_unknown_module(self):
        class CustomLayer(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x

        layer = CustomLayer()
        w = _get_layer_weight(layer)
        assert w is None


class TestReshapeWeight:
    """_reshape_weight_for_power_iteration."""

    def test_linear_2d_stays_2d(self):
        w = torch.randn(5, 10)
        reshaped = _reshape_weight_for_power_iteration(w)
        assert reshaped.shape == (5, 10)

    def test_conv_4d_flattens(self):
        w = torch.randn(16, 3, 3, 3)  # Conv2d weight
        reshaped = _reshape_weight_for_power_iteration(w)
        assert reshaped.ndim == 2
        assert reshaped.shape[0] == 16
        assert reshaped.shape[1] == 27  # 3*3*3


class TestEstimateLipschitz:
    """estimate_lipschitz via power iteration."""

    def test_linear_layer(self):
        layer = nn.Linear(10, 5)
        nn.init.eye_(layer.weight)
        lip = estimate_lipschitz(layer, iterations=10)
        assert isinstance(lip, float)
        assert lip > 0
        assert lip <= 1.0 + 1e-3  # identity matrix has spectral norm 1

    def test_identity_lipschitz(self):
        layer = nn.Linear(5, 5)
        nn.init.eye_(layer.weight)
        lip = estimate_lipschitz(layer, iterations=20)
        assert abs(lip - 1.0) < 0.05, f"Expected ~1.0, got {lip}"

    def test_spectral_normed_layer(self):
        layer = spectral_linear(10, 5, use_sn=True)
        lip = estimate_lipschitz(layer, iterations=5)
        assert isinstance(lip, float)
        assert lip > 0

    def test_conv2d_layer(self):
        layer = nn.Conv2d(3, 8, 3)
        lip = estimate_lipschitz(layer, iterations=5)
        assert isinstance(lip, float)
        assert lip > 0

    def test_small_layer_fast(self):
        """Tiny layer should converge quickly."""
        layer = nn.Linear(2, 2)
        nn.init.eye_(layer.weight)
        lip = estimate_lipschitz(layer, iterations=3)
        assert lip > 0
