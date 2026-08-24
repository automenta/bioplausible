"""Tests for EnergyTracker activation sparsity and hook lifecycle.

Verifies _estimate_activation_sparsity produces valid ranges and that
hooks are always cleaned up (no leak on exception).
"""

import pytest
import torch
from torch import nn

from computronium.core.profiling import EnergyTracker, _estimate_activation_sparsity


class TestActivationSparsity:
    """Direct tests of the _estimate_activation_sparsity function."""

    def test_returns_float_in_range(self):
        """Sparsity must be in [0, 1] for any model."""
        model = nn.Sequential(
            nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2)
        )
        x = torch.randn(4, 10)
        sparsity = _estimate_activation_sparsity(model, x)
        assert 0.0 <= sparsity <= 1.0

    def test_relu_gives_moderate_sparsity(self):
        """ReLU with random norm(0,1) input yields positive sparsity."""
        model = nn.Sequential(nn.Linear(100, 256), nn.ReLU(), nn.Linear(256, 10))
        x = torch.randn(8, 100)
        sparsity = _estimate_activation_sparsity(model, x)
        assert 0.15 <= sparsity <= 0.85

    def test_no_matching_modules_returns_zero(self):
        """Model without Linear/Conv/ReLU/GELU returns 0.0 sparsity."""

        class NoOpModel(nn.Module):
            def forward(self, x):
                return x

        model = NoOpModel()
        sparsity = _estimate_activation_sparsity(model, torch.randn(4, 10))
        assert sparsity == pytest.approx(0.0)

    def test_hook_cleanup_on_forward_exception(self):
        """Hooks are removed even when forward pass raises."""
        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
        before = sum(len(m._forward_hooks) for m in model.modules())

        bad_input = torch.randn(4, 999)
        with pytest.raises(RuntimeError):
            _estimate_activation_sparsity(model, bad_input)

        after = sum(len(m._forward_hooks) for m in model.modules())
        assert after == before, "Hooks leaked despite forward exception"


class TestEnergyTracker:
    """EnergyTracker integration with activation sparsity."""

    def test_tracker_sets_profile(self):
        """EnergyTracker records a valid EnergyProfile."""
        model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
        x = torch.randn(4, 10)

        with EnergyTracker(model, requires_backward=True) as et:
            out = model(x)
            out.sum().backward()

        prof = et.profile
        assert prof is not None
        assert 0.0 <= prof.activation_sparsity <= 1.0
        assert prof.forward_flops > 0
        assert prof.backward_flops > 0
        assert prof.param_count > 0
        assert prof.energy_proxy > 0

    def test_no_backward_no_flops(self):
        """requires_backward=False -> zero backward_flops."""
        model = nn.Linear(10, 2)
        with EnergyTracker(model, requires_backward=False) as et:
            model(torch.randn(4, 10))

        prof = et.profile
        assert prof is not None
        assert prof.backward_flops == 0
        assert not prof.requires_backward

    def test_conv_model_activation_sparsity(self):
        """_estimate_activation_sparsity works with Conv2d models directly."""
        model = nn.Sequential(
            nn.Conv2d(3, 8, 3),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 10),
        )
        x = torch.randn(2, 3, 16, 16)

        sparsity = _estimate_activation_sparsity(model, x)
        assert 0.0 <= sparsity <= 1.0

    def test_hook_cleanup_on_body_exception(self):
        """Hooks must be cleaned up when the with-body raises."""
        model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2))
        before = sum(len(m._forward_hooks) for m in model.modules())

        with pytest.raises(RuntimeError), EnergyTracker(model):
            msg = "simulated failure"
            raise RuntimeError(msg)

        after = sum(len(m._forward_hooks) for m in model.modules())
        assert after == before, "Hooks leaked despite with-body exception"

    def test_hook_cleanup_normal_exit(self):
        """Hooks are cleanly removed after normal exit."""
        model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2))
        before = sum(len(m._forward_hooks) for m in model.modules())

        with EnergyTracker(model):
            model(torch.randn(4, 10))

        after = sum(len(m._forward_hooks) for m in model.modules())
        assert after == before

    def test_gelu_model(self):
        """EnergyTracker handles GELU activations."""
        model = nn.Sequential(nn.Linear(20, 64), nn.GELU(), nn.Linear(64, 5))
        x = torch.randn(4, 20)

        with EnergyTracker(model) as et:
            model(x)

        assert et.profile is not None
        assert 0.0 <= et.profile.activation_sparsity <= 1.0
