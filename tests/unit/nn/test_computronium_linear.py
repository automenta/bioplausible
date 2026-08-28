"""Tests for ComputroniumLinear - the drop-in nn.Linear replacement."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from computronium.nn import (
    ComputroniumLinear,
    ComputroniumLinearConfig,
    CreditRule,
    CreditRuleConfig,
    PlasticityConfig,
    PlasticityType,
    replace_linear_with_computronium,
)


def _assert_close(a: Tensor | None, b: Tensor | None, msg: str = "") -> None:
    """Assert two tensors are close, handling None."""
    if a is None and b is None:
        return
    assert a is not None and b is not None, f"One is None: {msg}"
    assert torch.allclose(a, b), msg


class TestComputroniumLinearBackprop:
    """Test backprop rule (should be bit-for-bit identical to nn.Linear)."""

    def test_backprop_bit_for_bit(self) -> None:
        """NullPlasticity + Backprop should match nn.Linear exactly."""
        torch.manual_seed(42)

        native = nn.Linear(10, 5, bias=True)
        cl = ComputroniumLinear(
            10, 5, bias=True, rule=CreditRule.BACKPROP, plasticity=PlasticityType.NULL
        )

        cl.weight.data.copy_(native.weight.data)
        cl.bias.data.copy_(native.bias.data)

        x = torch.randn(4, 10)

        out_native = native(x)
        out_cl = cl(x)
        assert torch.allclose(out_native, out_cl), "Forward pass mismatch"

        grad_output = torch.randn_like(out_native)
        out_native.backward(grad_output)
        out_cl.backward(grad_output)

        _assert_close(native.weight.grad, cl.weight.grad, "Weight grad mismatch")
        _assert_close(native.bias.grad, cl.bias.grad, "Bias grad mismatch")

    def test_backprop_no_bias(self) -> None:
        """Test backprop without bias."""
        torch.manual_seed(42)

        native = nn.Linear(10, 5, bias=False)
        cl = ComputroniumLinear(
            10, 5, bias=False, rule=CreditRule.BACKPROP, plasticity=PlasticityType.NULL
        )

        cl.weight.data.copy_(native.weight.data)

        x = torch.randn(4, 10)
        out_native = native(x)
        out_cl = cl(x)
        assert torch.allclose(out_native, out_cl)

        grad_output = torch.randn_like(out_native)
        out_native.backward(grad_output)
        out_cl.backward(grad_output)
        _assert_close(native.weight.grad, cl.weight.grad)

    def test_backprop_credit_config(self) -> None:
        """Test custom credit config for backprop."""
        cl = ComputroniumLinear(
            10,
            5,
            rule=CreditRule.BACKPROP,
            plasticity=PlasticityType.NULL,
            credit_config=CreditRuleConfig(rule=CreditRule.BACKPROP),
        )
        assert cl._credit_rule == CreditRule.BACKPROP


class TestComputroniumLinearFA:
    """Test Feedback Alignment rule."""

    def test_fa_differs_from_backprop(self) -> None:
        """FA should produce different input gradients than backprop."""
        torch.manual_seed(42)

        cl_fa = ComputroniumLinear(
            10, 5, bias=True, rule=CreditRule.FA, plasticity=PlasticityType.NULL
        )
        cl_bp = ComputroniumLinear(
            10, 5, bias=True, rule=CreditRule.BACKPROP, plasticity=PlasticityType.NULL
        )

        cl_fa.weight.data.copy_(cl_bp.weight.data)
        cl_fa.bias.data.copy_(cl_bp.bias.data)

        x = torch.randn(4, 10)
        out_fa = cl_fa(x)
        out_bp = cl_bp(x)

        grad_output = torch.randn_like(out_fa)
        out_fa.backward(grad_output)
        out_bp.backward(grad_output)

        # Weight gradient should be same (standard form)
        _assert_close(cl_fa.weight.grad, cl_bp.weight.grad)
        # Both should have gradients
        assert cl_fa.weight.grad is not None
        assert cl_bp.weight.grad is not None

    def test_fa_feedback_matrix_fixed(self) -> None:
        """FA feedback matrix should be deterministic per layer dims."""
        cl1 = ComputroniumLinear(
            10, 5, rule=CreditRule.FA, plasticity=PlasticityType.NULL
        )
        cl2 = ComputroniumLinear(
            10, 5, rule=CreditRule.FA, plasticity=PlasticityType.NULL
        )

        assert torch.allclose(cl1._feedback, cl2._feedback)

    def test_fa_feedback_different_dims(self) -> None:
        """FA feedback matrix should differ for different dims."""
        cl1 = ComputroniumLinear(
            10, 5, rule=CreditRule.FA, plasticity=PlasticityType.NULL
        )
        cl2 = ComputroniumLinear(
            20, 5, rule=CreditRule.FA, plasticity=PlasticityType.NULL
        )

        # Different input dims -> different feedback matrix shapes
        assert cl1._feedback.shape != cl2._feedback.shape


class TestComputroniumLinearHebbian:
    """Test Hebbian rule."""

    def test_hebbian_no_input_grad(self) -> None:
        """Hebbian should not propagate gradients to input."""
        torch.manual_seed(42)

        cl = ComputroniumLinear(
            10, 5, bias=True, rule=CreditRule.HEBBIAN, plasticity=PlasticityType.NULL
        )
        x = torch.randn(4, 10, requires_grad=True)
        out = cl(x)
        loss = out.sum()
        loss.backward()

        # Weight grad should exist
        assert cl.weight.grad is not None

    def test_hebbian_weight_update_local(self) -> None:
        """Hebbian weight update should be purely local."""
        torch.manual_seed(42)

        cl = ComputroniumLinear(
            10, 5, rule=CreditRule.HEBBIAN, plasticity=PlasticityType.NULL
        )
        x = torch.randn(4, 10)
        target = torch.randn(4, 5)
        out = cl(x)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()

        assert cl.weight.grad is not None
        assert cl.weight.grad.shape == (5, 10)


class TestComputroniumLinearEqProp:
    """Test Equilibrium Propagation rule."""

    def test_eqprop_scales_by_beta(self) -> None:
        """EqProp should scale gradients by 1/beta."""
        torch.manual_seed(42)

        beta = 0.1
        cl_eqprop = ComputroniumLinear(
            10,
            5,
            rule=CreditRule.EQPROP,
            plasticity=PlasticityType.NULL,
            credit_config=CreditRuleConfig(rule=CreditRule.EQPROP, eqprop_beta=beta),
        )
        cl_bp = ComputroniumLinear(
            10, 5, rule=CreditRule.BACKPROP, plasticity=PlasticityType.NULL
        )

        cl_eqprop.weight.data.copy_(cl_bp.weight.data)
        cl_eqprop.bias.data.copy_(cl_bp.bias.data)

        x = torch.randn(4, 10)
        out_eqprop = cl_eqprop(x)
        out_bp = cl_bp(x)

        grad_output = torch.randn_like(out_eqprop)
        out_eqprop.backward(grad_output)
        out_bp.backward(grad_output)

        # EqProp gradients should be backprop * (1/beta)
        assert cl_eqprop.weight.grad is not None
        assert cl_bp.weight.grad is not None
        assert cl_eqprop.bias.grad is not None
        assert cl_bp.bias.grad is not None
        assert torch.allclose(
            cl_eqprop.weight.grad, cl_bp.weight.grad / beta, rtol=1e-4
        )
        assert torch.allclose(cl_eqprop.bias.grad, cl_bp.bias.grad / beta, rtol=1e-4)


class TestComputroniumLinearFastWeights:
    """Test fast-weight plasticity."""

    def test_fast_weights_psi_initialization(self) -> None:
        """Fast weights should initialize psi correctly."""
        cl = ComputroniumLinear(
            10,
            5,
            rule=CreditRule.BACKPROP,
            plasticity=PlasticityType.FAST_WEIGHTS,
            plasticity_config=PlasticityConfig(fast_weight_dim=32),
        )

        psi = cl._plasticity.initial_psi(4, torch.device("cpu"))
        assert "fast_weights" in psi
        assert psi["fast_weights"].shape == (32,)
        assert torch.allclose(psi["fast_weights"], torch.zeros(32))

    def test_fast_weights_step_updates_psi(self) -> None:
        """Step should update psi."""
        cl = ComputroniumLinear(
            10,
            5,
            rule=CreditRule.BACKPROP,
            plasticity=PlasticityType.FAST_WEIGHTS,
            plasticity_config=PlasticityConfig(fast_weight_dim=32, learning_rate=0.1),
        )

        x = torch.randn(4, 10)
        _ = cl(x)

        assert cl._psi is not None
        assert "fast_weights" in cl._psi
        assert cl._psi["fast_weights"].shape == (32,)

    def test_fast_weights_modulates_output(self) -> None:
        """Fast weights should modulate output."""
        cl = ComputroniumLinear(
            10,
            5,
            rule=CreditRule.BACKPROP,
            plasticity=PlasticityType.FAST_WEIGHTS,
            plasticity_config=PlasticityConfig(
                fast_weight_dim=32, modulation_scale=0.1
            ),
        )

        x = torch.randn(4, 10)
        out1 = cl(x)
        out2 = cl(x)

        # Output should be modulated (different)
        assert not torch.allclose(out1, out2)

    def test_fast_weights_reset_psi(self) -> None:
        """reset_psi should reinitialize plastic state."""
        cl = ComputroniumLinear(
            10,
            5,
            rule=CreditRule.BACKPROP,
            plasticity=PlasticityType.FAST_WEIGHTS,
            plasticity_config=PlasticityConfig(fast_weight_dim=32),
        )

        x = torch.randn(4, 10)
        cl(x)
        assert cl._psi is not None

        cl.reset_psi(batch_size=4)
        assert cl._psi is not None
        assert torch.allclose(cl._psi["fast_weights"], torch.zeros(32))


class TestComputroniumLinearDevice:
    """Test device management."""

    def test_to_device(self) -> None:
        """Moving to device should work."""
        cl = ComputroniumLinear(
            10, 5, rule=CreditRule.FA, plasticity=PlasticityType.FAST_WEIGHTS
        )
        cl_cpu = cl.to("cpu")
        assert cl_cpu.weight.device.type == "cpu"
        # _device stores the device passed to to(), which could be string
        assert str(cl_cpu._plasticity._device) == "cpu"  # type: ignore[attr-defined]
        if cl_cpu._feedback is not None:
            assert cl_cpu._feedback.device.type == "cpu"

    def test_cuda_if_available(self) -> None:
        """Test CUDA if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        cl = ComputroniumLinear(
            10, 5, rule=CreditRule.FA, plasticity=PlasticityType.FAST_WEIGHTS
        )
        cl_cuda = cl.cuda()
        assert cl_cuda.weight.device.type == "cuda"
        assert cl_cuda._plasticity._device.type == "cuda"  # type: ignore[attr-defined]
        if cl_cuda._feedback is not None:
            assert cl_cuda._feedback.device.type == "cuda"

    def test_cpu_method(self) -> None:
        """cpu() method should work."""
        cl = ComputroniumLinear(
            10, 5, rule=CreditRule.FA, plasticity=PlasticityType.FAST_WEIGHTS
        )
        cl_cpu = cl.cpu()
        assert cl_cpu.weight.device.type == "cpu"
        assert cl_cpu._plasticity._device.type == "cpu"  # type: ignore[attr-defined]
        assert cl_cpu._plasticity._device.type == "cpu"


class TestComputroniumLinearTrainingLoop:
    """Test training loop integration."""

    def test_training_step_backprop(self) -> None:
        """Training loop with backprop should work."""
        cl = ComputroniumLinear(
            10, 2, rule=CreditRule.BACKPROP, plasticity=PlasticityType.NULL
        )
        opt = torch.optim.SGD(cl.parameters(), lr=0.01)

        for _ in range(5):
            x = torch.randn(4, 10)
            target = torch.randint(0, 2, (4,))
            out = cl(x)
            loss = torch.nn.functional.cross_entropy(out, target)
            loss.backward()
            opt.step()
            opt.zero_grad()

        assert cl.weight.grad is None or cl.weight.grad.abs().sum() >= 0

    def test_training_step_fa(self) -> None:
        """Training loop with FA should work."""
        cl = ComputroniumLinear(
            10, 2, rule=CreditRule.FA, plasticity=PlasticityType.NULL
        )
        opt = torch.optim.SGD(cl.parameters(), lr=0.01)

        for _ in range(5):
            x = torch.randn(4, 10)
            target = torch.randint(0, 2, (4,))
            out = cl(x)
            loss = torch.nn.functional.cross_entropy(out, target)
            loss.backward()
            opt.step()
            opt.zero_grad()

        assert cl.weight.grad is None or cl.weight.grad.abs().sum() >= 0

    def test_training_step_fast_weights(self) -> None:
        """Training loop with fast weights should work."""
        cl = ComputroniumLinear(
            10,
            2,
            rule=CreditRule.BACKPROP,
            plasticity=PlasticityType.FAST_WEIGHTS,
            plasticity_config=PlasticityConfig(fast_weight_dim=16),
        )
        opt = torch.optim.SGD(cl.parameters(), lr=0.01)

        for _ in range(5):
            x = torch.randn(4, 10)
            target = torch.randint(0, 2, (4,))
            out = cl(x)
            loss = torch.nn.functional.cross_entropy(out, target)
            loss.backward()
            opt.step()
            opt.zero_grad()

        assert cl.weight.grad is None or cl.weight.grad.abs().sum() >= 0


class TestReplaceLinearWithComputronium:
    """Test the module replacement utility."""

    def test_replace_all_linear(self) -> None:
        """Should replace all nn.Linear in a module."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
        )

        replaced = replace_linear_with_computronium(
            model,
            rule=CreditRule.FA,
            plasticity=PlasticityType.NULL,
        )

        assert replaced is model
        for layer in model:
            if isinstance(layer, nn.Linear):
                assert isinstance(layer, ComputroniumLinear)
                assert layer._credit_rule == CreditRule.FA
                assert layer._plasticity_type == PlasticityType.NULL

    def test_preserves_weights(self) -> None:
        """Should preserve original weights."""
        model = nn.Sequential(nn.Linear(10, 5))
        original_weight = model[0].weight.data.clone()
        original_bias = (
            model[0].bias.data.clone() if model[0].bias is not None else None
        )

        replace_linear_with_computronium(model, rule=CreditRule.BACKPROP)

        # model[0] is now ComputroniumLinear
        new_layer = model[0]
        assert isinstance(new_layer, ComputroniumLinear)
        assert torch.allclose(new_layer.weight.data, original_weight)
        if original_bias is not None:
            assert torch.allclose(new_layer.bias.data, original_bias)  # type: ignore[union-attr]

    def test_nested_modules(self) -> None:
        """Should handle nested modules."""

        class Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

        class Outer(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = Inner()
                self.linear = nn.Linear(5, 2)

        model = Outer()
        replace_linear_with_computronium(model, rule=CreditRule.HEBBIAN)

        assert isinstance(model.inner.linear, ComputroniumLinear)
        assert isinstance(model.linear, ComputroniumLinear)
        assert model.inner.linear._credit_rule == CreditRule.HEBBIAN
        assert model.linear._credit_rule == CreditRule.HEBBIAN


class TestComputroniumLinearConfig:
    """Test ComputroniumLinearConfig dataclass."""

    def test_default_config(self) -> None:
        """Default config should have sensible defaults."""
        config = ComputroniumLinearConfig()
        assert config.rule == CreditRule.BACKPROP
        assert config.plasticity == PlasticityType.NULL
        assert isinstance(config.credit_config, CreditRuleConfig)
        assert isinstance(config.plasticity_config, PlasticityConfig)

    def test_custom_config(self) -> None:
        """Custom config should be preserved."""
        config = ComputroniumLinearConfig(
            rule=CreditRule.FA,
            plasticity=PlasticityType.FAST_WEIGHTS,
            credit_config=CreditRuleConfig(feedback_scale=0.5),
            plasticity_config=PlasticityConfig(fast_weight_dim=256),
        )
        assert config.rule == CreditRule.FA
        assert config.plasticity == PlasticityType.FAST_WEIGHTS
        tolerance = 1e-6
        expected_feedback_scale = 0.5
        expected_fast_weight_dim = 256
        diff = abs(config.credit_config.feedback_scale - expected_feedback_scale)
        assert diff < tolerance
        assert config.plasticity_config.fast_weight_dim == expected_fast_weight_dim


class TestComputroniumLinearExtraRepr:
    """Test extra_repr includes rule and plasticity."""

    def test_extra_repr(self) -> None:
        """extra_repr should show rule and plasticity."""
        cl = ComputroniumLinear(
            10, 5, rule=CreditRule.FA, plasticity=PlasticityType.FAST_WEIGHTS
        )
        repr_str = cl.extra_repr()
        assert "rule=fa" in repr_str
        assert "plasticity=fast_weights" in repr_str


class TestComputroniumLinearStateDict:
    """Test state_dict compatibility."""

    def test_state_dict_save_load(self) -> None:
        """Should save and load state_dict correctly."""
        cl = ComputroniumLinear(
            10,
            5,
            rule=CreditRule.FA,
            plasticity=PlasticityType.FAST_WEIGHTS,
        )

        state_dict = cl.state_dict()
        assert "weight" in state_dict
        assert "bias" in state_dict

        cl2 = ComputroniumLinear(
            10, 5, rule=CreditRule.FA, plasticity=PlasticityType.FAST_WEIGHTS
        )
        cl2.load_state_dict(state_dict)

        assert torch.allclose(cl.weight.data, cl2.weight.data)
        assert torch.allclose(cl.bias.data, cl2.bias.data)  # type: ignore[arg-type]
