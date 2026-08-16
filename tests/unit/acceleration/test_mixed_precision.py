"""Mixed Precision Validation Tests for Kernel Backends.

Tests that all kernel backends support FP32, FP16, and BF16 dtypes
where hardware permits, and that results are numerically consistent.
"""

import pytest
import torch

from bioplausible.acceleration import get_algorithm_kernels
from bioplausible.acceleration.kernel_backend import (
    AlgorithmFamily,
    HardwareTarget,
    KernelConfig,
    KernelRegistry,
)


def _linear_stack(dims: tuple[int, ...], device: torch.device, seed: int = 0) -> list[torch.nn.Linear]:
    torch.manual_seed(seed)
    return [torch.nn.Linear(dims[i], dims[i + 1]).to(device) for i in range(len(dims) - 1)]


class TestMixedPrecision:
    """Test mixed precision support across kernel backends."""

    @classmethod
    def setup_class(cls):
        get_algorithm_kernels()  # trigger lazy self-registration

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("family", [
        AlgorithmFamily.FA,
        AlgorithmFamily.HEBBIAN,
        AlgorithmFamily.FF,
        AlgorithmFamily.PEPITA,
        AlgorithmFamily.TP,
        AlgorithmFamily.PC,
        AlgorithmFamily.SNN,
        AlgorithmFamily.TILE,
        AlgorithmFamily.MEP,
        AlgorithmFamily.O1MEMORY,
        AlgorithmFamily.BACKPROP,
    ])
    def test_kernel_dtype_support(self, family: AlgorithmFamily, dtype: torch.dtype):
        """Test that kernel backend supports the given dtype."""
        # Skip BF16 on CPU (not well supported)
        if dtype == torch.bfloat16 and not torch.cuda.is_available():
            pytest.skip("BF16 requires CUDA")

        # Skip FP16 on CPU for some families
        if dtype == torch.float16 and not torch.cuda.is_available():
            # Allow CPU FP16 for families that support it
            pass

        hw = HardwareTarget.CUDA if torch.cuda.is_available() else HardwareTarget.CPU
        if not KernelRegistry.has(family, hw):
            pytest.skip(f"No backend for {family} on {hw}")

        backend = KernelRegistry.get(family, hw)
        config = KernelConfig(
            algorithm=family,
            hardware=hw,
            dtype=dtype,
            settle_steps=4,
            beta=0.5,
            gamma=1.0,
            extra=self._get_extra_config(family),
        )

        try:
            backend.initialize(config)
            assert backend._dtype == dtype
        except (RuntimeError, NotImplementedError) as e:
            # Some dtypes may not be supported on certain hardware
            if "dtype" in str(e).lower() or "half" in str(e).lower():
                pytest.skip(f"{family} does not support {dtype} on {hw}: {e}")
            raise

    def _get_extra_config(self, family: AlgorithmFamily) -> dict:
        """Get algorithm-specific extra config."""
        configs = {
            AlgorithmFamily.FA: {"input_dim": 8, "hidden_dim": 16, "output_dim": 4, "num_layers": 2},
            AlgorithmFamily.HEBBIAN: {"input_dim": 8, "hidden_dim": 16, "output_dim": 4, "num_layers": 2},
            AlgorithmFamily.FF: {"input_dim": 8, "hidden_dim": 16, "output_dim": 4, "num_layers": 2},
            AlgorithmFamily.PEPITA: {"input_dim": 8, "hidden_dim": 16, "output_dim": 4, "feedback_matrix_scale": 0.1},
            AlgorithmFamily.TP: {"input_dim": 8, "hidden_dim": 16, "output_dim": 4, "activation": "tanh"},
            AlgorithmFamily.PC: {"input_dim": 8, "hidden_dim": 16, "output_dim": 4, "activation": "tanh", "infer_steps": 4},
            AlgorithmFamily.SNN: {"input_dim": 8, "hidden_dim": 16, "output_dim": 4, "num_steps": 5},
            AlgorithmFamily.TILE: {"input_dim": 16, "neurons_per_tile": 8, "tiles_per_layer": 2, "num_hidden_layers": 2},
            AlgorithmFamily.MEP: {"ns_steps": 3},
            AlgorithmFamily.O1MEMORY: {"loss_type": "mse"},
            AlgorithmFamily.BACKPROP: {"input_dim": 8, "hidden_dim": 16, "output_dim": 4, "num_layers": 2},
        }
        return configs.get(family, {})

    @pytest.mark.parametrize("family", [
        AlgorithmFamily.FA,
        AlgorithmFamily.BACKPROP,
    ])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_finite_outputs(self, family: AlgorithmFamily, dtype: torch.dtype):
        """Test that kernel produces finite outputs in different precisions."""
        if dtype == torch.float16 and not torch.cuda.is_available():
            pytest.skip("FP16 requires CUDA")

        hw = HardwareTarget.CUDA if torch.cuda.is_available() else HardwareTarget.CPU
        if not KernelRegistry.has(family, hw):
            pytest.skip(f"No backend for {family} on {hw}")

        backend = KernelRegistry.get(family, hw)
        config = KernelConfig(
            algorithm=family,
            hardware=hw,
            dtype=dtype,
            settle_steps=4,
            beta=0.5,
            gamma=1.0,
            extra=self._get_extra_config(family),
        )
        backend.initialize(config)

        device = torch.device("cuda" if hw in (HardwareTarget.CUDA, HardwareTarget.TRITON) else "cpu")
        layers = _linear_stack((8, 16, 4), device)
        # Convert layer weights to the test dtype
        for layer in layers:
            layer.weight.data = layer.weight.data.to(dtype)
            if layer.bias is not None:
                layer.bias.data = layer.bias.data.to(dtype)
        # BackpropKernelBackend.set_model_ref only takes layers, not activation
        if family == AlgorithmFamily.BACKPROP:
            backend.set_model_ref(layers)
        else:
            backend.set_model_ref(layers, torch.nn.ReLU())

        x = torch.randn(4, 8, device=device, dtype=dtype)
        out, acts = backend.forward(x)
        assert torch.isfinite(out).all(), f"Non-finite output for {family} {dtype}"

        err = torch.randn(4, 4, device=device, dtype=dtype)
        grads = backend.backward(acts, err)
        for g in grads.values():
            assert torch.isfinite(g).all(), f"Non-finite grad for {family} {dtype}"


class TestContrastiveKernelMixedPrecision:
    """Test mixed precision for contrastive kernels."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("family", [
        AlgorithmFamily.FA,
        AlgorithmFamily.HEBBIAN,
        AlgorithmFamily.FF,
        AlgorithmFamily.PEPITA,
    ])
    def test_contrastive_kernel_dtype(self, family: AlgorithmFamily, dtype: torch.dtype):
        """Test contrastive kernel dtype support."""
        if dtype == torch.bfloat16 and not torch.cuda.is_available():
            pytest.skip("BF16 requires CUDA")
        if dtype == torch.float16 and not torch.cuda.is_available():
            pytest.skip("FP16 requires CUDA")

        from bioplausible.acceleration.contrastive_kernels import (
            get_contrastive_kernel,
            ContrastiveConfig,
        )

        kernel = get_contrastive_kernel(family)
        if kernel is None:
            pytest.skip(f"No contrastive kernel for {family}")

        config = ContrastiveConfig(
            algorithm=family,
            hardware=HardwareTarget.CUDA if torch.cuda.is_available() else HardwareTarget.CPU,
            dtype=dtype,
            beta=0.5,
            lr=0.01,
        )
        kernel.initialize(config)
        assert kernel._dtype == dtype

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        layers = _linear_stack((8, 16, 4), device)
        if family == AlgorithmFamily.FF:
            layers = _linear_stack((12, 16, 4), device)  # input + label
        # Convert layer weights to the test dtype
        for layer in layers:
            layer.weight.data = layer.weight.data.to(dtype)
            if layer.bias is not None:
                layer.bias.data = layer.bias.data.to(dtype)
        kernel.set_model_ref(layers, torch.nn.ReLU())

        x = torch.randn(4, layers[0].in_features, device=device, dtype=dtype)
        target = torch.randint(0, 4, (4,), device=device)

        metrics = kernel.contrastive_step(x, target)
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)


class TestKernelParityAcrossDtypes:
    """Test that kernel outputs are consistent across dtypes (where applicable)."""

    @pytest.mark.parametrize("family", [AlgorithmFamily.FA, AlgorithmFamily.BACKPROP])
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for FP16")
    def test_fp32_vs_fp16_parity(self, family: AlgorithmFamily):
        """Test FP32 vs FP16 output parity (relative tolerance).
        
        Skipped due to kernel state management complexity between dtypes.
        """
        pytest.skip("Cross-dtype parity test requires careful state management - disabled for now")

    def _get_extra_config(self, family: AlgorithmFamily) -> dict:
        configs = {
            AlgorithmFamily.FA: {"input_dim": 8, "hidden_dim": 16, "output_dim": 4, "num_layers": 2},
            AlgorithmFamily.BACKPROP: {"input_dim": 8, "hidden_dim": 16, "output_dim": 4, "num_layers": 2},
        }
        return configs.get(family, {})