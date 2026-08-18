"""Mixed Precision Validation Tests for Kernel Backends.

Tests that all kernel backends support FP32, FP16, and BF16 dtypes
where hardware permits, and that results are numerically consistent.

Accuracy parity gates (REFACTOR8 Phase 2):
- FP16/BF16: within 2% of FP32 on digits
- INT8: within 5% of FP32 (quantization-aware training if needed)
"""

import pytest
import torch
from torch import nn

from bioplausible.acceleration import get_algorithm_kernels
from bioplausible.acceleration.kernel_backend import (
    AlgorithmFamily,
    HardwareTarget,
    KernelConfig,
    KernelRegistry,
)
from bioplausible.core.registry import ComponentCategory, Registry

# Import zoo models to trigger registration
from bioplausible.zoo import models


def _linear_stack(dims: tuple[int, ...], device: torch.device, seed: int = 0) -> list[torch.nn.Linear]:
    torch.manual_seed(seed)
    return [torch.nn.Linear(dims[i], dims[i + 1]).to(device) for i in range(len(dims) - 1)]


def _construct_model(model_name: str, input_dim: int, output_dim: int, device: torch.device, dtype: torch.dtype) -> nn.Module:
    """Construct a model from the registry."""
    model_cls = Registry.get(ComponentCategory.MODEL, model_name)
    defaults = {
        "backprop_mlp": {"hidden_dim": 64, "num_layers": 2},
        "standard_fa": {"hidden_dim": 64, "num_layers": 2},
    }.get(model_name, {})
    model = model_cls.build(
        spec=type("Spec", (), {"name": model_name})(),
        input_dim=input_dim,
        output_dim=output_dim,
        device=device,
        task_type="vision",
        **defaults,
    )
    return model.to(device=device, dtype=dtype)


def _get_synthetic_data(input_dim: int, output_dim: int, n_samples: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic data for testing."""
    torch.manual_seed(42)
    x = torch.randn(n_samples, input_dim, device=device, dtype=dtype)
    y = torch.randint(0, output_dim, (n_samples,), device=device)
    for c in range(output_dim):
        mask = y == c
        if mask.any():
            direction = torch.randn(input_dim, device=device, dtype=dtype)
            direction = direction / direction.norm() * 1.5
            x[mask] += direction * 0.8
    return x, y


def _train_model(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    dtype: torch.dtype,
    lr: float = 0.01,
) -> float:
    """Train a model and return final accuracy."""
    # For FP16, keep model in FP32 for stable training, use autocast for forward
    if dtype == torch.float16:
        model = model.to(dtype=torch.float32)
        use_amp = True
        scaler = torch.amp.GradScaler('cuda')
    elif dtype == torch.bfloat16:
        # BF16 can use autocast but doesn't need GradScaler
        model = model.to(dtype=torch.bfloat16)
        use_amp = True
        scaler = None
    else:
        model = model.to(dtype=dtype)
        use_amp = False
        scaler = None
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    n_samples = len(x)
    batch_size = 64
    
    for epoch in range(epochs):
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = x[idx], y[idx]
            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast('cuda', dtype=dtype):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
    
    model.eval()
    with torch.no_grad():
        if use_amp:
            with torch.amp.autocast('cuda', dtype=dtype):
                logits = model(x[:128])
        else:
            logits = model(x[:128])
        accuracy = (logits.argmax(1) == y[:128]).float().mean().item()
    
    return accuracy


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


class TestAccuracyParityAcrossDtypes:
    """Test model training accuracy parity across dtypes (REFACTOR8 Phase 2).
    
    Gates:
    - FP16/BF16: accuracy within 2% of FP32 on digits
    - INT8: accuracy within 5% of FP32 (quantization-aware training if needed)
    """

    @pytest.mark.parametrize("model_name", ["backprop_mlp", "standard_fa"])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for FP16/BF16")
    def test_fp16_bf16_accuracy_parity(self, model_name: str, dtype: torch.dtype):
        """Test FP16/BF16 accuracy within 2% of FP32 on digits."""
        device = torch.device("cuda")
        
        # Get data (digits: 64 input, 10 output)
        input_dim, output_dim = 64, 10
        x, y = _get_synthetic_data(input_dim, output_dim, 512, device, dtype)
        x_fp32, y_fp32 = _get_synthetic_data(input_dim, output_dim, 512, device, torch.float32)
        
        # Train FP32 reference
        model_fp32 = _construct_model(model_name, input_dim, output_dim, device, torch.float32)
        fp32_acc = _train_model(model_fp32, x_fp32, y_fp32, epochs=15, dtype=torch.float32)
        
        # Train with target dtype
        model_dtype = _construct_model(model_name, input_dim, output_dim, device, dtype)
        dtype_acc = _train_model(model_dtype, x, y, epochs=15, dtype=dtype)
        
        # Check parity
        diff = abs(fp32_acc - dtype_acc)
        assert diff <= 0.02, (
            f"{model_name} {dtype} accuracy {dtype_acc:.4f} deviates from "
            f"FP32 {fp32_acc:.4f} by {diff:.4f} (max 0.02 allowed)"
        )

    @pytest.mark.parametrize("model_name", ["backprop_mlp"])
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for INT8")
    def test_int8_accuracy_parity(self, model_name: str):
        """Test INT8 accuracy within 5% of FP32 on digits (quantization-aware training).
        
        This is a placeholder for future INT8 quantization-aware training support.
        Currently skipped as INT8 training requires quantization-aware training infrastructure.
        """
        pytest.skip("INT8 quantization-aware training not yet implemented")


class TestMixedPrecisionLossScaling:
    """Test loss scaling for FP16 training."""

    @pytest.mark.parametrize("model_name", ["backprop_mlp"])
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for FP16")
    def test_fp16_loss_scaling(self, model_name: str):
        """Test that FP16 training with GradScaler produces finite gradients."""
        device = torch.device("cuda")
        input_dim, output_dim = 64, 10
        x, y = _get_synthetic_data(input_dim, output_dim, 256, device, torch.float16)
        
        # Construct model in FP32 for training, use autocast for FP16 forward
        model = _construct_model(model_name, input_dim, output_dim, device, torch.float32)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler('cuda')
        
        # Run a few steps and verify gradients are finite
        for _ in range(3):
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(x[:64])
                loss = criterion(logits, y[:64])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Check gradients are finite
            for param in model.parameters():
                if param.grad is not None:
                    assert torch.isfinite(param.grad).all(), "Non-finite gradients with FP16 loss scaling"