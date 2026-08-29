"""
Quantization utilities for INT8 and ternary quantization.

Provides:
- Post-Training Quantization (PTQ) to INT8
- Quantization-Aware Training (QAT) preparation for INT8
- Dynamic quantization to INT8 (weights only)
- Ternary quantization (weights {-1, 0, +1}) with Straight-Through Estimator
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import convert, get_default_qat_qconfig, get_default_qconfig, prepare, prepare_qat, quantize_dynamic


def quantize_model_int8_ptq(
    model: nn.Module,
    calibration_data: list[torch.Tensor] | None = None,
    backend: str = "fbgemm",
) -> nn.Module:
    """
    Post-Training Quantization (PTQ) to INT8.

    Args:
        model: Model to quantize (must be on CPU).
        calibration_data: List of input tensors for calibration.
        backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM).

    Returns:
        Quantized model.
    """
    model.eval()
    model.cpu()

    # Set quantization config
    model.qconfig = get_default_qconfig(backend)

    # Prepare for quantization
    prepare(model, inplace=True)

    # Calibrate with sample data
    if calibration_data is not None:
        with torch.no_grad():
            for x in calibration_data:
                model(x)

    # Convert to quantized model
    quantized_model = convert(model, inplace=False)

    return quantized_model


def quantize_model_int8_qat(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    backend: str = "fbgemm",
) -> nn.Module:
    """
    Quantization-Aware Training (QAT) preparation for INT8.

    Args:
        model: Model to prepare for QAT.
        optimizer: Optional optimizer (will be recreated for quantized model).
        backend: Quantization backend.

    Returns:
        QAT-prepared model (train this, then call convert).
    """
    model.train()
    model.cpu()

    # Set QAT config
    model.qconfig = get_default_qat_qconfig(backend)

    # Prepare for QAT (inserts fake quant observers)
    prepare_qat(model, inplace=True)

    return model


def convert_qat_model(model: nn.Module) -> nn.Module:
    """
    Convert QAT-prepared model to quantized INT8 model.

    Call after QAT training is complete.

    Args:
        model: QAT-prepared model.

    Returns:
        Fully quantized INT8 model.
    """
    model.eval()
    model.cpu()
    quantized_model = convert(model, inplace=False)
    return quantized_model


def quantize_model_dynamic_int8(model: nn.Module) -> nn.Module:
    """
    Dynamic quantization to INT8 (weights only, activations float).

    Simplest quantization - no calibration needed, weights quantized to INT8,
    activations remain float. Good for LSTM/Transformer models.

    Args:
        model: Model to quantize.

    Returns:
        Dynamically quantized model.
    """
    model.eval()
    model.cpu()

    # Quantize Linear and LSTM layers dynamically
    quantized_model = quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM, nn.GRU},
        dtype=torch.qint8,
    )

    return quantized_model


def save_quantized_model(
    model: nn.Module,
    path: str,
    model_name: str = "quantized_model",
    input_shape: tuple[int, ...] | None = None,
) -> None:
    """Save quantized model with metadata."""
    from computronium.core.checkpoint import save_checkpoint

    save_checkpoint(
        path,
        {
            "model_state_dict": model.state_dict(),
            "quantized": True,
            "model_name": model_name,
            "input_shape": input_shape,
        },
        mkdir=True,
    )


def load_quantized_model(
    path: str,
    model_class: type,
    model_params: dict[str, object],
    input_shape: tuple[int, ...] | None = None,
) -> nn.Module:
    """Load quantized model and prepare for inference."""
    from computronium.deployment import load_model

    model, _ = load_model(path)
    model.eval()
    return model


def benchmark_quantized_model(
    model: nn.Module,
    quantized_model: nn.Module,
    test_data: list[torch.Tensor],
    num_runs: int = 100,
) -> dict[str, float]:
    """
    Benchmark original vs quantized model.

    Returns:
        Dict with latency comparison and accuracy metrics.
    """
    import time

    model.eval()
    quantized_model.eval()

    # Warmup
    with torch.no_grad():
        for x in test_data[:5]:
            model(x)
            quantized_model(x)

    # Benchmark original
    times_orig = []
    for x in test_data[:num_runs]:
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        times_orig.append(time.perf_counter() - start)

    # Benchmark quantized
    times_quant = []
    for x in test_data[:num_runs]:
        start = time.perf_counter()
        with torch.no_grad():
            _ = quantized_model(x)
        times_quant.append(time.perf_counter() - start)

    return {
        "orig_mean_ms": sum(times_orig) / len(times_orig) * 1000,
        "quant_mean_ms": sum(times_quant) / len(times_quant) * 1000,
        "speedup": sum(times_orig) / sum(times_quant),
        "orig_p99_ms": sorted(times_orig)[int(len(times_orig) * 0.99)] * 1000,
        "quant_p99_ms": sorted(times_quant)[int(len(times_quant) * 0.99)] * 1000,
    }


# --- Ternary Quantization Utilities (P2.17) ---


class TernaryQuantize(torch.autograd.Function):
    """
    Ternary quantization with Straight-Through Estimator.

    Forward: Quantize weights to {-1, 0, +1}
    Backward: Pass gradients through unchanged (STE)
    """

    @staticmethod
    def forward(ctx, weight: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        ctx.save_for_backward(weight)

        ternary = torch.zeros_like(weight)
        ternary[weight > threshold] = 1.0
        ternary[weight < -threshold] = -1.0

        return ternary

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (_weight,) = ctx.saved_tensors
        grad_weight = grad_output.clone()
        return grad_weight, None


class TernaryLinear(nn.Module):
    """Linear layer with ternary weights."""

    def __init__(self, in_features: int, out_features: int, threshold: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        nn.init.xavier_uniform_(self.weight, gain=0.8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ternary_weight = TernaryQuantize.apply(self.weight, self.threshold)
        return F.linear(x, ternary_weight, self.bias)

    def get_weight_stats(self) -> dict:
        w = self.weight.detach()
        threshold = self.threshold

        n_pos = (w > threshold).sum().item()
        n_neg = (w < -threshold).sum().item()
        n_zero = w.numel() - n_pos - n_neg

        total = w.numel()
        return {
            "positive": n_pos / total,
            "zero": n_zero / total,
            "negative": n_neg / total,
            "sparsity": n_zero / total,
        }


def quantize_model_ternary(
    model: nn.Module,
    threshold: float = 0.5,
) -> nn.Module:
    """
    Convert a model's Linear layers to TernaryLinear (weights {-1, 0, +1}).

    This is a post-training quantization that replaces nn.Linear with TernaryLinear
    and quantizes the weights using the Straight-Through Estimator.

    Args:
        model: Model to quantize.
        threshold: Threshold for ternary quantization.

    Returns:
        Model with TernaryLinear layers.
    """
    model = copy.deepcopy(model)
    model.eval()

    def replace_linear(module: nn.Module, prefix: str = "") -> nn.Module:
        for name, child in list(module.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                ternary_layer = TernaryLinear(
                    child.in_features,
                    child.out_features,
                    threshold,
                )
                with torch.no_grad():
                    ternary_layer.weight.copy_(child.weight)
                    if child.bias is not None:
                        ternary_layer.bias.copy_(child.bias)
                setattr(module, name, ternary_layer)
            else:
                replace_linear(child, full_name)
        return module

    return replace_linear(model)


def quantize_model_ternary_inplace(
    model: nn.Module,
    threshold: float = 0.5,
) -> nn.Module:
    """
    Convert a model's Linear layers to TernaryLinear in-place.

    Args:
        model: Model to quantize (modified in-place).
        threshold: Threshold for ternary quantization.

    Returns:
        The same model with TernaryLinear layers.
    """

    def replace_linear(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                ternary_layer = TernaryLinear(
                    child.in_features,
                    child.out_features,
                    threshold,
                )
                with torch.no_grad():
                    ternary_layer.weight.copy_(child.weight)
                    if child.bias is not None:
                        ternary_layer.bias.copy_(child.bias)
                setattr(module, name, ternary_layer)
            else:
                replace_linear(child)

    replace_linear(model)
    return model


def count_ternary_operations(model: nn.Module, seq_len: int = 1) -> dict:
    """
    Count ternary operations in a model for efficiency analysis.

    Args:
        model: Model with TernaryLinear layers.
        seq_len: Sequence length for recurrent operations.

    Returns:
        Dict with operation counts and sparsity metrics.
    """
    total_macs = 0
    ternary_macs = 0
    total_params = 0
    ternary_params = 0

    for module in model.modules():
        if isinstance(module, TernaryLinear):
            in_f = module.in_features
            out_f = module.out_features
            macs = in_f * out_f * seq_len
            total_macs += macs
            stats = module.get_weight_stats()
            sparsity = stats["sparsity"]
            ternary_params += module.weight.numel()
            ternary_macs += int(macs * (1 - sparsity))
        elif isinstance(module, nn.Linear):
            total_macs += module.in_features * module.out_features * seq_len
            total_params += module.weight.numel()

    return {
        "total_macs": total_macs,
        "ternary_macs": ternary_macs,
        "speedup_factor": total_macs / ternary_macs
        if ternary_macs > 0
        else float("inf"),
        "sparsity": 1 - (ternary_macs / total_macs) if total_macs > 0 else 0,
    }


__all__ = [
    "TernaryLinear",
    "TernaryQuantize",
    "benchmark_quantized_model",
    "convert_qat_model",
    "count_ternary_operations",
    "load_quantized_model",
    "quantize_model_dynamic_int8",
    "quantize_model_int8_ptq",
    "quantize_model_int8_qat",
    "quantize_model_ternary",
    "quantize_model_ternary_inplace",
    "save_quantized_model",
]