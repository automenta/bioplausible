"""
EqProp-Torch Utilities

Helper functions for ONNX export, model verification, and training utilities.
"""

import os
import random
import sys
import time
from contextlib import contextmanager

from bioplausible.core.logging import get_logger

logger = get_logger()

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int = 42, device: str = "cpu") -> dict[str, str]:
    """
    Seed all random number generators for reproducibility.

    Single consolidated seeding API (Sprint 1.4): seeds Python's ``random``,
    NumPy, PyTorch (CPU), and — when ``device`` is ``cuda``/``gpu`` and CUDA is
    present — the CUDA generator(s) plus cuDNN deterministic/benchmark flags.
    Also captures the environment fingerprint so a ``biopl-repro-check`` run can
    prove two runs are bitwise identical.

    Args:
        seed: Random seed (default: 42)
        device: ``"cpu"`` (default), ``"cuda"``/``"gpu"`` (also seeds CUDA +
            cuDNN, and refuses to pretend determinism without CUDA).

    Returns:
        Environment fingerprint dict (see :func:`capture_environment`).

    Raises:
        RuntimeError: If ``device`` asks for CUDA seeding but CUDA is
            unavailable — a silent CPU fallback would silently defeat the
            bitwise-identical guarantee the caller is relying on.
    """
    want_cuda = device in ("cuda", "gpu") or device.startswith("cuda:")
    if want_cuda and not torch.cuda.is_available():
        raise RuntimeError(f"seed_everything device={device!r} but CUDA is unavailable")

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if want_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return capture_environment()


def capture_environment() -> dict[str, str]:
    """Capture a compact, hashable fingerprint of the execution environment.

    Returns a dict that stays stable within a machine/commit so the same-input
    guarantee can be asserted across two identical runs. Keys: ``git_commit``,
    ``torch_version``, ``cuda_version`` (or ``"n/a"``), ``python_version``.
    """
    git_commit = "unknown"
    try:
        import subprocess

        git_commit = (
            subprocess
            .check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except OSError, subprocess.CalledProcessError:
        pass

    return {
        "git_commit": git_commit,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.version.cuda else "n/a",
        "python_version": (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ),
    }


def deps_hash(environment: dict[str, str] | None = None) -> str:
    """Return a short digest of :func:`capture_environment` for rollup reporting."""
    import hashlib

    env = environment if environment is not None else capture_environment()
    canonical = "|".join(f"{k}={env[k]}" for k in sorted(env))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: tuple[int, ...],
    opset_version: int = 14,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    device: str = "cpu",
) -> None:
    """
    Export a PyTorch model to ONNX format.

    Args:
        model: PyTorch model to export
        output_path: Path to save .onnx file
        input_shape: Example input shape (e.g., (1, 784))
        opset_version: ONNX opset version
        dynamic_axes: Dynamic axis specification
        device: Device to use for export

    Example:
        >>> model = LoopedMLP(784, 256, 10)
        >>> export_to_onnx(model, "model.onnx", (1, 784))
    """
    # Handle compiled models
    model = _get_model_for_processing(model)

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # ONNX traces call every ``forward`` argument positionally — including the
    # keyword-only defaults (e.g. ``return_dynamics``) of equilibrium models —
    # which a ``forward(x, ..., *, kw_only)`` signature cannot accept. Wrap the
    # model so the exporter sees only the tensor input and inference defaults.
    model = _InferenceOnly(model)

    was_training = model.training
    model.eval()
    try:
        model = model.to(device)
        dummy_input = torch.randn(*input_shape, device=device)

        if dynamic_axes is None:
            dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}

        # Suppress PyTorch-internal buffer warnings from spectral_norm
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r".*cached_sn_weight.*assigned during export.*"
            )
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                opset_version=opset_version,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                dynamo=False,
            )
        logger.info("Model exported to %s", output_path)
    except (RuntimeError, ValueError, OSError) as e:
        raise RuntimeError(f"ONNX export failed: {e}") from e
    finally:
        if was_training:
            model.train()


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.

    Args:
        model: PyTorch model
        trainable_only: If True, only count trainable parameters

    Returns:
        Number of parameters
    """
    model = _get_model_for_processing(model)

    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def verify_spectral_norm(model: nn.Module) -> dict[str, float]:
    """
    Verify that all layers with spectral normalization have L <= 1.

    Args:
        model: PyTorch model

    Returns:
        Dict mapping layer names to their Lipschitz constants
    """
    model = _get_model_for_processing(model)

    lipschitz_values = {}

    for name, module in model.named_modules():
        # Check for spectral norm parametrization
        if _has_spectral_norm(module):
            spectral_norm = _compute_module_spectral_norm(module)
            if spectral_norm is not None:
                lipschitz_values[name] = spectral_norm

    return lipschitz_values


def _compute_module_spectral_norm(module: nn.Module) -> float | None:
    """Compute spectral norm for a module if possible."""
    with torch.no_grad():
        weight = getattr(module, "weight", None)
        if weight is not None and weight.dim() >= 2:
            return _compute_spectral_norm(weight)
    return None


def _has_spectral_norm(module: nn.Module) -> bool:
    """Check if a module has spectral normalization."""
    return hasattr(module, "parametrizations") and hasattr(
        module.parametrizations, "weight"
    )


def _compute_spectral_norm(weight: torch.Tensor) -> float:
    """Compute the spectral norm (largest singular value) of a weight tensor."""
    # Reshape for 2D computation if needed
    W_flat = weight.reshape(weight.shape[0], -1) if weight.dim() > 2 else weight
    s = torch.linalg.svdvals(W_flat)
    return s[0].item() if s.numel() > 0 else 0.0


def _get_model_for_processing(model: nn.Module) -> nn.Module:
    """Get the appropriate model for processing (handle compiled models)."""
    if hasattr(model, "_orig_mod"):
        return model._orig_mod
    return model


class _InferenceOnly(nn.Module):
    """Adapter exposing ``forward(x)`` only, for tracing/export.

    ONNX and torchscript tracing resolve every default argument of the wrapped
    model's ``forward`` and pass it positionally — including keyword-only
    flags — which multi-arg signatures (equilibrium models, etc.) reject. This
    adapter hides everything but the tensor input.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def compute_gradient_norm(model: nn.Module) -> float:
    """
    Compute the global gradient norm across all parameters.

    Args:
        model: PyTorch model

    Returns:
        Gradient norm
    """
    model = _get_model_for_processing(model)

    squared_norms = []
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            squared_norms.append(param_norm.item() ** 2)

    total_squared_norm = sum(squared_norms)
    return total_squared_norm**0.5


def estimate_memory_usage(
    model: nn.Module,
    input_shape: tuple[int, ...],
    batch_size: int = 1,
) -> dict[str, float]:
    """
    Estimate memory usage of a model.

    Args:
        model: PyTorch model
        input_shape: Input shape (without batch dimension)
        batch_size: Batch size for estimation

    Returns:
        Dict with memory estimates in MB
    """
    param_memory = _calculate_param_memory(model)
    grad_memory = param_memory  # Same as parameters
    activation_memory = _estimate_activation_memory(input_shape, batch_size)

    return {
        "parameters_mb": param_memory,
        "gradients_mb": grad_memory,
        "activations_mb": activation_memory,
        "total_mb": param_memory + grad_memory + activation_memory,
    }


def _calculate_param_memory(model: nn.Module) -> float:
    """Calculate memory used by model parameters."""
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6


def _estimate_activation_memory(input_shape: tuple[int, ...], batch_size: int) -> float:
    """Estimate memory used by activations."""
    # This is model-specific, here's a simple heuristic
    return batch_size * sum(input_shape) * 4 / 1e6  # 4 bytes per float32


def create_model_preset(preset_name: str, **overrides) -> nn.Module:
    """
    Create a model from a preset configuration.

    Args:
        preset_name: Name of preset ('mnist_small', 'mnist_large', 'cifar_conv', etc.)
        **overrides: Override default parameters

    Returns:
        Configured model

    Example:
        >>> model = create_model_preset("mnist_small", hidden_dim=512)
    """
    from bioplausible.zoo.models.eqprop.conv_eqprop import ConvEqProp
    from bioplausible.zoo.models.eqprop.looped_mlp import LoopedMLP

    presets = {
        "mnist_small": lambda: LoopedMLP(784, 128, 10, use_spectral_norm=True),
        "mnist_medium": lambda: LoopedMLP(784, 256, 10, use_spectral_norm=True),
        "mnist_large": lambda: LoopedMLP(784, 512, 10, use_spectral_norm=True),
        "cifar_conv": lambda: ConvEqProp(3, 64, 10),
        "cifar_mlp": lambda: LoopedMLP(3072, 512, 10, use_spectral_norm=True),
    }

    if preset_name not in presets:
        raise ValueError(
            f"Unknown preset '{preset_name}'. Available: {list(presets.keys())}"
        )

    # Note: overrides would require more sophisticated preset handling
    return presets[preset_name]()


# =============================================================================
# Profiling Utilities
# =============================================================================


@contextmanager
def simple_profiler(name: str):
    """
    Context manager for simple time profiling.

    Example:

        >>> with simple_profiler("Training Step"):
        >>>     train_step()
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        logger.debug("[%s] took %.2f ms", name, (end - start) * 1000)


def profile_model(
    model: nn.Module, input_shape: tuple[int, ...], device: str = "cpu", runs: int = 10
) -> dict[str, float]:
    """
    Run a simple performance profile on a model.

    Args:
        model: Model to profile
        input_shape: Input tensor shape (including batch dim)
        device: Device to run on
        runs: Number of runs for averaging

    Returns:
        Dictionary with 'avg_ms' and 'std_ms'
    """
    model = model.to(device)
    was_training = model.training
    model.eval()
    try:
        x = torch.randn(*input_shape, device=device)

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model(x)

        times = []
        with torch.no_grad():
            for _ in range(runs):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()

                _ = model(x)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)

        avg_ms = sum(times) / len(times)
        std_ms = (sum((t - avg_ms) ** 2 for t in times) / len(times)) ** 0.5

        return {"avg_ms": avg_ms, "std_ms": std_ms}
    finally:
        if was_training:
            model.train()


# Missing utils imported by models
def spectral_linear(
    in_features: int, out_features: int, use_sn: bool = True
) -> nn.Module:
    """Helper to create a linear layer with optional spectral normalization."""
    layer = nn.Linear(in_features, out_features)
    if use_sn:
        return torch.nn.utils.parametrizations.spectral_norm(layer)
    return layer


def spectral_conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    padding: int = 0,
    use_sn: bool = True,
) -> nn.Module:
    """Helper to create a conv2d layer with optional spectral normalization."""
    layer = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
    if use_sn:
        return torch.nn.utils.parametrizations.spectral_norm(layer)
    return layer


__all__ = [
    "capture_environment",
    "compute_gradient_norm",
    "count_parameters",
    "create_model_preset",
    "deps_hash",
    "estimate_memory_usage",
    "export_to_onnx",
    "profile_model",
    "seed_everything",
    "simple_profiler",
    "spectral_conv2d",
    "spectral_linear",
    "verify_spectral_norm",
]
