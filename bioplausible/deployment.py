"""
Bioplausible Deployment Utilities

Model export, serialization, and deployment utilities for production use.

Features:
- ONNX export for cross-platform deployment
- TorchScript compilation for optimized inference
- Model serialization/deserialization
- Inference optimization
- Batch prediction utilities
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from bioplausible.core.checkpoint import (
    load_checkpoint_into_model,
    save_checkpoint,
)
from bioplausible.core.logging import get_logger
from bioplausible.utils import count_parameters


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    """Request body for deployment prediction endpoint."""

    data: list[list[float]] | list[float]
    shape: list[int] | None = None


logger = get_logger()


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Metadata about an exported model."""

    model_name: str
    model_params: dict[str, object]
    optimizer_name: str | None
    optimizer_params: dict[str, object] | None
    training_metrics: dict[str, float]
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    num_parameters: int
    export_format: str
    export_path: str


class ModelExporter:
    """
    Export Bioplausible models for deployment.

    Supported formats:
    - ONNX: Cross-platform inference
    - TorchScript: PyTorch optimized inference
    - State dict: PyTorch checkpoint
    - JSON config: Model configuration

    Example usage:
        exporter = ModelExporter()

        # Export to multiple formats
        exporter.export(
            model=model,
            model_name='looped_mlp',
            model_params={'input_dim': 784, 'hidden_dim': 256, 'output_dim': 10},
            output_dir='./exports',
            formats=['onnx', 'torchscript', 'config'],
        )
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def export(
        self,
        model: nn.Module,
        model_name: str,
        model_params: dict[str, object],
        output_dir: str = "./exports",
        formats: list[str] = None,
        optimizer: object | None = None,
        optimizer_name: str | None = None,
        optimizer_params: dict[str, object] | None = None,
        training_metrics: dict[str, float] | None = None,
        input_shape: tuple[int, ...] = (1, 784),
        verbose: bool = True,
    ) -> ModelInfo:
        """
        Export model to multiple formats.

        Args:
            model: Model to export.
            model_name: Name of the model.
            model_params: Model parameters.
            output_dir: Output directory for exports.
            formats: List of formats ['onnx', 'torchscript', 'config', 'state'].
            optimizer: Optional optimizer for state export.
            optimizer_name: Name of optimizer.
            optimizer_params: Optimizer parameters.
            training_metrics: Training metrics to save.
            input_shape: Example input shape for tracing.
            verbose: Print progress.

        Returns:
            ModelInfo with export details.
        """
        if formats is None:
            formats = ["onnx", "torchscript", "config", "state"]

        Path(output_dir).mkdir(exist_ok=True, parents=True)
        model = model.to(self.device)
        model.eval()

        # Count parameters
        num_params = count_parameters(model, trainable_only=False)

        # Export to each format
        export_paths = {}

        if "onnx" in formats:
            try:
                path = self._export_onnx(model, output_dir, input_shape, verbose)
                export_paths["onnx"] = path
            except (RuntimeError, ValueError, OSError) as e:
                if verbose:
                    logger.warning("ONNX export failed: %s", e)

        if "torchscript" in formats:
            try:
                path = self._export_torchscript(
                    model, output_dir, input_shape, verbose, method="trace"
                )
                export_paths["torchscript"] = path
            except (RuntimeError, ValueError, OSError) as e:
                if verbose:
                    logger.warning("TorchScript export failed: %s", e)

        if "config" in formats:
            path = self._export_config(
                model_name,
                model_params,
                optimizer_name,
                optimizer_params,
                training_metrics,
                input_shape,
                output_dir,
                verbose,
            )
            export_paths["config"] = path

        if "state" in formats:
            path = self._export_state(model, optimizer, output_dir, verbose)
            export_paths["state"] = path

        # Create model info
        info = ModelInfo(
            model_name=model_name,
            model_params=model_params,
            optimizer_name=optimizer_name,
            optimizer_params=optimizer_params,
            training_metrics=training_metrics or {},
            input_shape=input_shape,
            output_shape=self._get_output_shape(model, input_shape),
            num_parameters=num_params,
            export_format=", ".join(export_paths.keys()),
            export_path=output_dir,
        )

        if verbose:
            logger.info("Exported %s to %s", model_name, output_dir)
            logger.info("  Formats: %s", info.export_format)
            logger.info("  Parameters: %s", f"{num_params:,}")

        return info

    def _export_onnx(
        self,
        model: nn.Module,
        output_dir: str,
        input_shape: tuple[int, ...],
        verbose: bool,
    ) -> str:
        """Export to ONNX format with dynamic axes and opset 17+."""
        path = str(Path(output_dir) / "model.onnx")

        model.eval()
        dummy_input = torch.randn(input_shape, device=self.device)

        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r".*cached_sn_weight.*assigned during export.*"
            )
            torch.onnx.export(
                model,
                dummy_input,
                path,
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"},
                },
                dynamo=True,
            )

        if verbose:
            logger.info("  ✓ ONNX (opset 17): %s", path)

        return path

    def _export_torchscript(
        self,
        model: nn.Module,
        output_dir: str,
        input_shape: tuple[int, ...],
        verbose: bool,
        method: str = "script",
    ) -> str:
        """Export to TorchScript format (script or trace).

        Args:
            model: Model to export.
            output_dir: Output directory.
            input_shape: Example input shape for tracing.
            verbose: Print progress.
            method: 'script' (torch.jit.script) or 'trace' (torch.jit.trace).

        Returns:
            Path to exported model.
        """
        path = str(Path(output_dir) / "model_ts.pt")

        model.eval()
        model = model.to(self.device)
        dummy_input = torch.randn(input_shape, device=self.device)

        if method == "script":
            scripted = torch.jit.script(model)
        elif method == "trace":
            scripted = torch.jit.trace(model, dummy_input)
        else:
            raise ValueError(f"Unknown TorchScript method: {method}")

        # Save TorchScript model
        scripted.save(path)

        if verbose:
            logger.info("  ✓ TorchScript (%s): %s", method, path)

        return path

    def _export_config(
        self,
        model_name: str,
        model_params: dict[str, object],
        optimizer_name: str | None,
        optimizer_params: dict[str, object] | None,
        training_metrics: dict[str, float] | None,
        input_shape: tuple[int, ...],
        output_dir: str,
        verbose: bool,
    ) -> str:
        """Export model configuration to JSON."""
        path = str(Path(output_dir) / "config.json")

        config = {
            "model_name": model_name,
            "model_params": model_params,
            "optimizer_name": optimizer_name,
            "optimizer_params": optimizer_params,
            "training_metrics": training_metrics,
            "input_shape": input_shape,
            "export_version": "1.0",
        }

        with Path(path).open("w") as f:
            json.dump(config, f, indent=2, default=str)

        if verbose:
            logger.info("  ✓ Config: %s", path)

        return path

    def _export_state(
        self,
        model: nn.Module,
        optimizer: object | None,
        output_dir: str,
        verbose: bool,
    ) -> str:
        """Export model and optimizer state."""
        path = str(Path(output_dir) / "checkpoint.pt")

        ckpt: dict = {
            "model_state_dict": model.state_dict(),
        }

        if optimizer is not None:
            ckpt["optimizer_state_dict"] = optimizer.state_dict()

        save_checkpoint(path, ckpt, mkdir=True)

        if verbose:
            logger.info("  ✓ State: %s", path)

        return path

    def _get_output_shape(
        self,
        model: nn.Module,
        input_shape: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Get model output shape."""
        model.eval()
        dummy_input = torch.randn(input_shape, device=self.device)

        with torch.no_grad():
            output = model(dummy_input)

        return tuple(output.shape)


class ModelLoader:
    """
    Load exported Bioplausible models.

    Example usage:
        loader = ModelLoader()

        # Load from config
        model, config = loader.load_from_config('./exports/config.json')

        # Load from checkpoint
        model = loader.load_from_checkpoint('./exports/checkpoint.pt', model_class)

        # Load ONNX for inference
        session = loader.load_onnx('./exports/model.onnx')
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def load_from_config(
        self,
        config_path: str,
    ) -> tuple[nn.Module, dict[str, object]]:
        """
        Load model from config file.

        Args:
            config_path: Path to config.json.

        Returns:
            Tuple of (model, config dict).
        """
        from bioplausible.core.registry import ComponentCategory, Registry

        with Path(config_path).open() as f:
            config = json.load(f)

        model_name = config["model_name"]
        model_params = config["model_params"]

        model_cls = Registry.get(ComponentCategory.MODEL, model_name)
        model = self._construct(model_cls, model_params, model_name)
        model = model.to(self.device)

        # Load state dict if available
        state_path = config_path.replace("config.json", "checkpoint.pt")
        if Path(state_path).exists():
            load_checkpoint_into_model(state_path, model, map_location=self.device)

        return model, config

    def _construct(
        self, model_cls: object, model_params: dict[str, object], model_name: str
    ) -> nn.Module:
        """Build a registered model through the single construction funnel."""
        from typing import cast

        from bioplausible.core.construction import construct_model

        input_dim = model_params.get("input_dim", 0)
        output_dim = model_params.get("output_dim", 0)
        return cast(
            "nn.Module",
            construct_model(
                model_cls,
                model_params,
                input_dim=int(input_dim or 0),
                output_dim=int(output_dim or 0),
                model_name=model_name,
            ),
        )

    def load_from_checkpoint(
        self,
        checkpoint_path: str,
        model_class: type,
        model_params: dict[str, object],
    ) -> nn.Module:
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint.pt.
            model_class: Model class.
            model_params: Model parameters.

        Returns:
            Loaded model.
        """
        model = self._construct(
            model_class, model_params, getattr(model_class, "__name__", "model")
        )
        model = model.to(self.device)

        load_checkpoint_into_model(checkpoint_path, model, map_location=self.device)

        return model

    def load_onnx(
        self,
        onnx_path: str,
    ) -> object:
        """
        Load ONNX model for inference.

        Args:
            onnx_path: Path to model.onnx.

        Returns:
            ONNX runtime session.
        """
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(onnx_path)
            return session
        except ImportError:
            raise ImportError("onnxruntime required: pip install onnxruntime")


class InferenceEngine:
    """
    Optimized inference engine for deployed models.

    Supports:
    - Batch prediction
    - Streaming prediction
    - Confidence scoring
    - Multiple input formats

    Example usage:
        engine = InferenceEngine.from_export('./exports')

        # Single prediction
        result = engine.predict(image)

        # Batch prediction
        results = engine.predict_batch(images)

        # With confidence
        pred, confidence = engine.predict_with_confidence(image)
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict[str, object],
        device: str = "auto",
    ):
        self.model = model
        self.config = config
        self.device = device

        if device == "auto":
            self.device = str(get_device())

        self.model = self.model.to(self.device)
        self.model.eval()

        self.input_shape = tuple(config.get("input_shape", (1, 784)))

    @classmethod
    def from_export(cls, export_dir: str, device: str = "auto") -> InferenceEngine:
        """
        Create inference engine from export directory.

        Args:
            export_dir: Directory with config.json and checkpoint.pt.
            device: Device for inference.

        Returns:
            InferenceEngine instance.
        """
        loader = ModelLoader(device="cpu")
        config_path = str(Path(export_dir) / "config.json")

        model, config = loader.load_from_config(config_path)

        return cls(model, config, device)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Single prediction.

        Args:
            x: Input tensor.

        Returns:
            Prediction tensor.
        """
        x = x.to(self.device)

        # Ensure correct shape
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[1:] != self.input_shape[1:]:
            x = x.view(-1, *self.input_shape[1:])

        output = self.model(x)
        return output

    @torch.no_grad()
    def predict_batch(
        self,
        xs: list[torch.Tensor],
        batch_size: int = 32,
    ) -> list[torch.Tensor]:
        """
        Batch prediction.

        Args:
            xs: List of input tensors.
            batch_size: Batch size for processing.

        Returns:
            List of predictions.
        """
        self.model.eval()
        predictions = []

        for i in range(0, len(xs), batch_size):
            batch = xs[i : i + batch_size]
            batched = torch.stack(batch).to(self.device)
            output = self.model(batched)
            predictions.extend(output.cpu().chunk(output.shape[0]))

        return predictions

    @torch.no_grad()
    def predict_with_confidence(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prediction with confidence score.

        Args:
            x: Input tensor.

        Returns:
            Tuple of (prediction, confidence).
        """
        output = self.predict(x)

        # Get confidence from softmax
        probs = torch.softmax(output, dim=-1)
        confidence, pred = probs.max(dim=-1)

        return pred, confidence

    @torch.no_grad()
    def predict_class(
        self,
        x: torch.Tensor,
    ) -> int:
        """
        Get predicted class index.

        Args:
            x: Input tensor.

        Returns:
            Class index.
        """
        output = self.predict(x)
        return output.argmax(dim=-1).item()

    @torch.no_grad()
    def predict_proba(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get class probabilities.

        Args:
            x: Input tensor.

        Returns:
            Probability distribution.
        """
        output = self.predict(x)
        return torch.softmax(output, dim=-1)


def export_model(
    model: nn.Module,
    model_name: str,
    model_params: dict[str, object],
    output_dir: str = "./exports",
    formats: list[str] = None,
    optimizer: object | None = None,
    training_metrics: dict[str, float] | None = None,
    verbose: bool = True,
) -> ModelInfo:
    """
    Convenience function to export a model.

    Args:
        model: Model to export.
        model_name: Name of the model.
        model_params: Model parameters.
        output_dir: Output directory.
        formats: Export formats.
        optimizer: Optional optimizer.
        training_metrics: Training metrics.
        verbose: Print progress.

    Returns:
        ModelInfo with export details.
    """
    exporter = ModelExporter()
    return exporter.export(
        model=model,
        model_name=model_name,
        model_params=model_params,
        output_dir=output_dir,
        formats=formats,
        optimizer=optimizer,
        optimizer_params=None,
        training_metrics=training_metrics,
        verbose=verbose,
    )


def load_model(
    export_dir: str,
    device: str = "auto",
) -> tuple[nn.Module, dict[str, object]]:
    """
    Convenience function to load a model.

    Args:
        export_dir: Export directory with config.json.
        device: Device for model.

    Returns:
        Tuple of (model, config).
    """
    loader = ModelLoader(device="cpu")
    config_path = str(Path(export_dir) / "config.json")
    return loader.load_from_config(config_path)


# ──────────────────────────────────────────────
# Merged from export.py
# ──────────────────────────────────────────────


def export_to_onnx(model, input_sample, path):
    """Export model to ONNX format with opset 17+."""
    import warnings

    model.eval()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r".*cached_sn_weight.*assigned during export.*"
        )
        torch.onnx.export(
            model,
            input_sample,
            path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            dynamo=True,
        )


def export_to_torchscript(model, input_sample, path, method: str = "script"):
    """Export model to TorchScript format.

    Args:
        model: Model to export.
        input_sample: Example input for tracing (required for trace method).
        path: Output path.
        method: 'script' (torch.jit.script) or 'trace' (torch.jit.trace).

    Returns:
        Path to exported model.
    """
    model.eval()

    if method == "script":
        scripted = torch.jit.script(model)
    elif method == "trace":
        scripted = torch.jit.trace(model, input_sample)
    else:
        raise ValueError(f"Unknown TorchScript method: {method}")

    scripted.save(path)
    return path


# --- Serving Logic (FastAPI) ---

import numpy as np
import uvicorn
from fastapi import FastAPI

from bioplausible.core.utils.device import get_device


class _AppState:
    """Encapsulates module-level state to avoid global keyword."""

    def __init__(self) -> None:
        self.app: FastAPI | None = None
        self.model_instance: object | None = None

    def get_app(self) -> FastAPI:
        """Lazy-initialized FastAPI app.

        Importing this module should not bind to a port or perform any
        I/O. The app is constructed on first access.
        """
        if self.app is None:
            self.app = self._build_app()
        return self.app

    def serve_model(
        self, model: object, host: str = "0.0.0.0", port: int = 8000
    ) -> None:
        """Run a FastAPI server for the model."""
        self.model_instance = model
        if hasattr(model, "eval"):
            model.eval()
        uvicorn.run(self.get_app(), host=host, port=port, log_level="info")

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="Bioplausible Inference API")

        @app.post("/predict")
        def predict(request: InferenceRequest):
            if not self.model_instance:
                return {"error": "No model loaded"}
            try:
                data = np.array(request.data, dtype=np.float32)
                if request.shape:
                    data = data.reshape(request.shape)
                elif hasattr(self.model_instance, "input_dim"):
                    if (
                        len(data.shape) == 1
                        and data.shape[0] == self.model_instance.input_dim
                    ):
                        data = data.reshape(1, -1)
                elif "Conv" in type(self.model_instance).__name__:
                    pass
                tensor = torch.from_numpy(data)
                if tensor.dim() == 1:
                    tensor = tensor.unsqueeze(0)
                device = next(self.model_instance.parameters()).device
                tensor = tensor.to(device)
                with torch.no_grad():
                    output = self.model_instance(tensor)
                return {"output": output.cpu().tolist()}
            except Exception as e:  # broad: best-effort
                return {"error": str(e)}

        @app.get("/health")
        def health():
            return {
                "status": "ok",
                "model": str(type(self.model_instance).__name__)
                if self.model_instance
                else "None",
            }

        return app


_app_state = _AppState()


def get_app() -> FastAPI:
    """Get the FastAPI app instance (lazy-initialized)."""
    return _app_state.get_app()


def serve_model(model: object, host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run a FastAPI server for the model."""
    _app_state.serve_model(model, host=host, port=port)


# --- Quantization Utilities ---


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
    import torch.quantization as quant

    model.eval()
    model.cpu()

    # Set quantization config
    model.qconfig = quant.get_default_qconfig(backend)

    # Prepare for quantization
    quant.prepare(model, inplace=True)

    # Calibrate with sample data
    if calibration_data is not None:
        with torch.no_grad():
            for x in calibration_data:
                model(x)

    # Convert to quantized model
    quantized_model = quant.convert(model, inplace=False)

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
    import torch.quantization as quant

    model.train()
    model.cpu()

    # Set QAT config
    model.qconfig = quant.get_default_qat_qconfig(backend)

    # Prepare for QAT (inserts fake quant observers)
    quant.prepare_qat(model, inplace=True)

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
    import torch.quantization as quant

    model.eval()
    model.cpu()
    quantized_model = quant.convert(model, inplace=False)
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
    import torch.quantization as quant

    model.eval()
    model.cpu()

    # Quantize Linear and LSTM layers dynamically
    quantized_model = quant.quantize_dynamic(
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
    from bioplausible.core.checkpoint import save_checkpoint

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
    from bioplausible.deployment import load_model

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


__all__ = [
    "InferenceEngine",
    "ModelExporter",
    "ModelInfo",
    "ModelLoader",
    "benchmark_quantized_model",
    "convert_qat_model",
    "export_model",
    "export_to_onnx",
    "export_to_torchscript",
    "get_app",
    "load_model",
    "load_quantized_model",
    "quantize_model_dynamic_int8",
    "quantize_model_int8_ptq",
    "quantize_model_int8_qat",
    "save_quantized_model",
    "serve_model",
]
