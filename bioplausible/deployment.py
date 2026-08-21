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
from torch.nn import functional as F

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

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from bioplausible.core.utils.device import get_device


class BatchInferenceRequest(BaseModel):
    """Batched request body for deployment prediction endpoint."""

    data: list[list[float]] = Field(..., description="List of input samples")
    shape: list[int] | None = Field(
        None, description="Optional reshape for each sample"
    )


class InferenceResponse(BaseModel):
    """Response body for inference."""

    outputs: list[list[float]]
    batch_size: int
    latency_ms: float


@dataclass(frozen=True, slots=True)
class TensorRTConfig:
    """TensorRT optimization configuration."""

    enabled: bool = False
    precision: str = "fp16"  # fp32, fp16, int8
    workspace_size: int = 1 << 30  # 1GB
    max_batch_size: int = 32


class InferenceServer:
    """
    Production-ready inference server with batching, TensorRT, and async support.

    Features:
    - Dynamic batching with configurable max batch size and timeout
    - TensorRT optimization (when available)
    - Async request handling for high throughput
    - Health checks and metrics
    - Graceful shutdown
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict[str, object],
        device: str = "auto",
        max_batch_size: int = 32,
        batch_timeout_ms: int = 10,
        tensorrt_config: TensorRTConfig | None = None,
    ):
        self.model = model
        self.config = config
        self.device = device if device != "auto" else str(get_device())
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.tensorrt_config = tensorrt_config or TensorRTConfig()

        self._batch_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._batch_task: asyncio.Task | None = None
        self._tensorrt_model: nn.Module | None = None

        self.model = self.model.to(self.device)
        self.model.eval()

        self.input_shape = tuple(config.get("input_shape", (1, 784)))

    async def _initialize_tensorrt(self) -> None:
        """Initialize TensorRT optimization if enabled."""
        if not self.tensorrt_config.enabled:
            return

        try:
            import torch_tensorrt  # type: ignore

            self.model.eval()
            example_input = torch.randn(
                self.tensorrt_config.max_batch_size,
                *self.input_shape[1:],
                device=self.device,
            )

            enabled_precisions = {
                torch.float32,
                torch.float16,
                torch.int8,
            }
            if self.tensorrt_config.precision == "fp16":
                enabled_precisions = {torch.float16, torch.float32}
            elif self.tensorrt_config.precision == "int8":
                enabled_precisions = {torch.int8, torch.float16, torch.float32}

            self._tensorrt_model = torch_tensorrt.compile(
                self.model,
                inputs=[
                    torch_tensorrt.Input(
                        min_shape=(1, *self.input_shape[1:]),
                        opt_shape=(
                            self.tensorrt_config.max_batch_size // 2,
                            *self.input_shape[1:],
                        ),
                        max_shape=(
                            self.tensorrt_config.max_batch_size,
                            *self.input_shape[1:],
                        ),
                        dtype=torch.float32,
                    )
                ],
                enabled_precisions=enabled_precisions,
                workspace_size=self.tensorrt_config.workspace_size,
                truncate_long_and_double=True,
            )
            print(
                f"TensorRT model compiled with {self.tensorrt_config.precision} precision"
            )
        except ImportError:
            print("torch_tensorrt not available, falling back to PyTorch")
            self.tensorrt_config.enabled = False
        except Exception as e:
            print(f"TensorRT compilation failed: {e}, falling back to PyTorch")
            self.tensorrt_config.enabled = False

    @property
    def _inference_model(self) -> nn.Module:
        """Get the active inference model (TensorRT or PyTorch)."""
        return self._tensorrt_model if self._tensorrt_model is not None else self.model

    async def _batch_worker(self) -> None:
        """Background worker that processes batched requests."""
        while self._running:
            batch_requests = []

            # Wait for first request
            try:
                first_req = await asyncio.wait_for(self._batch_queue.get(), timeout=0.1)
                batch_requests.append(first_req)
            except TimeoutError:
                continue

            # Collect additional requests up to max_batch_size
            while len(batch_requests) < self.max_batch_size:
                try:
                    req = await asyncio.wait_for(
                        self._batch_queue.get(),
                        timeout=self.batch_timeout_ms / 1000.0,
                    )
                    batch_requests.append(req)
                except TimeoutError:
                    break

            # Process batch
            if batch_requests:
                await self._process_batch(batch_requests)

    async def _process_batch(self, requests: list) -> None:
        """Process a batch of inference requests."""
        import time

        start_time = time.perf_counter()

        try:
            # Stack inputs
            batch_data = []
            for req in requests:
                data = np.array(req.data, dtype=np.float32)
                if req.shape:
                    data = data.reshape(req.shape)
                elif len(data.shape) == 1:
                    data = data.reshape(1, -1)
                batch_data.append(data)

            batched = np.stack(batch_data)
            tensor = torch.from_numpy(batched).to(self.device)

            with torch.no_grad():
                outputs = self._inference_model(tensor)

            outputs_list = outputs.cpu().tolist()
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Distribute results
            for i, req in enumerate(requests):
                req["future"].set_result(
                    InferenceResponse(
                        outputs=[outputs_list[i]],
                        batch_size=len(requests),
                        latency_ms=latency_ms,
                    )
                )
        except Exception as e:
            for req in requests:
                req["future"].set_exception(e)

    async def predict(self, request: BatchInferenceRequest) -> InferenceResponse:
        """Async prediction with dynamic batching."""
        if not self._running:
            raise HTTPException(status_code=503, detail="Server not running")

        future = asyncio.get_event_loop().create_future()
        await self._batch_queue.put({
            "data": request.data,
            "shape": request.shape,
            "future": future,
        })
        return await future

    def predict_sync(self, request: BatchInferenceRequest) -> InferenceResponse:
        """Synchronous prediction (bypasses batching queue)."""
        import time

        start_time = time.perf_counter()

        data = np.array(request.data, dtype=np.float32)
        if request.shape:
            data = data.reshape(request.shape)
        elif len(data.shape) == 1:
            data = data.reshape(1, -1)

        tensor = torch.from_numpy(data).to(self.device)

        with torch.no_grad():
            output = self._inference_model(tensor)

        outputs_list = output.cpu().tolist()
        latency_ms = (time.perf_counter() - start_time) * 1000

        return InferenceResponse(
            outputs=outputs_list,
            batch_size=len(request.data),
            latency_ms=latency_ms,
        )

    async def start(self) -> None:
        """Start the batching worker."""
        self._running = True
        await self._initialize_tensorrt()
        self._batch_task = asyncio.create_task(self._batch_worker())

    async def stop(self) -> None:
        """Stop the batching worker gracefully."""
        self._running = False
        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass


def create_inference_server(
    model: nn.Module,
    config: dict[str, object],
    device: str = "auto",
    max_batch_size: int = 32,
    batch_timeout_ms: int = 10,
    tensorrt_config: TensorRTConfig | None = None,
) -> InferenceServer:
    """Factory function to create an InferenceServer."""
    return InferenceServer(
        model=model,
        config=config,
        device=device,
        max_batch_size=max_batch_size,
        batch_timeout_ms=batch_timeout_ms,
        tensorrt_config=tensorrt_config,
    )


class _AppState:
    """Encapsulates module-level state to avoid global keyword."""

    def __init__(self) -> None:
        self.app: FastAPI | None = None
        self.server: InferenceServer | None = None

    def get_app(self) -> FastAPI:
        """Lazy-initialized FastAPI app."""
        if self.app is None:
            self.app = self._build_app()
        return self.app

    def serve_model(
        self,
        model: object,
        config: dict[str, object] | None = None,
        host: str = "0.0.0.0",
        port: int = 8000,
        max_batch_size: int = 32,
        batch_timeout_ms: int = 10,
        tensorrt_config: TensorRTConfig | None = None,
    ) -> None:
        """Run a FastAPI server for the model with batching support."""
        if config is None:
            config = {"input_shape": (1, 784)}

        self.server = create_inference_server(
            model=model,
            config=config,
            max_batch_size=max_batch_size,
            batch_timeout_ms=batch_timeout_ms,
            tensorrt_config=tensorrt_config,
        )

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await self.server.start()
            yield
            await self.server.stop()

        self.app = FastAPI(
            title="Bioplausible Inference API",
            version="1.0.0",
            lifespan=lifespan,
        )
        self._build_routes()

        uvicorn.run(self.app, host=host, port=port, log_level="info")

    def _build_app(self) -> FastAPI:
        return FastAPI(title="Bioplausible Inference API")

    def _build_routes(self) -> None:
        if not self.app or not self.server:
            return

        @self.app.post("/predict", response_model=InferenceResponse)
        async def predict(request: BatchInferenceRequest):
            if not self.server:
                raise HTTPException(status_code=503, detail="Server not initialized")
            return await self.server.predict(request)

        @self.app.post("/predict/sync", response_model=InferenceResponse)
        def predict_sync(request: BatchInferenceRequest):
            if not self.server:
                raise HTTPException(status_code=503, detail="Server not initialized")
            return self.server.predict_sync(request)

        @self.app.get("/health")
        def health():
            return {
                "status": "ok",
                "model": str(type(self.server.model).__name__)
                if self.server
                else "None",
                "device": self.server.device if self.server else "unknown",
                "tensorrt_enabled": self.server.tensorrt_config.enabled
                if self.server
                else False,
                "batching": {
                    "max_batch_size": self.server.max_batch_size if self.server else 0,
                    "batch_timeout_ms": self.server.batch_timeout_ms
                    if self.server
                    else 0,
                    "queue_size": self.server._batch_queue.qsize()
                    if self.server
                    else 0,
                },
            }

        @self.app.get("/metrics")
        def metrics():
            if not self.server:
                raise HTTPException(status_code=503, detail="Server not initialized")
            return {
                "max_batch_size": self.server.max_batch_size,
                "batch_timeout_ms": self.server.batch_timeout_ms,
                "tensorrt_enabled": self.server.tensorrt_config.enabled,
                "tensorrt_precision": self.server.tensorrt_config.precision,
                "device": self.server.device,
                "input_shape": self.server.input_shape,
            }


_app_state = _AppState()


def get_app() -> FastAPI:
    """Get the FastAPI app instance (lazy-initialized)."""
    return _app_state.get_app()


def serve_model(
    model: object,
    config: dict[str, object] | None = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    max_batch_size: int = 32,
    batch_timeout_ms: int = 10,
    tensorrt_config: TensorRTConfig | None = None,
) -> None:
    """Run a FastAPI server for the model with batching and TensorRT support."""
    _app_state.serve_model(
        model=model,
        config=config,
        host=host,
        port=port,
        max_batch_size=max_batch_size,
        batch_timeout_ms=batch_timeout_ms,
        tensorrt_config=tensorrt_config,
    )


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
    import copy

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
    "BatchInferenceRequest",
    "InferenceEngine",
    "InferenceResponse",
    "InferenceServer",
    "ModelExporter",
    "ModelInfo",
    "ModelLoader",
    "TensorRTConfig",
    "TernaryLinear",
    "TernaryQuantize",
    "benchmark_quantized_model",
    "convert_qat_model",
    "count_ternary_operations",
    "create_inference_server",
    "export_model",
    "export_to_onnx",
    "export_to_torchscript",
    "get_app",
    "load_model",
    "load_quantized_model",
    "quantize_model_dynamic_int8",
    "quantize_model_int8_ptq",
    "quantize_model_int8_qat",
    "quantize_model_ternary",
    "quantize_model_ternary_inplace",
    "save_quantized_model",
    "serve_model",
]
