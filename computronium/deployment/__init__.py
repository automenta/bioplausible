"""
Deployment Package

Model export, serialization, and deployment utilities for production use.

Features:
- ONNX export for cross-platform deployment
- ``torch.export`` (PT2) serialized program export
- Model serialization/deserialization
- Inference optimization
- Batch prediction utilities
"""

from computronium.deployment.exporter import ExportConfig, export_model, load_model
from computronium.deployment.onnx_export import export_to_onnx
from computronium.deployment.pt2_export import export_to_pt2
from computronium.deployment.quantization import (
    TernaryLinear,
    TernaryQuantize,
    benchmark_quantized_model,
    convert_qat_model,
    count_ternary_operations,
    load_quantized_model,
    quantize_model_dynamic_int8,
    quantize_model_int8_ptq,
    quantize_model_int8_qat,
    quantize_model_ternary,
    quantize_model_ternary_inplace,
    save_quantized_model,
)
from computronium.deployment.serialization import (
    BatchInferenceRequest,
    InferenceEngine,
    InferenceResponse,
    InferenceServer,
    ModelExporter,
    ModelInfo,
    ModelLoader,
    TensorRTConfig,
    create_inference_server,
    get_app,
    serve_model,
)

__all__ = [
    "BatchInferenceRequest",
    "ExportConfig",
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
    "export_to_pt2",
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