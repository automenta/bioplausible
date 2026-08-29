"""
ONNX export utilities.
"""

import warnings
from pathlib import Path
from typing import Any

import torch
from torch import nn


def export_to_onnx(
    model: nn.Module,
    input_sample: torch.Tensor | tuple[Any, ...],
    path: str | Path,
    opset_version: int = 17,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic_axes: dict | None = None,
) -> str:
    """
    Export model to ONNX format with opset 17+.

    Args:
        model: Model to export.
        input_sample: Example input for tracing.
        path: Output path.
        opset_version: ONNX opset version.
        input_names: Names for input tensors.
        output_names: Names for output tensors.
        dynamic_axes: Dynamic axis specifications.

    Returns:
        Path to exported model.
    """
    model.eval()
    path = str(path)

    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]
    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r".*cached_sn_weight.*assigned during export.*"
        )
        torch.onnx.export(
            model,
            input_sample,
            path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            dynamo=True,
        )

    return path


__all__ = [
    "export_to_onnx",
]
