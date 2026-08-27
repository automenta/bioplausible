"""Kernel export pipeline tests (REFACTOR7 Phase 11).

Verifies :func:`bioplausible.acceleration.export.export_kernel` writes a hardware
manifest + state dict for a bound kernel backend, plus a best-effort ONNX export,
and that the ``biopl-export-kernel`` CLI drives it end-to-end.
"""

from __future__ import annotations

import json

import pytest
import torch
from bioplausible.acceleration.export import _TARGET_SPECS, export_kernel
from bioplausible.acceleration.kernel_backend import (
    AlgorithmFamily,
    HardwareTarget,
    KernelConfig,
    KernelRegistry,
)
from torch import nn


@pytest.fixture(autouse=True)
def _clear_kernel_cache():
    """Clear kernel registry cache between tests to avoid state pollution."""
    KernelRegistry.clear_cache()
    yield
    KernelRegistry.clear_cache()


def _bound_backprop(tmp_path) -> object:
    """A BackpropKernelBackend bound to a small Linear stack."""
    from bioplausible.acceleration.backprop_kernels import BackpropKernelBackend

    backend = BackpropKernelBackend()
    config = KernelConfig(
        algorithm=AlgorithmFamily.BACKPROP, hardware=HardwareTarget.CPU
    )
    backend.initialize(config)
    stack = [nn.Linear(4, 8), nn.Linear(8, 3)]
    backend.set_model_ref(stack)
    return backend


def test_export_kernel_writes_manifest_and_state(tmp_path):
    """Export writes a JSON manifest + state dict and reports both paths."""
    backend = _bound_backprop(tmp_path)
    config = KernelConfig(
        algorithm=AlgorithmFamily.BACKPROP, hardware=HardwareTarget.CPU
    )
    result = export_kernel(
        backend, config, target=HardwareTarget.CPU, output_dir=str(tmp_path)
    )

    manifest = json.loads((tmp_path / "backprop_cpu_manifest.json").read_text())
    assert manifest["algorithm"] == "backprop"
    assert manifest["family"] == "backprop"
    assert manifest["hardware_target"] == "cpu"
    assert manifest["target_spec"] == "onnx"
    assert manifest["supports_autograd"] is True
    assert manifest["requires_settle"] is False
    assert manifest["state_dict"] == "backprop_cpu_state.pt"

    assert (tmp_path / manifest["state_dict"]).exists()
    assert result.state_dict_path == str(tmp_path / manifest["state_dict"])

    if result.onnx_path is not None:
        assert result.onnx_path.endswith(".onnx")
        assert (tmp_path / "backprop_cpu.onnx").exists()


def test_export_target_spec_mapping():
    """Hardware targets map to the documented descriptor formats."""
    assert _TARGET_SPECS[HardwareTarget.FPGA] == "hls"
    assert _TARGET_SPECS[HardwareTarget.NEUROMORPHIC] == "nxsdk"
    assert _TARGET_SPECS[HardwareTarget.CROSSBAR] == "spice"
    assert _TARGET_SPECS[HardwareTarget.OPTICAL] == "dsl"


def test_export_kernel_without_stack_writes_manifest_only(tmp_path):
    """A backend with no bound Linear stack still exports a manifest."""
    from bioplausible.acceleration.backprop_kernels import BackpropKernelBackend

    backend = BackpropKernelBackend()
    config = KernelConfig(
        algorithm=AlgorithmFamily.BACKPROP, hardware=HardwareTarget.CPU
    )
    backend.initialize(config)

    result = export_kernel(backend, config, output_dir=str(tmp_path))
    assert result.onnx_path is None
    assert (tmp_path / "backprop_cpu_manifest.json").exists()


def test_export_json_safe_tensors(tmp_path):
    """Tensors and torch.dtypes are serialized as JSON-safe primitives."""
    from bioplausible.acceleration.backprop_kernels import BackpropKernelBackend

    backend = BackpropKernelBackend()
    config = KernelConfig(
        algorithm=AlgorithmFamily.BACKPROP,
        hardware=HardwareTarget.CPU,
        dtype=torch.bfloat16,
        extra={"some_tensor": torch.zeros(2, 3)},
    )
    backend.initialize(config)
    result = export_kernel(backend, config, output_dir=str(tmp_path))
    manifest = json.loads((tmp_path / "backprop_cpu_manifest.json").read_text())
    assert manifest["dtype"] == "torch.bfloat16"
    assert manifest["extra"]["some_tensor"] == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_cli_export_kernel_main(tmp_path):
    """The ``biopl-export-kernel`` CLI exports a manifest for a family."""
    from bioplausible.cli.export_kernel import main

    out = str(tmp_path)
    rc = main(["--algorithm", "backprop", "--target", "cpu", "--output", out])
    assert rc == 0
    assert (tmp_path / "backprop_cpu_manifest.json").exists()
