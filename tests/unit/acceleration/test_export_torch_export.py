"""Tests for torch.export migration in export.py (REFACTOR8).

Verifies the new export path using torch.export.export() + torch.onnx.export_from_ep()
works correctly and falls back to legacy exporter when needed.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from bioplausible.acceleration.export import _TARGET_SPECS, _onnx_export, export_kernel
from bioplausible.acceleration.kernel_backend import (
    AlgorithmFamily,
    HardwareTarget,
    KernelConfig,
    KernelRegistry,
)


class TestOnnxExportTorchExport:
    """Test the new torch.export-based ONNX export."""

    def setup_method(self):
        """Clear kernel cache."""
        KernelRegistry.clear_cache()

    def test_onnx_export_simple_linear_stack(self, tmp_path):
        """_onnx_export should work with simple Linear stack via torch.export."""
        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
        )
        model.eval()

        sample = torch.zeros(1, 4)
        output_path = tmp_path / "test.onnx"

        # Should not raise
        _onnx_export(model, sample, output_path)

        assert output_path.exists()

        # Verify ONNX model loads
        import onnx
        onnx_model = onnx.load(str(output_path))
        assert onnx_model.graph.input[0].name == "input"
        assert onnx_model.graph.output[0].name == "output"

    def test_onnx_export_dynamic_batch(self, tmp_path):
        """ONNX export should support dynamic batch dimension."""
        model = nn.Sequential(nn.Linear(4, 3))
        model.eval()

        sample = torch.zeros(1, 4)
        output_path = tmp_path / "test.onnx"

        _onnx_export(model, sample, output_path)

        import onnx
        onnx_model = onnx.load(str(output_path))
        # Check dynamic axes
        input_shape = onnx_model.graph.input[0].type.tensor_type.shape
        batch_dim = input_shape.dim[0]
        assert batch_dim.dim_param == "batch"  # dynamic

    def test_onnx_export_with_custom_activation(self, tmp_path):
        """_onnx_export should work with various activations."""
        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.Tanh(),
            nn.Linear(8, 3),
        )
        model.eval()

        sample = torch.zeros(1, 4)
        output_path = tmp_path / "test.onnx"

        _onnx_export(model, sample, output_path)

        assert output_path.exists()


class TestExportKernelWithTorchExport:
    """Test export_kernel uses new torch.export path."""

    def setup_method(self):
        """Clear kernel cache."""
        KernelRegistry.clear_cache()

    def _bound_backprop(self):
        """Create a BackpropKernelBackend bound to a small Linear stack."""
        from bioplausible.acceleration.backprop_kernels import BackpropKernelBackend

        backend = BackpropKernelBackend()
        config = KernelConfig(algorithm=AlgorithmFamily.BACKPROP, hardware=HardwareTarget.CPU)
        backend.initialize(config)
        stack = [nn.Linear(4, 8), nn.Linear(8, 3)]
        backend.set_model_ref(stack)
        return backend

    def test_export_kernel_creates_onnx_with_torch_export(self, tmp_path):
        """export_kernel should create ONNX using torch.export path."""
        backend = self._bound_backprop()
        config = KernelConfig(algorithm=AlgorithmFamily.BACKPROP, hardware=HardwareTarget.CPU)

        result = export_kernel(backend, config, target=HardwareTarget.CPU, output_dir=str(tmp_path))

        assert result.onnx_path is not None
        assert (tmp_path / "backprop_cpu.onnx").exists()

        # Verify ONNX model is valid
        import onnx
        onnx_model = onnx.load(result.onnx_path)
        assert onnx_model.graph.input[0].name == "input"
        assert onnx_model.graph.output[0].name == "output"

    def test_export_kernel_state_dict_matches_model(self, tmp_path):
        """Exported state dict should match the bound model's weights."""
        backend = self._bound_backprop()
        config = KernelConfig(algorithm=AlgorithmFamily.BACKPROP, hardware=HardwareTarget.CPU)

        # Get original weights
        original_weights = {}
        for i, layer in enumerate(backend._layers):
            original_weights[f"{i*2}.weight"] = layer.weight.clone()  # Sequential: Linear, ReLU, Linear
            if layer.bias is not None:
                original_weights[f"{i*2}.bias"] = layer.bias.clone()

        result = export_kernel(backend, config, output_dir=str(tmp_path))

        # Load exported state dict
        exported = torch.load(result.state_dict_path)
        assert "state_dict" in exported

        # Compare weights (Sequential uses numeric indices: 0=Linear, 1=ReLU, 2=Linear)
        for key, original in original_weights.items():
            assert key in exported["state_dict"], f"Missing key: {key}"
            assert torch.allclose(exported["state_dict"][key], original, atol=1e-6)

    def test_export_kernel_with_fp16_dtype(self, tmp_path):
        """Export should respect dtype from config for sample input."""
        from bioplausible.acceleration.backprop_kernels import BackpropKernelBackend

        backend = BackpropKernelBackend()
        config = KernelConfig(
            algorithm=AlgorithmFamily.BACKPROP,
            hardware=HardwareTarget.CPU,
            dtype=torch.float16,
        )
        backend.initialize(config)
        stack = [nn.Linear(4, 8), nn.Linear(8, 3)]
        backend.set_model_ref(stack)

        result = export_kernel(backend, config, output_dir=str(tmp_path))

        # Manifest should record the requested dtype
        import json
        manifest = json.loads((tmp_path / "backprop_cpu_manifest.json").read_text())
        assert manifest["dtype"] == "torch.float16"

        # Note: The actual state dict uses the layer weights' dtype (float32)
        # since layers are created in float32 by default. The sample input
        # for ONNX export uses the config dtype.

    def test_export_kernel_onnx_fallback_legacy(self, tmp_path, monkeypatch):
        """Should fall back to legacy exporter if torch.export fails."""
        # Mock torch.export.export to raise AttributeError (simulating old PyTorch)
        def mock_export(*args, **kwargs):
            raise AttributeError("torch.export.export not available")

        monkeypatch.setattr(torch.export, "export", mock_export)

        backend = self._bound_backprop()
        config = KernelConfig(algorithm=AlgorithmFamily.BACKPROP, hardware=HardwareTarget.CPU)

        result = export_kernel(backend, config, output_dir=str(tmp_path))

        # Should still create ONNX via fallback
        assert result.onnx_path is not None
        assert (tmp_path / "backprop_cpu.onnx").exists()


class TestExportKernelEdgeCases:
    """Test export_kernel edge cases."""

    def setup_method(self):
        """Clear kernel cache."""
        KernelRegistry.clear_cache()

    def _bound_backprop(self):
        """Create a BackpropKernelBackend bound to a small Linear stack."""
        from bioplausible.acceleration.backprop_kernels import BackpropKernelBackend

        backend = BackpropKernelBackend()
        config = KernelConfig(algorithm=AlgorithmFamily.BACKPROP, hardware=HardwareTarget.CPU)
        backend.initialize(config)
        stack = [nn.Linear(4, 8), nn.Linear(8, 3)]
        backend.set_model_ref(stack)
        return backend

    def test_export_without_bound_stack(self, tmp_path):
        """Export without bound stack should write manifest only."""
        from bioplausible.acceleration.backprop_kernels import BackpropKernelBackend

        backend = BackpropKernelBackend()
        config = KernelConfig(algorithm=AlgorithmFamily.BACKPROP, hardware=HardwareTarget.CPU)
        backend.initialize(config)
        # Don't call set_model_ref

        result = export_kernel(backend, config, output_dir=str(tmp_path))

        assert result.onnx_path is None
        assert result.state_dict_path is not None  # Still creates empty state dict?

    def test_export_manifest_contains_metadata(self, tmp_path):
        """Manifest should contain all required metadata fields."""
        backend = self._bound_backprop()
        config = KernelConfig(
            algorithm=AlgorithmFamily.BACKPROP,
            hardware=HardwareTarget.CPU,
            dtype=torch.bfloat16,
            settle_steps=30,
            beta=0.5,
            gamma=1.0,
            spectral_norm=True,
            extra={"custom": "value"},
        )
        backend.initialize(config)

        result = export_kernel(backend, config, output_dir=str(tmp_path))

        import json
        manifest = json.loads((tmp_path / "backprop_cpu_manifest.json").read_text())

        assert manifest["algorithm"] == "backprop"
        assert manifest["family"] == "backprop"
        assert manifest["hardware_target"] == "cpu"
        assert manifest["dtype"] == "torch.bfloat16"
        assert manifest["settle_steps"] == 30
        assert manifest["beta"] == 0.5
        assert manifest["gamma"] == 1.0
        assert manifest["spectral_norm"] is True
        assert manifest["extra"]["custom"] == "value"
        assert manifest["supports_autograd"] is True
        assert manifest["requires_settle"] is False

    def test_export_different_hardware_targets(self, tmp_path):
        """Export should work for different hardware targets."""
        backend = self._bound_backprop()
        config = KernelConfig(algorithm=AlgorithmFamily.BACKPROP, hardware=HardwareTarget.CPU)

        for hw in [HardwareTarget.CPU, HardwareTarget.CUDA, HardwareTarget.TRITON]:
            hw_path = tmp_path / hw.value
            hw_path.mkdir()
            result = export_kernel(backend, config, target=hw, output_dir=str(hw_path))

            # ONNX file uses config.hardware for naming (not target)
            onnx_file = hw_path / f"backprop_{config.hardware.value}.onnx"
            assert result.onnx_path is not None
            assert onnx_file.exists()

            import json
            manifest = json.loads((hw_path / f"backprop_{config.hardware.value}_manifest.json").read_text())
            # Manifest records config.hardware for hardware_target (current behavior)
            # target_spec uses config.hardware for the descriptor (current behavior)
            assert manifest["hardware_target"] == config.hardware.value
            assert manifest["target_spec"] == _TARGET_SPECS.get(config.hardware, config.hardware.value)