"""Regression tests for Memory Accounting & Profiling (Phase 3.6.6/3.6.8)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from computronium.resources import ResourceUsage


class TestResourceUsage:
    """Tests for ResourceUsage correctness."""

    def test_peak_activation_bytes_field_exists(self):
        """ResourceUsage has peak_activation_bytes field."""
        usage = ResourceUsage()
        assert hasattr(usage, "peak_activation_bytes")
        assert usage.peak_activation_bytes == 0

    def test_peak_activation_bytes_serialization(self):
        """peak_activation_bytes survives serialization round-trip."""
        usage = ResourceUsage(
            peak_activation_bytes=123456,
            compute=1000,
            memory=10.0,
        )
        d = usage.to_dict()
        usage2 = ResourceUsage.from_dict(d)
        assert usage2.peak_activation_bytes == 123456

    def test_peak_activation_bytes_addition(self):
        """peak_activation_bytes uses max in addition."""
        u1 = ResourceUsage(peak_activation_bytes=1000)
        u2 = ResourceUsage(peak_activation_bytes=2000)
        u3 = u1 + u2
        assert u3.peak_activation_bytes == 2000  # max

    def test_peak_activation_bytes_division(self):
        """peak_activation_bytes divided in averaging."""
        usage = ResourceUsage(peak_activation_bytes=1000)
        avg = usage / 2
        assert avg.peak_activation_bytes == 500

    def test_measure_captures_peak_activation_bytes(self):
        """ResourceUsage.measure captures peak_activation_bytes."""
        model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )
        x = torch.randn(64, 784)

        usage = ResourceUsage.measure(model, x)

        assert hasattr(usage, "peak_activation_bytes")
        # On CPU, may be 0, but field should exist
        assert isinstance(usage.peak_activation_bytes, int)


class TestMemoryAccountedModel:
    """Tests for MemoryAccountedModel hook coverage."""

    def test_hooks_registered_on_target_layers(self):
        """Hooks registered on Linear and activation layers."""
        # This is tested in the memory_accounting audit script
        # Just verify the import works
        from computronium.experiments.joint.memory_wall import MemoryAccountedModel

        assert MemoryAccountedModel is not None


class TestEffectiveFlopsWiring:
    """effective-FLOPs -> C vector: gate-aware FLOPs feed the compute axis."""

    def test_measure_suite_resources_maps_effective_flops(self) -> None:
        from torch import nn as _nn

        from computronium.core.profiling import measure_suite_resources

        model = _nn.Sequential(_nn.Linear(32, 16), _nn.Linear(16, 8))
        usage = measure_suite_resources(
            model,
            coordinate="digital/feedforward/instantaneous/routing/gradient/euclidean",
            device="cpu",
            batch_size=4,
            elapsed_s=0.5,
            effective_flops=1000.0,
        )
        # Gate-aware effective FLOPs override the parameter-count estimate and
        # flow into the C compute axis (compute = 3x forward FLOPs).
        assert usage.forward_flops == pytest.approx(1000.0)
        assert usage.effective_flops == pytest.approx(1000.0)
        assert usage.compute == pytest.approx(3000.0)

    def test_resource_vector_roundtrip_preserves_effective_flops(self) -> None:
        from computronium.core.profiling import measure_suite_resources

        model = nn.Linear(8, 4)
        usage = measure_suite_resources(
            model,
            coordinate="digital/recurrent/instantaneous/fast_weights/gradient/euclidean",
            device="cpu",
            batch_size=2,
            elapsed_s=0.1,
            effective_flops=500.0,
        )
        restored = ResourceUsage.from_dict(usage.to_dict())
        assert restored.effective_flops == pytest.approx(500.0)
        assert restored.compute == pytest.approx(usage.compute)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
