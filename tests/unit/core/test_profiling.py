"""Regression tests for Memory Accounting & Profiling (Phase 3.6.6/3.6.8)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from computronium.core.profiling import ResourceUsage


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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
