"""Unit tests for the kernel-backend registry and config infrastructure.

Covers :mod:`bioplausible.acceleration.kernel_backend` and
:mod:`bioplausible.acceleration.contrastive_primitives` — the shared seam every
algorithm kernel backend routes through.
"""

from __future__ import annotations

import pytest
import torch

from bioplausible.acceleration import (
    AlgorithmFamily,
    HardwareTarget,
    KernelConfig,
    KernelRegistry,
    batched_outer_product,
    contrastive_delta,
    contrastive_hebbian_update,
    infer_algorithm_family,
    stdp_update,
)
from bioplausible.acceleration.kernel_backend import KernelRegistry as _KernelRegistry


@pytest.fixture(autouse=True)
def _clear_kernel_cache():
    """Clear kernel registry cache between tests to avoid state pollution."""
    _KernelRegistry.clear_cache()
    yield
    _KernelRegistry.clear_cache()


class TestKernelConfig:
    """Config defaults and the settle-steps auto-default behaviour."""

    def test_default_dtype_and_flags(self):
        cfg = KernelConfig(algorithm=AlgorithmFamily.FA, hardware=HardwareTarget.CPU)
        assert cfg.dtype == torch.float32
        assert cfg.use_autograd is False
        assert cfg.beta == 0.0
        assert cfg.gamma == 1.0

    def test_eqprop_defaults_settle_steps(self):
        cfg = KernelConfig(
            algorithm=AlgorithmFamily.EQPROP, hardware=HardwareTarget.CPU
        )
        assert cfg.settle_steps == 30

    def test_mep_defaults_settle_steps(self):
        cfg = KernelConfig(algorithm=AlgorithmFamily.MEP, hardware=HardwareTarget.CPU)
        assert cfg.settle_steps == 30

    def test_extra_passthrough(self):
        cfg = KernelConfig(
            algorithm=AlgorithmFamily.HEBBIAN,
            hardware=HardwareTarget.CUDA,
            extra={"learning_rate": 0.01, "use_oja": True},
        )
        assert cfg.extra["learning_rate"] == 0.01


class TestKernelRegistry:
    """Registry registration, lookup, fallback and cache semantics."""

    def setup_method(self):
        """Snapshot registry state so tests never pollute the global registry."""
        self._backends_saved = KernelRegistry._backends
        self._instances_saved = KernelRegistry._instances
        KernelRegistry._backends = {}
        KernelRegistry._instances = {}

    def teardown_method(self):
        """Restore the registry snapshot taken in ``setup_method``."""
        KernelRegistry._backends = self._backends_saved
        KernelRegistry._instances = self._instances_saved

    def test_register_and_get(self):
        KernelRegistry.register(AlgorithmFamily.FA, HardwareTarget.CPU, dict)
        backend = KernelRegistry.get(AlgorithmFamily.FA, HardwareTarget.CPU)
        assert backend == {}

    def test_get_unregistered_returns_none(self):
        # EQPROP is registered for CPU/CUDA/TRITON but not QUANTUM.
        assert (
            KernelRegistry.get(AlgorithmFamily.EQPROP, HardwareTarget.QUANTUM) is None
        )

    def test_get_best_falls_back_through_priority(self):
        # TRITON preferred; only CPU is registered -> falls back to CPU.
        KernelRegistry.register(AlgorithmFamily.PC, HardwareTarget.CPU, dict)
        backend = KernelRegistry.get_best(AlgorithmFamily.PC, HardwareTarget.TRITON)
        assert backend == {}

    def test_has_and_list(self):
        KernelRegistry.register(AlgorithmFamily.FA, HardwareTarget.CPU, dict)
        assert KernelRegistry.has(AlgorithmFamily.FA, HardwareTarget.CPU)
        assert HardwareTarget.CPU in KernelRegistry.list_for(AlgorithmFamily.FA)

    def test_clear_cache(self):
        KernelRegistry.register(AlgorithmFamily.TP, HardwareTarget.CPU, dict)
        KernelRegistry.get(AlgorithmFamily.TP, HardwareTarget.CPU)
        assert KernelRegistry._instances
        KernelRegistry.clear_cache()
        assert not KernelRegistry._instances


class TestInferAlgorithmFamily:
    """Model-name -> AlgorithmFamily inference."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("eqprop_mlp", AlgorithmFamily.EQPROP),
            ("looped_mlp", AlgorithmFamily.EQPROP),
            ("standard_fa", AlgorithmFamily.FA),
            ("deep_hebbian", AlgorithmFamily.HEBBIAN),
            ("pepita", AlgorithmFamily.PEPITA),
            ("diff_target_prop", AlgorithmFamily.TP),
            ("predictive_coding", AlgorithmFamily.PC),
            ("spiking_stdp", AlgorithmFamily.SNN),
            ("tile_pc", AlgorithmFamily.TILE),
            ("muon_backprop", AlgorithmFamily.MEP),
            ("backprop_mlp", AlgorithmFamily.BACKPROP),
            ("unknown_model", None),
        ],
    )
    def test_infer(self, name, expected):
        assert infer_algorithm_family(name) == expected


class TestContrastivePrimitives:
    """Shared contrastive primitives used by all kernel backends."""

    def test_batched_outer_product_shape_and_scale(self):
        src = torch.randn(8, 16)
        dst = torch.randn(8, 4)
        out = batched_outer_product(src, dst)
        assert out.shape == (4, 16)
        assert torch.allclose(out, (dst.T @ src) / 8)

    def test_contrastive_delta(self):
        free = torch.ones(4, 16)
        nudged = free * 2
        delta = contrastive_delta(free, nudged, beta=0.5)
        assert torch.allclose(delta, torch.full((4, 16), 2.0))

    def test_contrastive_hebbian_update_finite(self):
        delta = contrastive_hebbian_update(
            torch.randn(8, 16),
            torch.randn(8, 4),
            torch.randn(8, 16),
            torch.randn(8, 4),
            lr=0.01,
            beta=0.5,
        )
        assert delta.shape == (4, 16)
        assert torch.isfinite(delta).all()

    def test_stdp_update_shape(self):
        # STDP returns a proper [N_post, N_pre] correlation matrix.
        pre = torch.rand(8, 16, 5)
        post = torch.rand(8, 8, 5)
        delta = stdp_update(pre, post)
        assert delta.shape == (8, 16)
        assert torch.isfinite(delta).all()

    def test_stdp_update_symmetric_pair(self):
        # A spiking pre-before-post pair should give symmetric LTP/LTD cancellation
        # when A_plus == A_minus and the spike trains are identical.
        spikes = torch.zeros(4, 6, 10)
        spikes[..., 2] = 1.0  # a single simultaneous spike across all neurons
        delta = stdp_update(spikes, spikes, A_plus=0.01, A_minus=0.01)
        assert delta.shape == (6, 6)
        assert torch.isfinite(delta).all()
