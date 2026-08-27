"""Kernel Parity Test Infrastructure (REFACTOR7 Phase 1 CI gate).

A reusable base class (:class:`KernelParityBase`) that every algorithm-specific
kernel backend parity suite can inherit from, plus shared fixtures for
constructing a synthetic MLP task. The concrete tests here exercise the
registry-level contract (every family registers a backend that initialises and
produces finite outputs) and verify the exact-gradient reference (BACKPROP)
against ``torch.autograd``.

Design: kernel backends are a heterogeneous family (different forward/backward
signatures per algorithm), so the base class offers *helpers* rather than a
single rigid template — each concrete backend suite specialises
``_make_backend`` / ``_reference_grads`` as needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from bioplausible.acceleration import (
    AlgorithmFamily,
    HardwareTarget,
    KernelConfig,
    KernelRegistry,
    get_algorithm_kernels,
)
from bioplausible.acceleration.backprop_kernels import BackpropKernelBackend
from torch import nn


@pytest.fixture(scope="module", autouse=True)
def _populate_kernel_registry():
    """Trigger the (lazy) kernel module imports so backends self-register.

    Kernel modules register themselves at import time; this fixture forces the
    import up front so registry-backed assertions see every family.
    """
    get_algorithm_kernels()
    yield


def linear_stack_default() -> list[nn.Linear]:
    """A fresh 3-layer Linear stack for backends that need a model reference."""
    torch.manual_seed(0)
    return [nn.Linear(64, 32), nn.Linear(32, 32), nn.Linear(32, 8)]


@dataclass(frozen=True, slots=True)
class SyntheticTask:
    """A fixed-seed synthetic classification task for parity checks."""

    x: torch.Tensor
    y: torch.Tensor
    input_dim: int
    output_dim: int


@pytest.fixture(scope="module")
def synthetic_task() -> SyntheticTask:
    """64-dim, 8-class, 200-sample synthetic task (CPU, fixed seed)."""
    torch.manual_seed(42)
    n, input_dim, output_dim = 200, 64, 8
    x = torch.randn(n, input_dim)
    y = torch.randint(0, output_dim, (n,))
    return SyntheticTask(x=x, y=y, input_dim=input_dim, output_dim=output_dim)


@pytest.fixture(scope="module")
def linear_stack() -> list[nn.Linear]:
    """A 3-layer Linear stack used to set ``set_model_ref`` on backends."""
    torch.manual_seed(0)
    return [nn.Linear(64, 32), nn.Linear(32, 32), nn.Linear(32, 8)]


class KernelParityBase:
    """Shared helpers for kernel-backend parity suites.

    Subclasses set ``algorithm`` and implement ``_make_backend`` to return a
    backend instance wired to a synthetic model; the base then validates the
    registry contract (initialise + finite forward/backward + memory stats).
    """

    algorithm: AlgorithmFamily
    supported_hardware: tuple[HardwareTarget, ...] = (
        HardwareTarget.CPU,
        HardwareTarget.CUDA,
        HardwareTarget.TRITON,
    )

    def _make_backend(self):
        """Return an initialised backend instance for ``self.algorithm``."""
        raise NotImplementedError

    def test_registered(self):
        """The algorithm family must have at least one registered backend."""
        assert KernelRegistry.list_for(self.algorithm), (
            f"No kernel backend registered for {self.algorithm.value}"
        )

    def test_get_algorithm_kernels_exposes_family(self):
        """``get_algorithm_kernels`` must surface a backend for the family."""
        kernels = get_algorithm_kernels()
        assert self.algorithm.value in kernels, (
            f"get_algorithm_kernels missing {self.algorithm.value}"
        )

    def test_initialize_and_memory_stats(self):
        """``initialize`` must succeed and ``get_memory_stats`` be finite."""
        backend = self._make_backend()
        assert backend._config is not None
        stats = backend.get_memory_stats()
        assert stats and all(torch.isfinite(torch.tensor(v)) for v in stats.values())

    def test_forward_finite(self, synthetic_task):
        """A forward pass must produce finite output of the expected shape."""
        backend = self._make_backend()
        if not hasattr(backend, "forward"):
            pytest.skip(f"{self.algorithm.value} backend has no generic forward")
        out, acts = backend.forward(synthetic_task.x[:8])
        assert out.shape == (8, synthetic_task.output_dim)
        assert not torch.isnan(out).any() and not torch.isinf(out).any()


class TestBackpropKernelParity(KernelParityBase):
    """Exact-gradient reference: BACKPROP kernel vs ``torch.autograd``.

    The backprop kernel is the *reference* every fused/settled kernel must
    match, so its gradient parity gate is strict (tight tolerance).
    """

    algorithm = AlgorithmFamily.BACKPROP

    def _make_backend(self, layers=None):
        backend = BackpropKernelBackend()
        backend.set_model_ref(layers if layers is not None else linear_stack_default())
        backend.initialize(
            KernelConfig(
                algorithm=AlgorithmFamily.BACKPROP,
                hardware=HardwareTarget.CPU,
                extra={"activation": "relu"},
            )
        )
        return backend

    def test_gradient_parity_with_autograd(self, synthetic_task, linear_stack):
        """Kernel manual backprop must match ``torch.autograd`` gradients."""
        torch.manual_seed(0)
        layers = [nn.Linear(64, 32), nn.Linear(32, 32), nn.Linear(32, 8)]
        model = nn.Sequential(layers[0], nn.ReLU(), layers[1], nn.ReLU(), layers[2])

        backend = self._make_backend(layers)
        x = synthetic_task.x[:8]
        y = synthetic_task.y[:8]

        out, acts = backend.forward(x)
        err = torch.softmax(out, dim=-1) - nn.functional.one_hot(y, 8).float()
        grads = backend.backward(acts, err)

        model.train()
        model.zero_grad()
        nn.functional.cross_entropy(model(x), y).backward()

        for i, layer in enumerate(layers):
            w_grad = grads[f"layers.{i}.weight"]
            assert torch.allclose(w_grad, layer.weight.grad, atol=1e-4), (
                f"layer {i} weight grad mismatch"
            )
            b_grad = grads[f"layers.{i}.bias"]
            assert torch.allclose(b_grad, layer.bias.grad, atol=1e-4), (
                f"layer {i} bias grad mismatch"
            )

    def test_all_families_register_backends(self):
        """Every AlgorithmFamily providing a protocol backend is registered.

        ``EQPROP`` is intentionally absent: it uses the standalone NumPy/CuPy
        ``EqPropKernel`` engine (``acceleration/kernels.py``) rather than a
        ``KernelBackend`` protocol implementation, and is validated through
        ``test_model_kernel_api`` and the parity benchmark.
        """
        from bioplausible.acceleration.kernel_backend import AlgorithmFamily as AF

        unregistered = [
            f.value for f in AF if f != AF.EQPROP and not KernelRegistry.list_for(f)
        ]
        assert not unregistered, f"families missing a kernel backend: {unregistered}"
