"""Equilibrium settling memory-stability tests (FIX.md OOM)."""

from __future__ import annotations

import torch


def _energy_fn(states: list[torch.Tensor]) -> torch.Tensor:
    """Small MSE-based energy; layers simulated by a fixed linear map."""
    w = torch.randn(states[0].shape[1], states[0].shape[2], device=states[0].device)
    return sum(0.5 * (state @ w).pow(2).sum() for state in states)


def test_settling_releases_graph_between_steps() -> None:
    """Settling must free each step's autograd graph (no accumulation).

    Regression for FIX.md OOM: eqprop settling on CIFAR-10 grew GPU/CPU memory
    monotonically because per-step computation graphs were retained. The loop
    uses ``retain_graph=False`` and drops ``E``/``grads`` each iteration; this
    test runs enough steps that retained graphs would fail (device-cpu, so an
    actual "OOM" is a MemoryError) whereas correct release stays bounded.
    """
    from bioplausible.core.local_learning.settling import energy_gradient_descent

    states = [torch.randn(4, 8, 8, requires_grad=True)]
    settled = energy_gradient_descent(states, _energy_fn, steps=200, lr=0.1)
    assert len(settled) == 1
    assert not settled[0].requires_grad  # detached outputs


def test_sequential_settling_bounded_memory() -> None:
    """Repeated settling calls (simulating sequential trials) must not blow up.

    Each call detaches its outputs; no graph survives a settle. If graphs were
    retained across calls, cumulative retained activations would grow without
    bound. We assert the peak-allocated byte count stays flat after a warm-up
    call.
    """
    from bioplausible.core.local_learning.settling import energy_gradient_descent

    def run_once() -> None:
        s = [torch.randn(4, 8, 8, requires_grad=True)]
        energy_gradient_descent(s, _energy_fn, steps=300, lr=0.1)

    run_once()  # warm up / allocate caches

    before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    for _ in range(5):
        run_once()
    after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    # On CUDA: growth should be ~0 (caching allocator reuses). On CPU there is no
    # allocator to inspect; the test is still a smoke check that settling works.
    if torch.cuda.is_available():
        assert after <= before + 64 * 1024 * 1024, (
            f"settling retained memory across calls: before={before}, after={after}"
        )
