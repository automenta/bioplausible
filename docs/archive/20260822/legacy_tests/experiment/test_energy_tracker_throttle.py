"""EnergyTracker heavy-metric throttle (EXPERIMENT_PLAN5 §1).

The activation-sparsity forward and the weight-sparsity GPU reduction dominate
per-step measurement cost. The tracker must run them **once per probe** (first
step) and reuse the cached value thereafter, while standalone use (no step
counter) keeps the eager behaviour.
"""

from __future__ import annotations

from unittest.mock import patch

import torch
from bioplausible.core.profiling import EnergyTracker
from torch import nn


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(16, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(x)


def test_heavy_metrics_computed_once_per_probe() -> None:
    """With a step counter, activation sparsity is sampled only on step 0."""
    model = _MLP()
    with patch(
        "bioplausible.core.profiling._estimate_activation_sparsity",
        return_value=0.25,
    ) as spy:
        for step in range(5):
            with EnergyTracker(model, global_step=step):
                _ = model(torch.zeros(2, 16))
    # Heavy metric sampled exactly once across the whole probe.
    assert spy.call_count == 1

    # That single sample is reused on every later step's profile.
    with EnergyTracker(model, global_step=3) as et:
        pass
    assert et.profile is not None
    assert et.profile.activation_sparsity == 0.25


def test_standalone_tracker_always_measures() -> None:
    """Without a step counter (standalone use) every step samples sparsity."""
    model = _MLP()
    with patch(
        "bioplausible.core.profiling._estimate_activation_sparsity",
        return_value=0.0,
    ) as spy:
        for _ in range(3):
            with EnergyTracker(model):
                _ = model(torch.zeros(2, 16))
    assert spy.call_count == 3


def test_throttle_keeps_cheap_metrics() -> None:
    """Throttling must not drop the cheap per-step metrics (time/memory/flops)."""
    model = _MLP()
    with (
        patch(
            "bioplausible.core.profiling._estimate_activation_sparsity",
            return_value=0.0,
        ),
        EnergyTracker(model, global_step=2, requires_backward=True) as et,
    ):
        _ = model(torch.zeros(2, 16))
    assert et.profile is not None
    assert et.profile.param_count > 0
    assert et.profile.forward_flops > 0
    assert isinstance(et.profile.peak_memory_mb, float)
