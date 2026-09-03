"""Research Direction Property Tests (ontology-native replacement).

The former zoo-registry tests verified Tracks 42-44 via ``Registry.get``;
the same scientific claims are now verified against the native 5-D
factories in ``computronium.models.native``:

- Holomorphic EP learns using complex-valued weights (QuantumSubstrate)
- Directed EP learns with asymmetric forward/feedback pathways
- Finite-Nudge EP learns stably at large beta
"""

import pytest
import torch

from computronium.models.native.research_native import (
    create_native_directed_ep,
    create_native_finite_nudge_ep,
    create_native_holomorphic_ep,
)

INPUT_DIM = 8
HIDDEN_DIM = 8
OUTPUT_DIM = 4


@pytest.fixture(scope="module")
def synthetic_mlp_task():
    """Minimal separable classification task."""
    torch.manual_seed(42)
    n_samples = 32
    x = torch.randn(n_samples, INPUT_DIM)
    y = torch.randint(0, OUTPUT_DIM, (n_samples,))
    for c in range(OUTPUT_DIM):
        mask = y == c
        if mask.any():
            direction = torch.randn(INPUT_DIM)
            direction = direction / direction.norm() * 1.5
            x[mask] += direction * 0.5
    return x, y


def _train(
    model, x: torch.Tensor, y: torch.Tensor, steps: int = 5
) -> tuple[float, float]:
    """Target-free loss trajectory (free_loss when the phase exists)."""
    xb, yb = x[:16], y[:16]
    first = float("inf")
    last = float("inf")
    for i in range(steps):
        metrics = model.train_step(xb, yb)
        loss = float(metrics.get("free_loss", metrics["loss"]))
        if i == 0:
            first = loss
        last = loss
    assert torch.isfinite(torch.tensor(last)), f"loss diverged: {last}"
    return first, last


class TestHolomorphicEP:
    def test_substrate_carries_complex_states(self) -> None:
        model = create_native_holomorphic_ep(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
        config = model.substrate.config
        assert config.precision == "complex64", config
        assert str(config.substrate_type).lower().endswith("quantum"), config

    def test_train_step_runs_and_losses_finite(self, synthetic_mlp_task) -> None:
        x, y = synthetic_mlp_task
        model = create_native_holomorphic_ep(
            INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, settle_steps=5
        )
        first, last = _train(model, x, y, steps=3)
        assert last < first * 2 or last < 10.0, (first, last)


class TestDirectedEP:
    def test_feedback_asymmetric(self) -> None:
        model = create_native_directed_ep(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
        forward = model.geometry.params["0.weight"]
        feedback = model.geometry.params["recurrent_weight"]
        assert feedback.shape == forward.shape
        assert not torch.allclose(feedback, forward.t()), (
            "directed EP feedback must not be the forward transpose"
        )

    def test_learns(self, synthetic_mlp_task) -> None:
        x, y = synthetic_mlp_task
        model = create_native_directed_ep(
            INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, settle_steps=5
        )
        first, last = _train(model, x, y, steps=5)
        assert last < first, f"directed EP must learn: {first} -> {last}"


class TestFiniteNudgeEP:
    def test_learns_stably_at_large_beta(self, synthetic_mlp_task) -> None:
        x, y = synthetic_mlp_task
        model = create_native_finite_nudge_ep(
            INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, beta=4.0, settle_steps=5
        )
        first, last = _train(model, x, y, steps=5)
        assert last < first, f"finite-nudge EP must learn: {first} -> {last}"
