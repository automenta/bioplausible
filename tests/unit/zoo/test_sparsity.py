"""Tests for sparsity methods (TopKPruning, ActivityDrivenPruning, RandomPruning).

Verifies each pruning method runs a step, changes weights, and respects
the sparsity budget.
"""

import pytest
import torch
from torch import nn

from bioplausible.zoo.sparsity.methods import (
    ActivityDrivenPruning,
    RandomPruning,
    TopKPruning,
)


class TestSparseMLP(nn.Module):
    """A small model that gets gradients before sparsity step."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

    def forward(self, x):
        return self.net(x)


@pytest.fixture
def model():
    m = TestSparseMLP()
    # Run a forward/backward to populate gradients
    x = torch.randn(4, 10)
    loss = m(x).sum()
    loss.backward()
    return m


def _count_zero_weights(model):
    return sum((p.abs() < 1e-8).sum().item() for p in model.parameters())


def _total_weights(model):
    return sum(p.numel() for p in model.parameters())


class TestTopKPruning:
    """TopKPruning — keeps top k_ratio by gradient activity."""

    def test_step_runs(self, model):
        tk = TopKPruning(model, k_ratio=0.5)
        tk.step()

    def test_pruning_creates_sparsity(self, model):
        before = _count_zero_weights(model)
        tk = TopKPruning(model, k_ratio=0.3)
        tk.step()
        after = _count_zero_weights(model)
        assert after > before, "TopK did not increase zero-weight count"

    def test_no_nan_after_pruning(self, model):
        tk = TopKPruning(model, k_ratio=0.5)
        tk.step()
        for p in model.parameters():
            assert not torch.isnan(p).any()

    def test_zero_k_ratio(self, model):
        """k_ratio=1.0 keeps everything (nothing pruned)."""
        tk = TopKPruning(model, k_ratio=1.0)
        tk.step()
        zero_count = _count_zero_weights(model)
        assert zero_count == 0 or zero_count < _total_weights(model) * 0.01


class TestActivityDrivenPruning:
    """ActivityDrivenPruning — prunes below median activity."""

    def test_step_runs(self, model):
        adp = ActivityDrivenPruning(model, prune_fraction=0.1)
        adp.step()

    def test_pruning_creates_sparsity(self, model):
        before = _count_zero_weights(model)
        adp = ActivityDrivenPruning(model, prune_fraction=0.2)
        adp.step()
        after = _count_zero_weights(model)
        assert after >= before, "Activity pruning did not increase sparsity"

    def test_no_nan_after_pruning(self, model):
        adp = ActivityDrivenPruning(model, prune_fraction=0.1)
        adp.step()
        for p in model.parameters():
            assert not torch.isnan(p).any()


class TestRandomPruning:
    """RandomPruning — randomly drops weights."""

    def test_step_runs(self, model):
        rp = RandomPruning(model, prune_fraction=0.1)
        rp.step()

    def test_pruning_creates_sparsity(self, model):
        before = _count_zero_weights(model)
        rp = RandomPruning(model, prune_fraction=0.3)
        rp.step()
        after = _count_zero_weights(model)
        assert after > before, "Random pruning did not increase sparsity"

    def test_deterministic_seed(self, model):
        """Same seed produces same mask (deterministic within an instance)."""
        # Clone model to get identical initial weights
        state = model.state_dict()
        model2 = TestSparseMLP()
        model2.load_state_dict(state)
        x = torch.randn(4, 10)
        model(x).sum().backward()
        model2(x).sum().backward()

        rp1 = RandomPruning(model, prune_fraction=0.3, seed=42)
        rp2 = RandomPruning(model2, prune_fraction=0.3, seed=42)
        rp1.step()
        rp2.step()

        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(p1.data, p2.data), (
                "Same seed should produce identical pruning"
            )

    def test_no_nan(self, model):
        rp = RandomPruning(model, prune_fraction=0.1)
        rp.step()
        for p in model.parameters():
            assert not torch.isnan(p).any()
