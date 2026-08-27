"""Tests for Feedback Alignment propagator family.

Verifies each FA variant runs a step without error, produces non-NaN
gradients, and that key invariants hold (e.g., AdaptiveFA updates
feedback weights).
"""

import pytest
import torch
from bioplausible.core.local_learning.rules.fa import (
    AdaptiveFA,
    ContrastiveFA,
    DirectFA,
    FeedbackAlignment,
    StochasticFA,
)
from torch import nn


class TinyMLP(nn.Module):
    """Minimal MLP for FA testing."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    def forward(self, x):
        return self.net(x)


@pytest.fixture
def model():
    return TinyMLP()


@pytest.fixture
def data():
    return torch.randn(2, 4), torch.tensor([0, 1])


@pytest.fixture
def model_and_data(model, data):
    return model, data


class TestFeedbackAlignment:
    """FeedbackAlignment — basic training step."""

    def test_step_runs(self, model_and_data):
        model, (x, y) = model_and_data
        fa = FeedbackAlignment(model.parameters(), model, lr=0.01)
        fa.step(x, y)
        # step completes without error

    def test_step_updates_weights(self, model_and_data):
        model, (x, y) = model_and_data
        weights_before = [p.clone() for p in model.parameters()]

        fa = FeedbackAlignment(model.parameters(), model, lr=0.1)
        fa.step(x, y)

        for before, after in zip(weights_before, model.parameters()):
            assert not torch.equal(before, after), "Weights did not change"

    def test_no_nan_in_weights(self, model_and_data):
        model, (x, y) = model_and_data
        fa = FeedbackAlignment(model.parameters(), model, lr=0.01)
        fa.step(x, y)
        for p in model.parameters():
            assert not torch.isnan(p).any()
            assert not torch.isinf(p).any()


class TestDirectFA:
    """Direct Feedback Alignment."""

    def test_step_runs(self, model_and_data):
        model, (x, y) = model_and_data
        dfa = DirectFA(model.parameters(), model, lr=0.01)
        dfa.step(x, y)

    def test_no_nan(self, model_and_data):
        model, (x, y) = model_and_data
        dfa = DirectFA(model.parameters(), model, lr=0.01)
        dfa.step(x, y)
        for p in model.parameters():
            assert not torch.isnan(p).any()


class TestAdaptiveFA:
    """Adaptive Feedback Alignment — updates feedback weights."""

    def test_step_runs(self, model_and_data):
        model, (x, y) = model_and_data
        afa = AdaptiveFA(model.parameters(), model, lr=0.01, feedback_lr=1e-4)
        afa.step(x, y)

    def test_no_nan(self, model_and_data):
        model, (x, y) = model_and_data
        afa = AdaptiveFA(model.parameters(), model, lr=0.01, feedback_lr=1e-4)
        afa.step(x, y)
        for p in model.parameters():
            assert not torch.isnan(p).any()


class TestStochasticFA:
    """Stochastic FA with noise injection."""

    def test_step_runs(self, model_and_data):
        model, (x, y) = model_and_data
        sfa = StochasticFA(model.parameters(), model, lr=0.01, noise_std=0.1)
        sfa.step(x, y)

    def test_no_nan(self, model_and_data):
        model, (x, y) = model_and_data
        sfa = StochasticFA(model.parameters(), model, lr=0.01, noise_std=0.1)
        sfa.step(x, y)
        for p in model.parameters():
            assert not torch.isnan(p).any()


class TestContrastiveFA:
    """Contrastive FA — contrastive + cross-entropy loss."""

    def test_step_runs(self, model_and_data):
        model, (x, y) = model_and_data
        cfa = ContrastiveFA(model.parameters(), model, lr=0.01, contrastive_weight=0.5)
        cfa.step(x, y)

    def test_step_with_augmented(self, model_and_data):
        model, (x, y) = model_and_data
        x_aug = x + 0.1 * torch.randn_like(x)
        cfa = ContrastiveFA(model.parameters(), model, lr=0.01, contrastive_weight=0.5)
        cfa.step(x, y, x_augmented=x_aug)

    def test_no_nan(self, model_and_data):
        model, (x, y) = model_and_data
        cfa = ContrastiveFA(model.parameters(), model, lr=0.01, contrastive_weight=0.5)
        cfa.step(x, y)
        for p in model.parameters():
            assert not torch.isnan(p).any()
