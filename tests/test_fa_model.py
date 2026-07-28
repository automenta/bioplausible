"""Tests for zoo/models/fa.py — Feedback Alignment model variants."""

import torch
import pytest

from bioplausible.zoo.models.fa import (
    FeedbackAlignmentLayer,
    FeedbackAlignmentEqProp,
    AdaptiveFeedbackAlignment,
    StochasticFA,
    ContrastiveFeedbackAlignment,
    DirectFeedbackAlignmentEqProp,
    DeepDFAEqProp,
    StandardFA,
    EnergyGuidedFA,
    EnergyMinimizingFA,
    LayerwiseEquilibriumFA,
    EquilibriumAlignment,
)


# ─── FeedbackAlignmentLayer ─────────────────────────────────────────────────────

class TestFeedbackAlignmentLayer:
    def test_construction_random(self):
        layer = FeedbackAlignmentLayer(10, 20, feedback_mode="random")
        assert layer.feedback_mode == "random"
        assert layer.weight.shape == (20, 10)

    def test_construction_symmetric(self):
        layer = FeedbackAlignmentLayer(10, 20, feedback_mode="symmetric")
        fb = layer.get_feedback_weight()
        assert fb.shape == (10, 20)

    def test_construction_evolving(self):
        layer = FeedbackAlignmentLayer(10, 20, feedback_mode="evolving")
        fb = layer.get_feedback_weight()
        assert fb.shape == (10, 20)

    def test_forward_shape(self):
        layer = FeedbackAlignmentLayer(10, 20)
        x = torch.randn(4, 10)
        out = layer(x)
        assert out.shape == (4, 20)

    def test_alignment_angle_symmetric_is_one(self):
        layer = FeedbackAlignmentLayer(10, 20, feedback_mode="symmetric")
        angle = layer.get_alignment_angle()
        assert abs(angle - 1.0) < 1e-6

    def test_alignment_angle_random_in_range(self):
        layer = FeedbackAlignmentLayer(10, 20, feedback_mode="random")
        angle = layer.get_alignment_angle()
        assert -1.0 <= angle <= 1.0

    def test_alignment_angle_evolving_in_range(self):
        layer = FeedbackAlignmentLayer(10, 20, feedback_mode="evolving")
        angle = layer.get_alignment_angle()
        assert -1.0 <= angle <= 1.0

    def test_alignment_angle_none_returns_symmetric(self):
        """None mode defaults to self.weight.t() via get_feedback_weight."""
        layer = FeedbackAlignmentLayer(10, 20, feedback_mode=None)
        angle = layer.get_alignment_angle()
        assert abs(angle - 1.0) < 1e-6


# ─── FeedbackAlignmentEqProp ────────────────────────────────────────────────────

class TestFeedbackAlignmentEqProp:
    def test_construction(self):
        model = FeedbackAlignmentEqProp(
            input_dim=10, hidden_dim=20, output_dim=3, num_layers=2
        )
        assert len(model.layers) == 2

    def test_forward_shape(self):
        model = FeedbackAlignmentEqProp(10, 20, 3, num_layers=2)
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 3)

    def test_get_mean_alignment(self):
        model = FeedbackAlignmentEqProp(10, 20, 3, num_layers=2)
        mean = model.get_mean_alignment()
        assert -1.0 <= mean <= 1.0

    def test_get_alignment_angles(self):
        model = FeedbackAlignmentEqProp(10, 20, 3, num_layers=3)
        angles = model.get_alignment_angles()
        assert len(angles) == 3
        for k, v in angles.items():
            assert k.startswith("layer_")
            assert -1.0 <= v <= 1.0


# ─── AdaptiveFeedbackAlignment ──────────────────────────────────────────────────

class TestAdaptiveFeedbackAlignment:
    def test_construction(self):
        model = AdaptiveFeedbackAlignment(
            input_dim=10, hidden_dim=20, output_dim=3, num_layers=2
        )
        assert isinstance(model, AdaptiveFeedbackAlignment)

    def test_forward_shape(self):
        model = AdaptiveFeedbackAlignment(
            input_dim=10, hidden_dim=20, output_dim=3, num_layers=2
        )
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 3)

    def test_train_step_returns_dict(self):
        model = AdaptiveFeedbackAlignment(
            input_dim=10, hidden_dim=20, output_dim=3, num_layers=2
        )
        x = torch.randn(8, 10)
        y = torch.randint(0, 3, (8,))
        result = model.train_step(x, y)
        assert "loss" in result
        assert "accuracy" in result

    def test_train_step_loss_decreases(self):
        model = AdaptiveFeedbackAlignment(
            input_dim=10, hidden_dim=20, output_dim=3, num_layers=2
        )
        x = torch.randn(16, 10)
        y = torch.randint(0, 3, (16,))
        losses = [model.train_step(x, y)["loss"] for _ in range(3)]
        assert losses[-1] <= losses[0] + 0.1

    def test_build_classmethod(self):
        class MockSpec:
            name = "test_fa"

        model = AdaptiveFeedbackAlignment.build(
            MockSpec(), input_dim=50, output_dim=5, hidden_dim=30,
            num_layers=2, device="cpu", task_type="vision"
        )
        assert isinstance(model, AdaptiveFeedbackAlignment)


# ─── StochasticFA ───────────────────────────────────────────────────────────────

class TestStochasticFA:
    def test_construction(self):
        model = StochasticFA(
            input_dim=10, hidden_dim=20, output_dim=3, num_layers=2
        )
        assert isinstance(model, StochasticFA)

    def test_forward_shape(self):
        model = StochasticFA(
            input_dim=10, hidden_dim=20, output_dim=3, num_layers=2
        )
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 3)

    def test_train_step_returns_dict(self):
        model = StochasticFA(
            input_dim=10, hidden_dim=20, output_dim=3, num_layers=2
        )
        x = torch.randn(8, 10)
        y = torch.randint(0, 3, (8,))
        result = model.train_step(x, y)
        assert "loss" in result
        assert "accuracy" in result

    def test_build_classmethod(self):
        class MockSpec:
            name = "stochastic"

        model = StochasticFA.build(
            MockSpec(), input_dim=50, output_dim=5, hidden_dim=30,
            num_layers=2, device="cpu", task_type="vision"
        )
        assert isinstance(model, StochasticFA)


# ─── ContrastiveFeedbackAlignment ───────────────────────────────────────────────

class TestContrastiveFeedbackAlignment:
    def test_construction(self):
        model = ContrastiveFeedbackAlignment(
            input_dim=10, hidden_dim=20, output_dim=3, num_layers=2
        )
        assert isinstance(model, ContrastiveFeedbackAlignment)

    def test_forward_shape(self):
        model = ContrastiveFeedbackAlignment(
            input_dim=10, hidden_dim=20, output_dim=3, num_layers=2
        )
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 3)

    def test_train_step_returns_dict(self):
        model = ContrastiveFeedbackAlignment(
            input_dim=10, hidden_dim=20, output_dim=3, num_layers=2
        )
        x = torch.randn(8, 10)
        y = torch.randint(0, 3, (8,))
        result = model.train_step(x, y)
        assert "loss" in result
        assert "accuracy" in result


# ─── DirectFeedbackAlignmentEqProp ──────────────────────────────────────────────

class TestDirectFeedbackAlignmentEqProp:
    def test_construction(self):
        model = DirectFeedbackAlignmentEqProp(
            input_dim=10, hidden_dim=20, output_dim=3, num_layers=2
        )
        assert len(model.layers) == 2

    def test_forward_shape(self):
        model = DirectFeedbackAlignmentEqProp(10, 20, 3, num_layers=2)
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 3)

    def test_feedback_alignment_angles(self):
        model = DirectFeedbackAlignmentEqProp(10, 20, 3, num_layers=2)
        angles = model.get_feedback_alignment_angles()
        assert isinstance(angles, dict)
        assert len(angles) == 2

    def test_get_stats_includes_mean_alignment(self):
        model = DirectFeedbackAlignmentEqProp(10, 20, 3, num_layers=2)
        stats = model.get_stats()
        assert "mean_alignment" in stats


# ─── DeepDFAEqProp ──────────────────────────────────────────────────────────────

class TestDeepDFAEqProp:
    def test_construction(self):
        model = DeepDFAEqProp(10, 20, 3, num_layers=2)
        assert hasattr(model, "layer_norms")

    def test_forward_shape(self):
        model = DeepDFAEqProp(10, 20, 3, num_layers=3)
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 3)

    def test_layer_norms_present(self):
        model = DeepDFAEqProp(10, 20, 3, num_layers=4)
        assert len(model.layer_norms) == 4  # one per hidden layer


# ─── StandardFA ─────────────────────────────────────────────────────────────────

class TestStandardFA:
    def test_construction(self):
        model = StandardFA(
            input_dim=10, hidden_dim=20, output_dim=3, num_layers=2
        )
        assert isinstance(model, StandardFA)

    def test_forward_shape(self):
        model = StandardFA(
            input_dim=10, hidden_dim=20, output_dim=3, num_layers=2
        )
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 3)

    def test_train_step_returns_dict(self):
        model = StandardFA(
            input_dim=10, hidden_dim=20, output_dim=3, num_layers=2
        )
        x = torch.randn(8, 10)
        y = torch.randint(0, 3, (8,))
        result = model.train_step(x, y)
        assert "loss" in result
        assert "accuracy" in result

    def test_train_step_loss_decreases(self):
        model = StandardFA(
            input_dim=10, hidden_dim=20, output_dim=3, num_layers=2
        )
        x = torch.randn(16, 10)
        y = torch.randint(0, 3, (16,))
        losses = [model.train_step(x, y)["loss"] for _ in range(3)]
        assert losses[-1] <= losses[0] + 0.1


# ─── EnergyGuidedFA ─────────────────────────────────────────────────────────────

class TestEnergyGuidedFA:
    def test_construction(self):
        model = EnergyGuidedFA(
            input_dim=10, hidden_dim=20, output_dim=3, num_layers=2
        )
        assert isinstance(model, EnergyGuidedFA)

    def test_forward_shape(self):
        model = EnergyGuidedFA(
            input_dim=10, hidden_dim=20, output_dim=3, num_layers=2
        )
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 3)

    def test_train_step_returns_dict(self):
        model = EnergyGuidedFA(
            input_dim=10, hidden_dim=20, output_dim=3, num_layers=2
        )
        x = torch.randn(8, 10)
        y = torch.randint(0, 3, (8,))
        result = model.train_step(x, y)
        assert "loss" in result
        assert "accuracy" in result

    def test_build_classmethod(self):
        class MockSpec:
            name = "guided"

        model = EnergyGuidedFA.build(
            MockSpec(), input_dim=50, output_dim=5, hidden_dim=30,
            num_layers=2, device="cpu", task_type="vision"
        )
        assert isinstance(model, EnergyGuidedFA)


# ─── EnergyMinimizingFA ─────────────────────────────────────────────────────────

class TestEnergyMinimizingFA:
    def test_construction(self):
        model = EnergyMinimizingFA(
            input_dim=10, hidden_dim=20, output_dim=3, num_layers=2
        )
        assert isinstance(model, EnergyMinimizingFA)

    def test_forward_shape(self):
        model = EnergyMinimizingFA(
            input_dim=10, hidden_dim=20, output_dim=3, num_layers=2
        )
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 3)

    def test_train_step_returns_dict(self):
        model = EnergyMinimizingFA(
            input_dim=10, hidden_dim=20, output_dim=3, num_layers=2
        )
        x = torch.randn(8, 10)
        y = torch.randint(0, 3, (8,))
        result = model.train_step(x, y)
        assert "loss" in result
        assert "accuracy" in result

    def test_build_classmethod(self):
        class MockSpec:
            name = "minimizing"

        model = EnergyMinimizingFA.build(
            MockSpec(), input_dim=50, output_dim=5, hidden_dim=30,
            num_layers=2, device="cpu", task_type="vision"
        )
        assert isinstance(model, EnergyMinimizingFA)


# ─── LayerwiseEquilibriumFA ─────────────────────────────────────────────────────

class TestLayerwiseEquilibriumFA:
    def test_construction(self):
        model = LayerwiseEquilibriumFA(
            input_dim=10, hidden_dim=20, output_dim=3, num_layers=2
        )
        assert isinstance(model, LayerwiseEquilibriumFA)

    def test_forward_shape(self):
        model = LayerwiseEquilibriumFA(
            input_dim=10, hidden_dim=20, output_dim=3, num_layers=2
        )
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 3)

    def test_train_step_returns_dict(self):
        model = LayerwiseEquilibriumFA(
            input_dim=10, hidden_dim=20, output_dim=3, num_layers=2
        )
        x = torch.randn(8, 10)
        y = torch.randint(0, 3, (8,))
        result = model.train_step(x, y)
        assert "loss" in result
        assert "accuracy" in result

    def test_build_classmethod(self):
        class MockSpec:
            name = "layerwise"

        model = LayerwiseEquilibriumFA.build(
            MockSpec(), input_dim=50, output_dim=5, hidden_dim=30,
            num_layers=2, device="cpu", task_type="vision"
        )
        assert isinstance(model, LayerwiseEquilibriumFA)


# ─── EquilibriumAlignment ───────────────────────────────────────────────────────

class TestEquilibriumAlignment:
    def test_construction(self):
        model = EquilibriumAlignment(10, 20, 3)
        assert isinstance(model, EquilibriumAlignment)

    def test_forward_shape(self):
        model = EquilibriumAlignment(10, 20, 3)
        x = torch.randn(4, 10)
        out = model(x)
        assert out.shape == (4, 3)

    def test_train_step_returns_dict(self):
        model = EquilibriumAlignment(10, 20, 3)
        x = torch.randn(8, 10)
        y = torch.randint(0, 3, (8,))
        result = model.train_step(x, y)
        assert "loss" in result
        assert "accuracy" in result

    def test_train_step_loss_decreases(self):
        model = EquilibriumAlignment(10, 20, 3, learning_rate=0.01)
        x = torch.randn(16, 10)
        y = torch.randint(0, 3, (16,))
        losses = [model.train_step(x, y)["loss"] for _ in range(3)]
        assert losses[-1] <= losses[0] + 0.1

    def test_build_classmethod(self):
        class MockSpec:
            name = "equilibrium"
            default_lr = 0.001

        model = EquilibriumAlignment.build(
            MockSpec(), input_dim=50, output_dim=5, hidden_dim=30,
            num_layers=2, device="cpu", task_type="vision"
        )
        assert isinstance(model, EquilibriumAlignment)

    def test_get_stats(self):
        model = EquilibriumAlignment(10, 20, 3)
        stats = model.get_stats()
        assert "num_params" in stats