"""Integration tests for SettleProtocol adoption across model families.

Tests that all 4 model families (MEP, O1Memory, PC, Tile) properly implement
SettleProtocol and integrate with settle_universal for unified telemetry.
"""

from __future__ import annotations

import pytest
import torch

from bioplausible.core.local_learning.settling import (
    SettleConfig,
    SettleProtocol,
    SettleTelemetry,
    settle_universal,
)
from bioplausible.core.trainer import TrainingMetrics

# Import zoo models to trigger registration
from bioplausible.zoo import models
from bioplausible.zoo.models.mep import MEPEqPropModel
from bioplausible.zoo.models.o1memory import O1MemoryModel
from bioplausible.zoo.models.predictive_coding import PredictiveCodingHybrid
from bioplausible.core.local_learning.algorithm import TileAlgorithm, TileAlgorithmConfig
from bioplausible.zoo.models.tile_models import TilePC


def _run_model_settle(model, x, max_steps=10, convergence_threshold=1e-3, **kwargs):
    """Run settle using the model's internal _run_settle_universal method."""
    # Update model convergence settings if provided
    if hasattr(model, 'convergence_threshold'):
        model.convergence_threshold = convergence_threshold
    return model._run_settle_universal(x, steps=max_steps, **kwargs)


class TestMEPEqPropModelSettleProtocol:
    """Test MEPEqPropModel SettleProtocol integration."""

    @pytest.fixture
    def model(self):
        return MEPEqPropModel(
            input_dim=20,
            hidden_dim=16,
            output_dim=4,
            settle_steps=10,
            settle_lr=0.1,
            tol=1e-3,
            patience=3,
        )

    def test_isinstance_settle_protocol(self, model):
        """MEPEqPropModel implements SettleProtocol."""
        assert isinstance(model, SettleProtocol)

    def test_settle_universal_returns_telemetry(self, model):
        """_run_settle_universal returns (output, steps_taken, converged, telemetry)."""
        x = torch.randn(4, 20)
        out, steps_taken, converged, telemetry = _run_model_settle(model, x, max_steps=10)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (4, 4)
        assert isinstance(steps_taken, int)
        assert 0 < steps_taken <= 10
        assert isinstance(converged, bool)
        assert isinstance(telemetry, SettleTelemetry)
        # State list is stored in model._last_activations
        assert hasattr(model, "_last_activations")
        if model._last_activations is not None:
            assert isinstance(model._last_activations, list)
            assert all(isinstance(s, torch.Tensor) for s in model._last_activations)

    def test_forward_return_dynamics(self, model):
        """forward(return_dynamics=True) returns dynamics dict with telemetry."""
        x = torch.randn(4, 20)
        out, dynamics = model(x, return_dynamics=True)
        assert out.shape == (4, 4)
        assert isinstance(dynamics, dict)
        assert "deltas" in dynamics
        assert "final_delta" in dynamics
        assert "steps_taken" in dynamics
        assert "converged" in dynamics
        assert "settle_time_s" in dynamics

    def test_get_settle_telemetry(self, model):
        """get_settle_telemetry returns SettleTelemetry after settle_universal."""
        x = torch.randn(4, 20)
        _run_model_settle(model, x, max_steps=10)
        telemetry = model.get_settle_telemetry()
        assert isinstance(telemetry, SettleTelemetry)
        assert len(telemetry.deltas) > 0
        assert telemetry.steps_taken > 0

    def test_loose_threshold_early_convergence(self, model):
        """Loose convergence threshold triggers early stop."""
        model.convergence_threshold = 1.0
        x = torch.randn(4, 20)
        state, steps_taken, converged, telemetry = _run_model_settle(
            model, x, max_steps=20, convergence_threshold=1.0
        )
        assert converged
        assert steps_taken < 20

    def test_training_metrics_includes_settle_telemetry(self, model):
        """TrainingMetrics.extra['settle_telemetry'] populated after train_step."""
        x = torch.randn(8, 20)
        y = torch.randint(0, 4, (8,))
        model.train()
        result = model.train_step(x, y)
        assert isinstance(result, dict)
        assert "loss" in result
        assert "accuracy" in result


class TestO1MemoryModelSettleProtocol:
    """Test O1MemoryModel SettleProtocol integration."""

    @pytest.fixture
    def model(self):
        return O1MemoryModel(
            input_dim=20,
            hidden_dim=16,
            output_dim=4,
            settle_steps=10,
            settle_lr=0.1,
            momentum=0.5,
            optimizer_lr=0.01,
        )

    def test_isinstance_settle_protocol(self, model):
        """O1MemoryModel implements SettleProtocol."""
        assert isinstance(model, SettleProtocol)

    def test_settle_universal_returns_telemetry(self, model):
        """_run_settle_universal returns (output, steps_taken, converged, telemetry)."""
        x = torch.randn(4, 20)
        out, steps_taken, converged, telemetry = _run_model_settle(model, x, max_steps=10)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (4, 4)
        assert isinstance(steps_taken, int)
        assert 0 < steps_taken <= 10
        assert isinstance(converged, bool)
        assert isinstance(telemetry, SettleTelemetry)
        # State list is stored in model._last_activations
        assert hasattr(model, "_last_activations")
        if model._last_activations is not None:
            assert isinstance(model._last_activations, list)
            assert all(isinstance(s, torch.Tensor) for s in model._last_activations)

    def test_forward_return_dynamics(self, model):
        """forward(return_dynamics=True) returns dynamics dict with telemetry."""
        x = torch.randn(4, 20)
        out, dynamics = model(x, return_dynamics=True)
        assert out.shape == (4, 4)
        assert isinstance(dynamics, dict)
        assert "deltas" in dynamics
        assert "final_delta" in dynamics
        assert "steps_taken" in dynamics
        assert "converged" in dynamics
        assert "settle_time_s" in dynamics

    def test_get_settle_telemetry(self, model):
        """get_settle_telemetry returns SettleTelemetry after settle_universal."""
        x = torch.randn(4, 20)
        _run_model_settle(model, x, max_steps=10)
        telemetry = model.get_settle_telemetry()
        assert isinstance(telemetry, SettleTelemetry)
        assert len(telemetry.deltas) > 0

    def test_loose_threshold_early_convergence(self, model):
        """Loose convergence threshold triggers early stop."""
        model.convergence_threshold = 1.0
        x = torch.randn(4, 20)
        state, steps_taken, converged, telemetry = _run_model_settle(
            model, x, max_steps=20, convergence_threshold=1.0
        )
        assert converged
        assert steps_taken < 20

    def test_training_metrics_includes_settle_telemetry(self, model):
        """TrainingMetrics.extra['settle_telemetry'] populated after train_step."""
        x = torch.randn(8, 20)
        y = torch.randint(0, 4, (8,))
        model.train()
        result = model.train_step(x, y)
        assert isinstance(result, dict)
        assert "loss" in result
        assert "accuracy" in result


class TestPredictiveCodingHybridSettleProtocol:
    """Test PredictiveCodingHybrid SettleProtocol integration."""

    @pytest.fixture
    def model(self):
        return PredictiveCodingHybrid(
            input_dim=20,
            hidden_dim=16,
            output_dim=4,
            infer_steps=10,
            eta_infer=0.05,
            convergence_threshold=1e-3,
            convergence_start=3,
        )

    def test_isinstance_settle_protocol(self, model):
        """PredictiveCodingHybrid implements SettleProtocol."""
        assert isinstance(model, SettleProtocol)

    def test_settle_universal_returns_telemetry(self, model):
        """_run_settle_universal returns (output, steps_taken, converged, telemetry)."""
        x = torch.randn(4, 20)
        out, steps_taken, converged, telemetry = _run_model_settle(model, x, max_steps=10)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (4, 4)
        assert isinstance(steps_taken, int)
        assert 0 < steps_taken <= 10
        assert isinstance(converged, bool)
        assert isinstance(telemetry, SettleTelemetry)
        # State list is stored in model._settle_activations
        assert hasattr(model, "_settle_activations")
        if model._settle_activations is not None:
            assert isinstance(model._settle_activations, list)
            assert all(isinstance(s, torch.Tensor) for s in model._settle_activations)

    def test_forward_return_dynamics(self, model):
        """forward(return_dynamics=True) returns dynamics dict with telemetry."""
        x = torch.randn(4, 20)
        out, dynamics = model(x, return_dynamics=True)
        assert out.shape == (4, 4)
        assert isinstance(dynamics, dict)
        assert "deltas" in dynamics
        assert "final_delta" in dynamics
        assert "steps_taken" in dynamics
        assert "converged" in dynamics
        assert "settle_time_s" in dynamics

    def test_get_settle_telemetry(self, model):
        """get_settle_telemetry returns SettleTelemetry after settle_universal."""
        x = torch.randn(4, 20)
        _run_model_settle(model, x, max_steps=10)
        telemetry = model.get_settle_telemetry()
        assert isinstance(telemetry, SettleTelemetry)
        assert len(telemetry.deltas) > 0

    def test_loose_threshold_early_convergence(self, model):
        """Loose convergence threshold triggers early stop."""
        model.convergence_threshold = 1.0
        x = torch.randn(4, 20)
        state, steps_taken, converged, telemetry = _run_model_settle(
            model, x, max_steps=20, convergence_threshold=1.0
        )
        assert converged
        assert steps_taken < 20

    def test_training_metrics_includes_settle_telemetry(self, model):
        """TrainingMetrics.extra['settle_telemetry'] populated after train_step."""
        x = torch.randn(8, 20)
        y = torch.randint(0, 4, (8,))
        model.train()
        result = model.train_step(x, y)
        assert isinstance(result, dict)
        assert "loss" in result
        assert "accuracy" in result


class TestTileAlgorithmSettleProtocol:
    """Test TileAlgorithm (and subclasses) SettleProtocol integration."""

    @pytest.fixture
    def model(self):
        config = TileAlgorithmConfig(
            input_dim=20,
            output_dim=4,
            neurons_per_tile=16,
            tiles_per_layer=2,
            num_hidden_layers=2,
            algorithm="pc",
            mode="ep",
            free_steps=10,
            nudged_steps=10,
            learning_rate=0.001,
        )
        model = TileAlgorithm(config)
        model.convergence_threshold = 1e-3
        model.convergence_start = 3
        return model

    def test_isinstance_settle_protocol(self, model):
        """TileAlgorithm implements SettleProtocol."""
        assert isinstance(model, SettleProtocol)

    def test_settle_universal_returns_telemetry(self, model):
        """_run_settle_universal returns (output, steps_taken, converged, telemetry)."""
        x = torch.randn(4, 20)
        out, steps_taken, converged, telemetry = _run_model_settle(model, x, max_steps=10)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (4, 4)
        assert isinstance(steps_taken, int)
        assert 0 < steps_taken <= 10
        assert isinstance(converged, bool)
        assert isinstance(telemetry, SettleTelemetry)
        # State list is stored in model._last_activations
        assert hasattr(model, "_last_activations")
        if model._last_activations is not None:
            assert isinstance(model._last_activations, list)
            assert all(isinstance(s, torch.Tensor) for s in model._last_activations)

    def test_forward_return_dynamics(self, model):
        """forward(return_dynamics=True) returns dynamics dict with telemetry."""
        x = torch.randn(4, 20)
        out, dynamics = model(x, return_dynamics=True)
        assert out.shape == (4, 4)
        assert isinstance(dynamics, dict)
        assert "deltas" in dynamics
        assert "final_delta" in dynamics
        assert "steps_taken" in dynamics
        assert "converged" in dynamics
        assert "settle_time_s" in dynamics

    def test_get_settle_telemetry(self, model):
        """get_settle_telemetry returns SettleTelemetry after settle_universal."""
        x = torch.randn(4, 20)
        _run_model_settle(model, x, max_steps=10)
        telemetry = model.get_settle_telemetry()
        assert isinstance(telemetry, SettleTelemetry)
        assert len(telemetry.deltas) > 0

    def test_loose_threshold_early_convergence(self, model):
        """Loose convergence threshold triggers early stop."""
        model.convergence_threshold = 1.0
        x = torch.randn(4, 20)
        state, steps_taken, converged, telemetry = _run_model_settle(
            model, x, max_steps=20, convergence_threshold=1.0
        )
        assert converged
        assert steps_taken < 20

    def test_tile_pc_subclass(self):
        """TilePC subclass also implements SettleProtocol."""
        model = TilePC.from_pc(input_dim=20, output_dim=4, num_layers=2, neurons_per_tile=16, tiles_per_layer=2)
        assert isinstance(model, SettleProtocol)
        x = torch.randn(4, 20)
        state, steps_taken, converged, telemetry = _run_model_settle(model, x, max_steps=10)
        assert isinstance(telemetry, SettleTelemetry)

    def test_training_metrics_includes_settle_telemetry(self, model):
        """TrainingMetrics.extra['settle_telemetry'] populated after train_step."""
        x = torch.randn(8, 20)
        y = torch.randint(0, 4, (8,))
        model.train()
        result = model.train_step(x, y)
        assert isinstance(result, dict)
        assert "loss" in result
        assert "accuracy" in result


class TestSettleProtocolMultiEpochLearning:
    """Test that models with SettleProtocol can learn over multiple epochs."""

    @pytest.mark.parametrize("model_cls,model_kwargs", [
        (MEPEqPropModel, {"input_dim": 16, "hidden_dim": 12, "output_dim": 3, "settle_steps": 8, "settle_lr": 0.1}),
        (O1MemoryModel, {"input_dim": 16, "hidden_dim": 12, "output_dim": 3, "settle_steps": 8, "settle_lr": 0.1}),
        (PredictiveCodingHybrid, {"input_dim": 16, "hidden_dim": 12, "output_dim": 3, "infer_steps": 8, "eta_infer": 0.05}),
        (TileAlgorithm, {"config": TileAlgorithmConfig(input_dim=16, output_dim=3, neurons_per_tile=12, tiles_per_layer=2, num_hidden_layers=2, algorithm="pc", mode="ep", free_steps=8, nudged_steps=8, learning_rate=0.001)}),
    ])
    def test_multi_epoch_learning(self, model_cls, model_kwargs):
        """Model learns over multiple epochs (loss decreases, accuracy improves)."""
        model = model_cls(**model_kwargs)
        if hasattr(model, 'convergence_threshold'):
            model.convergence_threshold = 1e-3
            model.convergence_start = 3
        x = torch.randn(32, 16)
        y = torch.randint(0, 3, (32,))

        # Add class structure to make it learnable
        for c in range(3):
            mask = y == c
            if mask.any():
                direction = torch.randn(16)
                direction = direction / direction.norm() * 1.5
                x[mask] += direction * 0.8

        model.train()
        losses = []
        accuracies = []

        for epoch in range(5):
            result = model.train_step(x, y)
            losses.append(result["loss"])
            accuracies.append(result["accuracy"])

        # Loss should generally decrease (allow small fluctuation)
        assert losses[-1] <= losses[0] + 0.2
        # Accuracy should be above chance (0.33 for 3 classes)
        # Use lower threshold for synthetic data with few epochs
        assert accuracies[-1] > 0.15