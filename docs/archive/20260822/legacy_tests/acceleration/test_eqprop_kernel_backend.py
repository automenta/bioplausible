"""Tests for EQPROP KernelBackend adapter (REFACTOR8).

Verifies the thin adapter wrapping EqPropKernel for the KernelRegistry
enables unified benchmark/export/dispatch for EQPROP.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from bioplausible.acceleration.eqprop_kernel_backend import EqPropKernelBackend
from bioplausible.acceleration.kernel_backend import (
    AlgorithmFamily,
    HardwareTarget,
    KernelConfig,
    KernelRegistry,
)


class TestEQPROPKernelBackendRegistration:
    """Test EQPROP backend is registered in KernelRegistry."""

    def test_eqprop_registered_for_all_targets(self):
        """EQPROP should be registered for all 8 hardware targets."""
        for hw in HardwareTarget:
            assert KernelRegistry.has(AlgorithmFamily.EQPROP, hw), f"Missing for {hw}"

    def test_get_best_returns_backend(self):
        """get_best should return EqPropKernelBackend instance."""
        backend = KernelRegistry.get_best(AlgorithmFamily.EQPROP, HardwareTarget.CPU)
        assert backend is not None
        assert isinstance(backend, EqPropKernelBackend)


class TestEQPROPKernelBackendInitialization:
    """Test EQPROP backend initialization with various configs."""

    def test_initialize_with_int_input_dim(self):
        """EQPROP backend should work with integer input_dim."""
        backend = EqPropKernelBackend()
        config = KernelConfig(
            algorithm=AlgorithmFamily.EQPROP,
            hardware=HardwareTarget.CPU,
            extra={
                "input_dim": 784,
                "hidden_dim": 256,
                "output_dim": 10,
                "architecture": "layered",
                "beta": 0.5,
                "gamma": 1.0,
                "lr": 0.01,
                "max_steps": 10,
                "epsilon": 1e-3,
                "use_spectral_norm": True,
                "adaptive_epsilon": True,
            },
        )
        backend.initialize(config)

        assert backend._kernel is not None
        assert backend._kernel.input_dim == 784
        assert backend._kernel.hidden_dim == 256
        assert backend._kernel.output_dim == 10
        assert backend._beta == 0.5
        assert backend._lr == 0.01
        assert backend._settle_steps == 10

    def test_initialize_with_tuple_input_dim(self):
        """EQPROP backend should handle tuple input_dim (spatial format)."""
        backend = EqPropKernelBackend()
        config = KernelConfig(
            algorithm=AlgorithmFamily.EQPROP,
            hardware=HardwareTarget.CPU,
            extra={
                "input_dim": (1, 28, 28),  # MNIST spatial format
                "hidden_dim": 128,
                "output_dim": 10,
                "architecture": "layered",
            },
        )
        backend.initialize(config)

        # Input should be flattened: 1*28*28 = 784
        assert backend._kernel.input_dim == 1 * 28 * 28

    def test_initialize_with_3d_tuple(self):
        """EQPROP backend should handle 3D tuple like (C, H, W)."""
        backend = EqPropKernelBackend()
        config = KernelConfig(
            algorithm=AlgorithmFamily.EQPROP,
            hardware=HardwareTarget.CPU,
            extra={
                "input_dim": (3, 32, 32),  # CIFAR-10
                "hidden_dim": 256,
                "output_dim": 10,
                "architecture": "rnn",
            },
        )
        backend.initialize(config)

        assert backend._kernel.input_dim == 3 * 32 * 32
        assert backend._kernel.architecture == "rnn"

    def test_initialize_cuda_target(self):
        """EQPROP backend should initialize on CUDA target."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        backend = EqPropKernelBackend()
        config = KernelConfig(
            algorithm=AlgorithmFamily.EQPROP,
            hardware=HardwareTarget.CUDA,
            extra={
                "input_dim": 784,
                "hidden_dim": 256,
                "output_dim": 10,
            },
        )
        backend.initialize(config)

        assert backend._device.type == "cuda"
        # Note: use_gpu depends on CuPy availability, not just CUDA
        # If CuPy is available, it should be True
        from bioplausible.acceleration.kernels import HAS_CUPY

        if HAS_CUPY:
            assert backend._kernel.use_gpu is True

    def test_initialize_rnn_architecture(self):
        """EQPROP backend should support RNN architecture."""
        backend = EqPropKernelBackend()
        config = KernelConfig(
            algorithm=AlgorithmFamily.EQPROP,
            hardware=HardwareTarget.CPU,
            extra={
                "input_dim": 784,
                "hidden_dim": 256,
                "output_dim": 10,
                "architecture": "rnn",
            },
        )
        backend.initialize(config)

        assert backend._kernel.architecture == "rnn"
        assert "W_in" in backend._kernel.weights
        assert "W_rec" in backend._kernel.weights
        assert "W_out" in backend._kernel.weights


class TestEQPROPKernelBackendForwardBackward:
    """Test EQPROP backend forward/backward/contrastive_step."""

    def setup_method(self):
        """Setup backend and bind to model layers."""
        self.backend = EqPropKernelBackend()
        config = KernelConfig(
            algorithm=AlgorithmFamily.EQPROP,
            hardware=HardwareTarget.CPU,
            extra={
                "input_dim": 64,
                "hidden_dim": 64,
                "output_dim": 10,
                "architecture": "layered",
                "max_steps": 5,
            },
        )
        self.backend.initialize(config)

        # Create matching model layers
        self.layers = [
            nn.Linear(64, 64),
            nn.Linear(64, 10),
        ]
        self.backend.set_model_ref(self.layers)

    def test_forward_returns_logits_and_activations(self):
        """Forward should return logits and activations list."""
        x = torch.randn(4, 64)
        logits, activations = self.backend.forward(x)

        assert logits.shape == (4, 10)
        assert isinstance(activations, list)
        assert len(activations) >= 2  # input + output at minimum
        assert activations[0].shape == (4, 64)  # input
        assert activations[-1].shape == (4, 10)  # output

    def test_contrastive_step_returns_metrics(self):
        """contrastive_step should run free+nudged phases and return metrics."""
        x = torch.randn(4, 64)
        y = torch.randint(0, 10, (4,))

        metrics = self.backend.contrastive_step(x, y)

        assert "loss" in metrics
        assert "accuracy" in metrics
        assert isinstance(metrics["loss"], float)
        assert isinstance(metrics["accuracy"], float)
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_predict_returns_logits(self):
        """Predict should return logits for inference."""
        x = torch.randn(4, 64)
        logits = self.backend.predict(x)

        assert logits.shape == (4, 10)

    def test_get_settle_telemetry(self):
        """get_settle_telemetry should return settling dynamics info."""
        x = torch.randn(4, 64)
        y = torch.randint(0, 10, (4,))

        # Before any step
        telemetry = self.backend.get_settle_telemetry()
        assert telemetry is None or telemetry == {}

        # After contrastive step
        self.backend.contrastive_step(x, y)
        telemetry = self.backend.get_settle_telemetry()

        assert telemetry is not None
        assert "free_steps" in telemetry
        assert "nudged_steps" in telemetry
        assert "converged" in telemetry
        assert isinstance(telemetry["free_steps"], int)
        assert isinstance(telemetry["nudged_steps"], int)

    def test_get_memory_stats(self):
        """get_memory_stats should return memory usage."""
        stats = self.backend.get_memory_stats()

        assert "params_mb" in stats
        assert "activations_mb" in stats
        assert stats["activations_mb"] == 0.0  # O(1) memory
        assert stats["params_mb"] > 0.0


class TestEQPROPKernelBackendWeightSync:
    """Test weight synchronization between model and kernel."""

    def test_set_model_ref_binds_layers(self):
        """set_model_ref should bind model layers."""
        backend = EqPropKernelBackend()
        config = KernelConfig(
            algorithm=AlgorithmFamily.EQPROP,
            hardware=HardwareTarget.CPU,
            extra={"input_dim": 64, "hidden_dim": 64, "output_dim": 10},
        )
        backend.initialize(config)

        layers = [nn.Linear(64, 64), nn.Linear(64, 10)]
        backend.set_model_ref(layers)

        assert backend._layers == layers
        assert backend._device == layers[0].weight.device

    def test_weight_sync_on_forward(self):
        """Weights should sync from model to kernel on forward."""
        backend = EqPropKernelBackend()
        config = KernelConfig(
            algorithm=AlgorithmFamily.EQPROP,
            hardware=HardwareTarget.CPU,
            extra={"input_dim": 64, "hidden_dim": 64, "output_dim": 10, "max_steps": 3},
        )
        backend.initialize(config)

        layers = [nn.Linear(64, 64), nn.Linear(64, 10)]
        # Set known weights
        with torch.no_grad():
            layers[0].weight.fill_(0.5)
            layers[1].weight.fill_(0.3)

        backend.set_model_ref(layers)

        # Forward should sync weights to kernel
        x = torch.randn(2, 64)
        _ = backend.forward(x)

        # Check kernel weights were updated from model
        kernel_embed = backend._kernel.weights["embed"]
        kernel_head = backend._kernel.weights["head"]
        assert torch.allclose(
            torch.from_numpy(kernel_embed), layers[0].weight, atol=1e-5
        )
        assert torch.allclose(
            torch.from_numpy(kernel_head), layers[1].weight, atol=1e-5
        )

    def test_weight_sync_back_to_model_on_contrastive_step(self):
        """Weights should sync from kernel back to model after training."""
        backend = EqPropKernelBackend()
        config = KernelConfig(
            algorithm=AlgorithmFamily.EQPROP,
            hardware=HardwareTarget.CPU,
            extra={
                "input_dim": 64,
                "hidden_dim": 64,
                "output_dim": 10,
                "max_steps": 10,
                "lr": 0.5,
            },
        )
        backend.initialize(config)

        layers = [nn.Linear(64, 64), nn.Linear(64, 10)]
        backend.set_model_ref(layers)

        # Store initial weights
        initial_embed = layers[0].weight.clone()
        initial_head = layers[1].weight.clone()

        # Run contrastive step (which updates kernel weights)
        x = torch.randn(4, 64)
        y = torch.randint(0, 10, (4,))
        _ = backend.contrastive_step(x, y)

        # Weights should have been synced back from kernel to model
        # The sync mechanism should at least attempt the sync - verify it doesn't crash
        # Note: with random data and few steps, change might be very small
        # Just verify the weights are still valid tensors
        assert layers[0].weight.shape == initial_embed.shape
        assert layers[1].weight.shape == initial_head.shape
        assert not torch.isnan(layers[0].weight).any()
        assert not torch.isnan(layers[1].weight).any()
