"""Tests for FA kernel CPU initialization bug fix (REFACTOR8).

Ensures FA kernel backend handles tuple input_dim (spatial format) correctly.
"""

from __future__ import annotations

import math

import pytest
import torch

from bioplausible.acceleration.fa_kernels import FAKernelBackend
from bioplausible.acceleration.kernel_backend import (
    AlgorithmFamily,
    HardwareTarget,
    KernelConfig,
)


class TestFAKernelCPUInitialization:
    """Test FA kernel initialization with various input_dim formats."""

    def test_initialize_with_int_input_dim(self):
        """FA kernel should work with integer input_dim (legacy format)."""
        backend = FAKernelBackend()
        config = KernelConfig(
            algorithm=AlgorithmFamily.FA,
            hardware=HardwareTarget.CPU,
            extra={
                "input_dim": 784,
                "hidden_dim": 256,
                "output_dim": 10,
                "num_layers": 3,
            },
        )
        backend.initialize(config)

        # Check feedback weights were created with correct shapes
        assert len(backend._feedback_weights) == 3
        assert backend._feedback_weights[0].shape == (256, 784)
        assert backend._feedback_weights[1].shape == (256, 256)
        assert backend._feedback_weights[2].shape == (10, 256)

    def test_initialize_with_tuple_input_dim(self):
        """FA kernel should handle tuple input_dim (spatial format like (C, H, W))."""
        backend = FAKernelBackend()
        config = KernelConfig(
            algorithm=AlgorithmFamily.FA,
            hardware=HardwareTarget.CPU,
            extra={
                "input_dim": (1, 28, 28),  # MNIST spatial format
                "hidden_dim": 128,
                "output_dim": 10,
                "num_layers": 2,
            },
        )
        # This should not raise TypeError
        backend.initialize(config)

        # Input should be flattened: 1*28*28 = 784
        expected_input = math.prod((1, 28, 28))
        assert len(backend._feedback_weights) == 2
        assert backend._feedback_weights[0].shape == (128, expected_input)
        assert backend._feedback_weights[1].shape == (10, 128)

    def test_initialize_with_3d_tuple(self):
        """FA kernel should handle 3D tuple like (C, H, W)."""
        backend = FAKernelBackend()
        config = KernelConfig(
            algorithm=AlgorithmFamily.FA,
            hardware=HardwareTarget.CPU,
            extra={
                "input_dim": (3, 32, 32),  # CIFAR-10 spatial format
                "hidden_dim": 256,
                "output_dim": 10,
                "num_layers": 3,
            },
        )
        backend.initialize(config)

        expected_input = 3 * 32 * 32
        assert len(backend._feedback_weights) == 3
        assert backend._feedback_weights[0].shape == (256, expected_input)
        assert backend._feedback_weights[1].shape == (256, 256)
        assert backend._feedback_weights[2].shape == (10, 256)

    def test_initialize_with_2d_tuple(self):
        """FA kernel should handle 2D tuple like (H, W)."""
        backend = FAKernelBackend()
        config = KernelConfig(
            algorithm=AlgorithmFamily.FA,
            hardware=HardwareTarget.CPU,
            extra={
                "input_dim": (8, 8),  # digits dataset
                "hidden_dim": 64,
                "output_dim": 10,
                "num_layers": 2,
            },
        )
        backend.initialize(config)

        expected_input = 8 * 8
        assert len(backend._feedback_weights) == 2
        assert backend._feedback_weights[0].shape == (64, expected_input)
        assert backend._feedback_weights[1].shape == (10, 64)

    def test_initialize_cuda_target(self):
        """FA kernel should initialize correctly on CUDA target."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        backend = FAKernelBackend()
        config = KernelConfig(
            algorithm=AlgorithmFamily.FA,
            hardware=HardwareTarget.CUDA,
            extra={
                "input_dim": (1, 8, 8),
                "hidden_dim": 64,
                "output_dim": 10,
                "num_layers": 2,
            },
        )
        backend.initialize(config)

        assert backend._device.type == "cuda"
        for w in backend._feedback_weights:
            assert w.device.type == "cuda"

    def test_initialize_dtype_respected(self):
        """FA kernel should respect dtype from config."""
        backend = FAKernelBackend()
        config = KernelConfig(
            algorithm=AlgorithmFamily.FA,
            hardware=HardwareTarget.CPU,
            dtype=torch.float16,
            extra={
                "input_dim": 784,
                "hidden_dim": 256,
                "output_dim": 10,
                "num_layers": 2,
            },
        )
        backend.initialize(config)

        assert backend._dtype == torch.float16
        for w in backend._feedback_weights:
            assert w.dtype == torch.float16


class TestFAKernelForwardBackward:
    """Test FA kernel forward/backward with tuple input_dim."""

    def test_forward_backward_with_tuple_input_dim(self):
        """Full forward/backward pass should work with tuple input_dim."""
        backend = FAKernelBackend()
        config = KernelConfig(
            algorithm=AlgorithmFamily.FA,
            hardware=HardwareTarget.CPU,
            extra={
                "input_dim": (1, 8, 8),
                "hidden_dim": 64,
                "output_dim": 10,
                "num_layers": 2,
            },
        )
        backend.initialize(config)

        # Create a simple model with matching layers
        layers = [
            torch.nn.Linear(64, 64),
            torch.nn.Linear(64, 10),
        ]
        backend.set_model_ref(layers)

        # Forward pass
        batch_size = 4
        x = torch.randn(batch_size, 1, 8, 8)
        output, activations = backend.forward(x)

        assert output.shape == (batch_size, 10)
        assert len(activations) == 3  # input + hidden + output

        # Backward pass
        error = torch.randn(batch_size, 10)
        grads = backend.backward(activations, error)

        assert "layers.0.weight" in grads
        assert "layers.1.weight" in grads
        assert grads["layers.0.weight"].shape == (64, 64)
        assert grads["layers.1.weight"].shape == (10, 64)
