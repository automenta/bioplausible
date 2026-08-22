import sys
import unittest
from pathlib import Path

import pytest
import torch

# Add parent to path for in-package testing
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from bioplausible.acceleration.triton_kernels import HAS_TRITON, TritonEqPropOps
from bioplausible.zoo.models.eqprop._energy import EquilibriumMLP
from bioplausible.config.unified import ModelConfig

pytestmark = pytest.mark.gpu


class TestTritonIntegration(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.input_dim = 64
        self.hidden_dim = 128
        self.output_dim = 10
        config = ModelConfig(
            name="eqprop_mlp",
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dims=[self.hidden_dim],
            learning_rate=0.01,
            beta=0.5,
            max_steps=30,
            convergence_threshold=1e-4,
            convergence_start=5,
            use_spectral_norm=True,
            spectral_norm_power_iterations=5,
            activation="tanh",
            lipschitz_mode="power_iteration",
            output_scaling_mode="uniform",
            extra={
                "gradient_method": "equilibrium",
                "backend": "pytorch",
            },
        )
        self.model = EquilibriumMLP(config=config).to(self.device)

        # Reset functioning flag to ensure test isolation
        if HAS_TRITON:
            TritonEqPropOps._triton_functioning = True

    def test_triton_ops_availability(self):
        """Check if Triton ops are correctly detected."""
        if torch.cuda.is_available() and HAS_TRITON:
            self.assertTrue(TritonEqPropOps.is_available())
        else:
            self.assertFalse(TritonEqPropOps.is_available())

    def test_forward_step_execution(self):
        """
        Verify forward_step runs without error on the target device.
        This implicitly tests the Triton kernel path if available.
        """
        batch_size = 16
        x = torch.randn(batch_size, self.input_dim).to(self.device)

        # Initialize hidden state
        h = self.model._initialize_hidden_state(x)
        x_trans = self.model._transform_input(x)

        # Run one step
        h_next = self.model.forward_step(h, x_trans)

        self.assertEqual(h_next.shape, h.shape)
        self.assertEqual(h_next.device, h.device)

    def test_triton_logic_path(self):
        """
        Mock TritonEqPropOps.is_available to force/check logic path if possible,
        or just rely on the fact that if it crashes it failed.
        """
        # If we are on CUDA and have Triton, this executes the kernel
        if self.device == "cuda" and HAS_TRITON:
            batch_size = 16
            x = torch.randn(batch_size, self.input_dim).to(self.device)
            out = self.model(x)
            self.assertEqual(out.shape, (batch_size, self.output_dim))


if __name__ == "__main__":
    unittest.main()
