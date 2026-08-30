import unittest

import torch

from computronium.core.registry import ComponentCategory, Registry
from computronium.models.native.diffusion_eqprop_native import create_native_diffusion_eqprop


class TestDiffusionIntegration(unittest.TestCase):
    def test_factory_creation(self):
        """Test that the factory can create the diffusion model."""
        model = create_native_diffusion_eqprop(
            input_dim=10, hidden_dim=32, output_dim=10, num_layers=1
        )
        self.assertIsNotNone(model)
        # Check it has the expected components
        self.assertIsNotNone(model.substrate)
        self.assertIsNotNone(model.geometry)
        self.assertIsNotNone(model.dynamics)
        self.assertIsNotNone(model.credit)
        self.assertIsNotNone(model.update)

    def test_train_step(self):
        """Test a single training step."""
        model = create_native_diffusion_eqprop(
            input_dim=10, hidden_dim=8, output_dim=10, num_layers=1, diffusion_coeff=1.0
        )
        # Input: [B, input_dim]
        x = torch.randn(2, 10)
        y = torch.randint(0, 10, (2,))

        metrics = model.train_step(x, y)
        self.assertIn("loss", metrics)
        # Loss should be float or tensor
        self.assertTrue(
            isinstance(metrics["loss"], float)
            or isinstance(metrics["loss"], torch.Tensor)
        )

    def test_forward(self):
        """Test forward pass."""
        model = create_native_diffusion_eqprop(
            input_dim=10, hidden_dim=8, output_dim=10, num_layers=1
        )
        model.eval()  # type: ignore[attr-defined]

        x = torch.randn(2, 10)
        with torch.no_grad():
            out = model(x)  # type: ignore[operator]

        self.assertEqual(out.shape, (2, 10))

    def test_registry_lookup(self):
        """Test that native_diffusion_eqprop is registered."""
        from computronium.models.native.diffusion_eqprop_native import native_diffusion_eqprop
        self.assertIsNotNone(native_diffusion_eqprop)
        model = native_diffusion_eqprop(input_dim=10, hidden_dim=32, output_dim=10)
        self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()