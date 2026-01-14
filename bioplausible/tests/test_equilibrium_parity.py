import torch
import torch.nn as nn
import unittest
from bioplausible.models.looped_mlp import LoopedMLP

class TestEquilibriumParity(unittest.TestCase):
    def test_gradient_parity(self):
        print("\nTesting Gradient Parity between BPTT and Equilibrium Mode...")
        input_dim = 10
        hidden_dim = 20
        output_dim = 5
        batch_size = 4
        max_steps = 100 # Higher steps for better convergence

        # Fixed seed
        torch.manual_seed(42)

        # Data
        x = torch.randn(batch_size, input_dim)
        y = torch.randint(0, output_dim, (batch_size,))

        # Create two identical models
        # Disable spectral norm to strictly test the gradient logic without parametrization noise
        # (Though it should work with SN too)
        model_bptt = LoopedMLP(input_dim, hidden_dim, output_dim, max_steps=max_steps, gradient_method="bptt", use_spectral_norm=False)
        model_eq = LoopedMLP(input_dim, hidden_dim, output_dim, max_steps=max_steps, gradient_method="equilibrium", use_spectral_norm=False)

        # Copy weights
        model_eq.load_state_dict(model_bptt.state_dict())

        # Criterion
        criterion = nn.CrossEntropyLoss()

        # BPTT Pass
        model_bptt.zero_grad()
        out_bptt = model_bptt(x)
        loss_bptt = criterion(out_bptt, y)
        loss_bptt.backward()

        # Equilibrium Pass
        model_eq.zero_grad()
        out_eq = model_eq(x)
        loss_eq = criterion(out_eq, y)
        loss_eq.backward()

        # Compare losses
        print(f"Loss BPTT: {loss_bptt.item()}")
        print(f"Loss EqProp: {loss_eq.item()}")
        self.assertAlmostEqual(loss_bptt.item(), loss_eq.item(), places=5)

        # Compare gradients
        for (n1, p1), (n2, p2) in zip(model_bptt.named_parameters(), model_eq.named_parameters()):
            self.assertEqual(n1, n2)
            if p1.grad is not None and p2.grad is not None:
                # Relative error
                diff = (p1.grad - p2.grad).norm().item()
                scale = p1.grad.norm().item() + p2.grad.norm().item()
                if scale > 1e-9:
                    rel_err = diff / scale
                    print(f"Param {n1}: rel_err={rel_err:.6f}")
                    # Allow up to 10% relative error due to finite step approximation
                    self.assertLess(rel_err, 0.1, f"Gradient mismatch for {n1}")
            else:
                self.assertIsNone(p1.grad)
                self.assertIsNone(p2.grad)

if __name__ == '__main__':
    unittest.main()
