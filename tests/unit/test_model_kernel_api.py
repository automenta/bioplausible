import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from bioplausible.zoo.models.eqprop import (
    LoopedMLP,
)


class TestModelKernelAPI(unittest.TestCase):
    """Tests for the consolidated layered LoopedMLP API.

    The single-hidden CuPy ``EqPropKernel`` engine was removed when the six
    fundamental eqprop models (and ``LoopedMLP == eqprop_mlp``) were unified
    onto the layered ``EquilibriumMLP`` engine. ``backend`` is recorded for
    trainer compatibility but always runs the PyTorch layered path; ``W_in``
    is now ``layers[0]`` like every other eqprop model.
    """

    def setUp(self):
        self.input_dim = 10
        self.hidden_dim = 32
        self.output_dim = 2
        self.batch_size = 4

        self.x_np = np.random.randn(20, self.input_dim).astype(np.float32)
        self.y_np = np.random.randint(0, self.output_dim, (20,)).astype(np.int64)

        self.x_torch = torch.from_numpy(self.x_np)
        self.y_torch = torch.from_numpy(self.y_np)
        self.dataset = TensorDataset(self.x_torch, self.y_torch)
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def test_looped_mlp_kernel_backend_init(self):
        """backend='kernel' records the flag but routes to the layered engine."""
        model = LoopedMLP(
            self.input_dim, self.hidden_dim, self.output_dim, backend="kernel"
        )
        self.assertEqual(model.backend, "kernel")
        # The engine is the layered EquilibriumMLP: a layer stack, no numpy kernel.
        self.assertEqual(len(model.layers), 2)  # input→h, h→output
        self.assertFalse(hasattr(model, "_engine"))

    def test_looped_mlp_forward_kernel(self):
        """Forward pass on the backend='kernel' facade returns correct shape."""
        model = LoopedMLP(
            self.input_dim, self.hidden_dim, self.output_dim, backend="kernel"
        )
        x = self.x_torch[:2]
        out = model(x)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (2, self.output_dim))

    def test_model_kernel_forward_no_grad(self):
        """Forward without backward leaves parameter gradients unset."""
        model = LoopedMLP(
            self.input_dim, self.hidden_dim, self.output_dim, backend="kernel"
        )
        out = model(self.x_torch[:2])
        for param in model.parameters():
            self.assertIsNone(param.grad)
        self.assertIsInstance(out, torch.Tensor)

    def test_regression_pytorch_backend(self):
        """The standard PyTorch backend updates weights through the layer stack."""
        model = LoopedMLP(
            self.input_dim, self.hidden_dim, self.output_dim, backend="pytorch"
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = torch.nn.CrossEntropyLoss()

        x, y = next(iter(self.loader))
        # ``layers[0]`` is the input projection (was ``W_in`` pre-consolidation).
        w_before = model.layers[0].parametrizations.weight.original.clone()

        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

        w_after = model.layers[0].parametrizations.weight.original
        self.assertFalse(
            torch.allclose(w_before, w_after),
            "Weights did not update in PyTorch mode",
        )

    def test_num_layers_builds_multi_hidden_stack(self):
        """num_layers > 1 must build a deeper layer stack (no phantom depth)."""
        model = LoopedMLP(
            self.input_dim, self.hidden_dim, self.output_dim, num_layers=3
        )
        # input→h0→h1→h2→output
        self.assertEqual(len(model.layers), 4)
        self.assertEqual(len(model.W_rec), 3)


if __name__ == "__main__":
    unittest.main()
