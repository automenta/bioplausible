"""
Tests for EqPropKernel (NumPy/CuPy implementation).
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add parent to path for in-package testing
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from bioplausible.kernel import EqPropKernel, HAS_CUPY, spectral_normalize

class TestEqPropKernel(unittest.TestCase):
    """Test EqPropKernel functionality."""

    def setUp(self):
        self.input_dim = 10
        self.hidden_dim = 20
        self.output_dim = 5
        self.batch_size = 4

        self.kernel = EqPropKernel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            use_gpu=False # Force CPU for standard testing
        )

        self.x = np.random.randn(self.batch_size, self.input_dim).astype(np.float32)
        self.y = np.random.randint(0, self.output_dim, size=(self.batch_size,))

    def test_initialization(self):
        """Test kernel initialization."""
        self.assertIsNotNone(self.kernel)
        self.assertEqual(self.kernel.input_dim, self.input_dim)

        # Check weights exist
        self.assertTrue('W1' in self.kernel.weights)
        self.assertTrue('W2' in self.kernel.weights)

    def test_forward_step(self):
        """Test single forward step."""
        x_emb = self.kernel._compute_embedded_input(self.x)
        h = np.zeros((self.batch_size, self.hidden_dim), dtype=np.float32)

        h_next, activations = self.kernel.forward_step(h, x_emb, self.kernel.weights)

        self.assertEqual(h_next.shape, (self.batch_size, self.hidden_dim))
        self.assertTrue('ffn_hidden' in activations)

    def test_train_step(self):
        """Test full training step."""
        metrics = self.kernel.train_step(self.x, self.y)

        self.assertTrue('loss' in metrics)
        self.assertTrue('accuracy' in metrics)
        self.assertTrue('free_steps' in metrics)

        # Loss should be a number
        self.assertIsInstance(metrics['loss'], float)

    def test_spectral_normalize(self):
        """Test spectral normalization helper."""
        W = np.random.randn(10, 10).astype(np.float32)
        W_norm, u, sigma = spectral_normalize(W)

        self.assertEqual(W_norm.shape, W.shape)
        self.assertGreater(sigma, 0)

        # Run again with u
        W_norm_2, u_2, sigma_2 = spectral_normalize(W, u=u)
        self.assertEqual(W_norm_2.shape, W.shape)

    def test_predict(self):
        """Test prediction."""
        preds = self.kernel.predict(self.x)
        self.assertEqual(preds.shape, (self.batch_size,))
        self.assertTrue(np.all(preds >= 0))
        self.assertTrue(np.all(preds < self.output_dim))

if __name__ == '__main__':
    unittest.main()
