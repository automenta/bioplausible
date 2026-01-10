"""
Unit tests for bio-plausible research algorithms.

Tests that the 13 algorithms from algorithms/ are properly integrated
as first-class PyTorch models (same status as LoopedMLP).
"""

import unittest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent to path for in-package testing
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from eqprop_torch import HAS_BIOPLAUSIBLE

# Skip all tests if algorithms not available  
if not HAS_BIOPLAUSIBLE:
    raise unittest.SkipTest("Research algorithms not available")

from eqprop_torch import (
    BaseAlgorithm,
    StandardEqProp,
    StandardFA,
    AdaptiveFeedbackAlignment,
    ALGORITHM_REGISTRY,
)
from algorithms import AlgorithmConfig


class TestAlgorithmsAsFirstClassModels(unittest.TestCase):
    """Test that algorithms are first-class PyTorch models."""

    def test_algorithms_are_nn_modules(self):
        """Verify algorithms inherit from nn.Module."""
        config = AlgorithmConfig('eqprop', 50, [32], 5)
        model = StandardEqProp(config)
        
        self.assertIsInstance(model, nn.Module)
        self.assertIsInstance(model, BaseAlgorithm)

    def test_forward_pass(self):
        """Test standard forward() works."""
        config = AlgorithmConfig('eqprop', 50, [32], 5)
        model = StandardEqProp(config)
        
        x = torch.randn(4, 50)
        y = model(x)
        self.assertEqual(y.shape, (4, 5))

    def test_custom_train_step(self):
        """Test custom train_step() method."""
        config = AlgorithmConfig('eqprop', 50, [32], 5)
        model = StandardEqProp(config)
        
        x = torch.randn(4, 50)
        y = torch.randint(0, 5, (4,))
        
        metrics = model.train_step(x, y)
        self.assertIn('loss', metrics)
        self.assertIn('accuracy', metrics)


class TestAllAlgorithms(unittest.TestCase):
    """Test that all algorithms can be instantiated and used."""

    def setUp(self):
        self.input_dim = 50
        self.hidden_dims = [32]
        self.output_dim = 5

    def test_all_algorithms_instantiate(self):
        """Ensure all algorithms can be created."""
        from algorithms import create_model
        
        for algo_name in ALGORITHM_REGISTRY.keys():
            with self.subTest(algorithm=algo_name):
                try:
                    model = create_model(
                        algo_name,
                        self.input_dim,
                        self.hidden_dims,
                        self.output_dim
                    )
                    self.assertIsInstance(model, nn.Module)
                    self.assertIsInstance(model, BaseAlgorithm)
                except Exception as e:
                    self.fail(f"Failed to create {algo_name}: {e}")

    def test_all_algorithms_forward(self):
        """Ensure all algorithms can perform forward pass."""
        from algorithms import create_model
        
        x = torch.randn(4, self.input_dim)
        
        for algo_name in ALGORITHM_REGISTRY.keys():
            with self.subTest(algorithm=algo_name):
                try:
                    model = create_model(
                        algo_name,
                        self.input_dim,
                        self.hidden_dims,
                        self.output_dim
                    )
                    y = model(x)
                    self.assertEqual(y.shape, (4, self.output_dim))
                except Exception as e:
                    self.fail(f"Forward failed for {algo_name}: {e}")


class TestPyTorchCompatibility(unittest.TestCase):
    """Test PyTorch compatibility features."""

    def test_parameters_accessible(self):
        """Test that parameters() works."""
        config = AlgorithmConfig('eqprop', 50, [32], 5)
        model = StandardEqProp(config)
        
        params = list(model.parameters())
        self.assertGreater(len(params), 0)

    def test_state_dict(self):
        """Test saving/loading state dict."""
        config = AlgorithmConfig('eqprop', 50, [32], 5)
        model = StandardEqProp(config)
        
        state = model.state_dict()
        self.assertIsInstance(state, dict)
        self.assertGreater(len(state), 0)
        
        # Load back
        model.load_state_dict(state)

    def test_train_eval_modes(self):
        """Test train() and eval() modes."""
        config = AlgorithmConfig('eqprop', 50, [32], 5)
        model = StandardEqProp(config)
        
        model.train()
        self.assertTrue(model.training)
        
        model.eval()
        self.assertFalse(model.training)


if __name__ == '__main__':
    unittest.main()
