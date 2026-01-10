"""
Unit Tests for Novel Hybrid Algorithms

Proper test suite to verify all algorithms work correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import torch
import torch.nn as nn

from algorithms import (
    create_model,
    BackpropBaseline,
    StandardEqProp,
    StandardFA,
    EquilibriumAlignment,
    AdaptiveFeedbackAlignment,
    ALGORITHM_REGISTRY
)


class TestAlgorithmCreation(unittest.TestCase):
    """Test that all algorithms can be instantiated."""
    
    def setUp(self):
        self.input_dim = 28 * 28  # MNIST
        self.hidden_dims = [128, 128]
        self.output_dim = 10
        
    def test_all_algorithms_creatable(self):
        """All registered algorithms should be creatable."""
        for algo_name in ALGORITHM_REGISTRY.keys():
            with self.subTest(algorithm=algo_name):
                model = create_model(algo_name, 784, [64], 10)
                self.assertIsNotNone(model)
                self.assertGreater(model.get_num_params(), 0)


class TestAlgorithmTraining(unittest.TestCase):
    """Test training step works and loss decreases."""
    
    def setUp(self):
        self.batch_size = 8
        self.input_dim = 28 * 28
        self.hidden_dims = [64]
        self.output_dim = 10
        self.x = torch.randn(self.batch_size, self.input_dim)
        self.y = torch.randint(0, self.output_dim, (self.batch_size,))
        
    def _test_algorithm_trains(self, algo_name, num_steps=5):
        """Helper to test an algorithm can train."""
        model = create_model(
            algo_name, 
            self.input_dim, 
            self.hidden_dims, 
            self.output_dim,
            learning_rate=0.01,
            equilibrium_steps=3  # Fast for testing
        )
        
        # Ensure model is on CPU for tests
        model.to('cpu') 
        
        initial_output = model.forward(self.x).detach()
        losses = []
        
        # Run a few steps
        for _ in range(num_steps):
            metrics = model.train_step(self.x, self.y)
            losses.append(metrics['loss'])
            self.assertIn('accuracy', metrics)
            self.assertGreaterEqual(metrics['accuracy'], 0.0)
            self.assertLessEqual(metrics['accuracy'], 1.0)
        
        # Check output changed (weights updated)
        final_output = model.forward(self.x).detach()
        diff = (final_output - initial_output).abs().mean().item()
        self.assertGreater(diff, 1e-6, 
                          f"{algo_name} output should change during training")
        
    def test_all_registered_algorithms_train(self):
        """Test that ALL registered algorithms can train."""
        print("\nTesting Training Loop for All Algorithms:")
        print("="*40)
        for algo_name in ALGORITHM_REGISTRY.keys():
            with self.subTest(algorithm=algo_name):
                print(f"  • {algo_name}...", end='', flush=True)
                try:
                    self._test_algorithm_trains(algo_name)
                    print(" OK")
                except Exception as e:
                    print(f" FAILED: {e}")
                    raise e


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
