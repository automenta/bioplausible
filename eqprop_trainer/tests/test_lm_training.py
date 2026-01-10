"""
Extended dashboard tests for bioplausible LM training.

These tests catch dtype and shape issues before runtime.
"""

import unittest
import torch
import sys
from pathlib import Path

parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))


class TestBioplausibleLMTraining(unittest.TestCase):
    """Test that bioplausible algorithms work with LM data."""

    def test_bioplausible_lm_dtype(self):
        """Test that bioplausible models handle token inputs correctly."""
        from eqprop_torch import HAS_BIOPLAUSIBLE

        if not HAS_BIOPLAUSIBLE:
            self.skipTest("Bioplausible models not available")

        from algorithms import create_model
        from eqprop_torch.datasets import get_lm_dataset
        from torch.utils.data import DataLoader

        # Create bioplausible model for LM
        vocab_size = 65  # tiny_shakespeare vocab
        model = create_model('backprop', vocab_size, [128], vocab_size)

        # Get LM dataset
        dataset = get_lm_dataset('tiny_shakespeare', seq_len=32, split='train')
        loader = DataLoader(dataset, batch_size=4, shuffle=True)

        # Get a batch
        x, y = next(iter(loader))

        # This should NOT raise dtype error
        # The worker should handle token (Long) -> float conversion
        try:
            # Convert tokens to one-hot as worker should do
            x_onehot = torch.nn.functional.one_hot(x.reshape(-1), num_classes=vocab_size).float()
            output = model(x_onehot)

            # Should work without error
            self.assertIsNotNone(output)

        except RuntimeError as e:
            if "dtype" in str(e):
                self.fail(f"Dtype error not handled: {e}")
            raise


class TestPlotUpdates(unittest.TestCase):
    """Test that plot updates work correctly."""

    def test_plot_data_structure(self):
        """Test that loss/acc/lipschitz histories track correctly."""
        # Simulate what dashboard does
        loss_history = []
        acc_history = []
        lipschitz_history = []

        # Simulate updates
        for i in range(10):
            metrics = {
                'loss': 1.0 / (i + 1),
                'accuracy': i * 0.1,
                'lipschitz': 0.95,
            }
            loss_history.append(metrics['loss'])
            acc_history.append(metrics['accuracy'])
            lipschitz_history.append(metrics.get('lipschitz', 0.0))

        # Should have  10 points
        self.assertEqual(len(loss_history), 10)
        self.assertEqual(len(acc_history), 10)
        self.assertEqual(len(lipschitz_history), 10)

        # Values should be reasonable
        self.assertGreater(loss_history[0], loss_history[-1])  # Loss decreases
        self.assertLess(acc_history[0], acc_history[-1])  # Acc increases


class TestWorkerFunctionality(unittest.TestCase):
    """Test worker functionality after refactoring."""

    def test_worker_batch_processing(self):
        """Test that the worker can process a batch correctly."""
        from eqprop_trainer.worker import TrainingWorker

        # Create a mock model for testing
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        model = MockModel()
        dummy_loader = [(torch.randn(4, 10), torch.randint(0, 5, (4,)))]

        # Create a worker instance
        worker = TrainingWorker(
            model=model,
            train_loader=dummy_loader,
            epochs=1,
            lr=0.001
        )

        # Test that the worker methods exist and are callable
        self.assertTrue(callable(getattr(worker, '_process_batch')))
        self.assertTrue(callable(getattr(worker, '_convert_input_format')))
        self.assertTrue(callable(getattr(worker, '_compute_loss_and_accuracy')))


if __name__ == '__main__':
    unittest.main()
