"""
Tests for eqprop_trainer dashboard application.

These tests verify the dashboard components work before runtime.
"""

import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))


class TestDashboardModelCreation(unittest.TestCase):
    """Test model creation logic from dashboard."""

    def test_vision_model_creation(self):
        """Test creating vision models."""
        from eqprop_torch import LoopedMLP, ConvEqProp, BackpropMLP

        # Test LoopedMLP
        model = LoopedMLP(784, 256, 10)
        self.assertIsNotNone(model)

        # Test ConvEqProp
        model = ConvEqProp(1, 32, 10)
        self.assertIsNotNone(model)

        # Test BackpropMLP
        model = BackpropMLP(784, 256, 10)
        self.assertIsNotNone(model)

    def test_bioplausible_vision_models(self):
        """Test creating bioplausible models for vision."""
        from eqprop_torch import HAS_BIOPLAUSIBLE

        if not HAS_BIOPLAUSIBLE:
            self.skipTest("Bioplausible models not available")

        from algorithms import create_model

        # Test creating a research algorithm
        model = create_model('eqprop', 784, [256], 10)
        self.assertIsNotNone(model)

    def test_lm_dataset_loading(self):
        """Test LM dataset loading returns non-empty dataset."""
        from eqprop_torch.datasets import get_lm_dataset

        try:
            dataset = get_lm_dataset('tiny_shakespeare', seq_len=128, split='train')

            # Check dataset is not empty
            self.assertGreater(len(dataset), 0, "Dataset should have samples")

            # Check vocab_size exists
            self.assertTrue(hasattr(dataset, 'vocab_size'), "Dataset should have vocab_size")
            self.assertGreater(dataset.vocab_size, 0, "Vocab size should be positive")

        except Exception as e:
            self.fail(f"Dataset loading failed: {e}")

    def test_lm_model_creation(self):
        """Test creating LM models."""
        from eqprop_torch import HAS_LM_VARIANTS

        if not HAS_LM_VARIANTS:
            self.skipTest("LM variants not available")

        from eqprop_torch import get_eqprop_lm

        # Test creating an LM variant
        try:
            model = get_eqprop_lm(
                'full',
                vocab_size=256,
                hidden_dim=64,
                num_layers=2,
                max_seq_len=128,
                eq_steps=10
            )
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"LM model creation failed: {e}")


class TestDashboardDataLoading(unittest.TestCase):
    """Test data loading works correctly."""

    def test_vision_datasets(self):
        """Test loading vision datasets."""
        from eqprop_torch.datasets import get_vision_dataset
        from torch.utils.data import DataLoader

        for dataset_name in ['mnist']:  # Test at least one
            with self.subTest(dataset=dataset_name):
                try:
                    dataset = get_vision_dataset(dataset_name, train=True, flatten=True)
                    self.assertGreater(len(dataset), 0, f"{dataset_name} should have samples")

                    # Test DataLoader can be created
                    loader = DataLoader(dataset, batch_size=32, shuffle=True)
                    self.assertIsNotNone(loader)

                except Exception as e:
                    # Some datasets might not be downloaded
                    if "not found" not in str(e).lower():
                        self.fail(f"{dataset_name} failed: {e}")


class TestDashboardImports(unittest.TestCase):
    """Test all dashboard imports work."""

    def test_dashboard_imports(self):
        """Test importing dashboard module."""
        try:
            from eqprop_trainer import EqPropDashboard, main
            self.assertTrue(hasattr(EqPropDashboard, '__init__'))
        except ImportError as e:
            self.fail(f"Dashboard import failed: {e}")

    def test_worker_imports(self):
        """Test importing worker module."""
        try:
            from eqprop_trainer.worker import TrainingWorker
            self.assertTrue(hasattr(TrainingWorker, '__init__'))
        except ImportError as e:
            self.fail(f"Worker import failed: {e}")

    def test_generation_imports(self):
        """Test importing generation module."""
        try:
            from eqprop_trainer.generation import UniversalGenerator, SimpleCharTokenizer
            self.assertTrue(hasattr(UniversalGenerator, '__init__'))
            self.assertTrue(hasattr(SimpleCharTokenizer, '__init__'))
        except ImportError as e:
            self.fail(f"Generation import failed: {e}")

    def test_hyperparams_imports(self):
        """Test importing hyperparams module."""
        try:
            from eqprop_trainer.hyperparams import get_hyperparams_for_model, HyperparamSpec
            self.assertTrue(callable(get_hyperparams_for_model))
            self.assertTrue(hasattr(HyperparamSpec, '__init__'))
        except ImportError as e:
            self.fail(f"Hyperparams import failed: {e}")

    def test_viz_utils_imports(self):
        """Test importing viz_utils module."""
        try:
            from eqprop_trainer.viz_utils import extract_weights, format_weight_for_display
            self.assertTrue(callable(extract_weights))
            self.assertTrue(callable(format_weight_for_display))
        except ImportError as e:
            self.fail(f"Viz utils import failed: {e}")

    def test_dashboard_helpers_imports(self):
        """Test importing dashboard_helpers module."""
        try:
            from eqprop_trainer.dashboard_helpers import update_hyperparams_generic, generate_text_universal
            self.assertTrue(callable(update_hyperparams_generic))
            self.assertTrue(callable(generate_text_universal))
        except ImportError as e:
            self.fail(f"Dashboard helpers import failed: {e}")


if __name__ == '__main__':
    # Run tests
    unittest.main()
