import unittest
import torch
import torch.nn as nn
from bioplausible.models import LoopedMLP, ConvEqProp, TransformerEqProp
from bioplausible.core import EqPropTrainer
from bioplausible.algorithms import ALGORITHM_REGISTRY, create_model

class TestRefactor(unittest.TestCase):
    def test_imports_and_models(self):
        """Test that core models can be instantiated."""
        mlp = LoopedMLP(10, 20, 5)
        self.assertIsInstance(mlp, nn.Module)

        conv = ConvEqProp(1, 16, 5)
        self.assertIsInstance(conv, nn.Module)

        # Transformer requires more arguments or defaults are fine?
        # Let's check init: vocab_size, hidden_dim, output_dim are required
        trans = TransformerEqProp(100, 32, 5)
        self.assertIsInstance(trans, nn.Module)

    def test_trainer_init(self):
        """Test that trainer can be initialized."""
        mlp = LoopedMLP(10, 20, 5)
        trainer = EqPropTrainer(mlp, use_compile=False) # Disable compile to avoid overhead/issues in test env
        self.assertIsInstance(trainer, EqPropTrainer)

    def test_algorithms_registry(self):
        """Test that algorithms are available."""
        self.assertTrue(len(ALGORITHM_REGISTRY) > 0)
        self.assertIn('eqprop', ALGORITHM_REGISTRY)

        model = create_model('eqprop', 10, [20], 5)
        self.assertIsInstance(model, nn.Module)

if __name__ == '__main__':
    unittest.main()
