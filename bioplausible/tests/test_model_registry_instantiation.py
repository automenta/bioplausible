import torch
import unittest
import numpy as np
from bioplausible.models.registry import get_model_spec
from bioplausible.hyperopt.experiment import ExperimentAlgorithm
from bioplausible.config import GLOBAL_CONFIG

class TestModelRegistryInstantiation(unittest.TestCase):
    def setUp(self):
        GLOBAL_CONFIG.quick_mode = True

    def test_holomorphic_ep_instantiation(self):
        print("\nTesting Holomorphic EP...")
        spec = get_model_spec("Holomorphic EqProp")
        algo = ExperimentAlgorithm(
            spec=spec,
            output_dim=10,
            input_dim=32,
            hidden_dim=16,
            num_layers=2,
            device="cpu",
            task_type="vision"
        )
        x = torch.randn(4, 32)
        y = torch.randint(0, 10, (4,))
        state = algo.train_step(x, y, 0)
        self.assertIsNotNone(state.loss)

    def test_directed_ep_instantiation(self):
        print("\nTesting Directed EP...")
        spec = get_model_spec("Directed EqProp (Deep EP)")
        algo = ExperimentAlgorithm(
            spec=spec,
            output_dim=10,
            input_dim=32,
            hidden_dim=16,
            num_layers=2,
            device="cpu",
            task_type="vision"
        )
        x = torch.randn(4, 32)
        y = torch.randint(0, 10, (4,))
        state = algo.train_step(x, y, 0)
        self.assertIsNotNone(state.loss)

    def test_finite_nudge_ep_instantiation(self):
        print("\nTesting Finite-Nudge EP...")
        spec = get_model_spec("Finite-Nudge EqProp")
        algo = ExperimentAlgorithm(
            spec=spec,
            output_dim=10,
            input_dim=32,
            hidden_dim=16,
            num_layers=2,
            device="cpu",
            task_type="vision"
        )
        x = torch.randn(4, 32)
        y = torch.randint(0, 10, (4,))
        state = algo.train_step(x, y, 0)
        self.assertIsNotNone(state.loss)

    def test_modern_conv_eqprop_instantiation(self):
        print("\nTesting Conv EqProp (CIFAR-10)...")
        spec = get_model_spec("Conv EqProp (CIFAR-10)")
        algo = ExperimentAlgorithm(
            spec=spec,
            output_dim=10,
            input_dim=None,
            hidden_dim=64,
            num_layers=2,
            device="cpu",
            task_type="vision"
        )
        x = torch.randn(4, 3, 32, 32)
        y = torch.randint(0, 10, (4,))
        state = algo.train_step(x, y, 0)
        self.assertIsNotNone(state.loss)

    def test_hybrid_models(self):
        models_to_test = [
            "Adaptive Feedback Alignment",
            "Equilibrium Alignment",
            "Layerwise Equilibrium FA",
            "Energy Guided FA",
            "Predictive Coding Hybrid",
            "Sparse Equilibrium",
            "Momentum Equilibrium",
            "Stochastic FA",
            "Energy Minimizing FA"
        ]

        for model_name in models_to_test:
            print(f"\nTesting {model_name}...")
            spec = get_model_spec(model_name)
            algo = ExperimentAlgorithm(
                spec=spec,
                output_dim=10,
                input_dim=32,
                hidden_dim=16,
                num_layers=2,
                device="cpu",
                task_type="vision"
            )
            x = torch.randn(4, 32)
            y = torch.randint(0, 10, (4,))
            try:
                state = algo.train_step(x, y, 0)
                self.assertIsNotNone(state.loss)
                print(f"  Passed: {model_name} Loss={state.loss}")
            except Exception as e:
                self.fail(f"Failed {model_name}: {e}")

if __name__ == '__main__':
    unittest.main()
