import unittest
import torch
import shutil
from pathlib import Path
from bioplausible.hyperopt.experiment import TrialRunner, ExperimentAlgorithm
from bioplausible.models.registry import MODEL_REGISTRY, get_model_spec
from bioplausible.hyperopt.storage import HyperoptStorage

class TestHyperoptIntegration(unittest.TestCase):
    def setUp(self):
        self.test_db = "test_hyperopt_integration.db"
        self.storage = HyperoptStorage(self.test_db)
        self.storage.clear_all_trials()

    def tearDown(self):
        self.storage.close()
        if Path(self.test_db).exists():
            Path(self.test_db).unlink()

    def test_instantiate_all_models_lm(self):
        """Test instantiation of all model types for LM task."""
        vocab_size = 65
        for spec in MODEL_REGISTRY:
            if spec.model_type in ["deep_hebbian", "eqprop_mlp"]:
               # deep_hebbian and eqprop_mlp in current registry config are set up for vision/vector tasks primarily
               # but ExperimentAlgorithm handles them for LM by adding embedding.
               pass

            try:
                algo = ExperimentAlgorithm(
                    spec,
                    output_dim=vocab_size,
                    hidden_dim=32,
                    num_layers=2,
                    device="cpu",
                    task_type="lm"
                )
                self.assertIsNotNone(algo.model)
                if spec.model_type in ["eqprop_mlp", "dfa", "chl", "deep_hebbian"]:
                    self.assertTrue(algo.has_embed)
                    self.assertIsNotNone(algo.embed)
            except Exception as e:
                self.fail(f"Failed to instantiate {spec.name} for LM: {e}")

    def test_instantiate_vision_models(self):
        """Test instantiation for Vision tasks (vector input)."""
        input_dim = 784
        output_dim = 10
        # Only test models compatible with vector input (usually MLPs)
        mlp_specs = [s for s in MODEL_REGISTRY if s.model_type in ["backprop", "eqprop_mlp", "dfa", "chl", "deep_hebbian"]]

        for spec in mlp_specs:
            try:
                algo = ExperimentAlgorithm(
                    spec,
                    output_dim=output_dim,
                    input_dim=input_dim,
                    hidden_dim=32,
                    num_layers=2,
                    device="cpu",
                    task_type="vision"
                )
                self.assertIsNotNone(algo.model)
                self.assertFalse(algo.has_embed)
            except Exception as e:
                self.fail(f"Failed to instantiate {spec.name} for Vision: {e}")

    def test_rl_runner_step(self):
        """Test RL runner execution (integration with RLTrainer)."""
        # Create a runner for CartPole
        runner = TrialRunner(
            storage=self.storage,
            device="cpu",
            task="cartpole",
            quick_mode=True
        )

        # Override epochs to 1 for speed
        runner.epochs = 1

        # Pick a compatible model (e.g., EqProp MLP)
        spec = [m for m in MODEL_REGISTRY if m.model_type == "eqprop_mlp"][0]

        config = {
            "hidden_dim": 32,
            "num_layers": 1,
            "lr": 0.01,
            "steps": 5
        }

        trial_id = self.storage.create_trial(spec.name, config)

        # Run
        success = runner.run_trial(trial_id)
        self.assertTrue(success)

        # Check results
        trial = self.storage.get_trial(trial_id)
        self.assertEqual(trial.status, "completed")
        self.assertIsNotNone(trial.accuracy) # Reward

if __name__ == '__main__':
    unittest.main()
