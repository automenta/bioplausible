import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch.nn as nn

# Add parent to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))


class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


class TestTaskAbstractions(unittest.TestCase):

    def test_lm_task_setup_fail(self):
        from bioplausible.hyperopt.tasks import LMTask

        task = LMTask("invalid_dataset_123")

        # Should raise exception on setup but handle printing gracefully
        with self.assertRaises(Exception):
            task.setup()

    def test_rl_task_trainer_creation(self):
        import gymnasium as gym

        from bioplausible.hyperopt.tasks import RLTask

        # Mock gym environment
        with patch("gymnasium.make") as mock_make:
            mock_env = MagicMock()
            mock_env.action_space.n = 2
            mock_env.observation_space.shape = (4,)
            mock_make.return_value = mock_env

            task = RLTask("CartPole-v1")
            task.setup()

            # Mock model with parameters
            model = MockModel()

            # Test kwargs filtering
            trainer = task.create_trainer(model, lr=0.01, episodes=100, invalid_arg=555)
            self.assertIsNotNone(trainer)
            # RLTrainer stores optimizer, we can check that
            self.assertEqual(trainer.optimizer.param_groups[0]["lr"], 0.01)

    def test_evolution_p2p_seeding(self):
        from bioplausible.hyperopt.engine import (EvolutionaryOptimizer,
                                                  OptimizationConfig)

        # Mock P2P controller
        mock_p2p = MagicMock()
        mock_dht = MagicMock()
        mock_p2p.dht = mock_dht

        # Mock return from DHT
        best_config = {
            "model_name": "EqProp MLP",  # Use a valid name
            "accuracy": 0.9,
            "config": {"hidden_dim": 128, "model_name": "EqProp MLP"},
        }
        mock_dht.get_best_model.return_value = best_config

        config = OptimizationConfig(use_p2p=True, task="test_task")

        # Mock storage to avoid DB writes
        with patch("bioplausible.hyperopt.engine.HyperoptStorage") as MockStorage:
            storage = MockStorage()
            optimizer = EvolutionaryOptimizer(
                model_names=["EqProp MLP"],
                config=config,
                storage=storage,
                p2p_controller=mock_p2p,
            )

            # Mock search space
            with patch(
                "bioplausible.hyperopt.engine.get_search_space"
            ) as mock_get_space:
                mock_space = MagicMock()
                mock_space.sample.return_value = {"hidden_dim": 64}
                mock_get_space.return_value = mock_space

                # Initialize
                optimizer.initialize_population("EqProp MLP")

                # Verify DHT was called with correct task
                mock_dht.get_best_model.assert_called_with("test_task")

                # Verify storage.create_trial was called
                self.assertTrue(storage.create_trial.called)


if __name__ == "__main__":
    unittest.main()
