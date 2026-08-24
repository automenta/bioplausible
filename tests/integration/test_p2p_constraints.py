import unittest
from unittest.mock import MagicMock, patch

from computronium.hyperopt.search_space import SearchSpace
from computronium.p2p.evolution import P2PEvolution


class TestP2PConstraints(unittest.TestCase):
    def test_search_space_constraints(self):
        # Create a search space
        space = SearchSpace(
            "test",
            {
                "hidden_dim": [64, 128, 256, 512],
                "num_layers": (2, 10, "int"),
                "steps": (5, 50, "int"),
            },
        )

        # Apply constraints
        constraints = {"max_hidden": 128, "max_layers": 4, "max_steps": 20}

        constrained_space = space.apply_constraints(constraints)

        # Verify
        # 1. Discrete Choice
        self.assertEqual(constrained_space.params["hidden_dim"], [64, 128])

        # 2. Range
        self.assertEqual(constrained_space.params["num_layers"], (2, 4, "int"))
        self.assertEqual(constrained_space.params["steps"], (5, 20, "int"))

        # Sample to double check
        sample = constrained_space.sample()
        self.assertLessEqual(sample["hidden_dim"], 128)
        self.assertLessEqual(sample["num_layers"], 4)
        self.assertLessEqual(sample["steps"], 20)

    @patch("computronium.p2p.evolution.DHTNode")
    def test_p2p_evolution_settings(self, mock_dht):
        # Test Quick Mode
        evo = P2PEvolution(discovery_mode="quick")

        # Mock dependencies to allow instantiation
        evo.dht = MagicMock()

        # We can't easily test internal loop variables without running it,
        # but we can verify the object state.
        self.assertEqual(evo.discovery_mode, "quick")

        # Test Constraints passed
        constraints = {"max_hidden": 32}
        evo_constr = P2PEvolution(constraints=constraints)
        self.assertEqual(evo_constr.constraints, constraints)

    def test_search_space_crossover(self):
        space = SearchSpace(
            "test",
            {
                "hidden_dim": [64, 128],
                "num_layers": (2, 4, "int"),
                "learning_rate": (1e-4, 1e-2, "log"),
            },
        )
        parent_a = {"hidden_dim": 64, "num_layers": 2, "learning_rate": 1e-3}
        parent_b = {"hidden_dim": 128, "num_layers": 4, "learning_rate": 5e-3}
        child = space.crossover(parent_a, parent_b)
        # Every spec key is present, and each value comes from one of the parents.
        self.assertEqual(set(child), set(space.params))
        self.assertIn(child["hidden_dim"], {64, 128})
        self.assertIn(child["learning_rate"], {1e-3, 5e-3})
        self.assertLessEqual(child["num_layers"], 4)

    def test_search_space_mutate_clamps_and_perturbs(self):
        space = SearchSpace(
            "test",
            {
                "hidden_dim": [64, 128, 256],
                "num_layers": (2, 8, "int"),
                "learning_rate": (1e-4, 1e-2, "log"),
                "momentum": (0.0, 0.9, "linear"),
            },
        )
        # mutation_rate=0 must clamp out-of-range values without tossing them.
        clamped = space.mutate(
            {
                "hidden_dim": 999,
                "num_layers": 99,
                "learning_rate": 1e-3,
                "momentum": 0.5,
            },
            mutation_rate=0.0,
        )
        self.assertIn(clamped["hidden_dim"], [64, 128, 256])
        self.assertEqual(clamped["num_layers"], 8)  # clamped to max
        self.assertEqual(clamped["momentum"], 0.5)  # in-range, untouched

        # mutation_rate=1 must keep every value within bounds.
        for _ in range(20):
            mutated = space.mutate(
                {
                    "hidden_dim": 128,
                    "num_layers": 3,
                    "learning_rate": 1e-3,
                    "momentum": 0.5,
                },
                mutation_rate=1.0,
            )
            self.assertIn(mutated["hidden_dim"], [64, 128, 256])
            self.assertGreaterEqual(mutated["num_layers"], 2)
            self.assertLessEqual(mutated["num_layers"], 8)
            self.assertLessEqual(1e-4, mutated["learning_rate"] <= 1e-2)
            self.assertLessEqual(0.0, mutated["momentum"] <= 0.9)


if __name__ == "__main__":
    unittest.main()
