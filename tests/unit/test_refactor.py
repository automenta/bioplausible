import unittest

from computronium.core.registry import ComponentCategory, Registry
from computronium.models.native import registration  # noqa: F401  (populates Registry)


class TestRefactor(unittest.TestCase):
    def test_imports_and_models(self):
        """Test that native 5-D compositions can be instantiated."""
        # Use native 5-D composition for eqprop_mlp
        system = Registry.to_system(
            "eqprop_mlp", input_dim=10, hidden_dim=20, output_dim=5
        )
        self.assertTrue(hasattr(system, "forward"))
        self.assertTrue(hasattr(system, "train_step"))

    def test_models_registry(self):
        """Test that models are registered with the unified Registry."""
        # 'eqprop_mlp' is the native factory returning a System
        names = Registry.list(ComponentCategory.MODEL).get("model", [])
        self.assertIn("eqprop_mlp", names)

        eqprop_factory = Registry.get(ComponentCategory.MODEL, "eqprop_mlp")
        self.assertIsNotNone(eqprop_factory)

        # And we can instantiate by name - returns a System
        system = eqprop_factory(input_dim=10, hidden_dim=20, output_dim=5)
        self.assertTrue(hasattr(system, "forward"))
        self.assertTrue(hasattr(system, "train_step"))


if __name__ == "__main__":
    unittest.main()
