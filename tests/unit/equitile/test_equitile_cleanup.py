import unittest

from bioplausible.equitile.core import EquiTile
from bioplausible.equitile.core.config import EquiTileConfig
from bioplausible.equitile.deployments.rl import RLEquiTile, RLEquiTileConfig
from bioplausible.equitile.deployments.vision import ConvEquiTile, ConvEquiTileConfig
from bioplausible.zoo.models.tile_lm import TileLM


class TestEquiTileCleanup(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"

    def test_vision_kwargs(self):
        """Test passing kwargs to ConvEquiTile."""
        config = ConvEquiTileConfig(
            input_channels=1,
            input_size=28,
            num_classes=10,
            equitile_kwargs={"sparsity_threshold": 0.5},
        )
        model = ConvEquiTile(config)
        self.assertEqual(model.head.get_config().extra["sparsity_threshold"], 0.5)

    def test_lm_kwargs(self):
        """Test substrate LM knobs flow through config."""
        model = TileLM.from_lm(
            vocab_size=100,
            embed_dim=16,
            num_layers=1,
            importance_lr=0.05,
        )
        config = model.get_config()
        self.assertEqual(config.importance_lr, 0.05)
        self.assertEqual(config.extra["vocab_size"], 100)

    def test_rl_kwargs(self):
        """Test passing kwargs to RLEquiTile."""
        config = RLEquiTileConfig(
            obs_dim=8,
            action_dim=4,
            equitile_kwargs={"dropout": 0.3},
        )
        model = RLEquiTile(config)
        self.assertEqual(model.feature_extractor.get_config().extra["dropout"], 0.3)

    def test_core_get_config(self):
        """Test get_config on Core EquiTile."""
        model = EquiTile(neurons_per_tile=16)
        config = model.get_config()
        self.assertIsInstance(config, EquiTileConfig)
        self.assertEqual(config.neurons_per_tile, 16)

    def test_input_output_dim_consistency(self):
        """Test that input_dim and output_dim are consistent."""
        model = EquiTile(input_dim=12, output_dim=6)
        self.assertEqual(model.input_dim, 12)
        self.assertEqual(model.output_dim, 6)
        self.assertEqual(model.config.input_dim, 12)
        self.assertEqual(model.config.output_dim, 6)


if __name__ == "__main__":
    unittest.main()
