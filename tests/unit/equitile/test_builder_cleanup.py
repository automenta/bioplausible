import unittest

from bioplausible.equitile.deployments.graph import GraphEquiTile, GraphEquiTileConfig
from bioplausible.equitile.deployments.timeseries import (
    TimeSeriesConfig,
    TimeSeriesEquiTile,
)


class TestDeploymentConfigCleanup(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"

    def test_graph_equitile_config_cleanup(self):
        """Test GraphEquiTile works with the consolidated deployment config."""
        config = GraphEquiTileConfig(node_features=5, hidden_dim=16, num_classes=2)
        # Consolidated onto the unified deployment config (backprop-capable).
        self.assertTrue(hasattr(config, "mode"))

        model = GraphEquiTile(config)
        self.assertIsInstance(model, GraphEquiTile)

    def test_timeseries_equitile_config_cleanup(self):
        """Test TimeSeriesEquiTile works with the consolidated deployment config."""
        config = TimeSeriesConfig(input_dim=5, seq_len=10, output_dim=1)
        # Consolidated onto the unified deployment config (backprop-capable).
        self.assertTrue(hasattr(config, "mode"))

        model = TimeSeriesEquiTile(config)
        self.assertIsInstance(model, TimeSeriesEquiTile)


if __name__ == "__main__":
    unittest.main()
