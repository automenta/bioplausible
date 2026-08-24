import unittest

from computronium.zoo.models.deployments.graph import GraphTileNet, GraphTileNetConfig
from computronium.zoo.models.deployments.timeseries import (
    TimeSeriesConfig,
    TimeSeriesTileNet,
)


class TestDeploymentConfigCleanup(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"

    def test_graph_equitile_config_cleanup(self):
        """Test GraphTileNet works with the consolidated deployment config."""
        config = GraphTileNetConfig(node_features=5, hidden_dim=16, num_classes=2)
        # Consolidated onto the unified deployment config (backprop-capable).
        self.assertTrue(hasattr(config, "mode"))

        model = GraphTileNet(config)
        self.assertIsInstance(model, GraphTileNet)

    def test_timeseries_equitile_config_cleanup(self):
        """Test TimeSeriesTileNet works with the consolidated deployment config."""
        config = TimeSeriesConfig(input_dim=5, seq_len=10, output_dim=1)
        # Consolidated onto the unified deployment config (backprop-capable).
        self.assertTrue(hasattr(config, "mode"))

        model = TimeSeriesTileNet(config)
        self.assertIsInstance(model, TimeSeriesTileNet)


if __name__ == "__main__":
    unittest.main()
