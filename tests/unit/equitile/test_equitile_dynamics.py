import unittest

import torch

from bioplausible.analysis.tile_dynamics import (
    DynamicTileAlgorithm,
    DynamicTileConfig,
    TileGrowthConfig,
)
from bioplausible.core.local_learning.algorithm import TileAlgorithm


class TestTileAlgorithmDynamics(unittest.TestCase):
    def setUp(self):
        self.model = TileAlgorithm.from_ep(
            input_dim=8,
            output_dim=4,
            neurons_per_tile=16,
            num_layers=2,
            tiles_per_layer=2,
        )
        self.growth_config = TileGrowthConfig(
            growth_enabled=True,
            prune_enabled=True,
            growth_threshold=0.1,  # Low threshold to trigger growth
            prune_threshold=0.01,
            growth_cooldown=0,  # No cooldown for testing
            min_age_for_modify=0,
        )
        self.dynamic_config = DynamicTileConfig(growth=self.growth_config)
        self.dynamic = DynamicTileAlgorithm(self.model, config=self.dynamic_config)

    def test_add_tile_via_api(self):
        """Test adding a tile via core API."""
        initial_tiles = len(self.model.graph.tiles)
        new_id = self.model.add_tile(neurons=16, layer_id=1, pos_x=0.5, pos_y=0.5)
        self.assertIn(new_id, self.model.graph.tiles)
        self.assertEqual(len(self.model.graph.tiles), initial_tiles + 1)
        # Check if optimizers reset (check if params are in optimizer)
        found = False
        for group in self.model._optim_importance.param_groups:
            for p in group["params"]:
                if p.shape == self.model.tile_importance.shape:
                    found = True
        self.assertTrue(found)

    def test_remove_tile_via_api(self):
        """Test removing a tile via core API."""
        # Add a dummy tile first
        new_id = self.model.add_tile(neurons=16, layer_id=1)
        initial_tiles = len(self.model.graph.tiles)

        self.model.remove_tile(new_id)
        self.assertNotIn(new_id, self.model.graph.tiles)
        self.assertEqual(len(self.model.graph.tiles), initial_tiles - 1)

    def test_growth_manager(self):
        """Test TileGrowthManager logic."""
        # Fake high error on a tile
        target_tile_id = self.model.graph.tiles[
            self.model.graph.input_tile_ids[0]
        ].fwd_neighbors[0]
        self.dynamic.growth_manager.error_ema[target_tile_id] = 1.0  # High error

        # Trigger step
        stats = self.dynamic.step()

        # Should have grown
        self.assertEqual(stats["grown"], 1)
        self.assertTrue(self.dynamic.tile_modified)

    def test_add_remove_edge(self):
        """Test adding/removing edges."""
        src = self.model.graph.input_tile_ids[0]
        dst = self.model.graph.output_tile_ids[0]

        # Ensure no edge initially (skip hidden)
        if (src, dst) in self.model.graph._edge_set:
            self.model.remove_edge(src, dst)

        self.model.add_edge(src, dst)
        self.assertIn((src, dst), self.model.graph._edge_set)
        self.assertIn(f"{src}_{dst}", self.model._tile_weights)

        self.model.remove_edge(src, dst)
        self.assertNotIn((src, dst), self.model.graph._edge_set)
        self.assertNotIn(f"{src}_{dst}", self.model._tile_weights)

    def test_train_step_after_growth(self):
        """Substrate still trains after dynamic topology mutation."""
        self.dynamic.growth_manager.error_ema[
            self.model.graph.tiles[self.model.graph.input_tile_ids[0]].fwd_neighbors[0]
        ] = 1.0
        self.dynamic.step()

        x = torch.randn(4, 8)
        y = torch.randint(0, 4, (4,))
        stats = self.model.train_step(x, y)
        self.assertIn("loss", stats)
        self.assertIn("accuracy", stats)


if __name__ == "__main__":
    unittest.main()
