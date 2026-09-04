"""Depth metrics (R11.3.13) and μPC depth-scaled init (R11.3.11) locks."""

import pytest
import torch
from torch import nn

from computronium import (
    FeedforwardGeometry,
    FixedDepth,
    GeometryConfig,
    GraphGeometry,
    LongestPathDepth,
    ShortestPathDepth,
)


class TestDepthMetrics:
    def test_fixed_depth(self):
        torch.testing.assert_close(FixedDepth(4).per_node(3), torch.full((3,), 4.0))

    def test_shortest_path_bfs(self):
        # 0 -> 1 -> 2 and 0 -> 3; node 4 unreachable
        edge = torch.tensor([[1, 2, 3], [0, 1, 0]])
        d = ShortestPathDepth(edge).per_node(5)
        torch.testing.assert_close(d, torch.tensor([0.0, 1.0, 2.0, 1.0, float("inf")]))

    def test_shortest_path_multi_source(self):
        edge = torch.tensor([[2, 3], [0, 1]])
        d = ShortestPathDepth(edge, sources=(0, 1)).per_node(4)
        torch.testing.assert_close(d, torch.tensor([0.0, 0.0, 1.0, 1.0]))

    def test_shortest_path_rejects_bad_source(self):
        edge = torch.zeros(2, 0, dtype=torch.long)
        with pytest.raises(ValueError, match="source"):
            ShortestPathDepth(edge, sources=(2,)).per_node(2)

    def test_longest_path_dag(self):
        # diamond: 0 -> 1 -> 3, 0 -> 2 -> 3
        edge = torch.tensor([[1, 2, 3, 3], [0, 0, 1, 2]])
        d = LongestPathDepth(edge).per_node(4)
        torch.testing.assert_close(d, torch.tensor([0.0, 1.0, 1.0, 2.0]))

    def test_longest_path_rejects_cycle(self):
        edge = torch.tensor([[1, 0], [0, 1]])
        with pytest.raises(ValueError, match="cyclic"):
            LongestPathDepth(edge).per_node(2)

    def test_bad_edge_index_shape(self):
        with pytest.raises(ValueError, match=r"\[2, E\]"):
            LongestPathDepth(torch.zeros(3, 2, dtype=torch.long)).per_node(4)


class TestGraphGeometryDepths:
    def test_num_nodes_and_node_depths(self):
        g = GraphGeometry(
            GeometryConfig.graph(
                input_dim=4,
                output_dim=2,
                edge_index=[[1, 2, 3, 3], [0, 0, 1, 2]],
            )
        )
        assert g.num_nodes == 4
        torch.testing.assert_close(
            g.node_depths(LongestPathDepth(g._edge_index)),
            torch.tensor([0.0, 1.0, 1.0, 2.0]),
        )


DIMS = (16, 32, 32, 10)
NUM_HIDDEN = 2


def _linear_weights(config: GeometryConfig) -> list[torch.Tensor]:
    geom = FeedforwardGeometry(config)
    return [m.weight for m in geom._layers if isinstance(m, nn.Linear)]


class TestMupcInit:
    def _config(self, **overrides: object) -> GeometryConfig:
        return GeometryConfig.feedforward(
            input_dim=DIMS[0],
            output_dim=DIMS[-1],
            hidden_dims=DIMS[1:-1],
            **overrides,  # type: ignore[arg-type]
        )

    def test_default_bitwise_unchanged(self):
        torch.manual_seed(42)
        via_classmethod = _linear_weights(self._config())
        torch.manual_seed(42)
        manual = _linear_weights(
            GeometryConfig(
                input_dim=DIMS[0],
                output_dim=DIMS[-1],
                hidden_dims=DIMS[1:-1],
                num_layers=NUM_HIDDEN,
                topology_type="feedforward",
                connectivity=None,
                recurrent_weight=None,
            )
        )
        for a, b in zip(via_classmethod, manual, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_mupc_hidden_scale(self):
        torch.manual_seed(0)
        weights = _linear_weights(self._config(init_scheme="mupc"))
        n, depth = DIMS[1], NUM_HIDDEN
        for w in weights[:-1]:
            assert w.std().item() == pytest.approx(1.0 / (n * depth) ** 0.5, rel=0.15)

    def test_mupc_output_scale(self):
        torch.manual_seed(0)
        out = _linear_weights(self._config(init_scheme="mupc"))[-1]
        assert out.std().item() == pytest.approx(1.0 / DIMS[1], rel=0.15)

    def test_graph_geometry_mupc(self):
        torch.manual_seed(0)
        g = GraphGeometry(
            GeometryConfig.graph(
                input_dim=4,
                output_dim=2,
                edge_index=[[1, 2], [0, 0]],
                hidden_dims=(8, 8),
                init_scheme="mupc",
            )
        )
        n, depth = 8, 2
        for key in g._layer_weights:
            std = g._layer_weights[key].std().item()
            assert std == pytest.approx(1.0 / (n * depth) ** 0.5, rel=0.3)
        assert g._head.weight.std().item() == pytest.approx(1.0 / n, rel=0.3)


class TestMupcRecurrentCoherence:
    """Recurrent matrices keep the EqProp small-recurrent convention under
    both init schemes (mupc scales the feedforward stack only)."""

    def test_recurrent_weight_untouched_by_mupc(self):
        torch.manual_seed(7)
        from computronium import RecurrentGeometry

        default = RecurrentGeometry(
            GeometryConfig.recurrent(input_dim=4, output_dim=2, hidden_dims=(8,))
        )
        torch.manual_seed(7)
        mupc = RecurrentGeometry(
            GeometryConfig.recurrent(
                input_dim=4,
                output_dim=2,
                hidden_dims=(8,),
                init_scheme="mupc",
            )
        )
        # Same distribution (init_scale x 0.1), not bitwise — the schemes
        # consume the global RNG differently in the linear stack.
        torch.testing.assert_close(
            default._recurrent_weight.std(),
            mupc._recurrent_weight.std(),
            rtol=0.15,
            atol=0,
        )

    def test_recurrent_linear_layers_do_use_mupc(self):
        from computronium import RecurrentGeometry

        torch.manual_seed(0)
        geom = RecurrentGeometry(
            GeometryConfig.recurrent(
                input_dim=16, output_dim=10, hidden_dims=(32,), init_scheme="mupc"
            )
        )
        hidden_linear = [m.weight for m in geom._layers if isinstance(m, nn.Linear)][
            :-1
        ]
        n, depth = 32, 1
        for w in hidden_linear:
            assert w.std().item() == pytest.approx(1.0 / (n * depth) ** 0.5, rel=0.3)
