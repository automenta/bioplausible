"""Property lock: tile-mesh block settle ≡ per-edge reference (R11.1.4).

Kernel-equivalence locks (max_diff < 1e-5) are the acceptance bar.
The tile block relaxation through the substrate operator must match
the per-edge reference relaxation up to floating-point reordering.
"""

import pytest
import torch

from computronium.core.tile.topology import TileGraph
from computronium.models.native.tile_native import (
    create_native_tile_ep,
    create_native_tile_fa,
    create_native_tile_tp,
    create_native_tile_snn,
    create_native_tile_hebbian,
    create_native_tile_pc,
    create_native_tile_gnn,
)
from computronium.ontology import DigitalSubstrate
from computronium.ontology._tile_blocks import build_block_view, assemble_transition_blocks


def _make_tile_graph():
    g = TileGraph()
    g.build_layered(
        input_dim=20,
        output_dim=5,
        neurons_per_tile=8,
        num_hidden_layers=1,
        tiles_per_layer=2,
    )
    return g


@pytest.fixture(scope="module")
def tile_graph():
    return _make_tile_graph()


@pytest.fixture(scope="module")
def block_view(tile_graph):
    return build_block_view(tile_graph)


def test_block_view_structure(block_view):
    """The block view has the expected transition count."""
    assert block_view.block_act_count == 5
    assert len(block_view.edge_slots) == 2
    assert len(block_view.edge_slots[0]) == 6
    assert len(block_view.edge_slots[1]) == 2


def test_block_vs_per_edge_equivalence(block_view, tile_graph):
    """Assembled block matmul matches per-edge sum (Digital substrate)."""
    substrate = DigitalSubstrate()
    op = substrate.get_forward_operator()

    per_edge = {}
    for src_id, dst_id in tile_graph.edges:
        src = tile_graph.tiles[src_id]
        dst = tile_graph.tiles[dst_id]
        w = torch.randn(dst.neurons, src.neurons) * 0.01
        per_edge[f"tile_weight.{src_id}_{dst_id}"] = w
    per_edge["input_proj.weight"] = torch.randn(20, 20) * 0.01
    per_edge["output_proj.weight"] = torch.randn(5, 5) * 0.01

    blocks = assemble_transition_blocks(block_view, per_edge)

    x = torch.randn(4, 20)
    z = op(x, per_edge["input_proj.weight"])
    for k, slots in enumerate(block_view.edge_slots, start=1):
        z_edge = torch.zeros_like(z[:, :block_view.layer_widths[k]])
        for name, d_off, d_n, s_off, s_n in slots:
            z_edge[:, d_off : d_off + d_n] += op(
                z[:, s_off : s_off + s_n], per_edge[name]
            )
        z_block = op(z, blocks[k])
        diff = (z_edge - z_block).abs().max().item()
        assert diff < 1e-5, f"transition {k}: max diff {diff} >= 1e-5"
        z = z_block


def test_free_vs_nudged_contrast():
    """Settled nudged state differs from free at the output layer."""
    from computronium.ontology.dynamics._dynamics import (
        EnergyMinimizationDynamics,
        StateDynamicsConfig,
    )
    from computronium.ontology.system import SystemState

    m = create_native_tile_ep(
        input_dim=20,
        hidden_dim=16,
        output_dim=5,
        num_layers=3,
        neurons_per_tile=8,
        tiles_per_layer=2,
        lr=1e-2,
        beta=0.1,
        settle_steps=5,
    )
    g, s, d, c, u = m.geometry, m.substrate, m.dynamics, m.credit, m.update

    x = torch.randn(4, 20)
    y = torch.randint(0, 5, (4,))

    free_st = SystemState(x=x, y=y)
    free_settled = d.settle(free_st, g, s, target=None)

    nudged_st = SystemState(x=x, y=y)
    nudged_settled = d.settle(nudged_st, g, s, target=y)

    free_out = free_settled.activations[-1]
    nudged_out = nudged_settled.activations[-1]
    diff = (free_out - nudged_out).abs().max().item()
    assert diff > 1e-5, "nudged and free output acts must differ"


def test_all_seven_factories_learn():
    """Smoke: all seven tile factories move parameters."""
    x = torch.randn(8, 10)
    y = torch.randint(0, 3, (8,))
    factories = [
        (create_native_tile_ep, dict(beta=0.1)),
        (create_native_tile_fa, {}),
        (create_native_tile_tp, dict(beta=0.1)),
        (create_native_tile_snn, {}),
        (create_native_tile_hebbian, {}),
        (create_native_tile_pc, {}),
        (create_native_tile_gnn, {}),
    ]

    for f, kwargs in factories:
        m = f(
            10, 8, 3, num_layers=2, neurons_per_tile=8, tiles_per_layer=2, lr=1e-2, **kwargs
        )
        before = [p.detach().clone() for p in m.parameters()]
        m.train_step(x, y)
        moved = any(not torch.equal(p.detach(), b) for p, b in zip(m.parameters(), before))
        assert moved, f"{f.__name__}: no parameters moved"