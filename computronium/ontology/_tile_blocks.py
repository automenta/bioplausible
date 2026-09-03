"""Tile-mesh block view: the layered contract for the settle family.

The tile DAG (adjacent-layer edges, layer-ordered tiles) admits a per-
transition block-matrix view: each act transition assembles into one dense
``(N_post, N_pre)`` matrix whose rows/cols are the dst/src tile offsets.
``SubstrateSettleKernel`` then relaxes the mesh exactly like a layered
geometry, with substrate physics applied to the assembled blocks. Credits
walk the same blocks and scatter per-transition pseudo-gradients back to
per-edge parameters.

Block layout of settled acts: ``[x, z_0, ..., z_{L-1}, output]`` where
``z_k`` concatenates layer ``k``'s tile activities in ``layer_ids[k]``
order — transition 0 is the input projection, transitions 1..L-1 are the
tile edges, transition ``L`` the output projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from computronium.ontology._settle_kernel import LayeredParams

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from computronium.core.tile.topology import TileGraph
    from computronium.ontology.substrate import Substrate

_INPUT_PROJ = "input_proj.weight"
_OUTPUT_PROJ = "output_proj.weight"

# Edge slot: (weight_name, dst_offset, dst_neurons, src_offset, src_neurons)
type EdgeSlot = tuple[str, int, int, int, int]


@dataclass(frozen=True, slots=True)
class TileBlockView:
    """Static block-view structure of a layered tile graph."""

    layer_spans: tuple[tuple[tuple[int, int, int], ...], ...]
    layer_widths: tuple[int, ...]
    edge_slots: tuple[tuple[EdgeSlot, ...], ...]
    block_act_count: int


def build_block_view(graph: TileGraph) -> TileBlockView:
    """Derive the block view; rejects graphs whose edges skip layers."""
    layer_spans: list[tuple[tuple[int, int, int], ...]] = []
    rank: dict[int, int] = {}
    for k, layer in enumerate(graph.layer_ids):
        spans: list[tuple[int, int, int]] = []
        for tid in layer:
            tile = graph.tiles[tid]
            if tile.layer_id != k:
                raise ValueError(
                    f"tile block view requires consistent layer_ids; tile "
                    f"{tid} declares layer {tile.layer_id}, sits in {k}"
                )
            spans.append((tid, sum(n for _, _, n in spans), tile.neurons))
            rank[tid] = k
        layer_spans.append(tuple(spans))
    for tid, tile in graph.tiles.items():
        if rank.get(tid) != tile.layer_id:
            raise ValueError(
                f"tile block view requires every tile in layer_ids; tile "
                f"{tid} declares layer {tile.layer_id}"
            )

    per_layer: dict[int, list[EdgeSlot]] = {}
    for src_id, dst_id in graph.edges:
        src, dst = graph.tiles[src_id], graph.tiles[dst_id]
        if dst.layer_id != src.layer_id + 1:
            raise ValueError(
                "tile block view requires adjacent-layer edges (no skip "
                f"connections); edge {src_id}->{dst_id} spans "
                f"{src.layer_id}->{dst.layer_id}"
            )
        d_off, d_n = _locate(layer_spans[dst.layer_id], dst_id)
        s_off, s_n = _locate(layer_spans[src.layer_id], src_id)
        per_layer.setdefault(dst.layer_id, []).append(
            (f"tile_weight.{src_id}_{dst_id}", d_off, d_n, s_off, s_n)
        )

    edge_slots = tuple(
        tuple(per_layer.get(k, ())) for k in range(1, len(layer_spans))
    )
    return TileBlockView(
        layer_spans=tuple(layer_spans),
        layer_widths=tuple(sum(n for _, _, n in spans) for spans in layer_spans),
        edge_slots=edge_slots,
        block_act_count=len(layer_spans) + 2,
    )


def _locate(
    spans: tuple[tuple[int, int, int], ...], tid: int
) -> tuple[int, int]:
    for span_tid, offset, neurons in spans:
        if span_tid == tid:
            return offset, neurons
    raise KeyError(tid)


def assemble_transition_blocks(
    view: TileBlockView, named: Mapping[str, Tensor]
) -> tuple[Tensor, ...]:
    """Assemble per-transition block matrices from named per-edge tensors.

    Projection transitions pass their named tensor through; tile-edge
    transitions scatter the per-edge weights into a dense block (in-graph:
    gradients flow to the per-edge parameters).
    """
    blocks = [named[_INPUT_PROJ]]
    for k, slots in enumerate(view.edge_slots, start=1):
        rows, cols = view.layer_widths[k], view.layer_widths[k - 1]
        if slots:
            first = named[slots[0][0]]
            block = torch.zeros(rows, cols, dtype=first.dtype, device=first.device)
            for name, d_off, d_n, s_off, s_n in slots:
                block[d_off : d_off + d_n, s_off : s_off + s_n] = named[name]
        else:
            block = torch.zeros(
                rows, cols, dtype=blocks[0].dtype, device=blocks[0].device
            )
        blocks.append(block)
    blocks.append(named[_OUTPUT_PROJ])
    return tuple(blocks)


def assemble_transition_biases(
    view: TileBlockView, named: Mapping[str, Tensor | None]
) -> tuple[Tensor | None, ...]:
    """(input_proj.bias, per-layer concatenated tile biases, output_proj.bias)."""
    result: list[Tensor | None] = [named.get("input_proj.bias")]
    for k in range(1, len(view.layer_spans)):
        parts = [
            bias
            for tid, _, _ in view.layer_spans[k]
            if (bias := named.get(f"tile_bias.{tid}")) is not None
        ]
        result.append(torch.cat(parts) if parts else None)
    result.append(named.get("output_proj.bias"))
    return tuple(result)


def tile_layered_params(
    view: TileBlockView, named: Mapping[str, Tensor]
) -> LayeredParams:
    """``LayeredParams`` for the settle kernel: assembled weight blocks."""
    return LayeredParams(
        weights=assemble_transition_blocks(view, named),
        biases=assemble_transition_biases(view, named),
        activations=(),
        recurrent_weight=None,
    )


def tile_settle_block_acts(
    view: TileBlockView, geometry: object, x: Tensor, substrate: Substrate
) -> list[Tensor]:
    """Initial block activities: [x, z0, ..., z_{L-1}, output]."""
    op = substrate.get_forward_operator()
    params: Mapping[str, Tensor] = geometry.params  # type: ignore[attr-defined]
    blocks = assemble_transition_blocks(view, params)
    biases = assemble_transition_biases(view, params)
    z = geometry._input_projection(x)  # type: ignore[attr-defined]
    acts = [x, z]
    for k in range(1, len(view.layer_spans)):
        z = op(z, blocks[k])
        bias = biases[k]
        if bias is not None:
            z = z + bias
        acts.append(z)
    acts.append(geometry._output_projection(z))  # type: ignore[attr-defined]
    return acts


def scatter_block_grads(
    view: TileBlockView,
    block_grads: Sequence[Tensor],
    order: Sequence[str],
) -> list[Tensor]:
    """Scatter per-transition block grads to per-edge params in ``order``."""
    by_name: dict[str, Tensor] = {
        _INPUT_PROJ: block_grads[0],
        _OUTPUT_PROJ: block_grads[-1],
    }
    for k, slots in enumerate(view.edge_slots, start=1):
        block = block_grads[k]
        for name, d_off, d_n, s_off, s_n in slots:
            by_name[name] = block[d_off : d_off + d_n, s_off : s_off + s_n]
    return [by_name[name] for name in order]


def tile_hopfield_energy(
    view: TileBlockView, acts: list[Tensor], named: Mapping[str, Tensor]
) -> Tensor:
    """Hopfield energy over the block layout: 0.5·Σ‖z‖² − Σ transitions
    post·(pre@Wᵀ) − Σ post·b, meaned per sample."""
    blocks = assemble_transition_blocks(view, named)
    biases = assemble_transition_biases(view, named)
    fields = acts[1:]
    total = torch.zeros((), device=fields[0].device, dtype=fields[0].dtype)
    for field in fields:
        total = total + 0.5 * field.pow(2).sum()
    for i, block in enumerate(blocks):
        pre, post = acts[i], acts[i + 1]
        total = total - (post * (pre @ block.T)).sum()
        bias = biases[i]
        if bias is not None:
            total = total - (post * bias).sum()
    return total / fields[0].shape[0]
