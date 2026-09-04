"""Per-node effective-depth metrics for arbitrary graph topologies.

Layer-counting fails on non-layered topologies (graphs, tile meshes,
fabrics): "depth" is a per-node property. These metrics compute it from
connectivity, feeding depth-scaled (μPC-style) init and depth-scaling
studies on `GraphGeometry`/`TileGeometry`.

Attribution: μPC depth-scaled init, Ernoult et al., arXiv:2505.13124.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class DepthMetric(Protocol):
    """Compute a depth value per node of a topology with ``num_nodes`` nodes."""

    def per_node(self, num_nodes: int) -> Tensor: ...


@dataclass(frozen=True, slots=True)
class FixedDepth:
    """Constant depth for every node (layer-based stacks)."""

    depth: int

    def per_node(self, num_nodes: int) -> Tensor:
        return torch.full((num_nodes,), float(self.depth))


def _validate_edge_index(edge_index: Tensor, num_nodes: int) -> None:
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must be [2, E], got {tuple(edge_index.shape)}")
    if edge_index.numel() and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes
    ):
        raise ValueError("edge_index references a node outside [0, num_nodes)")


@dataclass(frozen=True, slots=True)
class ShortestPathDepth:
    """BFS depth from a set of source nodes; unreachable nodes get inf.

    Treats edges as directed (``row`` receives from ``col``), matching
    `GraphGeometry._aggregate`'s message direction.
    """

    edge_index: Tensor
    sources: tuple[int, ...] = (0,)

    def per_node(self, num_nodes: int) -> Tensor:
        _validate_edge_index(self.edge_index, num_nodes)
        if not self.sources:
            raise ValueError("sources must be non-empty")
        if any(s < 0 or s >= num_nodes for s in self.sources):
            raise ValueError("source outside [0, num_nodes)")
        row, col = self.edge_index
        depth = torch.full((num_nodes,), float("inf"))
        frontier = list(dict.fromkeys(self.sources))
        seen = set(frontier)
        d = 0
        while frontier:
            for s in frontier:
                depth[s] = float(d)
            outgoing = row[torch.isin(col, torch.tensor(frontier))]
            frontier = [int(n) for n in torch.unique(outgoing) if int(n) not in seen]
            seen.update(frontier)
            d += 1
        return depth


@dataclass(frozen=True, slots=True)
class LongestPathDepth:
    """DAG longest-path depth per node (Kahn topological order).

    Raises ``ValueError`` on cyclic graphs — longest path is undefined
    there, and silent wraparound would corrupt depth-scaled init.
    """

    edge_index: Tensor

    def per_node(self, num_nodes: int) -> Tensor:
        _validate_edge_index(self.edge_index, num_nodes)
        row, col = self.edge_index
        indegree = torch.zeros(num_nodes, dtype=torch.long)
        for dst in row.tolist():
            indegree[dst] += 1
        depth = torch.zeros(num_nodes)
        frontier = [n for n in range(num_nodes) if indegree[n] == 0]
        seen = len(frontier)
        while frontier:
            nxt: list[int] = []
            for src in frontier:
                outgoing = row[col == src]
                for tgt in outgoing.tolist():
                    indegree[tgt] -= 1
                    depth[tgt] = torch.maximum(depth[tgt], depth[src] + 1.0)
                    if indegree[tgt] == 0:
                        nxt.append(tgt)
            seen += len(nxt)
            frontier = nxt
        if seen != num_nodes:
            cyclic = int((indegree > 0).sum())
            raise ValueError(f"graph is cyclic ({cyclic} nodes unresolved)")
        return depth


def max_depth(metric: DepthMetric, num_nodes: int) -> int:
    """Integer depth ceiling used by depth-scaled init."""
    return int(metric.per_node(num_nodes).max().item())
