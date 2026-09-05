"""Performance hunt, round 4: the FULL update axis across the geometries.

Two user directives: don't fixate on conv, and don't put all eggs in the
Muon basket. This sweep runs the capacity-identical swap (same geometry,
swapped update/credit — params identical per geometry) over FOUR
geometries × FOUR update rules:

    geometry ∈ {mlp_d2_w64, attention, graph_grid8x4, lattice3d}
    update   ∈ {euclidean lr 0.1, muon lr 0.02, spectral lr 0.1,
                natural lr 0.1}
    credit   = bp (the standard rule), plus ff/muon as the local reference

mnist quick, 150 batches, 1 epoch, seeds 0-2, TEST accuracy. The grid is
the U-axis coverage map the ontology claims but never measured.
Capacity-matched: all four geometries land within 1.15x of each other
(48k-57k params, per the D8-D12 fairness convention).
"""

from itertools import islice
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
import torch

from computronium import (
    AdamUpdate,
    AttentionGeometry,
    BackpropCredit,
    CreditAssignmentConfig,
    DigitalSubstrate,
    FeedforwardGeometry,
    GeometryConfig,
    GraphGeometry,
    InstantaneousDynamics,
    LocalGoodnessCredit,
    NaturalGradientUpdate,
    ParameterUpdateConfig,
    SpatialLattice3DGeometry,
    SpectralConstrainedUpdate,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemState,
    SystemTrainer,
    SystemTrainerConfig,
    compose_system,
    create_task,
)
from computronium.ontology.update import EuclideanUpdate, RiemannianOrthogonalUpdate

SEEDS = (0, 1, 2)
BATCH_CAP = 150


def _grid_edges(h: int = 8, w: int = 4) -> list[list[int]]:
    src: list[int] = []
    dst: list[int] = []
    for r in range(h):
        for c in range(w):
            i = r * w + c
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    j = nr * w + nc
                    src.extend((i, j))
                    dst.extend((j, i))
    return [src, dst]


def _geometries() -> dict[str, Callable]:
    return {
        "mlp_d2_w64": lambda: FeedforwardGeometry(
            GeometryConfig.feedforward(
                input_dim=784, output_dim=10, hidden_dims=(64, 64)
            )
        ),
        "attention": lambda: AttentionGeometry(
            GeometryConfig.attention(
                input_dim=784, output_dim=10, hidden_dim=32, num_layers=2, num_heads=4
            )
        ),
        "graph_grid8x4": lambda: GraphGeometry(
            GeometryConfig.graph(
                input_dim=784,
                output_dim=10,
                edge_index=_grid_edges(),
                hidden_dims=(56, 56),
            )
        ),
        "lattice3d": lambda: SpatialLattice3DGeometry(
            GeometryConfig.spatial_lattice(
                input_dim=784,
                output_dim=10,
                lattice_dims=(4, 3, 3),
                hidden_dims=(2,),
                connectivity_radius=1,
            )
        ),
    }


def _updates() -> dict[str, Callable]:
    return {
        "euclid": lambda: EuclideanUpdate(
            ParameterUpdateConfig.euclidean(step_size=0.1)
        ),
        "muon": lambda: RiemannianOrthogonalUpdate(
            ParameterUpdateConfig.riemannian_orthogonal(step_size=0.02, momentum=0.9)
        ),
        "spectral": lambda: SpectralConstrainedUpdate(
            ParameterUpdateConfig.spectral_constrained(step_size=0.1)
        ),
        "natural": lambda: NaturalGradientUpdate(
            ParameterUpdateConfig.natural_gradient(step_size=0.1)
        ),
        "adam": lambda: AdamUpdate(ParameterUpdateConfig.adam(step_size=1e-3)),
    }


def _credit(name: str):
    if name == "bp":
        return BackpropCredit()
    objective = "ff" if name == "ff" else "pepita"
    return LocalGoodnessCredit(
        CreditAssignmentConfig.local_goodness(
            feedback_scale=0.01, local_objective=objective
        )
    )


def _run(
    geometry_fn: Callable, credit: str, update: str, seed: int, train_data, test_batches
) -> float:
    torch.manual_seed(seed)
    system = compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device="cpu")),
        geometry=geometry_fn(),
        dynamics=InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
        credit=_credit(credit),
        update=_updates()[update](),
    )
    SystemTrainer(
        system=system,
        config=SystemTrainerConfig(max_epochs=1, device="cpu", seed=42),
        train_data=train_data,
    ).fit()
    ok = tot = 0
    with torch.no_grad():
        for batch_x, batch_y in test_batches:
            state = system.dynamics.settle(
                SystemState(x=batch_x),
                system.geometry,
                system.substrate,
                None,
            )
            acts = state.activations
            out = acts[-1] if isinstance(acts, list) else acts
            ok += (out.argmax(1) == batch_y).sum().item()
            tot += batch_y.size(0)
    return ok / tot


if __name__ == "__main__":
    task = create_task("mnist", device="cpu", quick_mode=True, num_workers=0)
    task.setup()
    torch.manual_seed(0)  # seed BEFORE the loader draw (D8 trap)
    # GraphGeometry contract: node dim == batch dim (D9 semantics) — each
    # 32-sample batch is one graph over an 8x4 grid of batch positions.
    train_data = [
        (xb.view(xb.size(0), -1), yb)
        for xb, yb in islice(task.get_dataloader("train"), BATCH_CAP)
    ]
    test_batches = [
        (xb.view(xb.size(0), -1), yb)
        for xb, yb in task.get_dataloader("test")
        if xb.size(0) == 32
    ]

    for name, geometry_fn in _geometries().items():
        n_params = sum(p.numel() for p in geometry_fn().params.values())
        for update in _updates():
            accs = [
                _run(geometry_fn, "bp", update, s, train_data, test_batches)
                for s in SEEDS
            ]
            print(
                f"{name:>12} bp/{update:>7}: {np.mean(accs):.3f} "
                f"± {np.std(accs):.3f} {accs} params={n_params}",
                flush=True,
            )
        # the local reference on the same geometry
        accs = [
            _run(geometry_fn, "ff", "muon", s, train_data, test_batches) for s in SEEDS
        ]
        print(
            f"{name:>12} ff/muon   : {np.mean(accs):.3f} "
            f"± {np.std(accs):.3f} {accs} params={n_params}",
            flush=True,
        )
        # the learning-algorithm hunt: local credit on the Adam family
        accs = [
            _run(geometry_fn, "ff", "adam", s, train_data, test_batches) for s in SEEDS
        ]
        print(
            f"{name:>12} ff/adam   : {np.mean(accs):.3f} "
            f"± {np.std(accs):.3f} {accs} params={n_params}",
            flush=True,
        )
