"""Learning-algorithm hunt, round 4: the NS-orthogonalized-Adam cell.

Queued from TODO11's hunt list: "NS-orthogonalized-Adam variant (SVD per
step is the deep-sweep cost center — NS is Muon's cheaper recipe;
measure whether it preserves the OrthoAdam lift, the D13 whitening
question for the hybrid)."

`OrthoAdamUpdate.ortho_steps` (landed this session): 0 = exact SVD polar
factor (configuration of record for D15/D16), >0 = canonical Muon
Newton–Schulz quintic iteration on the bias-corrected first moment.
D13's whitening lesson: NS preserves BP×Muon but collapses FF×Muon
(0.29 vs 0.838 at width 32) — the local-credit lift is full-spectrum-
whitening-driven. Does the same split hold for the hybrid?

Arms (D16 regime: mnist quick 150 batches, 1 epoch, bp credit unless
stated, seeds 0-2, TEST accuracy via the free settle):
  4 geometries x {ortho_adam SVD, ortho_adam NS5}  — the lift question
  ff credit x {ortho_adam SVD, ortho_adam NS5} on mlp_d2_w64 — whitening
  walltime per arm on stdout only (never a record).

Reference (D16 record, same regime): ortho_adam SVD mlp 0.930 /
attention 0.911 / lattice 0.924 / graph 0.411.

Run: uv run python scripts/probes/hunt_ns_adam.py
"""

from itertools import islice
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
import torch

from computronium import (
    AttentionGeometry,
    BackpropCredit,
    DigitalSubstrate,
    FeedforwardGeometry,
    GeometryConfig,
    GraphGeometry,
    InstantaneousDynamics,
    LocalGoodnessCredit,
    OrthoAdamUpdate,
    ParameterUpdateConfig,
    SpatialLattice3DGeometry,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemState,
    SystemTrainer,
    SystemTrainerConfig,
    compose_system,
    create_task,
)

SEEDS = (0, 1, 2)
BATCH_CAP = 150
ORTHO_LR = 3e-3  # the config of record (hunt_hybrid calibration)


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


def _run(
    update_fn: Callable,
    geometry_fn: Callable,
    seed: int,
    train_data,
    test_batches,
    credit=None,
) -> tuple[float, float]:
    torch.manual_seed(seed)
    system = compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device="cpu")),
        geometry=geometry_fn(),
        dynamics=InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
        credit=credit or BackpropCredit(),
        update=update_fn(),
    )
    t0 = perf_counter()
    SystemTrainer(
        system=system,
        config=SystemTrainerConfig(max_epochs=1, device="cpu", seed=42),
        train_data=train_data,
    ).fit()
    walltime = perf_counter() - t0
    ok = tot = 0
    with torch.no_grad():
        for batch_x, batch_y in test_batches:
            state = system.dynamics.settle(
                SystemState(x=batch_x), system.geometry, system.substrate, None
            )
            out = state.activations[-1]
            ok += (out.argmax(1) == batch_y).sum().item()
            tot += batch_y.size(0)
    return ok / tot, walltime


def _ortho(steps: int) -> Callable:
    return lambda: OrthoAdamUpdate(
        ParameterUpdateConfig.ortho_adam(ortho_lr=ORTHO_LR, ortho_steps=steps)
    )


if __name__ == "__main__":
    task = create_task("mnist", device="cpu", quick_mode=True, num_workers=0)
    task.setup()
    torch.manual_seed(0)  # seed BEFORE the loader draw (D8 trap)
    train_data = [
        (xb.view(xb.size(0), -1), yb)
        for xb, yb in islice(task.get_dataloader("train"), BATCH_CAP)
    ]
    test_batches = [
        (xb.view(xb.size(0), -1), yb)
        for xb, yb in task.get_dataloader("test")
        if xb.size(0) == 32
    ]

    print("=== 4 geometries x {svd, ns5}, bp credit, seeds 0-2 ===", flush=True)
    for name, geometry_fn in _geometries().items():
        for label, steps in (("svd", 0), ("ns5", 5)):
            accs, walls = zip(
                *(
                    _run(_ortho(steps), geometry_fn, s, train_data, test_batches)
                    for s in SEEDS
                ),
                strict=True,
            )
            print(
                f"{name:>14} ortho_adam[{label:>3}]: {np.mean(accs):.3f} "
                f"± {np.std(accs):.3f} {tuple(round(a, 3) for a in accs)}  "
                f"({np.mean(walls):.1f} s/arm)",
                flush=True,
            )

    print(
        "\n=== the D13 whitening question: ff credit x {svd, ns5}, mlp ===", flush=True
    )
    credit = LocalGoodnessCredit()
    for label, steps in (("svd", 0), ("ns5", 5)):
        accs, _ = zip(
            *(
                _run(
                    _ortho(steps),
                    _geometries()["mlp_d2_w64"],
                    s,
                    train_data,
                    test_batches,
                    credit=credit,
                )
                for s in SEEDS
            ),
            strict=True,
        )
        print(
            f"ff ortho_adam[{label:>3}]: {np.mean(accs):.3f} ± {np.std(accs):.3f} "
            f"{tuple(round(a, 3) for a in accs)}",
            flush=True,
        )
