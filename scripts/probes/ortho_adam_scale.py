"""Registered-scale OrthoAdam probe: does the hybrid's dominance survive
depth and width?

Demo-scale finding (hunt_hybrid.py, D16): OrthoAdam beats both parents on
mlp/attention/lattice at 47-57k params. D15 showed the depth frontier is
where optimizer regimes separate (Euclid cliffs by depth 16, Muon
generalizes). This probe runs the capacity-matched depth grid:

    geometry = feedforward, depth ∈ {8, 16} × width 128 (D15 regime),
    update ∈ {adam 1e-3, muon 0.02, ortho_adam 3e-3}, credit = bp,
    mnist quick 300 batches, 1 epoch, seeds 0-2, TEST accuracy.

Plus the local-credit cells at depth 4 / width 128 (ff × {muon,
ortho_adam, adam}) — does the hybrid rescue local credit at scale, and
does the D13 whitening lesson (SVD-polar load-bearing for FF) transfer
to Adam-momentum whitening?

D15 reference (same regime, mnist quick 300 batches, TEST):
depth 16 / width 128 — bp/muon 0.834 ± 0.036, bp/euclid 0.114.
"""

from itertools import islice
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
import torch

from computronium import (
    AdamUpdate,
    BackpropCredit,
    CreditAssignmentConfig,
    DigitalSubstrate,
    FeedforwardGeometry,
    GeometryConfig,
    InstantaneousDynamics,
    LocalGoodnessCredit,
    OrthoAdamUpdate,
    ParameterUpdateConfig,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemState,
    SystemTrainer,
    SystemTrainerConfig,
    compose_system,
    create_task,
)
from computronium.ontology.update import RiemannianOrthogonalUpdate

SEEDS = (0, 1, 2)
BATCH_CAP = 300


def _geometry(depth: int) -> Callable:
    return lambda: FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=784, output_dim=10, hidden_dims=(128,) * depth
        )
    )


def _updates() -> dict[str, Callable]:
    return {
        "adam": lambda: AdamUpdate(ParameterUpdateConfig.adam(step_size=1e-3)),
        "muon": lambda: RiemannianOrthogonalUpdate(
            ParameterUpdateConfig.riemannian_orthogonal(step_size=0.02, momentum=0.9)
        ),
        "ortho_adam": lambda: OrthoAdamUpdate(
            ParameterUpdateConfig.ortho_adam(step_size=1e-3, ortho_lr=3e-3)
        ),
    }


def _run(credit: str, update: str, depth: int, seed: int, train_data, test_batches):
    torch.manual_seed(seed)
    if credit == "bp":
        credit_obj = BackpropCredit()
    else:
        credit_obj = LocalGoodnessCredit(
            CreditAssignmentConfig.local_goodness(
                feedback_scale=0.01, local_objective="ff"
            )
        )
    system = compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device="cpu")),
        geometry=_geometry(depth)(),
        dynamics=InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
        credit=credit_obj,
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
                SystemState(x=batch_x), system.geometry, system.substrate, None
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
    train_data = [
        (xb.view(xb.size(0), -1), yb)
        for xb, yb in islice(task.get_dataloader("train"), BATCH_CAP)
    ]
    test_batches = [
        (xb.view(xb.size(0), -1), yb)
        for xb, yb in task.get_dataloader("test")
        if xb.size(0) == 32
    ]

    cells = [
        ("bp", "adam", 8),
        ("bp", "muon", 8),
        ("bp", "ortho_adam", 8),
        ("bp", "adam", 16),
        ("bp", "muon", 16),
        ("bp", "ortho_adam", 16),
        ("ff", "muon", 4),
        ("ff", "adam", 4),
        ("ff", "ortho_adam", 4),
    ]
    for credit, update, depth in cells:
        accs = [_run(credit, update, depth, s, train_data, test_batches) for s in SEEDS]
        print(
            f"{credit:>2} x {update:>10} d{depth:>2} w128: {np.mean(accs):.3f} "
            f"± {np.std(accs):.3f} {[round(a, 3) for a in accs]}",
            flush=True,
        )
