"""Learning-algorithm hunt, round 2: the unexplored credit x optimizer cells.

Queued from TODO11's D16 landing: the optimizer-family axis is now a
first-class coordinate and the unexplored cells are the next pull order:

1. pepita x Adam — does Adam rescue realized PEPITA's slow demo-budget
   learning (D13: 0.226 Muon / 0.106 Euclid, 1 epoch)?
2. spectral x Adam — the spectral rule under the Adam family.
3. natural-gradient lr sweep — at chance on mlp/graph/lattice at lr 0.1
   (D16 OPEN regime question): lr/scaling defect or a real boundary?
   Sweep lr and readout-scale before any verdict (R11.5.5a).

Same harness as performance_hunt4: mnist quick, 150 batches, 1 epoch,
seeds 0-2, TEST accuracy, cpu.
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
    NaturalGradientUpdate,
    ParameterUpdateConfig,
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


def _credit(name: str):
    if name == "bp":
        return BackpropCredit()
    objective = "ff" if name == "ff" else "pepita"
    return LocalGoodnessCredit(
        CreditAssignmentConfig.local_goodness(
            feedback_scale=0.01, local_objective=objective
        )
    )


def _updates() -> dict[str, Callable]:
    return {
        "euclid.2": lambda: EuclideanUpdate(
            ParameterUpdateConfig.euclidean(step_size=0.2)
        ),
        "muon": lambda: RiemannianOrthogonalUpdate(
            ParameterUpdateConfig.riemannian_orthogonal(step_size=0.02, momentum=0.9)
        ),
        "adam": lambda: AdamUpdate(ParameterUpdateConfig.adam(step_size=1e-3)),
        "adam3": lambda: AdamUpdate(ParameterUpdateConfig.adam(step_size=3e-3)),
        "spectral": lambda: SpectralConstrainedUpdate(
            ParameterUpdateConfig.spectral_constrained(step_size=0.1)
        ),
        "nat.01": lambda: NaturalGradientUpdate(
            ParameterUpdateConfig.natural_gradient(step_size=0.01)
        ),
        "nat.1": lambda: NaturalGradientUpdate(
            ParameterUpdateConfig.natural_gradient(step_size=0.1)
        ),
        "nat1": lambda: NaturalGradientUpdate(
            ParameterUpdateConfig.natural_gradient(step_size=1.0)
        ),
    }


def _run(credit: str, update: str, seed: int, train_data, test_batches) -> float:
    torch.manual_seed(seed)
    system = compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device="cpu")),
        geometry=FeedforwardGeometry(
            GeometryConfig.feedforward(
                input_dim=784, output_dim=10, hidden_dims=(64, 64)
            )
        ),
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
        ("pepita", "euclid.2"),
        ("pepita", "muon"),
        ("pepita", "adam"),
        ("pepita", "adam3"),
        ("spectral", "spectral"),
        ("spectral", "adam"),
        ("bp", "nat.01"),
        ("bp", "nat.1"),
        ("bp", "nat1"),
    ]
    for credit, update in cells:
        accs = [_run(credit, update, s, train_data, test_batches) for s in SEEDS]
        print(
            f"{credit:>7} x {update:>9}: {np.mean(accs):.3f} ± {np.std(accs):.3f} {accs}",
            flush=True,
        )
