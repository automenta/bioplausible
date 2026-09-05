"""U-axis learning probe (D13 candidate — the update axis has zero live
learning demos; every demo uses Euclidean).

Arms: one coordinate (BP × feedforward, width 32, MNIST quick 60 batches),
one swapped `update` argument: euclidean, riemannian_orthogonal (Muon-class
QR orthogonalization), spectral_constrained (Lipschitz-bounded),
natural_gradient (Fisher-scaled). Depths 2 and 8 — orthogonalization's
claimed advantage is at depth.

Run: uv run python scripts/probes/uaxis_learning.py
"""

from itertools import islice

import torch

from computronium import (
    BackpropCredit,
    DigitalSubstrate,
    FeedforwardGeometry,
    GeometryConfig,
    InstantaneousDynamics,
    ParameterUpdateConfig,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemTrainer,
    SystemTrainerConfig,
    compose_system,
    create_task,
)
from computronium.ontology.update import (
    EuclideanUpdate,
    NaturalGradientUpdate,
    RiemannianOrthogonalUpdate,
    SpectralConstrainedUpdate,
)

WIDTH = 32
DEPTHS = (2, 8)
BATCH_CAP = 60

ARMS = (
    ("euclidean", EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.2))),
    (
        "muon",
        RiemannianOrthogonalUpdate(
            ParameterUpdateConfig.riemannian_orthogonal(step_size=0.1)
        ),
    ),
    (
        "spectral",
        SpectralConstrainedUpdate(
            ParameterUpdateConfig.spectral_constrained(step_size=0.2)
        ),
    ),
    (
        "natural",
        NaturalGradientUpdate(ParameterUpdateConfig.natural_gradient(step_size=0.2)),
    ),
)


def _flatten(loader, cap):
    for x, y in islice(loader, cap):
        yield x.view(x.size(0), -1), y


def main() -> None:
    task = create_task("mnist", device="cpu", quick_mode=True, num_workers=0)
    task.setup()
    loader = task.get_dataloader("train")
    substrate = DigitalSubstrate(SubstrateConfig.digital(device="cpu"))
    config = SystemTrainerConfig(max_epochs=1, device="cpu", seed=42)
    torch.manual_seed(0)
    train_data = list(_flatten(loader, BATCH_CAP))

    for depth in DEPTHS:
        for name, update in ARMS:
            torch.manual_seed(0)
            geometry = FeedforwardGeometry(
                GeometryConfig.feedforward(
                    input_dim=784, output_dim=10, hidden_dims=(WIDTH,) * depth
                )
            )
            system = compose_system(
                substrate=substrate,
                geometry=geometry,
                dynamics=InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
                credit=BackpropCredit(),
                update=update,
            )
            acc = SystemTrainer(
                system=system, config=config, train_data=train_data
            ).fit()[-1]["train_acc"]
            print(f"depth {depth:>2} {name:>10}: acc {acc:.3f}", flush=True)


if __name__ == "__main__":
    main()
