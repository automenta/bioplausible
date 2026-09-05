"""Local-credit × Muon probe (user hypothesis: FF/PEPITA with Muon-class
orthogonalized updates — an untested combination; Muon is credit-agnostic,
so the composition is native here).

Arms: bp / ff / pepita credit × euclidean / muon update, one swapped
argument, feedforward width 32, MNIST quick, 150 batches. Instrument note:
muon's effective step differs (orthogonalized gradient, fixed spectral
norm), so each arm runs a small lr grid; compare best-vs-best and same-lr.

Run: uv run python scripts/probes/local_credit_muon.py
"""

from itertools import islice

import torch

from computronium import (
    BackpropCredit,
    CreditAssignmentConfig,
    DigitalSubstrate,
    EuclideanUpdate,
    FeedforwardGeometry,
    GeometryConfig,
    InstantaneousDynamics,
    LocalGoodnessCredit,
    ParameterUpdateConfig,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemTrainer,
    SystemTrainerConfig,
    compose_system,
    create_task,
)
from computronium.ontology.update import RiemannianOrthogonalUpdate

WIDTH = 32
DEPTHS = (2, 8)
BATCH_CAP = 150
LRS = {"euclidean": (0.2,), "muon": (0.02, 0.1, 0.5)}


def _flatten(loader, cap):
    for x, y in islice(loader, cap):
        yield x.view(x.size(0), -1), y


def _credit(name: str):
    if name == "bp":
        return BackpropCredit()
    return LocalGoodnessCredit(
        CreditAssignmentConfig.local_goodness(
            feedback_scale=0.01,
            local_objective="pepita" if name == "pepita" else "ff",
        )
    )


def main() -> None:
    task = create_task("mnist", device="cpu", quick_mode=True, num_workers=0)
    task.setup()
    loader = task.get_dataloader("train")
    substrate = DigitalSubstrate(SubstrateConfig.digital(device="cpu"))
    config = SystemTrainerConfig(max_epochs=1, device="cpu", seed=42)
    torch.manual_seed(0)
    train_data = list(_flatten(loader, BATCH_CAP))

    for depth in DEPTHS:
        for credit_name in ("bp", "ff", "pepita"):
            for update_name, lrs in LRS.items():
                for lr in lrs:
                    torch.manual_seed(0)
                    geometry = FeedforwardGeometry(
                        GeometryConfig.feedforward(
                            input_dim=784,
                            output_dim=10,
                            hidden_dims=(WIDTH,) * depth,
                        )
                    )
                    if update_name == "muon":
                        update = RiemannianOrthogonalUpdate(
                            ParameterUpdateConfig.riemannian_orthogonal(step_size=lr)
                        )
                    else:
                        update = EuclideanUpdate(
                            ParameterUpdateConfig.euclidean(step_size=lr, momentum=0.0)
                        )
                    system = compose_system(
                        substrate=substrate,
                        geometry=geometry,
                        dynamics=InstantaneousDynamics(
                            StateDynamicsConfig.instantaneous()
                        ),
                        credit=_credit(credit_name),
                        update=update,
                    )
                    acc = SystemTrainer(
                        system=system, config=config, train_data=train_data
                    ).fit()[-1]["train_acc"]
                    print(
                        f"depth {depth} {credit_name:>6} {update_name:>9} "
                        f"lr {lr:<5}: acc {acc:.3f}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
