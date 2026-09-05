"""F1 audit (TODO11 plan item 2): does ePC close the sPC depth wall?

F1's depth wall is sPC-specific — under the layered settle the thermo-
contrast credit reaches only the last weight matrix (hidden norms exactly
0.00). ePC (Goemaere et al., D12) was written for exactly this pathology:
its error reparameterization lets the nudged signal reach every hidden
layer at 1/3 the settle budget.

Arms: ePC at depths 2/8/20 (sPC at 2/8 for reference), same F1 regime
(width 32, 60 MNIST quick batches, lr 0.2, beta 0.5, ePC budget 5 vs sPC
15 = the 1/3 D12 budget regime). Measures: per-layer credit norms (does
the signal reach layer 1?) and train accuracy (does ePC learn at depth
8-20 where sPC walls?).

Run: uv run python scripts/probes/f1_epc_depth.py
"""

from itertools import islice

import torch

from computronium import (
    CreditAssignmentConfig,
    DigitalSubstrate,
    ErrorPredictiveCodingDynamics,
    EuclideanUpdate,
    FeedforwardGeometry,
    GeometryConfig,
    ParameterUpdateConfig,
    PredictiveSettlingDynamics,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemState,
    SystemTrainer,
    SystemTrainerConfig,
    ThermodynamicContrast,
    compose_system,
    create_task,
)
from computronium.ontology.credit import Phase

WIDTH = 32
DEPTHS = (2, 8, 20)
BATCH_CAP = 60
LR = 0.2
BETA = 0.5


def _flatten(loader, cap):
    for x, y in islice(loader, cap):
        yield x.view(x.size(0), -1), y


def _credit_norms(system, substrate, x, y) -> list[float]:
    free = system.dynamics.settle(
        SystemState(x=x), system.geometry, substrate, target=None
    )
    nudged = system.dynamics.settle(
        SystemState(x=x), system.geometry, substrate, target=y
    )
    grads = system.credit.compute_pseudo_gradient(
        {Phase.FREE: free, Phase.NUDGED: nudged}, None, system.geometry
    )
    return [g.norm().item() for g in grads]


def main() -> None:
    task = create_task("mnist", device="cpu", quick_mode=True, num_workers=0)
    task.setup()
    torch.manual_seed(0)  # seed BEFORE the loader draw (D8 trap)
    train_data = list(_flatten(task.get_dataloader("train"), BATCH_CAP))
    substrate = DigitalSubstrate(SubstrateConfig.digital(device="cpu"))
    config = SystemTrainerConfig(max_epochs=1, device="cpu", seed=42)
    x, y = train_data[0]

    for depth in DEPTHS:
        for name, dynamics, settle_budget in (
            (
                "epc",
                lambda: ErrorPredictiveCodingDynamics(
                    StateDynamicsConfig.error_predictive_coding(
                        max_steps=5, step_size=0.5, beta=BETA
                    )
                ),
                5,
            ),
            (
                "spc",
                lambda: PredictiveSettlingDynamics(
                    StateDynamicsConfig.predictive_settling(max_steps=15)
                ),
                15,
            ),
        ):
            if name == "spc" and depth == 20:
                continue  # sPC wall at 8 already measured in F1
            torch.manual_seed(0)
            geometry = FeedforwardGeometry(
                GeometryConfig.feedforward(
                    input_dim=784, output_dim=10, hidden_dims=(WIDTH,) * depth
                )
            )
            system = compose_system(
                substrate=substrate,
                geometry=geometry,
                dynamics=dynamics(),
                credit=ThermodynamicContrast(
                    CreditAssignmentConfig.thermodynamic_contrast(beta=BETA)
                ),
                update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=LR)),
            )
            acc = SystemTrainer(
                system=system, config=config, train_data=train_data
            ).fit()[-1]["train_acc"]
            norms = _credit_norms(system, substrate, x, y)
            hidden_reach = min(norms[:-1]) if len(norms) > 1 else float("nan")
            print(
                f"depth {depth:>2} {name}: acc {acc:.3f}  "
                f"settle_budget {settle_budget}  "
                f"credit norms {[f'{n:.2e}' for n in norms]}  "
                f"min-hidden {hidden_reach:.2e}",
                flush=True,
            )


if __name__ == "__main__":
    main()
