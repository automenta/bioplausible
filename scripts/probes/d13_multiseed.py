"""D13 multi-seed audit (TODO11 plan item 1): does the local-credit x Muon
lift survive across seeds?

Grid: seeds 0-4 x {bp, ff, pepita} x {euclidean lr 0.2, muon lr 0.02},
demo regime (width 32, depth 2, 150 MNIST quick batches, one epoch).
D13's headline (FF/PEPITA x Muon ~ 0.85 vs ~ 0.26 on Euclidean) is
single-seed; the muPC audit showed seed noise can fake large effects, so
the 0.85 number is quoted nowhere until this probe says whether it holds.

Run: uv run python scripts/probes/d13_multiseed.py
"""

from itertools import islice
from statistics import mean, stdev

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
DEPTH = 2
BATCH_CAP = 150
SEEDS = range(5)
ARMS = {
    "bp/euclidean": lambda: (
        BackpropCredit(),
        EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.2)),
    ),
    "bp/muon": lambda: (
        BackpropCredit(),
        RiemannianOrthogonalUpdate(
            ParameterUpdateConfig.riemannian_orthogonal(step_size=0.02, momentum=0.9)
        ),
    ),
    "ff/euclidean": lambda: (
        _local("ff"),
        EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.2)),
    ),
    "ff/muon": lambda: (
        _local("ff"),
        RiemannianOrthogonalUpdate(
            ParameterUpdateConfig.riemannian_orthogonal(step_size=0.02, momentum=0.9)
        ),
    ),
    "pepita/euclidean": lambda: (
        _local("pepita"),
        EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.2)),
    ),
    "pepita/muon": lambda: (
        _local("pepita"),
        RiemannianOrthogonalUpdate(
            ParameterUpdateConfig.riemannian_orthogonal(step_size=0.02, momentum=0.9)
        ),
    ),
}


def _local(objective: str):
    return LocalGoodnessCredit(
        CreditAssignmentConfig.local_goodness(
            feedback_scale=0.01, local_objective=objective
        )
    )


def _flatten(loader, cap):
    for x, y in islice(loader, cap):
        yield x.view(x.size(0), -1), y


def main() -> None:
    task = create_task("mnist", device="cpu", quick_mode=True, num_workers=0)
    task.setup()
    config = SystemTrainerConfig(max_epochs=1, device="cpu", seed=42)
    torch.manual_seed(0)  # seed BEFORE the loader draw (D8 trap)
    train_data = list(_flatten(task.get_dataloader("train"), BATCH_CAP))

    results: dict[str, list[float]] = {name: [] for name in ARMS}
    for seed in SEEDS:
        for arm, make in ARMS.items():
            torch.manual_seed(seed)
            geometry = FeedforwardGeometry(
                GeometryConfig.feedforward(
                    input_dim=784, output_dim=10, hidden_dims=(WIDTH,) * DEPTH
                )
            )
            credit, update = make()
            system = compose_system(
                substrate=DigitalSubstrate(SubstrateConfig.digital(device="cpu")),
                geometry=geometry,
                dynamics=InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
                credit=credit,
                update=update,
            )
            acc = SystemTrainer(
                system=system, config=config, train_data=train_data
            ).fit()[-1]["train_acc"]
            results[arm].append(acc)
            print(f"seed {seed} {arm:>18}: {acc:.3f}", flush=True)

    print("\n=== summary (mean +/- stdev over 5 seeds) ===")
    for arm, accs in results.items():
        print(
            f"{arm:>18}: {mean(accs):.3f} +/- {stdev(accs):.3f}  "
            f"{[round(a, 3) for a in accs]}"
        )
    for local in ("ff", "pepita"):
        muon = results[f"{local}/muon"]
        euclid = results[f"{local}/euclidean"]
        lifts = [m - e for m, e in zip(muon, euclid)]
        print(
            f"{local} muon-lift per seed: "
            f"{[round(x, 3) for x in lifts]} (min {min(lifts):.3f})"
        )


if __name__ == "__main__":
    main()
