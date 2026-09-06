"""P1(a) — Fundamental-Research Focus (TODO11): does the FF hybrid
(readout_error=True: FF layer-local goodness + CE on the free logits) beat
pure FF on MNIST at the D13 demo regime?

The hybrid is the strongest unique LM result (6.74 vs bp 5.82 ppl at 2.5 min
on the transformer). If ff_hybrid×Muon beats D13's flagship FF×Muon 0.838
here, the hybrid upgrades every local-credit claim in the D-table.

Grid: seeds 0-4 x {ff, ff_hybrid} x {euclidean lr 0.2, muon lr 0.02},
demo regime (width 32, depth 2, 150 MNIST quick batches, one epoch) —
byte-identical to `d13_multiseed.py`'s ff arms.

Run: uv run python scripts/probes/p1a_ff_hybrid_mnist.py
"""

from itertools import islice
from statistics import mean, stdev

import torch

from computronium import (
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
    "ff/euclidean": lambda: (
        _local(readout_error=False),
        EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.2)),
    ),
    "ff/muon": lambda: (
        _local(readout_error=False),
        RiemannianOrthogonalUpdate(
            ParameterUpdateConfig.riemannian_orthogonal(step_size=0.02, momentum=0.9)
        ),
    ),
    "ff_hybrid/euclidean": lambda: (
        _local(readout_error=True),
        EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.2)),
    ),
    "ff_hybrid/muon": lambda: (
        _local(readout_error=True),
        RiemannianOrthogonalUpdate(
            ParameterUpdateConfig.riemannian_orthogonal(step_size=0.02, momentum=0.9)
        ),
    ),
}


def _local(readout_error: bool):
    return LocalGoodnessCredit(
        CreditAssignmentConfig.local_goodness(
            feedback_scale=0.01, local_objective="ff", readout_error=readout_error
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
            print(f"seed {seed} {arm:>20}: {acc:.3f}", flush=True)

    print("\n=== summary (mean +/- stdev over 5 seeds) ===")
    for arm, accs in results.items():
        print(
            f"{arm:>20}: {mean(accs):.3f} +/- {stdev(accs):.3f}  "
            f"{[round(a, 3) for a in accs]}"
        )
    hybrid_muon = results["ff_hybrid/muon"]
    ff_muon = results["ff/muon"]
    lifts = [h - f for h, f in zip(hybrid_muon, ff_muon)]
    print(
        f"ff_hybrid/muon lift over ff/muon per seed: "
        f"{[round(x, 3) for x in lifts]} (min {min(lifts):.3f})"
    )


if __name__ == "__main__":
    main()
