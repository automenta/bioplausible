"""D2 — One trainer, every credit rule.

The same coordinate — Digital, Recurrent, EnergyMinimization, Null,
Euclidean — is trained three times through byte-identical ``SystemTrainer``
wiring; the only difference between arms is the single credit-assignment
constructor argument. All three credit rules (gradient / ThermodynamicContrast
/ RandomProjections) demonstrably learn. The comparison is one line.

Demonstrated regime (pinned 2026-09-02): MNIST quick-mode, 1 epoch, hidden
``(32,)``, ``EnergyMinimization(max_steps=3, beta=0.5)``, Euclidean step 0.1
-> accuracy ≈ 0.96 / 0.92 / 0.38 (chance 0.1).
"""

import torch

from computronium import (
    BackpropCredit,
    DigitalSubstrate,
    EnergyMinimizationDynamics,
    EuclideanUpdate,
    GeometryConfig,
    NullPlasticity,
    ParameterUpdateConfig,
    RandomProjectionsCredit,
    RecurrentGeometry,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemTrainer,
    SystemTrainerConfig,
    ThermodynamicContrast,
    compose_joint_system,
    create_task,
)

CREDIT_ARMS = (
    ("gradient", BackpropCredit()),
    ("thermodynamic_contrast", ThermodynamicContrast()),
    ("random_projections", RandomProjectionsCredit()),
)


def _flatten(loader):
    for x, y in loader:
        yield x.view(x.size(0), -1), y


def test_demo_swap_credit(emit_run_record) -> None:
    task = create_task("mnist", device="cpu", quick_mode=True)
    task.setup()
    train_loader = task.get_dataloader("train")
    config = SystemTrainerConfig(max_epochs=1, device="cpu", seed=42)

    record: dict = {"arms": {}}
    for name, credit in CREDIT_ARMS:
        torch.manual_seed(0)
        system = compose_joint_system(
            substrate=DigitalSubstrate(SubstrateConfig.digital(device="cpu")),
            geometry=RecurrentGeometry(
                GeometryConfig.recurrent(
                    input_dim=784, output_dim=10, hidden_dims=(32,)
                )
            ),
            dynamics=EnergyMinimizationDynamics(
                StateDynamicsConfig.energy_minimization(max_steps=3, beta=0.5)
            ),
            plasticity=NullPlasticity(),
            credit=credit,  # the one swapped argument
            update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.1)),
        )
        metrics = SystemTrainer(
            system=system, config=config, train_data=_flatten(train_loader)
        ).fit()[-1]
        print(f"{name}: {metrics['train_acc']:.1%}")
        record["arms"][name] = {"train_acc": metrics["train_acc"]}
        assert metrics["train_acc"] > 0.25, f"{name} must learn above 2.5x chance"

    emit_run_record("D2", "swap_credit", record)
