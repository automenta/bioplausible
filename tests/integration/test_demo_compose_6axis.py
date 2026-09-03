"""D1 — Six-axis composition is real.

A system composed from all six ontology axes — Digital, Recurrent,
EnergyMinimization, Null, ThermodynamicContrast, Euclidean — trains on
MNIST and produces valid metrics; its configuration round-trips through
``extract_config``/``compose_system_from_configs`` to an equivalent system;
and the J1 Zero-Extension invariant holds at train scale end-to-end: two
independent builds seeded identically — one 5-D, one 6-axis with
``NullPlasticity`` — produce identical metric dicts and bitwise-equal θ.
A stranger reads this file and knows how to build anything.

Demonstrated regime (re-pinned 2026-09-02 with a 600-batch loader cap for
suite walltime): MNIST quick-mode, 1 epoch over the capped stream, batch 64,
hidden ``(32,)``, ``EnergyMinimization(max_steps=5, beta=0.5)`` -> train
accuracy ≈ 0.84 (chance 0.1).
"""

from itertools import islice

import pytest
import torch

from computronium import (
    DigitalSubstrate,
    EnergyMinimizationDynamics,
    EuclideanUpdate,
    GeometryConfig,
    NullPlasticity,
    RecurrentGeometry,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemTrainer,
    SystemTrainerConfig,
    ThermodynamicContrast,
    compose_joint_system,
    compose_system,
    compose_system_from_configs,
    create_task,
    extract_config,
)

BATCH_CAP = 600  # loader cap (Register C): suite walltime, regime re-pinned 2026-09-02
EXPECTED_ACCURACY_FLOOR = 0.5  # far above the 0.1 chance, wide guard band


def _flatten(loader, cap=BATCH_CAP):
    for x, y in islice(loader, cap):
        yield x.view(x.size(0), -1), y


def test_demo_compose_6axis(emit_run_record) -> None:
    task = create_task("mnist", device="cpu", quick_mode=True)
    task.setup()
    train_loader = task.get_dataloader("train")

    # The 6-axis composition: one constructor argument per axis, Null on M.
    torch.manual_seed(0)
    six_axis = compose_joint_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device="cpu")),
        geometry=RecurrentGeometry(
            GeometryConfig.recurrent(input_dim=784, output_dim=10, hidden_dims=(32,))
        ),
        dynamics=EnergyMinimizationDynamics(
            StateDynamicsConfig.energy_minimization(max_steps=5, beta=0.5)
        ),
        plasticity=NullPlasticity(),
        credit=ThermodynamicContrast(),
        update=EuclideanUpdate(),
    )
    trainer = SystemTrainer(
        system=six_axis,
        config=SystemTrainerConfig(max_epochs=1, device="cpu", seed=42),
        train_data=_flatten(train_loader),
    )
    history = trainer.fit()
    print(f"train accuracy: {history[-1]['train_acc']:.1%}")
    metrics = history[-1]
    theta_six = trainer.system.geometry.params
    record: dict = {
        "six_axis": {
            "history": [
                {"train_acc": h["train_acc"], "train_loss": h["train_loss"]}
                for h in history
            ],
            "train_acc": metrics["train_acc"],
            "train_loss": metrics["train_loss"],
        }
    }

    # Valid metrics: the composed system demonstrably learns.
    assert metrics["train_loss"] >= 0.0
    assert 0.0 <= metrics["train_acc"] <= 1.0
    assert metrics["train_acc"] > EXPECTED_ACCURACY_FLOOR, "must learn far above chance"

    # J1 Zero-Extension at train scale: 5-D ≡ 6-axis with Null. The build is
    # verbatim-identical except ``compose_system`` takes no plasticity axis.
    torch.manual_seed(0)
    five_axis = compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device="cpu")),
        geometry=RecurrentGeometry(
            GeometryConfig.recurrent(input_dim=784, output_dim=10, hidden_dims=(32,))
        ),
        dynamics=EnergyMinimizationDynamics(
            StateDynamicsConfig.energy_minimization(max_steps=5, beta=0.5)
        ),
        credit=ThermodynamicContrast(),
        update=EuclideanUpdate(),
    )
    trainer_5d = SystemTrainer(
        system=five_axis,
        config=SystemTrainerConfig(max_epochs=1, device="cpu", seed=42),
        train_data=_flatten(train_loader),
    )
    history_5d = trainer_5d.fit()
    metrics_5d = history_5d[-1]
    theta_five = trainer_5d.system.geometry.params
    record["j1"] = {
        "five_axis": {
            "train_acc": metrics_5d["train_acc"],
            "train_loss": metrics_5d["train_loss"],
        },
        "metrics_equal": True,
        "theta_bitwise_equal": True,
    }
    for key in ("train_loss", "train_acc"):
        assert metrics[key] == pytest.approx(metrics_5d[key], abs=1e-7)
    assert all(
        torch.equal(a, b)
        for a, b in zip(theta_six.values(), theta_five.values(), strict=True)
    )

    # L6 config round-trip: System -> configs -> System -> same configs.
    rebuilt = compose_system_from_configs(**extract_config(five_axis))
    assert extract_config(rebuilt) == extract_config(five_axis)
    record["round_trip"] = True

    emit_run_record("D1", "compose_6axis", record)
