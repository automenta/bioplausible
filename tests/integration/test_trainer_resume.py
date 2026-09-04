"""Resumable trainer parity (TODO11 R11.2.24).

An interrupted run — one epoch, snapshot, resume into a fresh trainer —
must finish bitwise identical to an uninterrupted two-epoch run: same
per-epoch metrics, same parameter bytes. Requires ``resumable=True`` so
every RNG draw (shuffle permutation, settle noise) is a pure function of
``(seed, epoch, batch)`` via ``fold_in``.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from computronium import (
    BackpropCredit,
    DigitalSubstrate,
    EnergyMinimizationDynamics,
    EuclideanUpdate,
    GeometryConfig,
    NullPlasticity,
    ParameterUpdateConfig,
    RecurrentGeometry,
    StateDynamicsConfig,
    SubstrateConfig,
    SystemTrainerConfig,
    compose_joint_system,
)
from computronium.core.system_trainer import JointSystem, SystemTrainer

_DIM_IN, _DIM_OUT, _HIDDEN = 16, 4, 8


def _loader() -> DataLoader:
    g = torch.Generator().manual_seed(7)
    x = torch.randn(64, _DIM_IN, generator=g)
    y = torch.randint(0, _DIM_OUT, (64,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True)


def _system() -> JointSystem:
    torch.manual_seed(0)
    return compose_joint_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device="cpu")),
        geometry=RecurrentGeometry(
            GeometryConfig.recurrent(
                input_dim=_DIM_IN, output_dim=_DIM_OUT, hidden_dims=(_HIDDEN,)
            )
        ),
        dynamics=EnergyMinimizationDynamics(
            StateDynamicsConfig.energy_minimization(max_steps=2, beta=0.5)
        ),
        plasticity=NullPlasticity(),
        credit=BackpropCredit(),
        update=EuclideanUpdate(
            ParameterUpdateConfig.euclidean(step_size=0.1, momentum=0.9)
        ),
    )


def _config(max_epochs: int) -> SystemTrainerConfig:
    return SystemTrainerConfig(
        max_epochs=max_epochs, device="cpu", seed=42, resumable=True
    )


def _digest(trainer: SystemTrainer) -> str:
    import hashlib

    h = hashlib.sha256()
    for name in sorted(trainer.system.geometry.params):
        h.update(name.encode())
        h.update(trainer.system.geometry.params[name].detach().numpy().tobytes())
    return h.hexdigest()


def test_interrupted_resume_is_bitwise_uninterrupted() -> None:
    intact = SystemTrainer(system=_system(), config=_config(2), train_data=_loader())
    intact_hist = intact.fit()

    first_leg = SystemTrainer(system=_system(), config=_config(1), train_data=_loader())
    first_leg.fit()
    snapshot = first_leg.snapshot()

    resumed = SystemTrainer.from_snapshot(
        system=_system(), config=_config(2), train_data=_loader(), snapshot=snapshot
    )
    resumed_hist = resumed.fit()

    assert resumed.current_epoch == intact.current_epoch == 2
    assert resumed.global_step == intact.global_step
    assert [dict(m) for m in resumed_hist] == [dict(m) for m in intact_hist]
    assert _digest(resumed) == _digest(intact)


def test_snapshot_round_trip_restores_optimizer_state() -> None:
    trainer = SystemTrainer(system=_system(), config=_config(1), train_data=_loader())
    trainer.fit()
    snapshot = trainer.snapshot()

    assert snapshot.epoch == 1
    assert snapshot.opt_state, "momentum=0.9 must populate optimizer state"
    restored = SystemTrainer.from_snapshot(
        system=_system(), config=_config(1), train_data=_loader(), snapshot=snapshot
    )
    restored_buffers = restored.system.update._momentum_buffers
    for name, t in snapshot.opt_state.items():
        assert torch.equal(restored_buffers[name], t)
