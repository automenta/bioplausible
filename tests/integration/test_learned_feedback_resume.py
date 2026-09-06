"""TODO12 B1 — TrainerSnapshot captures credit-internal state.

A learned-feedback PEPITA system's snapshot carries the learned B matrices
and step counter; resuming a fresh trainer restores them bitwise (the
carried-queue lesson: credit state must not be silently dropped on resume,
the A1 optimizer-state failure mode).
"""

import hashlib

import torch
from torch.utils.data import DataLoader, TensorDataset

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
)

_DIM_IN, _DIM_OUT, _HIDDEN = 16, 4, 8


def _loader() -> DataLoader:
    g = torch.Generator().manual_seed(7)
    x = torch.randn(64, _DIM_IN, generator=g)
    y = torch.randint(0, _DIM_OUT, (64,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True)


def _system():
    torch.manual_seed(0)
    return compose_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device="cpu")),
        geometry=FeedforwardGeometry(
            GeometryConfig.feedforward(
                input_dim=_DIM_IN, output_dim=_DIM_OUT, hidden_dims=(_HIDDEN,)
            )
        ),
        dynamics=InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
        credit=LocalGoodnessCredit(
            CreditAssignmentConfig.local_goodness(
                local_objective="pepita",
                learned_feedback=True,
                feedback_lr=0.5,
                feedback_update_every=1,
            )
        ),
        update=EuclideanUpdate(
            ParameterUpdateConfig.euclidean(step_size=0.05, momentum=0.0)
        ),
    )


def _config(max_epochs: int) -> SystemTrainerConfig:
    return SystemTrainerConfig(
        max_epochs=max_epochs, device="cpu", seed=42, resumable=True
    )


def _digest(system) -> str:
    h = hashlib.sha256()
    for name in sorted(system.geometry.params):
        h.update(name.encode())
        h.update(system.geometry.params[name].detach().numpy().tobytes())
    return h.hexdigest()


def test_snapshot_carries_learned_feedback_state() -> None:
    trainer = SystemTrainer(system=_system(), config=_config(1), train_data=_loader())
    trainer.fit()
    snap = trainer.snapshot()
    assert snap.credit_state, "learned-feedback credit state must be captured"
    assert "learned_feedback" in snap.credit_state
    assert snap.credit_state["step"]["counter"].item() > 0


def test_resume_restores_credit_state_bitwise() -> None:
    intact = SystemTrainer(system=_system(), config=_config(2), train_data=_loader())
    intact.fit()
    first_leg = SystemTrainer(system=_system(), config=_config(1), train_data=_loader())
    first_leg.fit()
    resumed = SystemTrainer.from_snapshot(
        system=_system(),
        config=_config(2),
        train_data=_loader(),
        snapshot=first_leg.snapshot(),
    )
    resumed.fit()
    assert _digest(resumed.system) == _digest(intact.system), (
        "resume must restore learned-B state bitwise"
    )
