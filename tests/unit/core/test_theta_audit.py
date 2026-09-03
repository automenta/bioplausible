"""PR-2 θ-invariance audit harness: exact-diff as a library feature.

Control arm proves the harness isn't vacuous (training moves θ and the
audit says so); the frozen arm proves a genuinely θ-invariant episode
reports clean — the same instrument D5 demonstrates on Z3 machinery.
"""

import pytest
import torch

from computronium import (
    BackpropCredit,
    DigitalSubstrate,
    EuclideanUpdate,
    FeedforwardGeometry,
    GeometryConfig,
    InstantaneousDynamics,
    NullPlasticity,
    ParameterUpdateConfig,
    SubstrateConfig,
    SystemTrainerConfig,
    compose_joint_system,
    theta_audit,
)
from computronium.core.system_trainer import SystemTrainer


def _tiny_system():
    torch.manual_seed(0)
    return compose_joint_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device="cpu")),
        geometry=FeedforwardGeometry(
            GeometryConfig.feedforward(input_dim=8, output_dim=4, hidden_dims=(8,))
        ),
        dynamics=InstantaneousDynamics(),
        plasticity=NullPlasticity(),
        credit=BackpropCredit(),
        update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.1)),
    )


def test_audit_flags_moved_theta() -> None:
    system = _tiny_system()
    with theta_audit(system, label="training", seed=42) as audit:
        x = torch.randn(16, 8)
        y = torch.randint(0, 4, (16,))
        SystemTrainer(
            system=system,
            config=SystemTrainerConfig(max_epochs=1, device="cpu", seed=42),
            train_data=[(x, y)],
        ).fit()
    report = audit.report
    assert report.theta_sha256_before != report.theta_sha256_after
    assert report.moved, "training must move θ for this control arm"
    with pytest.raises(AssertionError, match="training"):
        report.assert_invariant()


def test_audit_passes_free_episode() -> None:
    system = _tiny_system()
    with theta_audit(system, label="inference", seed=42) as audit, torch.no_grad():
        system.forward(torch.randn(16, 8))
    report = audit.report
    assert report.invariant
    report.assert_invariant()
    assert report.seed == 42


def test_audit_accepts_plain_mapping() -> None:
    params = {"w": torch.randn(3, 3)}
    with theta_audit(params, label="mapping") as audit:
        params["w"].add_(1.0)
    # the tensor was mutated in place, so the audit must flag it by name
    assert audit.report.moved == ("w",)
    assert not audit.report.invariant
