"""Regression tests for Plasticity Correctness (Phase 3.6.3).

These tests lock in the plasticity correctness audit checks to prevent regressions.
"""

from __future__ import annotations

import pytest
import torch

from computronium.core.continual.system import ContinualJointSystem
from computronium.core.joint.transition import NullPlasticity
from computronium.core.plasticity.fast_weights import FastWeightPlasticity
from computronium.core.plasticity.routing import RoutingPlasticity
from computronium.core.plasticity.rule_state import RuleStatePlasticity
from computronium.core.system_trainer import compose_joint_system
from computronium.ontology import (
    BackpropCredit,
    CreditAssignmentConfig,
    DigitalSubstrate,
    EuclideanUpdate,
    FeedforwardGeometry,
    GeometryConfig,
    InstantaneousDynamics,
    ParameterUpdateConfig,
    StateDynamicsConfig,
    SubstrateConfig,
)
from computronium.state import CompositeState

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def joint_system(device):
    """Create a joint system with fast weight plasticity."""
    joint = compose_joint_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device=str(device))),
        geometry=FeedforwardGeometry(
            GeometryConfig.feedforward(
                input_dim=784, output_dim=10, hidden_dims=(256, 128)
            )
        ),
        dynamics=InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
        plasticity=FastWeightPlasticity(
            fast_weight_dim=512, decay=0.9, learning_rate=0.1
        ),
        credit=BackpropCredit(CreditAssignmentConfig.thermodynamic_contrast()),
        update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.01)),
    )
    continual = ContinualJointSystem.from_joint_system(joint)
    continual = continual.to(device)
    return continual


@pytest.fixture
def context(joint_system):
    return joint_system.context


# ============================================================
# FastWeightPlasticity Tests
# ============================================================


class TestFastWeightPlasticity:
    """Tests for FastWeightPlasticity correctness."""

    def test_initial_psi_returns_correct_shape(self, context, device):
        """initial_psi returns fast_weights with correct shape."""
        plasticity = FastWeightPlasticity(
            fast_weight_dim=512, decay=0.9, learning_rate=0.1
        )
        plasticity = plasticity.to(device)

        psi = plasticity.initial_psi(context, batch_size=4)

        assert "fast_weights" in psi
        assert psi["fast_weights"].shape == (4, 512)
        assert psi["fast_weights"].device.type == device.type

    def test_projection_matrix_fixed_per_outer_dim(self, device):
        """Projection matrix is deterministic and fixed per outer_dim."""
        plasticity = FastWeightPlasticity(
            fast_weight_dim=512, decay=0.9, learning_rate=0.1
        )
        plasticity = plasticity.to(device)

        outer_dim = 7840
        proj1 = plasticity._get_proj_matrix(outer_dim, device)
        proj2 = plasticity._get_proj_matrix(outer_dim, device)

        # Same outer_dim -> same matrix
        assert torch.allclose(proj1, proj2)

        # Different outer_dim -> different matrix (different shape)
        proj3 = plasticity._get_proj_matrix(outer_dim + 1, device)
        assert proj3.shape != proj1.shape

    def test_projection_matrix_shape(self, device):
        """Projection matrix has correct shape [fast_weight_dim, outer_dim]."""
        plasticity = FastWeightPlasticity(
            fast_weight_dim=512, decay=0.9, learning_rate=0.1
        )
        plasticity = plasticity.to(device)

        outer_dim = 7840
        proj = plasticity._get_proj_matrix(outer_dim, device)

        assert proj.shape == (512, outer_dim)

    def test_decay_property_zero_activity(self, context, device):
        """With zero activity, fast weights decay exponentially."""
        plasticity = FastWeightPlasticity(
            fast_weight_dim=512, decay=0.9, learning_rate=0.1
        )
        plasticity = plasticity.to(device)

        batch_size = 4
        psi = plasticity.initial_psi(context, batch_size=batch_size)
        psi["fast_weights"] = torch.randn(batch_size, 512, device=device)
        initial_norm = psi["fast_weights"].norm(dim=1).mean().item()

        # CompositeState with NO activity
        z = CompositeState(activity={}, plastic={}, substrate={})

        # Step N times
        N = 10
        for _ in range(N):
            psi = plasticity.step(psi, z, context)

        final_norm = psi["fast_weights"].norm(dim=1).mean().item()
        expected_norm = initial_norm * (0.9**N)
        relative_error = abs(final_norm - expected_norm) / expected_norm

        assert relative_error <= 1e-6

    def test_step_updates_fast_weights(self, context, device):
        """Step updates fast weights with decay + Hebbian update."""
        plasticity = FastWeightPlasticity(
            fast_weight_dim=512, decay=0.9, learning_rate=0.1
        )
        plasticity = plasticity.to(device)

        batch_size = 4
        psi = plasticity.initial_psi(context, batch_size=batch_size)

        z = CompositeState(
            activity={
                "x": torch.randn(batch_size, 784, device=device),
                "y": torch.randint(0, 10, (batch_size,), device=device),
            },
            plastic={},
            substrate={},
        )

        psi_after = plasticity.step(psi, z, context)

        # Fast weights should change
        diff = (psi_after["fast_weights"] - psi["fast_weights"]).abs().mean().item()
        assert diff > 1e-6

    def test_to_device_moves_projection_matrices(self, device):
        """Calling .to(device) moves projection matrices to device."""
        device_cuda = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        plasticity = FastWeightPlasticity(
            fast_weight_dim=512, decay=0.9, learning_rate=0.1
        )
        _ = plasticity._get_proj_matrix(7840, device)
        plasticity = plasticity.to(device_cuda)

        for v in plasticity._proj_matrices.values():
            assert v.device.type == device_cuda.type

    def test_forward_modulation_changes_output(self, joint_system, device):
        """Fast weight modulation changes model output."""
        plasticity = joint_system.plasticity
        context = joint_system.context
        batch_size = 4

        # Create psi with non-zero fast weights
        psi = plasticity.initial_psi(context, batch_size=batch_size)

        z = CompositeState(
            activity={
                "x": torch.randn(batch_size, 784, device=device),
                "y": torch.randint(0, 10, (batch_size,), device=device),
            },
            plastic={},
            substrate={},
        )

        psi_after = plasticity.step(psi, z, context)

        x = torch.randn(batch_size, 784, device=device)

        # Forward without psi
        joint_system._psi = None
        output_without = joint_system.forward(x)

        # Forward with psi
        joint_system._psi = psi_after
        output_with = joint_system.forward(x)

        output_diff = (output_with - output_without).abs().max().item()
        assert output_diff > 1e-4


# ============================================================
# NullPlasticity Tests
# ============================================================


class TestNullPlasticity:
    """Tests for NullPlasticity correctness."""

    def test_initial_psi_empty(self, context):
        """initial_psi returns empty dict."""
        plasticity = NullPlasticity()
        psi = plasticity.initial_psi(context, batch_size=4)
        assert psi == {}

    def test_step_returns_empty(self, context):
        """step returns empty dict."""
        plasticity = NullPlasticity()
        psi = plasticity.initial_psi(context, batch_size=4)
        z = CompositeState(activity={}, plastic={}, substrate={})

        psi_after = plasticity.step(psi, z, context)
        assert psi_after == {}

        # Repeated steps still empty
        psi_after2 = plasticity.step(psi_after, z, context)
        assert psi_after2 == {}


# ============================================================
# RuleStatePlasticity Tests
# ============================================================


class TestRuleStatePlasticity:
    """Tests for RuleStatePlasticity correctness."""

    def test_freeze_unfreeze_theta(self, device):
        """freeze_theta and unfreeze_theta control requires_grad."""
        plasticity = RuleStatePlasticity(
            num_operators=8, operator_dim=64, device=device
        )

        # Initially unfrozen (trainable)
        assert not plasticity.verify_theta_frozen()

        # Freeze
        plasticity.freeze_theta()
        assert plasticity.verify_theta_frozen()

        # Unfreeze
        plasticity.unfreeze_theta()
        assert not plasticity.verify_theta_frozen()

    def test_step_updates_operator_logits(self, device):
        """step updates operator_logits and controller_state."""
        # Create context with correct input dim for RuleStatePlasticity
        joint = compose_joint_system(
            substrate=DigitalSubstrate(SubstrateConfig.digital(device=str(device))),
            geometry=FeedforwardGeometry(
                GeometryConfig.feedforward(
                    input_dim=64, output_dim=2, hidden_dims=(128,)
                )
            ),
            dynamics=InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
            plasticity=NullPlasticity(),
            credit=BackpropCredit(CreditAssignmentConfig.thermodynamic_contrast()),
            update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.01)),
        )
        context = joint.context

        plasticity = RuleStatePlasticity(
            num_operators=8, operator_dim=64, controller_hidden=128, device=device
        )

        batch_size = 4
        psi = plasticity.initial_psi(context, batch_size=batch_size)

        z = CompositeState(
            activity={"x": torch.randn(batch_size, 64, device=device)},
            plastic={},
            substrate={},
        )

        psi_after = plasticity.step(psi, z, context)

        # operator_logits should change
        assert not torch.allclose(psi_after["operator_logits"], psi["operator_logits"])
        # controller_state should change
        assert not torch.allclose(
            psi_after["controller_state"], psi["controller_state"]
        )

    def test_device_management(self, device):
        """RuleStatePlasticity device is set at construction."""
        device_cuda = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        plasticity = RuleStatePlasticity(
            num_operators=8, operator_dim=64, device=device_cuda
        )

        assert plasticity._operator_embeddings.device.type == device_cuda.type
        for p in plasticity._controller.parameters():
            assert p.device.type == device_cuda.type


# ============================================================
# RoutingPlasticity Tests
# ============================================================


class TestRoutingPlasticity:
    """Tests for RoutingPlasticity correctness."""

    def test_initial_psi(self, context, device):
        """initial_psi returns gate_logits and active_routes."""
        plasticity = RoutingPlasticity(gate_dim=64, decay=0.99, learning_rate=0.01)

        psi = plasticity.initial_psi(context, batch_size=4)

        assert "gate_logits" in psi
        assert "active_routes" in psi
        assert psi["gate_logits"].shape == (4, 64)
        assert psi["active_routes"].shape == (4, 64)

    def test_step_updates_gate_logits(self, context, device):
        """step updates gate_logits and computes active_routes."""
        plasticity = RoutingPlasticity(gate_dim=64, decay=0.99, learning_rate=0.01)

        batch_size = 4
        psi = plasticity.initial_psi(context, batch_size=batch_size)

        z = CompositeState(
            activity={"x": torch.randn(batch_size, 784, device=device)},
            plastic={},
            substrate={},
        )

        psi_after = plasticity.step(psi, z, context)

        # gate_logits should change
        assert not torch.allclose(psi_after["gate_logits"], psi["gate_logits"])
        # active_routes should be computed
        assert psi_after["active_routes"].shape == (batch_size, 64)


# ============================================================
# Device Consistency Tests (for all plasticity types)
# ============================================================


class TestPlasticityDeviceConsistency:
    """Test device consistency for all plasticity types."""

    @pytest.mark.parametrize(
        "plasticity_factory",
        [
            lambda d: FastWeightPlasticity(
                fast_weight_dim=512, decay=0.9, learning_rate=0.1
            ).to(d),
            lambda d: RoutingPlasticity(gate_dim=64, decay=0.99, learning_rate=0.01),  # ruff: ignore[unused-lambda-argument]
            lambda d: RuleStatePlasticity(num_operators=8, operator_dim=64, device=d),
            lambda d: NullPlasticity(),  # ruff: ignore[unused-lambda-argument]
        ],
    )
    def test_plasticity_device_consistency(self, plasticity_factory, device):
        """All plasticity types handle device correctly."""
        device_cuda = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        plasticity = plasticity_factory(device)

        # For FastWeightPlasticity, create proj matrix then move
        if isinstance(plasticity, FastWeightPlasticity):
            _ = plasticity._get_proj_matrix(7840, device)
            plasticity = plasticity.to(device_cuda)
            for v in plasticity._proj_matrices.values():
                assert v.device.type == device_cuda.type

        # For RuleStatePlasticity, construct on target device
        elif isinstance(plasticity, RuleStatePlasticity):
            plasticity = RuleStatePlasticity(
                num_operators=8, operator_dim=64, device=device_cuda
            )
            assert plasticity._operator_embeddings.device.type == device_cuda.type
            for p in plasticity._controller.parameters():
                assert p.device.type == device_cuda.type

        # NullPlasticity has no internal state
        elif isinstance(plasticity, NullPlasticity) or isinstance(  # ruff: ignore[duplicate-isinstance-call]
            plasticity, RoutingPlasticity
        ):
            pass  # Nothing to check


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
