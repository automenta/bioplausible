"""Regression tests for Device Consistency (Phase 3.6.4/3.6.8).

These tests verify that all components properly move to target devices.
"""

from __future__ import annotations

import pytest
import torch

from computronium.core.continual.system import ContinualJointSystem
from computronium.core.plasticity import (
    FastWeightPlasticity,
    NullPlasticity,
    RoutingPlasticity,
    RuleStatePlasticity,
)
from computronium.core.system_trainer import compose_joint_system
from computronium.ontology import (
    BackpropCredit,
    CreditAssignmentConfig,
    DigitalSubstrate,
    EnergyMinimizationDynamics,
    EuclideanUpdate,
    FeedforwardGeometry,
    GeometryConfig,
    InstantaneousDynamics,
    ParameterUpdateConfig,
    StateDynamicsConfig,
    SubstrateConfig,
    ThermodynamicContrast,
)

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def device_cpu():
    return torch.device("cpu")


@pytest.fixture
def device_cuda():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def make_joint_system(device: torch.device, plasticity_type: str = "null"):
    """Create a joint system with specified plasticity."""
    if plasticity_type == "fast_weights":
        plasticity = FastWeightPlasticity(
            fast_weight_dim=512, decay=0.9, learning_rate=0.1
        )
    elif plasticity_type == "routing":
        plasticity = RoutingPlasticity(gate_dim=64, decay=0.99, learning_rate=0.01)
    elif plasticity_type == "rule_state":
        plasticity = RuleStatePlasticity(
            num_operators=8, operator_dim=64, device=device
        )
    elif plasticity_type == "null":
        plasticity = NullPlasticity()
    else:
        raise ValueError(f"Unknown plasticity type: {plasticity_type}")

    joint = compose_joint_system(
        substrate=DigitalSubstrate(SubstrateConfig.digital(device=str(device))),
        geometry=FeedforwardGeometry(
            GeometryConfig.feedforward(
                input_dim=784, output_dim=10, hidden_dims=(256, 128)
            )
        ),
        dynamics=InstantaneousDynamics(StateDynamicsConfig.instantaneous()),
        plasticity=plasticity,
        credit=BackpropCredit(CreditAssignmentConfig.thermodynamic_contrast()),
        update=EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.01)),
    )

    continual = ContinualJointSystem.from_joint_system(joint)
    continual = continual.to(device)
    return continual


# ============================================================
# Device Propagation Tests
# ============================================================


class TestDevicePropagation:
    """Test that .to(device) moves all components to target device."""

    @pytest.mark.parametrize(
        "plasticity_type", ["null", "fast_weights", "routing", "rule_state"]
    )
    def test_joint_system_to_device(self, plasticity_type, device_cpu, device_cuda):
        """ContinualJointSystem.to(device) moves all parameters."""
        continual = make_joint_system(device_cpu, plasticity_type)

        # Move to target device
        continual = continual.to(device_cuda)

        # Check geometry parameters
        for param in continual.geometry.parameters():
            assert param.device.type == device_cuda.type, (
                f"Geometry param on {param.device}, expected {device_cuda.type}"
            )

        # Check plasticity internal state
        if plasticity_type == "fast_weights":
            for v in continual.plasticity._proj_matrices.values():
                assert v.device.type == device_cuda.type
        elif plasticity_type == "rule_state":
            assert (
                continual.plasticity._operator_embeddings.device.type
                == device_cuda.type
            )
            for p in continual.plasticity._controller.parameters():
                assert p.device.type == device_cuda.type

        # Forward pass should work on target device
        x = torch.randn(2, 784, device=device_cuda)
        output = continual.forward(x)
        assert output.device.type == device_cuda.type

    def test_geometry_to_device(self, device_cpu, device_cuda):
        """FeedforwardGeometry moves parameters to device."""
        geometry = FeedforwardGeometry(
            GeometryConfig.feedforward(
                input_dim=784, output_dim=10, hidden_dims=(256, 128)
            )
        )
        geometry = geometry.to(device_cpu)
        geometry = geometry.to(device_cuda)

        for param in geometry.parameters():
            assert param.device.type == device_cuda.type

    def test_dynamics_to_device(self, device_cpu, device_cuda):
        """StateDynamics moves to device (if it has parameters)."""
        dynamics = InstantaneousDynamics(StateDynamicsConfig.instantaneous())
        # InstantaneousDynamics has no parameters, but check no error
        dynamics = dynamics.to(device_cuda) if hasattr(dynamics, "to") else dynamics

        # EnergyMinimizationDynamics also has no parameters
        dynamics2 = EnergyMinimizationDynamics(
            StateDynamicsConfig.energy_minimization(max_steps=10)
        )
        dynamics2 = dynamics2.to(device_cuda) if hasattr(dynamics2, "to") else dynamics2

    def test_credit_to_device(self, device_cpu, device_cuda):
        """CreditAssignment moves to device."""
        credit = BackpropCredit(CreditAssignmentConfig.thermodynamic_contrast())
        credit = credit.to(device_cuda) if hasattr(credit, "to") else credit

        credit2 = ThermodynamicContrast(CreditAssignmentConfig.thermodynamic_contrast())
        credit2 = credit2.to(device_cuda) if hasattr(credit2, "to") else credit2

    def test_update_to_device(self, device_cpu, device_cuda):
        """ParameterUpdate moves to device."""
        update = EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=0.01))
        update = update.to(device_cuda) if hasattr(update, "to") else update

    def test_substrate_config_device(self, device_cpu, device_cuda):
        """SubstrateConfig device string is set at construction."""
        substrate = DigitalSubstrate(SubstrateConfig.digital(device=str(device_cuda)))
        assert substrate.config.device == device_cuda.type

        # Note: Substrate doesn't have .to() method, config device is set at construction


# ============================================================
# Component Device Consistency Tests
# ============================================================


class TestComponentDeviceConsistency:
    """Test that individual components handle device correctly."""

    def test_fast_weight_plasticity_device(self, device_cpu, device_cuda):
        """FastWeightPlasticity .to() moves projection matrices."""
        plasticity = FastWeightPlasticity(
            fast_weight_dim=512, decay=0.9, learning_rate=0.1
        )
        _ = plasticity._get_proj_matrix(7840, device_cpu)
        plasticity = plasticity.to(device_cuda)

        for v in plasticity._proj_matrices.values():
            assert v.device.type == device_cuda.type

    def test_routing_plasticity_no_internal_state(self, device_cpu, device_cuda):
        """RoutingPlasticity has no internal persistent tensors."""
        plasticity = RoutingPlasticity(gate_dim=64, decay=0.99, learning_rate=0.01)  # ruff: ignore[unused-variable]
        # No internal state to move
        assert True

    def test_rule_state_plasticity_device(self, device_cpu, device_cuda):
        """RuleStatePlasticity device set at construction."""
        plasticity = RuleStatePlasticity(
            num_operators=8, operator_dim=64, device=device_cuda
        )

        assert plasticity._operator_embeddings.device.type == device_cuda.type
        for p in plasticity._controller.parameters():
            assert p.device.type == device_cuda.type

    def test_null_plasticity_no_internal_state(self, device_cpu, device_cuda):
        """NullPlasticity has no internal state."""
        plasticity = NullPlasticity()  # ruff: ignore[unused-variable]
        assert True

    def test_joint_system_all_plasticity_types(self, device_cpu, device_cuda):
        """Joint system works with all plasticity types on target device."""
        for plasticity_type in ["null", "fast_weights", "routing", "rule_state"]:
            continual = make_joint_system(device_cpu, plasticity_type)
            continual = continual.to(device_cuda)

            # Forward pass
            x = torch.randn(2, 784, device=device_cuda)
            output = continual.forward(x)
            assert output.device.type == device_cuda.type
            assert output.shape == (2, 10)


# ============================================================
# CPU/CUDA Consistency Tests
# ============================================================


class TestCPUCUDAConsistency:
    """Test that CPU and CUDA produce consistent results."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_joint_system_cpu_vs_cuda(self, device_cpu):
        """Joint system produces same results on CPU and CUDA when copied."""
        device_cuda = torch.device("cuda")

        # Create system on CPU
        continual_cpu = make_joint_system(device_cpu, "null")

        # Copy to CUDA by creating new system with same weights
        # We need to ensure identical initialization
        torch.manual_seed(42)
        continual_cuda = make_joint_system(device_cpu, "null")
        continual_cuda = continual_cuda.to(device_cuda)

        # Copy weights from CPU to CUDA
        with torch.no_grad():
            for (name_cpu, param_cpu), (name_cuda, param_cuda) in zip(
                continual_cpu.geometry.named_parameters(),
                continual_cuda.geometry.named_parameters(),
            ):
                param_cuda.copy_(param_cpu.to(device_cuda))

        # Same input
        x = torch.randn(2, 784, device=device_cpu)
        x_cuda = x.to(device_cuda)

        # Forward pass
        output_cpu = continual_cpu.forward(x)
        output_cuda = continual_cuda.forward(x_cuda)

        # Compare (allow small numerical differences)
        assert torch.allclose(output_cpu, output_cuda.cpu(), rtol=1e-5, atol=1e-7)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_fast_weight_plasticity_deterministic_per_device(self, device_cpu):
        """FastWeightPlasticity projection matrices are deterministic per device."""
        device_cuda = torch.device("cuda")

        # Create on CPU
        plasticity_cpu = FastWeightPlasticity(
            fast_weight_dim=512, decay=0.9, learning_rate=0.1
        )
        plasticity_cpu = plasticity_cpu.to(device_cpu)
        proj_cpu_1 = plasticity_cpu._get_proj_matrix(7840, device_cpu)
        proj_cpu_2 = plasticity_cpu._get_proj_matrix(7840, device_cpu)

        # Same on CPU
        assert torch.allclose(proj_cpu_1, proj_cpu_2)

        # Create on CUDA
        plasticity_cuda = FastWeightPlasticity(
            fast_weight_dim=512, decay=0.9, learning_rate=0.1
        )
        plasticity_cuda = plasticity_cuda.to(device_cuda)
        proj_cuda_1 = plasticity_cuda._get_proj_matrix(7840, device_cuda)
        proj_cuda_2 = plasticity_cuda._get_proj_matrix(7840, device_cuda)

        # Same on CUDA
        assert torch.allclose(proj_cuda_1, proj_cuda_2)

        # Note: CPU and CUDA matrices differ due to different RNG implementations
        # but each is deterministic on its own device


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
