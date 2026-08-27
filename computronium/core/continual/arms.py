"""Continual learning arm factories (6-D Joint Systems)."""

from __future__ import annotations

import torch

from computronium.core.continual.buffers import ReplayBuffer
from computronium.core.continual.constants import CL_TOTAL_CLASSES
from computronium.core.continual.losses import LwFLoss, SynapticIntelligence
from computronium.core.continual.system import ContinualJointSystem
from computronium.core.joint.transition import NullPlasticity, PlasticityConfig
from computronium.core.ontology import (
    BackpropCredit,
    CreditAssignmentConfig,
    DigitalSubstrate,
    ElasticConsolidationUpdate,
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
from computronium.core.plasticity import create_fast_weight_plasticity


def _get_compose_joint_system():
    """Lazy import to avoid circular dependency."""
    from computronium.core.system_trainer import compose_joint_system
    return compose_joint_system


def create_fast_weight_arm(
    input_dim: int = 784,
    hidden_dim: int = 256,
    output_dim: int = CL_TOTAL_CLASSES,  # 10 classes
    device: str = "cpu",
) -> ContinualJointSystem:
    """Create FastWeightPlasticity arm (ψ/θ decoupling).

    Uses EnergyMinimizationDynamics with ThermodynamicContrast for proper
    free/nudged settling dynamics required by the contrastive credit assignment.
    """
    compose_joint_system = _get_compose_joint_system()
    substrate = DigitalSubstrate(
        SubstrateConfig.digital(precision="float32", noise_level=0.0, device=device)
    )

    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=(hidden_dim, hidden_dim),
            init_scale=0.1,
        )
    )

    dynamics = EnergyMinimizationDynamics(
        StateDynamicsConfig.energy_minimization(max_steps=30, beta=0.5)
    )

    credit = ThermodynamicContrast(
        CreditAssignmentConfig.thermodynamic_contrast(beta=0.5)
    )

    update = EuclideanUpdate(
        ParameterUpdateConfig.euclidean(step_size=0.001, momentum=0.0)
    )

    plasticity = create_fast_weight_plasticity(
        PlasticityConfig.fast_weights(fast_weight_dim=512, decay=0.9, learning_rate=0.1)
    )

    joint = compose_joint_system(substrate, geometry, dynamics, plasticity, credit, update)
    return ContinualJointSystem.from_joint_system(joint).to(device)


def create_ewc_arm(
    input_dim: int = 784,
    hidden_dim: int = 256,
    output_dim: int = CL_TOTAL_CLASSES,
    device: str = "cpu",
    ewc_lambda: float = 1000.0,
) -> tuple[ContinualJointSystem, ElasticConsolidationUpdate]:
    """Create ElasticConsolidationUpdate (EWC) arm.

    Uses EnergyMinimizationDynamics with ThermodynamicContrast for proper
    free/nudged settling dynamics required by the contrastive credit assignment.
    """
    compose_joint_system = _get_compose_joint_system()
    substrate = DigitalSubstrate(
        SubstrateConfig.digital(precision="float32", noise_level=0.0, device=device)
    )

    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=(hidden_dim, hidden_dim),
            init_scale=0.1,
        )
    )

    dynamics = EnergyMinimizationDynamics(
        StateDynamicsConfig.energy_minimization(max_steps=30, beta=0.5)
    )

    credit = ThermodynamicContrast(
        CreditAssignmentConfig.thermodynamic_contrast(beta=0.5)
    )

    update = ElasticConsolidationUpdate(
        ParameterUpdateConfig.elastic_consolidation(
            step_size=0.001, momentum=0.0, ewc_lambda=ewc_lambda
        )
    )

    joint = compose_joint_system(
        substrate, geometry, dynamics, NullPlasticity(), credit, update
    )
    system = ContinualJointSystem.from_joint_system(joint).to(device)

    # Return the update object for consolidation at task boundaries
    return system, update


def create_backprop_arm(
    input_dim: int = 784,
    hidden_dim: int = 256,
    output_dim: int = CL_TOTAL_CLASSES,
    device: str = "cpu",
) -> ContinualJointSystem:
    """Create Backprop+SGD control arm."""
    compose_joint_system = _get_compose_joint_system()
    substrate = DigitalSubstrate(
        SubstrateConfig.digital(precision="float32", noise_level=0.0, device=device)
    )

    geometry = FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=(hidden_dim, hidden_dim),
            init_scale=0.1,
        )
    )

    dynamics = InstantaneousDynamics(StateDynamicsConfig.instantaneous())

    credit = BackpropCredit(CreditAssignmentConfig.gradient())

    update = EuclideanUpdate(
        ParameterUpdateConfig.euclidean(step_size=0.001)
    )

    joint = compose_joint_system(substrate, geometry, dynamics, NullPlasticity(), credit, update)
    return ContinualJointSystem.from_joint_system(joint).to(device)


def create_replay_arm(
    input_dim: int = 784,
    hidden_dim: int = 256,
    output_dim: int = CL_TOTAL_CLASSES,
    device: str = "cpu",
    buffer_capacity: int = 5000,
) -> tuple[ContinualJointSystem, ReplayBuffer]:
    """Create replay buffer arm (matched total memory)."""
    system = create_backprop_arm(input_dim, hidden_dim, output_dim, device)
    # Fast weight plasticity has ~512 * 4 bytes * batch_size = ~2KB per sample
    # Match replay buffer capacity to equivalent memory
    buffer = ReplayBuffer(buffer_capacity, (input_dim,), torch.device(device))
    return system, buffer


def create_lwf_arm(
    input_dim: int = 784,
    hidden_dim: int = 256,
    output_dim: int = CL_TOTAL_CLASSES,
    device: str = "cpu",
) -> tuple[ContinualJointSystem, LwFLoss]:
    """Create LwF arm."""
    system = create_backprop_arm(input_dim, hidden_dim, output_dim, device)
    lwf_loss = LwFLoss(temperature=2.0, lambda_lwf=1.0)
    return system, lwf_loss


def create_si_arm(
    input_dim: int = 784,
    hidden_dim: int = 256,
    output_dim: int = CL_TOTAL_CLASSES,
    device: str = "cpu",
) -> tuple[ContinualJointSystem, SynapticIntelligence]:
    """Create Synaptic Intelligence arm."""
    system = create_backprop_arm(input_dim, hidden_dim, output_dim, device)
    si = SynapticIntelligence(system, xi=0.1)
    return system, si


__all__ = [
    "create_backprop_arm",
    "create_ewc_arm",
    "create_fast_weight_arm",
    "create_lwf_arm",
    "create_replay_arm",
    "create_si_arm",
]
