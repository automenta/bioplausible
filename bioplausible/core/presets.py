"""Preset Factory Functions for Common Bioplausible Systems.

Provides one-line system construction for common 5-D and 6-D coordinates.
Instead of 20 lines of config, users can call:
    system = create_backprop_mlp(input_dim=784, hidden_dims=(256, 128), output_dim=10)
    system = create_eqprop_mlp(input_dim=784, hidden_dims=(256, 128), output_dim=10, beta=0.5, n_iters=20)
    system = create_fa_mlp(input_dim=784, hidden_dims=(256, 128), output_dim=10)
    system = create_routing_mlp(input_dim=784, hidden_dims=(256, 128), output_dim=10)
    system = create_fast_weight_mlp(input_dim=784, hidden_dims=(256, 128), output_dim=10)
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch

from bioplausible.core.ontology import (
    BackpropCredit,
    CreditAssignmentConfig,
    DigitalSubstrate,
    EnergyMinimizationDynamics,
    EuclideanUpdate,
    FeedforwardGeometry,
    GeometryConfig,
    InstantaneousDynamics,
    ParameterUpdateConfig,
    RandomProjectionsCredit,
    RecurrentGeometry,
    StateDynamicsConfig,
    SubstrateConfig,
    ThermodynamicContrast,
)
from bioplausible.core.plasticity import (
    FastWeightPlasticity,
    FastWeightPlasticityConfig,
    RoutingPlasticity,
    RoutingPlasticityConfig,
)
from bioplausible.core.system_trainer import (
    compose_joint_system,
    compose_system,
)

if TYPE_CHECKING:
    from bioplausible.core.ontology import System
    from bioplausible.core.system_trainer import JointSystem


def _default_substrate(device: str = "cpu") -> DigitalSubstrate:
    """Create a default digital substrate."""
    return DigitalSubstrate(
        SubstrateConfig(
            precision="float32",
            noise_level=0.0,
            weight_bounds=None,
            sparsity=0.0,
            device=device,
        )
    )


def _mlp_geometry(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    init_scale: float = 0.1,
) -> FeedforwardGeometry:
    """Create a standard MLP feedforward geometry."""
    return FeedforwardGeometry(
        GeometryConfig.feedforward(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            init_scale=init_scale,
        )
    )


def _recurrent_geometry(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    init_scale: float = 0.1,
) -> RecurrentGeometry:
    """Create a recurrent geometry for EqProp."""
    # Use the last hidden dim as the recurrent state dimension
    hidden_dim = hidden_dims[-1] if hidden_dims else output_dim
    return RecurrentGeometry(
        GeometryConfig.recurrent(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            init_scale=init_scale,
        ),
        hidden_dim=hidden_dim,
    )


def _default_credit() -> BackpropCredit:
    """Create default backprop credit assignment."""
    return BackpropCredit(
        CreditAssignmentConfig(
            credit_type="gradient",
            beta=0.5,
            feedback_matrix=None,
            local_objective="mse",
            orthogonal_init=False,
            feedback_scale=0.01,
        )
    )


def _eqprop_credit(beta: float = 0.5) -> ThermodynamicContrast:
    """Create EqProp thermodynamic contrast credit assignment."""
    return ThermodynamicContrast(
        CreditAssignmentConfig(
            credit_type="thermodynamic_contrast",
            beta=beta,
            feedback_matrix=None,
            local_objective="mse",
            orthogonal_init=False,
            feedback_scale=0.01,
        )
    )


def _default_update(lr: float = 0.001) -> EuclideanUpdate:
    """Create default Euclidean update."""
    return EuclideanUpdate(
        ParameterUpdateConfig.euclidean(step_size=lr)
    )


# ============================================================
# 5-D System Factories (Standard Ontology)
# ============================================================


def create_backprop_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    lr: float = 0.001,
    init_scale: float = 0.1,
    device: str = "cpu",
) -> "System":
    """Create a standard Backprop MLP system (5-D coordinate).

    Args:
        input_dim: Input dimension (e.g., 784 for MNIST)
        hidden_dims: Tuple of hidden layer dimensions (e.g., (256, 128))
        output_dim: Output dimension (e.g., 10 for MNIST)
        lr: Learning rate
        init_scale: Weight initialization scale
        device: Target device ("cpu" or "cuda")

    Returns:
        A composed 5-D System with Backprop credit assignment.
    """
    substrate = _default_substrate(device)
    geometry = _mlp_geometry(input_dim, hidden_dims, output_dim, init_scale)
    dynamics = InstantaneousDynamics(StateDynamicsConfig.instantaneous())
    credit = _default_credit()
    update = _default_update(lr)

    return compose_system(substrate, geometry, dynamics, credit, update)


def create_eqprop_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    beta: float = 0.5,
    n_iters: int = 20,
    lr: float = 0.01,
    init_scale: float = 0.1,
    device: str = "cpu",
) -> "System":
    """Create an Equilibrium Propagation MLP system (5-D coordinate).

    Args:
        input_dim: Input dimension (e.g., 784 for MNIST)
        hidden_dims: Tuple of hidden layer dimensions (e.g., (256, 128))
        output_dim: Output dimension (e.g., 10 for MNIST)
        beta: Nudge strength for EqProp
        n_iters: Number of settling iterations
        lr: Learning rate
        init_scale: Weight initialization scale
        device: Target device ("cpu" or "cuda")

    Returns:
        A composed 5-D System with ThermodynamicContrast credit assignment
        and EnergyMinimization dynamics.
    """
    substrate = _default_substrate(device)
    geometry = _recurrent_geometry(input_dim, hidden_dims, output_dim, init_scale)
    dynamics = EnergyMinimizationDynamics(
        StateDynamicsConfig.energy_minimization(
            max_steps=n_iters,
            convergence_threshold=1e-4,
            convergence_start=5,
            step_size=0.1,
            beta=beta,
            track_free_energy_per_iter=False,
        )
    )
    credit = _eqprop_credit(beta)
    update = _default_update(lr)

    return compose_system(substrate, geometry, dynamics, credit, update)


def create_fa_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    lr: float = 0.001,
    init_scale: float = 0.1,
    feedback_scale: float = 0.01,
    device: str = "cpu",
) -> "System":
    """Create a Feedback Alignment MLP system (5-D coordinate).

    Args:
        input_dim: Input dimension (e.g., 784 for MNIST)
        hidden_dims: Tuple of hidden layer dimensions (e.g., (256, 128))
        output_dim: Output dimension (e.g., 10 for MNIST)
        lr: Learning rate
        init_scale: Weight initialization scale
        feedback_scale: Scale for random feedback matrix initialization
        device: Target device ("cpu" or "cuda")

    Returns:
        A composed 5-D System with RandomProjections (FA) credit assignment.
    """
    substrate = _default_substrate(device)
    geometry = _mlp_geometry(input_dim, hidden_dims, output_dim, init_scale)
    dynamics = InstantaneousDynamics(StateDynamicsConfig.instantaneous())
    credit = RandomProjectionsCredit(
        CreditAssignmentConfig(
            credit_type="random_projections",
            beta=0.5,
            feedback_matrix=None,
            local_objective="mse",
            orthogonal_init=False,
            feedback_scale=feedback_scale,
        )
    )
    update = _default_update(lr)

    return compose_system(substrate, geometry, dynamics, credit, update)


# ============================================================
# 6-D Joint System Factories (Extended Ontology with Plasticity)
# ============================================================


def create_routing_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    lr: float = 0.001,
    init_scale: float = 0.1,
    gate_dim: int = 64,
    gate_init_scale: float = 0.1,
    device: str = "cpu",
) -> "JointSystem":
    """Create an MLP with RoutingPlasticity (6-D coordinate).

    Args:
        input_dim: Input dimension
        hidden_dims: Tuple of hidden layer dimensions
        output_dim: Output dimension
        lr: Learning rate
        init_scale: Weight initialization scale
        gate_dim: Dimension of routing gates
        gate_init_scale: Initial scale for gate logits
        device: Target device

    Returns:
        A composed 6-D JointSystem with RoutingPlasticity.
    """
    substrate = _default_substrate(device)
    geometry = _mlp_geometry(input_dim, hidden_dims, output_dim, init_scale)
    dynamics = InstantaneousDynamics(StateDynamicsConfig.instantaneous())
    credit = _default_credit()
    update = _default_update(lr)

    plasticity = RoutingPlasticity(
        RoutingPlasticityConfig(
            gate_dim=gate_dim,
            temperature=1.0,
            decay=0.99,
            learning_rate=0.01,
        )
    )

    return compose_joint_system(substrate, geometry, dynamics, plasticity, credit, update)


def create_fast_weight_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    lr: float = 0.001,
    init_scale: float = 0.1,
    fast_weight_dim: int = 512,
    decay: float = 0.9,
    learning_rate: float = 0.1,
    device: str = "cpu",
) -> "JointSystem":
    """Create an MLP with FastWeightPlasticity (6-D coordinate).

    Args:
        input_dim: Input dimension
        hidden_dims: Tuple of hidden layer dimensions
        output_dim: Output dimension
        lr: Learning rate
        init_scale: Weight initialization scale
        fast_weight_dim: Dimension of fast weight matrix
        decay: Decay factor for fast weights
        learning_rate: Hebbian learning rate for fast weights
        device: Target device

    Returns:
        A composed 6-D JointSystem with FastWeightPlasticity.
    """
    substrate = _default_substrate(device)
    geometry = _mlp_geometry(input_dim, hidden_dims, output_dim, init_scale)
    dynamics = InstantaneousDynamics(StateDynamicsConfig.instantaneous())
    credit = _default_credit()
    update = _default_update(lr)

    plasticity = FastWeightPlasticity(
        FastWeightPlasticityConfig(
            fast_weight_dim=fast_weight_dim,
            decay=decay,
            learning_rate=learning_rate,
            outer_product_scale=1.0,
        )
    )

    return compose_joint_system(substrate, geometry, dynamics, plasticity, credit, update)


__all__ = [
    # 5-D factories
    "create_backprop_mlp",
    "create_eqprop_mlp",
    "create_fa_mlp",
    # 6-D factories
    "create_routing_mlp",
    "create_fast_weight_mlp",
]