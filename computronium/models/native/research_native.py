"""Native Research Direction models using 5-D Ontology composition.

These replace the legacy HolomorphicEP, DirectedEP, FiniteNudgeEP
with direct composition of the 5 Protocols, bypassing ModelAdapter.
"""

from __future__ import annotations

import torch

from computronium.core.system_trainer import compose_system
from computronium.ontology import (
    CreditAssignmentConfig,
    DigitalSubstrate,
    EnergyMinimizationDynamics,
    EuclideanUpdate,
    GeometryConfig,
    ParameterUpdateConfig,
    QuantumSubstrate,
    RandomProjectionsCredit,
    RecurrentGeometry,
    StateDynamicsConfig,
    System,
    ThermodynamicContrast,
)


def create_native_holomorphic_ep(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 1,
    beta: float = 0.5,
    settle_steps: int = 30,
    lr: float = 0.01,
    device: str | torch.device = "cpu",
) -> System:
    """Create a Holomorphic Equilibrium Propagation system using native 5-D composition.

    Holomorphic EP uses complex-valued weights and states with thermodynamic contrast
    credit assignment. The complex domain enables holomorphic (analytic) activation
    functions and conjugate-transpose feedback pathways.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_layers: Number of hidden layers
        beta: Nudge strength for energy-based methods
        settle_steps: Maximum settling iterations
        lr: Learning rate
        device: Target device for parameter placement

    Returns:
        A composed System with QuantumSubstrate (complex64) + RecurrentGeometry
        + EnergyMinimizationDynamics + ThermodynamicContrast + EuclideanUpdate
    """
    # Build hidden dims list
    hidden_dims = tuple([hidden_dim] * max(num_layers, 1))

    geometry_cfg = GeometryConfig.recurrent(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        init_scale=0.1,
    )

    # Use QuantumSubstrate for complex64 precision
    substrate = QuantumSubstrate()
    geometry = RecurrentGeometry(geometry_cfg, hidden_dim=hidden_dim)
    dynamics = EnergyMinimizationDynamics(
        StateDynamicsConfig.energy_minimization(
            max_steps=settle_steps,
            beta=beta,
        )
    )
    credit = ThermodynamicContrast(
        CreditAssignmentConfig.thermodynamic_contrast(
            beta=beta,
        )
    )
    update = EuclideanUpdate(
        ParameterUpdateConfig.euclidean(
            step_size=lr,
        )
    )

    return compose_system(
        substrate, geometry, dynamics, credit, update, device=device
    )


def create_native_directed_ep(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 1,
    beta: float = 0.5,
    settle_steps: int = 30,
    lr: float = 0.01,
    feedback_scale: float = 0.01,
    device: str | torch.device = "cpu",
) -> System:
    """Create a Directed Equilibrium Propagation system using native 5-D composition.

    Directed EP uses asymmetric forward/feedback weights (no weight transport),
    implementing the Feedback Alignment principle within an energy-based framework.
    The feedback matrices are random and fixed, not transposes of forward weights.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_layers: Number of hidden layers
        beta: Nudge strength for energy-based methods
        settle_steps: Maximum settling iterations
        lr: Learning rate
        feedback_scale: Scaling factor for random feedback matrices
        device: Target device for parameter placement

    Returns:
        A composed System with DigitalSubstrate + RecurrentGeometry
        + EnergyMinimizationDynamics + RandomProjectionsCredit + EuclideanUpdate
    """
    hidden_dims = tuple([hidden_dim] * max(num_layers, 1))

    geometry_cfg = GeometryConfig.recurrent(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        init_scale=0.1,
    )

    substrate = DigitalSubstrate()
    geometry = RecurrentGeometry(geometry_cfg, hidden_dim=hidden_dim)
    dynamics = EnergyMinimizationDynamics(
        StateDynamicsConfig.energy_minimization(
            max_steps=settle_steps,
            beta=beta,
        )
    )
    # RandomProjectionsCredit implements Feedback Alignment: fixed random B matrices
    credit = RandomProjectionsCredit(
        CreditAssignmentConfig.random_projections(
            beta=beta,
            feedback_scale=feedback_scale,
            orthogonal_init=True,  # Use orthogonal initialization for stability
        )
    )
    update = EuclideanUpdate(
        ParameterUpdateConfig.euclidean(
            step_size=lr,
        )
    )

    return compose_system(
        substrate, geometry, dynamics, credit, update, device=device
    )


def create_native_finite_nudge_ep(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 1,
    beta: float = 1.0,  # Large beta (finite nudge) is the defining feature
    settle_steps: int = 30,
    lr: float = 0.01,
    device: str | torch.device = "cpu",
) -> System:
    """Create a Finite-Nudge Equilibrium Propagation system using native 5-D composition.

    Finite-Nudge EP uses a large beta (finite nudge) instead of the infinitesimal
    limit. This enables stable learning with stronger supervision signals while
    maintaining the equilibrium propagation dynamics.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_layers: Number of hidden layers
        beta: Nudge strength (typically >= 1.0 for finite nudge)
        settle_steps: Maximum settling iterations
        lr: Learning rate
        device: Target device for parameter placement

    Returns:
        A composed System with DigitalSubstrate + RecurrentGeometry
        + EnergyMinimizationDynamics + ThermodynamicContrast (large beta) + EuclideanUpdate
    """
    hidden_dims = tuple([hidden_dim] * max(num_layers, 1))

    geometry_cfg = GeometryConfig.recurrent(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        init_scale=0.1,
    )

    substrate = DigitalSubstrate()
    geometry = RecurrentGeometry(geometry_cfg, hidden_dim=hidden_dim)
    dynamics = EnergyMinimizationDynamics(
        StateDynamicsConfig.energy_minimization(
            max_steps=settle_steps,
            beta=beta,
        )
    )
    credit = ThermodynamicContrast(
        CreditAssignmentConfig.thermodynamic_contrast(
            beta=beta,  # Large beta = finite nudge regime
        )
    )
    update = EuclideanUpdate(
        ParameterUpdateConfig.euclidean(
            step_size=lr,
        )
    )

    return compose_system(
        substrate, geometry, dynamics, credit, update, device=device
    )


# Aliases for registry registration
native_holomorphic_ep = create_native_holomorphic_ep
native_directed_ep = create_native_directed_ep
native_finite_nudge_ep = create_native_finite_nudge_ep
