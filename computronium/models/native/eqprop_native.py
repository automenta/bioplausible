"""Native Equilibrium Propagation model using 5-D Ontology composition.

This replaces the legacy LoopedMLP/EquilibriumMLP with a direct
composition of the 5 Protocols, bypassing ModelAdapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from computronium.core.system_trainer import compose_system
from computronium.ontology import (
    CreditAssignmentConfig,
    DigitalSubstrate,
    EnergyMinimizationDynamics,
    EuclideanUpdate,
    GeometryConfig,
    ParameterUpdateConfig,
    RecurrentGeometry,
    StateDynamicsConfig,
    System,
    ThermodynamicContrast,
)

if TYPE_CHECKING:
    import torch


def create_native_eqprop_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 1,
    beta: float = 0.5,
    settle_steps: int = 30,
    lr: float = 0.01,
    device: str | torch.device = "cpu",
) -> System:
    """Create an Equilibrium Propagation system using native 5-D composition.

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
        A composed System with RecurrentGeometry + EnergyMinimizationDynamics
        + ThermodynamicContrast + EuclideanUpdate
    """
    # Build hidden dims list
    hidden_dims = tuple([hidden_dim] * max(num_layers, 1))

    geometry_cfg = GeometryConfig.recurrent(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
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
            beta=beta,
        )
    )
    update = EuclideanUpdate(
        ParameterUpdateConfig.euclidean(
            step_size=lr,
        )
    )

    return compose_system(substrate, geometry, dynamics, credit, update, device=device)


# Alias for registry registration
native_eqprop_mlp = create_native_eqprop_mlp
