"""Native Momentum Equilibrium Propagation using 5-D Ontology composition.

This replaces the legacy MomentumEquilibrium variant with a direct composition
of the 5 Protocols, using EnergyMinimizationDynamics with momentum (heavy-ball
acceleration) for faster convergence during settling.
"""

from __future__ import annotations

from bioplausible.core.ontology import (
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
from bioplausible.core.system_trainer import compose_system


def create_native_momentum_eqprop(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 1,
    beta: float = 0.5,
    settle_steps: int = 30,
    lr: float = 0.01,
    momentum: float = 0.5,
    **kwargs,
) -> System:
    """Create a Momentum Equilibrium Propagation system using native 5-D composition.

    Momentum Equilibrium adds heavy-ball momentum to the settling dynamics:
    h_{t+1} = h_t + step_size * (f(h_t) - h_t) + momentum * (h_t - h_{t-1})

    This accelerates convergence of the energy minimization, particularly
    for ill-conditioned energy landscapes.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_layers: Number of hidden layers
        beta: Nudge strength for energy-based methods
        settle_steps: Maximum settling iterations
        lr: Learning rate
        momentum: Momentum coefficient for heavy-ball dynamics (0 to 1)
        **kwargs: Additional arguments (ignored, for compatibility)

    Returns:
        A composed System with DigitalSubstrate + RecurrentGeometry
        + EnergyMinimizationDynamics (with momentum) + ThermodynamicContrast + EuclideanUpdate
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
            momentum=momentum,
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

    return compose_system(substrate, geometry, dynamics, credit, update)


# Alias for registry registration
native_momentum_eqprop = create_native_momentum_eqprop
