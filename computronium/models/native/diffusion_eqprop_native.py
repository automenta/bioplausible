"""Native Diffusion Equilibrium Propagation using 5-D Ontology composition.

This replaces the legacy EqPropDiffusion with a direct composition of the
5 Protocols, using DiffusionDynamics for continuous-time Langevin dynamics
settling with stochastic noise injection.
"""

from __future__ import annotations

from computronium.core.ontology import (
    CreditAssignmentConfig,
    DiffusionDynamics,
    DigitalSubstrate,
    EuclideanUpdate,
    GeometryConfig,
    ParameterUpdateConfig,
    RecurrentGeometry,
    StateDynamicsConfig,
    System,
    ThermodynamicContrast,
)
from computronium.core.system_trainer import compose_system


def create_native_diffusion_eqprop(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 1,
    beta: float = 0.5,
    settle_steps: int = 30,
    lr: float = 0.01,
    diffusion_coeff: float = 1.0,
    **kwargs,
) -> System:
    """Create a Diffusion Equilibrium Propagation system using native 5-D composition.

    Diffusion Equilibrium Propagation uses Langevin dynamics (stochastic
    differential equation) for settling:

    dh/dt = -∇E(h) + √(2D) ξ(t)

    where E is the energy function, D is the diffusion coefficient, and
    ξ(t) is white noise. This models stochastic settling where the network
    samples from the Boltzmann distribution p(h) ∝ exp(-E(h)/D).

    The discrete-time update (Euler-Maruyama):
    h_{t+1} = h_t - step_size * ∇E(h_t) + √(2 * step_size * D) * N(0, I)

    This provides a principled way to add noise for exploration and can
    escape local minima during settling.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_layers: Number of hidden layers
        beta: Nudge strength for energy-based methods
        settle_steps: Maximum settling iterations
        lr: Learning rate
        diffusion_coeff: Diffusion coefficient D (noise scale)
        **kwargs: Additional arguments (ignored, for compatibility)

    Returns:
        A composed System with DigitalSubstrate + RecurrentGeometry
        + DiffusionDynamics + ThermodynamicContrast + EuclideanUpdate
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
    dynamics = DiffusionDynamics(
        StateDynamicsConfig.diffusion(
            max_steps=settle_steps,
            beta=beta,
            step_size=0.1,  # Default step size for diffusion
        )
    )
    # Store diffusion coefficient on the dynamics instance
    dynamics._diffusion_coeff = diffusion_coeff

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
native_diffusion_eqprop = create_native_diffusion_eqprop
