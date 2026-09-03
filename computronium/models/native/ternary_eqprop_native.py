"""Native Ternary Equilibrium Propagation using 5-D Ontology composition.

This replaces the legacy TernaryEqProp with a direct composition of the
5 Protocols, using TernarySubstrate for { -1, 0, +1 } weight quantization
with Straight-Through Estimator (STE) gradient estimation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from computronium.core.substrates.ternary_substrate import TernarySubstrate
from computronium.core.system_trainer import compose_system
from computronium.ontology import (
    CreditAssignmentConfig,
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


def create_native_ternary_eqprop(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 1,
    beta: float = 0.5,
    settle_steps: int = 30,
    lr: float = 0.01,
    ternary_type: str = "standard",
    threshold_init: float = 0.05,
    learn_threshold: bool = False,
    weight_decay: float = 0.0,
    alpha_init: float = 1.0,
    device: str | torch.device = "cpu",
) -> System:
    """Create a Ternary Equilibrium Propagation system using native 5-D composition.

    Ternary EP uses ternary weight quantization {-α, 0, +α} with STE for
    gradient backpropagation through the quantization function. This enables
    extreme quantization for FPGA/ASIC deployment while maintaining learning
    capability through the STE.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_layers: Number of hidden layers
        beta: Nudge strength for energy-based methods
        settle_steps: Maximum settling iterations
        lr: Learning rate
        ternary_type: "standard", "delta", or "trained_threshold"
        threshold_init: Initial threshold for ternary quantization
        learn_threshold: Whether to learn the threshold via STE
        weight_decay: Weight decay on latent full-precision weights
        alpha_init: Initial scale factor α for ternary values
        device: Target device for parameter placement

    Returns:
        A composed System with TernarySubstrate + RecurrentGeometry
        + EnergyMinimizationDynamics + ThermodynamicContrast + EuclideanUpdate
    """
    hidden_dims = tuple([hidden_dim] * max(num_layers, 1))

    geometry_cfg = GeometryConfig.recurrent(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        init_scale=0.1,
    )

    # Use TernarySubstrate for ternary weight quantization with STE
    substrate = TernarySubstrate(
        ternary_type=ternary_type,  # type: ignore[arg-type]
        threshold_init=threshold_init,
        learn_threshold=learn_threshold,
        weight_decay=weight_decay,
        alpha_init=alpha_init,
    )
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
native_ternary_eqprop = create_native_ternary_eqprop
