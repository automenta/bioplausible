"""Native PEPITA model using 5-D Ontology composition.

This replaces the legacy PEPITA with a direct
composition of the 5 Protocols, bypassing ModelAdapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from computronium.core.system_trainer import compose_system
from computronium.ontology import (
    CreditAssignmentConfig,
    DigitalSubstrate,
    EuclideanUpdate,
    FeedforwardGeometry,
    GeometryConfig,
    InstantaneousDynamics,
    LocalGoodnessCredit,
    ParameterUpdateConfig,
    StateDynamicsConfig,
    System,
)

if TYPE_CHECKING:
    import torch


def create_native_pepita_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 2,
    lr: float = 0.01,
    device: str | torch.device = "cpu",
) -> System:
    """Create a PEPITA system using native 5-D composition.

    PEPITA uses forward-only local learning with error-modulated input
    perturbation and layer-local contrastive updates.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_layers: Number of hidden layers
        lr: Learning rate
        device: Target device for parameter placement

    Returns:
        A composed System with FeedforwardGeometry + InstantaneousDynamics
        + LocalGoodnessCredit + EuclideanUpdate
    """
    # Build hidden dims list
    hidden_dims = tuple([hidden_dim] * num_layers)

    geometry_cfg = GeometryConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
        num_layers=num_layers,
        topology_type="feedforward",
        connectivity=None,
        recurrent_weight=None,
    )

    substrate = DigitalSubstrate()
    geometry = FeedforwardGeometry(geometry_cfg)
    dynamics = InstantaneousDynamics(StateDynamicsConfig.instantaneous())
    credit = LocalGoodnessCredit(
        CreditAssignmentConfig.local_goodness(
            feedback_scale=0.01,
            local_objective="pepita",
        )
    )
    update = EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=lr))

    return compose_system(substrate, geometry, dynamics, credit, update, device=device)


# Alias for registry registration
native_pepita_mlp = create_native_pepita_mlp
