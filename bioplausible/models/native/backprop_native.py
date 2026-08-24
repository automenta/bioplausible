"""Native Backprop model using 5-D Ontology composition.

This replaces the legacy BackpropMLP with a direct
composition of the 5 Protocols, bypassing ModelAdapter.
"""

from __future__ import annotations

from bioplausible.core.ontology import (
    BackpropCredit,
    CreditAssignmentConfig,
    DigitalSubstrate,
    EuclideanUpdate,
    FeedforwardGeometry,
    GeometryConfig,
    InstantaneousDynamics,
    ParameterUpdateConfig,
    StateDynamicsConfig,
    System,
)
from bioplausible.core.system_trainer import compose_system


def create_native_backprop_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 2,
    lr: float = 0.001,
    **kwargs,
) -> System:
    """Create a standard Backprop system using native 5-D composition.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_layers: Number of hidden layers
        lr: Learning rate
        **kwargs: Additional arguments (ignored, for compatibility)

    Returns:
        A composed System with FeedforwardGeometry + InstantaneousDynamics
        + BackpropCredit + EuclideanUpdate
    """
    # Build hidden dims list
    hidden_dims = tuple([hidden_dim] * max(num_layers - 1, 1))

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
    credit = BackpropCredit(CreditAssignmentConfig.gradient())
    update = EuclideanUpdate(ParameterUpdateConfig.euclidean(step_size=lr))

    return compose_system(substrate, geometry, dynamics, credit, update)


# Alias for registry registration
native_backprop_mlp = create_native_backprop_mlp
