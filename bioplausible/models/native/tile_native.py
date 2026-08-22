"""Native Tile model using 5-D Ontology composition.

This replaces the legacy TileAlgorithm family with a direct
composition of the 5 Protocols, bypassing ModelAdapter.
"""

from __future__ import annotations

from bioplausible.core.ontology import (
    CreditAssignmentConfig,
    DigitalSubstrate,
    EnergyMinimizationDynamics,
    EuclideanUpdate,
    GeometryConfig,
    InstantaneousDynamics,
    LocalGoodnessCredit,
    ParameterUpdateConfig,
    PredictiveSettlingDynamics,
    RandomProjectionsCredit,
    SpikeIntegrationDynamics,
    StateDynamicsConfig,
    System,
    TargetInversionCredit,
    ThermodynamicContrast,
    TileGeometry,
)
from bioplausible.core.system_trainer import compose_system


def create_native_tile_ep(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 3,
    neurons_per_tile: int = 48,
    tiles_per_layer: int = 4,
    lr: float = 0.001,
    beta: float = 0.1,
    settle_steps: int = 30,
    **kwargs,
) -> System:
    """Create a Tile EP (Equilibrium Propagation) system using native 5-D composition.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension (used for backwards compat)
        output_dim: Output dimension
        num_layers: Number of hidden layers
        neurons_per_tile: Neurons per tile
        tiles_per_layer: Tiles per layer
        lr: Learning rate
        beta: Nudge strength
        settle_steps: Maximum settling iterations
        **kwargs: Additional arguments (ignored, for compatibility)

    Returns:
        A composed System with TileGeometry + EnergyMinimizationDynamics
        + ThermodynamicContrast + EuclideanUpdate
    """
    geometry_cfg = GeometryConfig.tile_mesh(
        input_dim=input_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        neurons_per_tile=neurons_per_tile,
        tiles_per_layer=tiles_per_layer,
    )

    substrate = DigitalSubstrate()
    geometry = TileGeometry(
        geometry_cfg,
        neurons_per_tile=neurons_per_tile,
        tiles_per_layer=tiles_per_layer,
    )
    dynamics = EnergyMinimizationDynamics(
        StateDynamicsConfig(
            dynamics_type="energy_minimization",
            max_steps=settle_steps,
            beta=beta,
        )
    )
    credit = ThermodynamicContrast(
        CreditAssignmentConfig(
            credit_type="thermodynamic_contrast",
            beta=beta,
        )
    )
    update = EuclideanUpdate(
        ParameterUpdateConfig(
            update_type="euclidean",
            step_size=lr,
        )
    )

    return compose_system(substrate, geometry, dynamics, credit, update)


def create_native_tile_fa(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 3,
    neurons_per_tile: int = 48,
    tiles_per_layer: int = 4,
    lr: float = 0.001,
    **kwargs,
) -> System:
    """Create a Tile FA (Feedback Alignment) system using native 5-D composition.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension (used for backwards compat)
        output_dim: Output dimension
        num_layers: Number of hidden layers
        neurons_per_tile: Neurons per tile
        tiles_per_layer: Tiles per layer
        lr: Learning rate
        **kwargs: Additional arguments (ignored, for compatibility)

    Returns:
        A composed System with TileGeometry + InstantaneousDynamics
        + RandomProjectionsCredit + EuclideanUpdate
    """
    geometry_cfg = GeometryConfig.tile_mesh(
        input_dim=input_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        neurons_per_tile=neurons_per_tile,
        tiles_per_layer=tiles_per_layer,
    )

    substrate = DigitalSubstrate()
    geometry = TileGeometry(
        geometry_cfg,
        neurons_per_tile=neurons_per_tile,
        tiles_per_layer=tiles_per_layer,
    )
    dynamics = InstantaneousDynamics(
        StateDynamicsConfig(
            dynamics_type="instantaneous",
            max_steps=1,
            beta=0.1,
        )
    )
    credit = RandomProjectionsCredit(
        CreditAssignmentConfig(
            credit_type="random_projections",
            feedback_scale=0.01,
        )
    )
    update = EuclideanUpdate(
        ParameterUpdateConfig(
            update_type="euclidean",
            step_size=lr,
        )
    )

    return compose_system(substrate, geometry, dynamics, credit, update)


def create_native_tile_tp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 3,
    neurons_per_tile: int = 48,
    tiles_per_layer: int = 4,
    lr: float = 0.001,
    beta: float = 0.1,
    settle_steps: int = 30,
    **kwargs,
) -> System:
    """Create a Tile TP (Target Propagation) system using native 5-D composition.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension (used for backwards compat)
        output_dim: Output dimension
        num_layers: Number of hidden layers
        neurons_per_tile: Neurons per tile
        tiles_per_layer: Tiles per layer
        lr: Learning rate
        beta: Nudge strength
        settle_steps: Maximum settling iterations
        **kwargs: Additional arguments (ignored, for compatibility)

    Returns:
        A composed System with TileGeometry + PredictiveSettlingDynamics
        + TargetInversionCredit + EuclideanUpdate
    """
    geometry_cfg = GeometryConfig.tile_mesh(
        input_dim=input_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        neurons_per_tile=neurons_per_tile,
        tiles_per_layer=tiles_per_layer,
    )

    substrate = DigitalSubstrate()
    geometry = TileGeometry(
        geometry_cfg,
        neurons_per_tile=neurons_per_tile,
        tiles_per_layer=tiles_per_layer,
    )
    dynamics = PredictiveSettlingDynamics(
        StateDynamicsConfig(
            dynamics_type="predictive_settling",
            max_steps=settle_steps,
            beta=beta,
        )
    )
    credit = TargetInversionCredit(
        CreditAssignmentConfig(
            credit_type="target_inversion",
            feedback_scale=0.01,
        )
    )
    update = EuclideanUpdate(
        ParameterUpdateConfig(
            update_type="euclidean",
            step_size=lr,
        )
    )

    return compose_system(substrate, geometry, dynamics, credit, update)


def create_native_tile_snn(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 3,
    neurons_per_tile: int = 48,
    tiles_per_layer: int = 4,
    lr: float = 0.001,
    **kwargs,
) -> System:
    """Create a Tile SNN (Spiking Neural Network) system using native 5-D composition.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension (used for backwards compat)
        output_dim: Output dimension
        num_layers: Number of hidden layers
        neurons_per_tile: Neurons per tile
        tiles_per_layer: Tiles per layer
        lr: Learning rate
        **kwargs: Additional arguments (ignored, for compatibility)

    Returns:
        A composed System with TileGeometry + SpikeIntegrationDynamics
        + LocalGoodnessCredit + EuclideanUpdate
    """
    geometry_cfg = GeometryConfig.tile_mesh(
        input_dim=input_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        neurons_per_tile=neurons_per_tile,
        tiles_per_layer=tiles_per_layer,
    )

    substrate = DigitalSubstrate()
    geometry = TileGeometry(
        geometry_cfg,
        neurons_per_tile=neurons_per_tile,
        tiles_per_layer=tiles_per_layer,
    )
    dynamics = SpikeIntegrationDynamics(
        StateDynamicsConfig(
            dynamics_type="spike_integration",
            max_steps=30,
            beta=0.1,
        )
    )
    credit = LocalGoodnessCredit(
        CreditAssignmentConfig(
            credit_type="local_goodness",
            feedback_scale=0.01,
        )
    )
    update = EuclideanUpdate(
        ParameterUpdateConfig(
            update_type="euclidean",
            step_size=lr,
        )
    )

    return compose_system(substrate, geometry, dynamics, credit, update)


# Aliases for registry registration
native_tile_ep = create_native_tile_ep
native_tile_fa = create_native_tile_fa
native_tile_tp = create_native_tile_tp
native_tile_snn = create_native_tile_snn
