"""Serialization/deserialization utilities for System and JointSystem."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from computronium.ontology import (
    BackpropCredit,
    CreditAssignmentConfig,
    DiffusionDynamics,
    ElasticConsolidationUpdate,
    EnergyMinimizationDynamics,
    EuclideanUpdate,
    FeedforwardGeometry,
    GeometryConfig,
    HomeostaticCredit,
    InstantaneousDynamics,
    LocalGoodnessCredit,
    NaturalGradientUpdate,
    ParameterUpdateConfig,
    PredictiveSettlingDynamics,
    RandomProjectionsCredit,
    RecurrentGeometry,
    RiemannianOrthogonalUpdate,
    SpectralConstrainedUpdate,
    SpikeIntegrationDynamics,
    StateDynamicsConfig,
    SubstrateConfig,
    TargetInversionCredit,
    TemporalTraceCredit,
    ThermodynamicContrast,
    TileGeometry,
    substrate_from_config,
)

if TYPE_CHECKING:
    from computronium.core.joint.transition import PlasticityConfig, PlasticityPrimitive
    from computronium.core.system_trainer.protocol import JointSystem
    from computronium.ontology import (
        CreditAssignment,
        Geometry,
        ParameterUpdate,
        StateDynamics,
        Substrate,
        System,
    )


def extract_config(system: System) -> dict[str, object]:
    """Extract configuration from a composed System.

    Returns a dictionary mapping layer names to their configuration objects.
    This enables round-trip: System -> configs -> System.

    Args:
        system: A composed System instance.

    Returns:
        Dictionary with keys: substrate, geometry, dynamics, credit, update.
    """
    return {
        "substrate": system.substrate.config,
        "geometry": system.geometry.config,
        "dynamics": system.dynamics.config,
        "credit": system.credit.config,
        "update": system.update.config,
    }


def _geometry_from_config(geometry: GeometryConfig) -> Geometry:
    """Instantiate geometry from config."""
    topology_type = geometry.topology_type.lower()
    if topology_type in ("recurrent", "recurrent_attractor"):  # ruff: ignore[literal-membership]
        hidden_dim = geometry.hidden_dims[-1] if geometry.hidden_dims else None
        recurrent_weight = None
        if geometry.recurrent_weight is not None:
            recurrent_weight = torch.tensor(geometry.recurrent_weight)
        return RecurrentGeometry(
            geometry, hidden_dim=hidden_dim, recurrent_weight=recurrent_weight
        )
    elif topology_type in ("tile_mesh", "tile"):  # ruff: ignore[literal-membership]
        return TileGeometry(
            geometry,
            neurons_per_tile=8,
            tiles_per_layer=2,
        )
    elif topology_type == "feedforward":
        return FeedforwardGeometry(geometry)
    else:
        raise ValueError(f"Unknown topology_type: {topology_type!r}")


def _dynamics_from_config(dynamics: StateDynamicsConfig) -> StateDynamics:
    """Instantiate dynamics from config."""
    dynamics_type = dynamics.dynamics_type.lower()
    if dynamics_type == "energy_minimization":
        return EnergyMinimizationDynamics(dynamics)
    elif dynamics_type == "predictive_settling":
        return PredictiveSettlingDynamics(dynamics)
    elif dynamics_type == "spike_integration":
        return SpikeIntegrationDynamics(dynamics)
    elif dynamics_type == "diffusion":
        return DiffusionDynamics(dynamics)
    elif dynamics_type == "instantaneous":
        return InstantaneousDynamics(dynamics)
    else:
        raise ValueError(f"Unknown dynamics_type: {dynamics_type!r}")


def _credit_from_config(config: CreditAssignmentConfig):  # ruff: ignore[too-many-return-statements]
    """Instantiate the credit implementation named by ``config.credit_type``."""
    match config.credit_type.lower():
        case "thermodynamic_contrast" | "equilibrium":
            return ThermodynamicContrast(config)
        case (
            "random_projections" | "feedback_alignment" | ("direct_feedback_alignment")
        ):
            return RandomProjectionsCredit(config)
        case "local_goodness" | "forward_only":
            return LocalGoodnessCredit(config)
        case "temporal_trace" | "spiking":
            return TemporalTraceCredit(config)
        case "target_inversion" | "target_prop":
            return TargetInversionCredit(config)
        case "homeostatic":
            return HomeostaticCredit(config)
        case "gradient" | "backprop":
            return BackpropCredit(config)
        case other:
            raise ValueError(f"Unknown credit_type: {other!r}")


def _update_from_config(update: ParameterUpdateConfig):
    """Instantiate update from config."""
    update_type = update.update_type.lower()
    if update_type in ("riemannian_orthogonal", "muon"):  # ruff: ignore[literal-membership]
        return RiemannianOrthogonalUpdate(update)
    elif update_type in ("spectral_constrained", "spectral"):  # ruff: ignore[literal-membership]
        return SpectralConstrainedUpdate(update)
    elif update_type in ("natural_gradient", "fisher"):  # ruff: ignore[literal-membership]
        return NaturalGradientUpdate(update)
    elif update_type in ("elastic_consolidation", "ewc"):  # ruff: ignore[literal-membership]
        return ElasticConsolidationUpdate(update)
    elif update_type == "euclidean":
        return EuclideanUpdate(update)
    else:
        raise ValueError(f"Unknown update_type: {update_type!r}")


def _plasticity_from_config(plasticity: PlasticityConfig):
    """Instantiate plasticity from config."""
    from computronium.core.plasticity import (
        NullPlasticity,
        create_fast_weight_plasticity,
        create_routing_plasticity,
        create_rule_state_plasticity,
        create_substrate_coupled_plasticity,
    )

    plasticity_type = plasticity.plasticity_type.lower()
    if plasticity_type == "routing":
        return create_routing_plasticity(plasticity)
    elif plasticity_type == "fast_weights":
        return create_fast_weight_plasticity(plasticity)
    elif plasticity_type == "substrate_coupled":
        return create_substrate_coupled_plasticity(plasticity)
    elif plasticity_type == "rule_state":
        return create_rule_state_plasticity(plasticity)
    elif plasticity_type == "null":
        return NullPlasticity()
    else:
        raise ValueError(f"Unknown plasticity_type: {plasticity_type!r}")


def compose_system_from_configs(
    substrate: SubstrateConfig,
    geometry: GeometryConfig,
    dynamics: StateDynamicsConfig,
    credit: CreditAssignmentConfig,
    update: ParameterUpdateConfig,
) -> System:
    """Compose a System from five configuration objects.

    This is the inverse of extract_config(), enabling the round-trip:
    System --extract_config--> configs --compose_system_from_configs--> System

    Args:
        substrate: Substrate configuration
        geometry: Geometry configuration
        dynamics: StateDynamics configuration
        credit: CreditAssignment configuration
        update: ParameterUpdate configuration

    Returns:
        A composed System with default implementations for each layer.
    """
    # Instantiate substrate from config (class named by the explicit type tag)
    substrate_instance = substrate_from_config(substrate)

    # Instantiate geometry from config
    geometry_instance = _geometry_from_config(geometry)

    # Instantiate dynamics from config
    dynamics_instance = _dynamics_from_config(dynamics)

    # Instantiate credit from config
    credit_instance = _credit_from_config(credit)

    # Instantiate update from config
    update_instance = _update_from_config(update)

    from computronium.core.system_trainer.factory import compose_system

    return compose_system(
        substrate_instance,
        geometry_instance,
        dynamics_instance,
        credit_instance,
        update_instance,
    )


def compose_joint_system_from_configs(
    substrate: SubstrateConfig,
    geometry: GeometryConfig,
    dynamics: StateDynamicsConfig,
    plasticity: PlasticityConfig,
    credit: CreditAssignmentConfig,
    update: ParameterUpdateConfig,
) -> JointSystem[
    Substrate,
    Geometry,
    StateDynamics,
    PlasticityPrimitive,
    CreditAssignment,
    ParameterUpdate,
]:
    """Compose a JointSystem from six configuration objects.

    This is the inverse of extract_config(), enabling the round-trip:
    JointSystem --extract_config--> configs --compose_joint_system_from_configs--> JointSystem

    Args:
        substrate: Substrate configuration
        geometry: Geometry configuration
        dynamics: StateDynamics configuration
        plasticity: Plasticity configuration
        credit: CreditAssignment configuration
        update: ParameterUpdate configuration

    Returns:
        A composed JointSystem with default implementations for each layer.
    """
    # Instantiate substrate from config (class named by the explicit type tag)
    substrate_instance = substrate_from_config(substrate)

    # Instantiate geometry from config
    geometry_instance = _geometry_from_config(geometry)

    # Instantiate dynamics from config
    dynamics_instance = _dynamics_from_config(dynamics)

    # Instantiate credit from config
    credit_instance = _credit_from_config(credit)

    # Instantiate update from config
    update_instance = _update_from_config(update)

    # Instantiate plasticity from config
    plasticity_instance = _plasticity_from_config(plasticity)

    from computronium.core.system_trainer.joint import compose_joint_system

    return compose_joint_system(
        substrate_instance,
        geometry_instance,
        dynamics_instance,
        plasticity_instance,
        credit_instance,
        update_instance,
    )


__all__ = [
    "compose_joint_system_from_configs",
    "compose_system_from_configs",
    "extract_config",
]
