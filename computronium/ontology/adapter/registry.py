"""Registry metadata extraction for ModelAdapter.

Extracts ontology axis information from ComponentMetadata for native models.
"""

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING

from computronium.core.registry import ComponentMetadata

if TYPE_CHECKING:
    from computronium.ontology import (
        CreditAssignmentConfig,
        GeometryConfig,
        ParameterUpdateConfig,
        StateDynamicsConfig,
        SubstrateConfig,
    )


def extract_ontology_axes(metadata: ComponentMetadata) -> dict[str, str]:
    """Extract the 5-D ontology axis assignments from ComponentMetadata.

    For native models with explicit ontology_axes fields, returns those directly.
    For legacy models, returns empty dict to signal heuristic fallback is needed.

    Args:
        metadata: ComponentMetadata from registry

    Returns:
        Dictionary with keys: substrate, geometry, dynamics, credit, update
        Values are the ontology config class names (e.g., "DigitalSubstrate",
        "FeedforwardGeometry", "EnergyMinimizationDynamics", etc.)
    """
    axes = {}
    for axis in ("substrate", "geometry", "dynamics", "credit", "update"):
        field_name = f"ontology_{axis}"
        value = getattr(metadata, field_name, "")
        if value:
            axes[axis] = value
    return axes


def has_explicit_ontology_axes(metadata: ComponentMetadata) -> bool:
    """Check if ComponentMetadata has explicit ontology axis assignments.

    Args:
        metadata: ComponentMetadata from registry

    Returns:
        True if at least one ontology axis is explicitly set
    """
    return any(
        getattr(metadata, f"ontology_{axis}", "")
        for axis in ("substrate", "geometry", "dynamics", "credit", "update")
    )


def get_native_config_classes(
    metadata: ComponentMetadata,
) -> dict[str, type] | None:
    """Get the ontology config classes for a native model.

    Uses explicit ontology_axes fields to determine the exact config classes.

    Args:
        metadata: ComponentMetadata with ontology_axes populated

    Returns:
        Dictionary mapping axis name to config class, or None if not native
    """
    if not has_explicit_ontology_axes(metadata):
        return None

    # Lazy imports to avoid circular dependencies
    from computronium.ontology import (
        CreditAssignmentConfig,
        GeometryConfig,
        ParameterUpdateConfig,
        StateDynamicsConfig,
        SubstrateConfig,
    )

    # Map ontology axis values to config classes
    # This is a simplified mapping - in practice, the axis values are the
    # config class names (e.g., "DigitalSubstrate", "FeedforwardGeometry")
    axis_to_config = {
        "substrate": SubstrateConfig,
        "geometry": GeometryConfig,
        "dynamics": StateDynamicsConfig,
        "credit": CreditAssignmentConfig,
        "update": ParameterUpdateConfig,
    }

    result = {}
    for axis, config_class in axis_to_config.items():
        field_name = f"ontology_{axis}"
        value = getattr(metadata, field_name, "")
        if value:
            result[axis] = config_class

    return result if result else None