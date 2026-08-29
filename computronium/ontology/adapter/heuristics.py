"""Heuristic inference for ModelAdapter fallback.

Provides family/name-based fallback inference when registry metadata
is missing or incomplete. Used for backward compatibility with legacy models.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from computronium.core.registry import ComponentMetadata, LocalityLevel

if TYPE_CHECKING:
    from computronium.ontology import (
        CreditAssignment,
        CreditAssignmentConfig,
        Geometry,
        GeometryConfig,
        ParameterUpdate,
        ParameterUpdateConfig,
        StateDynamics,
        StateDynamicsConfig,
        Substrate,
        SubstrateConfig,
    )

# Family-specific tolerances for validation (copied from system.py)
FAMILY_TOLERANCES: dict[str, tuple[float, float]] = {
    "eqprop": (0.15, 1e-2),
    "equilibrium": (0.15, 1e-2),
    "ep": (0.15, 1e-2),
    "chl": (0.15, 1e-2),
    "fa": (0.1, 5e-3),
    "feedback_alignment": (0.1, 5e-3),
    "dfa": (0.1, 5e-3),
    "forward_only": (0.05, 1e-3),
    "ff": (0.05, 1e-3),
    "pepita": (0.05, 1e-3),
    "hebbian": (0.2, 1e-2),
    "target_prop": (0.1, 5e-3),
    "target_inversion": (0.1, 5e-3),
    "spiking": (0.2, 1e-2),
    "stdp": (0.2, 1e-2),
    "snn": (0.2, 1e-2),
    "predictive_coding": (0.1, 5e-3),
    "pc": (0.1, 5e-3),
    "backprop": (0.01, 1e-4),
    "gradient": (0.01, 1e-4),
    "mep": (0.1, 5e-3),
    "equitile": (0.1, 5e-3),
    "tile": (0.1, 5e-3),
    "default": (0.1, 1e-3),
}


def get_family_tolerances(family: str | None) -> tuple[float, float]:
    """Get family-specific tolerances based on model metadata."""
    if family:
        family_lower = family.lower()
        # Check for exact match first
        if family_lower in FAMILY_TOLERANCES:
            return FAMILY_TOLERANCES[family_lower]
        # Check for partial matches
        for key, tol in FAMILY_TOLERANCES.items():
            if key != "default" and key in family_lower:
                return tol
    return FAMILY_TOLERANCES["default"]


def infer_substrate_from_metadata(metadata: ComponentMetadata | None) -> "Substrate | None":
    """Infer substrate from registry metadata compute_profile."""
    if not (metadata and metadata.compute_profile):
        return None

    from computronium.ontology.substrate import (
        AnalogSubstrate,
        ComputeProfile,
        DigitalSubstrate,
        MemristiveSubstrate,
        NeuromorphicSubstrate,
        OpticalSubstrate,
        SubstrateConfig,
    )

    profile = metadata.compute_profile
    if profile == ComputeProfile.ANALOG:
        return AnalogSubstrate(SubstrateConfig.analog(device="cpu"))
    if profile == ComputeProfile.OPTICAL:
        return OpticalSubstrate(SubstrateConfig(
            substrate_type=SubstrateType.OPTICAL,
            precision="float32",
            noise_level=0.0,
            weight_bounds=None,
            sparsity=0.0,
            device="cpu",
        ))
    if profile == ComputeProfile.MEMRISTOR:
        return MemristiveSubstrate(SubstrateConfig(
            substrate_type=SubstrateType.MEMRISTIVE,
            precision="float32",
            noise_level=0.0,
            weight_bounds=None,
            sparsity=0.0,
            device="cpu",
        ))
    if profile == ComputeProfile.NEUROMORPHIC:
        return NeuromorphicSubstrate(SubstrateConfig.neuromorphic(device="cpu"))

    # Default to digital for GPU/CPU
    return DigitalSubstrate(SubstrateConfig.digital())


def infer_substrate_from_backend(model: nn.Module) -> "Substrate | None":
    """Infer substrate from model backend attribute."""
    if not hasattr(model, "backend"):
        return None

    backend = getattr(model, "backend", "").lower()
    from computronium.ontology.substrate import (
        AnalogSubstrate,
        DigitalSubstrate,
        MemristiveSubstrate,
        NeuromorphicSubstrate,
        OpticalSubstrate,
        QuantumSubstrate,
        SubstrateConfig,
        SubstrateType,
    )

    if "analog" in backend:
        return AnalogSubstrate(SubstrateConfig.analog(device="cpu"))
    if "memrist" in backend:
        return MemristiveSubstrate(SubstrateConfig(
            substrate_type=SubstrateType.MEMRISTIVE,
            precision="float32",
            noise_level=0.0,
            weight_bounds=None,
            sparsity=0.0,
            device="cpu",
        ))
    if "neuromorph" in backend:
        return NeuromorphicSubstrate(SubstrateConfig.neuromorphic(device="cpu"))
    if "optical" in backend or "photonic" in backend:
        return OpticalSubstrate(SubstrateConfig(
            substrate_type=SubstrateType.OPTICAL,
            precision="float32",
            noise_level=0.0,
            weight_bounds=None,
            sparsity=0.0,
            device="cpu",
        ))
    if "quantum" in backend:
        return QuantumSubstrate(SubstrateConfig(
            substrate_type=SubstrateType.QUANTUM,
            precision="complex64",
            noise_level=0.0,
            weight_bounds=None,
            sparsity=0.0,
            device="cpu",
        ))

    return None


def infer_substrate_from_family(metadata: ComponentMetadata | None) -> "Substrate | None":
    """Infer substrate from family tag."""
    if not (metadata and metadata.family):
        return None

    from computronium.ontology.substrate import DigitalSubstrate, NeuromorphicSubstrate, SubstrateConfig

    family = metadata.family.lower()
    if "spiking" in family or "snn" in family or "stdp" in family:
        return NeuromorphicSubstrate(SubstrateConfig.neuromorphic())
    if "tile" in family or "equitile" in family:
        return DigitalSubstrate(SubstrateConfig.digital())

    return None


def infer_geometry_from_metadata(metadata: ComponentMetadata | None) -> "Geometry | None":
    """Infer geometry from metadata - simplified fallback."""
    if not metadata:
        return None

    from computronium.ontology.geometry import FeedforwardGeometry, GeometryConfig

    # Default to feedforward
    return FeedforwardGeometry(GeometryConfig.feedforward(input_dim=784, output_dim=10, hidden_dims=(256, 128)))


def infer_dynamics_from_metadata(metadata: ComponentMetadata | None) -> "StateDynamics | None":
    """Infer dynamics from metadata - simplified fallback."""
    if not metadata:
        return None

    from computronium.ontology.dynamics import InstantaneousDynamics, StateDynamicsConfig

    # Default to instantaneous
    return InstantaneousDynamics(StateDynamicsConfig.instantaneous())


def infer_credit_from_metadata(metadata: ComponentMetadata | None) -> "CreditAssignment | None":
    """Infer credit assignment from metadata - simplified fallback."""
    if not metadata:
        return None

    from computronium.ontology.credit import CreditAssignmentConfig, ThermodynamicContrast

    # Default to thermodynamic contrast (works with energy-based dynamics)
    return ThermodynamicContrast(CreditAssignmentConfig.thermodynamic_contrast())


def infer_update_from_metadata(metadata: ComponentMetadata | None) -> "ParameterUpdate | None":
    """Infer parameter update from metadata - simplified fallback."""
    if not metadata:
        return None

    from computronium.ontology.update import EuclideanUpdate, ParameterUpdateConfig

    # Default to euclidean
    return EuclideanUpdate(ParameterUpdateConfig.euclidean())


def infer_all_axes(
    model: nn.Module,
    metadata: ComponentMetadata | None,
) -> tuple[
    "Substrate",
    "Geometry",
    "StateDynamics",
    "CreditAssignment",
    "ParameterUpdate",
]:
    """Infer all 5 ontology axes using heuristic fallbacks.

    Priority order:
    1. Registry metadata (compute_profile, family)
    2. Model attributes (backend)
    3. Defaults

    Args:
        model: The model to adapt
        metadata: Optional registry metadata

    Returns:
        Tuple of (substrate, geometry, dynamics, credit, update)
    """
    # Substrate inference
    substrate = (
        infer_substrate_from_metadata(metadata)
        or infer_substrate_from_backend(model)
        or infer_substrate_from_family(metadata)
        or _default_substrate()
    )

    # Other axes use simple defaults for legacy models
    geometry = infer_geometry_from_metadata(metadata) or _default_geometry()
    dynamics = infer_dynamics_from_metadata(metadata) or _default_dynamics()
    credit = infer_credit_from_metadata(metadata) or _default_credit()
    update = infer_update_from_metadata(metadata) or _default_update()

    return substrate, geometry, dynamics, credit, update


def _default_substrate() -> "Substrate":
    from computronium.ontology.substrate import DigitalSubstrate, SubstrateConfig
    return DigitalSubstrate(SubstrateConfig.digital())


def _default_geometry() -> "Geometry":
    from computronium.ontology.geometry import FeedforwardGeometry, GeometryConfig
    return FeedforwardGeometry(GeometryConfig.feedforward(input_dim=784, output_dim=10, hidden_dims=(256, 128)))


def _default_dynamics() -> "StateDynamics":
    from computronium.ontology.dynamics import InstantaneousDynamics, StateDynamicsConfig
    return InstantaneousDynamics(StateDynamicsConfig.instantaneous())


def _default_credit() -> "CreditAssignment":
    from computronium.ontology.credit import CreditAssignmentConfig, ThermodynamicContrast
    return ThermodynamicContrast(CreditAssignmentConfig.thermodynamic_contrast())


def _default_update() -> "ParameterUpdate":
    from computronium.ontology.update import EuclideanUpdate, ParameterUpdateConfig
    return EuclideanUpdate(ParameterUpdateConfig.euclidean())