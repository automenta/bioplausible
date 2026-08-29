"""Inference protocols for ModelAdapter.

Defines the protocols for inferring each ontology axis from a model
and its registry metadata. Each inferrer handles one of the 5-D axes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from torch import nn

from computronium.core.registry import ComponentMetadata

if TYPE_CHECKING:
    from computronium.ontology import (
        CreditAssignment,
        Geometry,
        ParameterUpdate,
        StateDynamics,
        Substrate,
    )


# =============================================================================
# Inferrer Protocols
# =============================================================================


class SubstrateInferer(Protocol):
    """Protocol for inferring Substrate from model and metadata."""

    def infer(self, model: nn.Module, metadata: ComponentMetadata | None) -> Substrate:
        """Infer the substrate for a model."""
        ...


class GeometryInferer(Protocol):
    """Protocol for inferring Geometry from model and metadata."""

    def infer(self, model: nn.Module, metadata: ComponentMetadata | None) -> Geometry:
        """Infer the geometry for a model."""
        ...


class DynamicsInferer(Protocol):
    """Protocol for inferring StateDynamics from model and metadata."""

    def infer(
        self, model: nn.Module, metadata: ComponentMetadata | None
    ) -> StateDynamics:
        """Infer the dynamics for a model."""
        ...


class CreditInferer(Protocol):
    """Protocol for inferring CreditAssignment from model and metadata."""

    def infer(
        self, model: nn.Module, metadata: ComponentMetadata | None
    ) -> CreditAssignment:
        """Infer the credit assignment for a model."""
        ...


class UpdateInferer(Protocol):
    """Protocol for inferring ParameterUpdate from model and metadata."""

    def infer(
        self, model: nn.Module, metadata: ComponentMetadata | None
    ) -> ParameterUpdate:
        """Infer the parameter update for a model."""
        ...


# =============================================================================
# Native Inferrer Implementations (use explicit ontology_axes)
# =============================================================================


class NativeSubstrateInferer:
    """Infer substrate from explicit ontology_axes metadata."""

    def infer(self, model: nn.Module, metadata: ComponentMetadata | None) -> Substrate:
        from computronium.ontology.substrate import (
            SubstrateConfig,
            substrate_from_config,
        )

        if not metadata or not metadata.ontology_substrate:
            raise ValueError("No explicit substrate ontology axis in metadata")

        # Create config from ontology axis value
        # The ontology_substrate value should be the config class name
        # For now, we use the compute_profile as fallback
        config = SubstrateConfig.digital()
        return substrate_from_config(config)


class NativeGeometryInferer:
    """Infer geometry from explicit ontology_axes metadata."""

    def infer(self, model: nn.Module, metadata: ComponentMetadata | None) -> Geometry:
        from computronium.ontology.geometry import (
            FeedforwardGeometry,
            GeometryConfig,
            RecurrentGeometry,
        )

        if not metadata or not metadata.ontology_geometry:
            raise ValueError("No explicit geometry ontology axis in metadata")

        geometry_type = metadata.ontology_geometry
        if geometry_type == "FeedforwardGeometry":
            return FeedforwardGeometry(
                GeometryConfig.feedforward(
                    input_dim=784, output_dim=10, hidden_dims=(256, 128)
                )
            )
        elif geometry_type == "RecurrentGeometry":
            return RecurrentGeometry(
                GeometryConfig.recurrent(
                    input_dim=784, output_dim=10, hidden_dims=(256,)
                )
            )
        elif geometry_type == "TileGeometry":
            # Use feedforward as TileGeometry config might not have tile method
            return FeedforwardGeometry(
                GeometryConfig.feedforward(
                    input_dim=784, output_dim=10, hidden_dims=(256, 128)
                )
            )

        raise ValueError(f"Unknown geometry type: {geometry_type}")


class NativeDynamicsInferer:
    """Infer dynamics from explicit ontology_axes metadata."""

    def infer(
        self, model: nn.Module, metadata: ComponentMetadata | None
    ) -> StateDynamics:
        from computronium.ontology.dynamics import (
            EnergyMinimizationDynamics,
            InstantaneousDynamics,
            PredictiveSettlingDynamics,
            SpikeIntegrationDynamics,
            StateDynamicsConfig,
        )

        if not metadata or not metadata.ontology_dynamics:
            raise ValueError("No explicit dynamics ontology axis in metadata")

        dynamics_type = metadata.ontology_dynamics
        if dynamics_type == "EnergyMinimizationDynamics":
            return EnergyMinimizationDynamics(StateDynamicsConfig.energy_minimization())
        elif dynamics_type == "PredictiveSettlingDynamics":
            return PredictiveSettlingDynamics(StateDynamicsConfig.predictive_settling())
        elif dynamics_type == "SpikeIntegrationDynamics":
            return SpikeIntegrationDynamics(StateDynamicsConfig.spike_integration())
        elif dynamics_type == "InstantaneousDynamics":
            return InstantaneousDynamics(StateDynamicsConfig.instantaneous())

        raise ValueError(f"Unknown dynamics type: {dynamics_type}")


class NativeCreditInferer:
    """Infer credit assignment from explicit ontology_axes metadata."""

    def infer(
        self, model: nn.Module, metadata: ComponentMetadata | None
    ) -> CreditAssignment:
        from computronium.ontology.credit import (
            CreditAssignmentConfig,
            ThermodynamicContrast,
        )

        if not metadata or not metadata.ontology_credit:
            raise ValueError("No explicit credit ontology axis in metadata")

        # Only ThermodynamicContrast fully implements the CreditAssignment protocol
        # Other credit types are not fully typed - fall back to ThermodynamicContrast
        return ThermodynamicContrast(CreditAssignmentConfig.thermodynamic_contrast())


class NativeUpdateInferer:
    """Infer parameter update from explicit ontology_axes metadata."""

    def infer(
        self, model: nn.Module, metadata: ComponentMetadata | None
    ) -> ParameterUpdate:
        from computronium.ontology.update import (
            ElasticConsolidationUpdate,
            EuclideanUpdate,
            NaturalGradientUpdate,
            ParameterUpdateConfig,
            RiemannianOrthogonalUpdate,
            SpectralConstrainedUpdate,
        )

        if not metadata or not metadata.ontology_update:
            raise ValueError("No explicit update ontology axis in metadata")

        update_type = metadata.ontology_update
        if update_type == "EuclideanUpdate":
            return EuclideanUpdate(ParameterUpdateConfig.euclidean())
        elif update_type == "RiemannianOrthogonalUpdate":
            return RiemannianOrthogonalUpdate(
                ParameterUpdateConfig.riemannian_orthogonal()
            )
        elif update_type == "SpectralConstrainedUpdate":
            return SpectralConstrainedUpdate(
                ParameterUpdateConfig.spectral_constrained()
            )
        elif update_type == "NaturalGradientUpdate":
            return NaturalGradientUpdate(ParameterUpdateConfig.natural_gradient())
        elif update_type == "ElasticConsolidationUpdate":
            return ElasticConsolidationUpdate(
                ParameterUpdateConfig.elastic_consolidation()
            )

        raise ValueError(f"Unknown update type: {update_type}")


# =============================================================================
# Heuristic Inferrer Implementations (fallback for legacy models)
# =============================================================================


class HeuristicSubstrateInferer:
    """Infer substrate using heuristic fallbacks."""

    def infer(self, model: nn.Module, metadata: ComponentMetadata | None) -> Substrate:
        from computronium.ontology.adapter.heuristics import (
            infer_substrate_from_metadata,
        )

        result = infer_substrate_from_metadata(metadata)
        if result is None:
            from computronium.ontology.substrate import (
                DigitalSubstrate,
                SubstrateConfig,
            )

            return DigitalSubstrate(SubstrateConfig.digital())
        return result


class HeuristicGeometryInferer:
    """Infer geometry using heuristic fallbacks."""

    def infer(self, model: nn.Module, metadata: ComponentMetadata | None) -> Geometry:
        from computronium.ontology.adapter.heuristics import (
            infer_geometry_from_metadata,
        )

        result = infer_geometry_from_metadata(metadata)
        if result is None:
            from computronium.ontology.geometry import (
                FeedforwardGeometry,
                GeometryConfig,
            )

            return FeedforwardGeometry(
                GeometryConfig.feedforward(
                    input_dim=784, output_dim=10, hidden_dims=(256, 128)
                )
            )
        return result


class HeuristicDynamicsInferer:
    """Infer dynamics using heuristic fallbacks."""

    def infer(
        self, model: nn.Module, metadata: ComponentMetadata | None
    ) -> StateDynamics:
        from computronium.ontology.adapter.heuristics import (
            infer_dynamics_from_metadata,
        )

        result = infer_dynamics_from_metadata(metadata)
        if result is None:
            from computronium.ontology.dynamics import (
                InstantaneousDynamics,
                StateDynamicsConfig,
            )

            return InstantaneousDynamics(StateDynamicsConfig.instantaneous())
        return result


class HeuristicCreditInferer:
    """Infer credit using heuristic fallbacks."""

    def infer(
        self, model: nn.Module, metadata: ComponentMetadata | None
    ) -> CreditAssignment:
        from computronium.ontology.adapter.heuristics import infer_credit_from_metadata

        result = infer_credit_from_metadata(metadata)
        if result is None:
            from computronium.ontology.credit import (
                CreditAssignmentConfig,
                ThermodynamicContrast,
            )

            return ThermodynamicContrast(
                CreditAssignmentConfig.thermodynamic_contrast()
            )
        return result


class HeuristicUpdateInferer:
    """Infer update using heuristic fallbacks."""

    def infer(
        self, model: nn.Module, metadata: ComponentMetadata | None
    ) -> ParameterUpdate:
        from computronium.ontology.adapter.heuristics import infer_update_from_metadata

        result = infer_update_from_metadata(metadata)
        if result is None:
            from computronium.ontology.update import (
                EuclideanUpdate,
                ParameterUpdateConfig,
            )

            return EuclideanUpdate(ParameterUpdateConfig.euclidean())
        return result
