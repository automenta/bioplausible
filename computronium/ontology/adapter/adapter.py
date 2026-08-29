"""Main ModelAdapter: coordinates inferrers to build System from legacy models.

This is the primary facade for the Strangler Fig migration pattern.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from computronium.core.registry import ComponentMetadata

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
        System,
    )


@dataclass(frozen=True, slots=True)
class AdapterConfig:
    """Configuration for ModelAdapter behavior."""

    # Whether to use native (explicit ontology_axes) inference when available
    prefer_native: bool = True

    # Whether to validate the adapted system against the legacy model
    validate_on_adapt: bool = False

    # Validation tolerances (None = use family-specific)
    validation_rtol: float | None = None
    validation_atol: float | None = None


class ModelAdapter:
    """Adapt an existing registered model to the 5-D System interface.

    This enables the Strangler Fig migration: existing models stay registered
    and functional, but can be projected into the ontology for AutoScientist
    queries and cross-axis ablation studies.

    Inference priority:
    1. Registry metadata (most reliable - from @register_model decorator)
    2. Model attributes (backend, gradient_method, max_steps, etc.)
    3. Heuristics from class name / family tag
    4. Defaults (DigitalSubstrate, FeedforwardGeometry, InstantaneousDynamics, etc.)
    """

    def __init__(
        self,
        model: nn.Module,
        metadata: ComponentMetadata | None = None,
        config: AdapterConfig | None = None,
    ):
        self.model = model
        self._metadata = metadata
        self._config = config or AdapterConfig()

        # Lazy-loaded inferrers
        self._substrate_inferer: SubstrateInferer | None = None
        self._geometry_inferer: GeometryInferer | None = None
        self._dynamics_inferer: DynamicsInferer | None = None
        self._credit_inferer: CreditInferer | None = None
        self._update_inferer: UpdateInferer | None = None

    def to_system(self) -> "System":
        """Project model into 5-D ontology (best-effort inference)."""
        substrate = self._infer_substrate()
        geometry = self._infer_geometry()
        dynamics = self._infer_dynamics()
        credit = self._infer_credit()
        update = self._infer_update()

        return _AdaptedSystem(
            substrate=substrate,
            geometry=geometry,
            dynamics=dynamics,
            credit=credit,
            update=update,
            model=self.model,
        )

    def _infer_substrate(self) -> "Substrate":
        inferer = self._get_substrate_inferer()
        return inferer.infer(self.model, self._metadata)

    def _infer_geometry(self) -> "Geometry":
        inferer = self._get_geometry_inferer()
        return inferer.infer(self.model, self._metadata)

    def _infer_dynamics(self) -> "StateDynamics":
        inferer = self._get_dynamics_inferer()
        return inferer.infer(self.model, self._metadata)

    def _infer_credit(self) -> "CreditAssignment":
        inferer = self._get_credit_inferer()
        return inferer.infer(self.model, self._metadata)

    def _infer_update(self) -> "ParameterUpdate":
        inferer = self._get_update_inferer()
        return inferer.infer(self.model, self._metadata)

    def _get_substrate_inferer(self) -> "SubstrateInferer":
        if self._substrate_inferer is not None:
            return self._substrate_inferer

        if self._config.prefer_native and self._metadata and self._metadata.ontology_substrate:
            from computronium.ontology.adapter.inference import NativeSubstrateInferer
            self._substrate_inferer = NativeSubstrateInferer()
        else:
            from computronium.ontology.adapter.inference import HeuristicSubstrateInferer
            self._substrate_inferer = HeuristicSubstrateInferer()
        return self._substrate_inferer

    def _get_geometry_inferer(self) -> "GeometryInferer":
        if self._geometry_inferer is not None:
            return self._geometry_inferer

        if self._config.prefer_native and self._metadata and self._metadata.ontology_geometry:
            from computronium.ontology.adapter.inference import NativeGeometryInferer
            self._geometry_inferer = NativeGeometryInferer()
        else:
            from computronium.ontology.adapter.inference import HeuristicGeometryInferer
            self._geometry_inferer = HeuristicGeometryInferer()
        return self._geometry_inferer

    def _get_dynamics_inferer(self) -> "DynamicsInferer":
        if self._dynamics_inferer is not None:
            return self._dynamics_inferer

        if self._config.prefer_native and self._metadata and self._metadata.ontology_dynamics:
            from computronium.ontology.adapter.inference import NativeDynamicsInferer
            self._dynamics_inferer = NativeDynamicsInferer()
        else:
            from computronium.ontology.adapter.inference import HeuristicDynamicsInferer
            self._dynamics_inferer = HeuristicDynamicsInferer()
        return self._dynamics_inferer

    def _get_credit_inferer(self) -> "CreditInferer":
        if self._credit_inferer is not None:
            return self._credit_inferer

        if self._config.prefer_native and self._metadata and self._metadata.ontology_credit:
            from computronium.ontology.adapter.inference import NativeCreditInferer
            self._credit_inferer = NativeCreditInferer()
        else:
            from computronium.ontology.adapter.inference import HeuristicCreditInferer
            self._credit_inferer = HeuristicCreditInferer()
        return self._credit_inferer

    def _get_update_inferer(self) -> "UpdateInferer":
        if self._update_inferer is not None:
            return self._update_inferer

        if self._config.prefer_native and self._metadata and self._metadata.ontology_update:
            from computronium.ontology.adapter.inference import NativeUpdateInferer
            self._update_inferer = NativeUpdateInferer()
        else:
            from computronium.ontology.adapter.inference import HeuristicUpdateInferer
            self._update_inferer = HeuristicUpdateInferer()
        return self._update_inferer

    def validate(
        self,
        x: Tensor | None = None,
        y: Tensor | None = None,
        rtol: float | None = None,
        atol: float | None = None,
    ) -> dict[str, object]:
        """Validate the 5-D projection against the legacy model.

        Runs a forward/backward pass on both the legacy model and the adapted
        System, comparing key metrics (loss, gradients) to ensure the ontology
        projection preserves the model's learning behavior.

        Args:
            x: Input tensor. If None, generates synthetic data based on
               inferred input_dim.
            y: Target tensor. If None, generates synthetic labels based on
               inferred output_dim.
            rtol: Relative tolerance for metric comparison. If None, uses
                  family-specific tolerance from FAMILY_TOLERANCES.
            atol: Absolute tolerance for metric comparison. If None, uses
                  family-specific tolerance from FAMILY_TOLERANCES.

        Returns:
            Dictionary with validation results:
            - "passed": bool indicating if all checks passed
            - "legacy_metrics": metrics from legacy model train_step
            - "system_metrics": metrics from System train_step
            - "differences": dict of metric differences
            - "details": additional diagnostic info
        """
        from computronium.ontology.adapter.heuristics import get_family_tolerances

        # Use family-specific tolerances if not explicitly provided
        if rtol is None or atol is None:
            family_rtol, family_atol = get_family_tolerances(self._metadata.family if self._metadata else None)
            rtol = rtol if rtol is not None else family_rtol
            atol = atol if atol is not None else family_atol

        # Generate test data if not provided
        if x is None:
            input_dim = getattr(self.model, "input_dim", 10)
            x = torch.randn(4, input_dim)
        if y is None:
            output_dim = getattr(self.model, "output_dim", 3)
            y = torch.randint(0, output_dim, (x.shape[0],))

        # Ensure model is in train mode
        self.model.train()

        # Run legacy model train_step
        legacy_metrics: dict[str, object] = {}
        legacy_train_step = getattr(self.model, "train_step", None)
        if callable(legacy_train_step):
            try:
                legacy_result = legacy_train_step(x, y)
                if legacy_result is not None:
                    legacy_metrics = legacy_result  # type: ignore[assignment]
            except Exception as e:
                legacy_metrics = {"error": str(e)}

        # Run System train_step
        system = self.to_system()
        system_metrics: dict[str, object] = {}
        try:
            system_metrics = system.train_step(x, y)  # type: ignore[assignment]
        except Exception as e:
            system_metrics = {"error": str(e)}

        # Compare metrics
        differences, all_passed = self._compare_metrics(
            legacy_metrics, system_metrics, rtol, atol
        )

        return {
            "passed": all_passed,
            "legacy_metrics": legacy_metrics,
            "system_metrics": system_metrics,
            "differences": differences,
            "details": {
                "rtol": rtol,
                "atol": atol,
                "input_shape": tuple(x.shape),
                "target_shape": tuple(y.shape),
                "family": self._metadata.family if self._metadata else "unknown",
            },
        }

    @staticmethod
    def _compare_metrics(
        legacy: dict[str, object],
        system: dict[str, object],
        rtol: float,
        atol: float,
    ) -> tuple[dict[str, dict[str, object]], bool]:
        """Compare legacy and system metrics, return differences and pass status."""
        differences: dict[str, dict[str, object]] = {}
        all_passed = True

        for key in set(legacy.keys()) | set(system.keys()):
            legacy_val = legacy.get(key)
            system_val = system.get(key)
            if legacy_val is not None and system_val is not None:
                if isinstance(legacy_val, (int, float)) and isinstance(system_val, (int, float)):
                    diff = abs(legacy_val - system_val)
                    rel_diff = diff / (abs(legacy_val) + atol)
                    differences[key] = {
                        "legacy": legacy_val,
                        "system": system_val,
                        "abs_diff": diff,
                        "rel_diff": rel_diff,
                    }
                    if rel_diff > rtol and diff > atol:
                        all_passed = False
                elif isinstance(legacy_val, Tensor) and isinstance(system_val, Tensor):
                    diff = (legacy_val - system_val).abs().max().item()
                    differences[key] = {"abs_diff": diff}
                    if diff > atol:
                        all_passed = False
                else:
                    differences[key] = {
                        "legacy": legacy_val,
                        "system": system_val,
                        "type_mismatch": True,
                    }
                    all_passed = False
            else:
                differences[key] = {
                    "legacy": legacy_val,
                    "system": system_val,
                    "missing": True,
                }
                all_passed = False

        return differences, all_passed


class _AdaptedSystem:
    """Internal adapter wrapping a model as a System."""

    def __init__(
        self,
        substrate: "Substrate",
        geometry: "Geometry",
        dynamics: "StateDynamics",
        credit: "CreditAssignment",
        update: "ParameterUpdate",
        model: nn.Module,
    ):
        self.substrate = substrate
        self.geometry = geometry
        self.dynamics = dynamics
        self.credit = credit
        self.update = update
        self._model = model

    def train_step(self, x: Tensor, y: Tensor) -> dict[str, float]:
        # Delegate to model's training step if available
        if hasattr(self._model, "train_step"):
            return self._model.train_step(x, y)
        return {"loss": 0.0}

    def forward(self, x: Tensor) -> Tensor:
        return self._model(x)

    def to_spec(self) -> dict:
        return {
            "schema_version": "1.0",
            "substrate": self.substrate.config.__dict__,
            "geometry": self.geometry.config.__dict__,
            "dynamics": self.dynamics.config.__dict__,
            "credit": self.credit.config.__dict__,
            "update": self.update.config.__dict__,
        }

    @classmethod
    def from_spec(cls, spec: dict) -> "System":
        raise NotImplementedError("Cannot reconstruct adapted system from spec")


# Re-export inferrer protocols for external use
from computronium.ontology.adapter.inference import (
    SubstrateInferer,
    GeometryInferer,
    DynamicsInferer,
    CreditInferer,
    UpdateInferer,
)

__all__ = [
    "AdapterConfig",
    "ModelAdapter",
    "SubstrateInferer",
    "GeometryInferer",
    "DynamicsInferer",
    "CreditInferer",
    "UpdateInferer",
]