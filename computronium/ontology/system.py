"""System Composition: 6-D Ontology (S ⊗ G ⊗ D ⊗ M ⊗ C ⊗ U)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, TypeVar, cast, runtime_checkable

import torch
from torch import Tensor, nn

from computronium.core.registry import ComponentMetadata, ComputeProfile
from computronium.ontology.credit import (
    CreditAssignment,
    CreditAssignmentConfig,
    GradientCredit,
)
from computronium.ontology.dynamics import (
    InstantaneousDynamics,
    StateDynamics,
    StateDynamicsConfig,
)
from computronium.ontology.geometry import FeedforwardGeometry, Geometry, GeometryConfig
from computronium.ontology.substrate import (
    AnalogSubstrate,
    DigitalSubstrate,
    MemristiveSubstrate,
    NeuromorphicSubstrate,
    OpticalSubstrate,
    QuantumSubstrate,
    Substrate,
    SubstrateConfig,
)
from computronium.ontology.update import (
    EuclideanUpdate,
    ParameterUpdate,
    ParameterUpdateConfig,
)
from computronium.state import PlasticityConfig

if TYPE_CHECKING:
    from collections.abc import Callable

# ============================================================
# SystemState: Mutable state for 5-D pipeline
# ============================================================


@dataclass(frozen=False, slots=True)
class SystemState:
    """Mutable state carried through the 5-layer pipeline.

    This is the single state object threaded through Substrate → Geometry
    → StateDynamics → CreditAssignment → ParameterUpdate. Each layer reads
    and writes a defined subset of fields.

    Attributes:
        x: Input batch
        y: Target batch
        activations: Current layer activations (list for multi-layer, tensor for single)
        free_state: Settled free-phase state
        nudged_state: Settled nudged-phase state
        pseudo_gradients: Computed pseudo-gradients per layer
        energy: Current energy value
        loss: Current loss value
        metrics: Accumulated metrics dict
        spike_counts: Per-step per-neuron spike counts (for spiking dynamics)
    """

    x: Tensor | None = None
    y: Tensor | None = None
    activations: list[Tensor] | Tensor | None = None
    free_state: list[Tensor] | Tensor | None = None
    nudged_state: list[Tensor] | Tensor | None = None
    pseudo_gradients: list[Tensor] | None = None
    energy: Tensor | float | None = None
    loss: Tensor | float | None = None
    metrics: dict[str, float] = None  # type: ignore[assignment]
    spike_counts: list[Tensor] | None = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


# ============================================================
# Phase enum (re-exported for convenience)
# ============================================================


Phase = StrEnum(
    "Phase",
    {
        "FREE": "free",
        "NUDGED": "nudged",
    },
)


# ============================================================
# System Protocol: The 5-Layer Tensor Product
# ============================================================


TS = TypeVar("TS", bound=Substrate)
TG = TypeVar("TG", bound=Geometry)
TD = TypeVar("TD", bound=StateDynamics)
TC = TypeVar("TC", bound=CreditAssignment)
TU = TypeVar("TU", bound=ParameterUpdate)


@runtime_checkable
class System(Protocol[TS, TG, TD, TC, TU]):
    """Composable computronium system: S ⊗ G ⊗ D ⊗ C ⊗ U.

    The tensor product of five orthogonal primitives. Every model in the
    zoo is a coordinate in this 5-D space. The System orchestrates the
    strict pipeline:

        Substrate.forward_op → Geometry.route → StateDynamics.settle
        → CreditAssignment.compute_pseudo_gradient → ParameterUpdate.step

    Type parameters ensure only valid compositions compile:
    - SpikingDynamics requires STDP CreditAssignment
    - EnergyMinimization requires ThermodynamicContrast
    - TileMesh Geometry requires Tile-aware StateDynamics
    """

    substrate: TS
    geometry: TG
    dynamics: TD
    credit: TC
    update: TU

    def train_step(self, x: Tensor, y: Tensor) -> dict[str, float]:
        """Execute one training step through the family-neutral pipeline."""
        from computronium.core.pipeline import run_train_step

        return run_train_step(
            self.substrate,
            self.geometry,
            self.dynamics,
            self.credit,
            self.update,
            x,
            y,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Inference forward pass (free phase only, no weight updates)."""
        from computronium.core.pipeline import run_forward

        return run_forward(self.substrate, self.geometry, self.dynamics, x)

    def to_spec(self) -> dict[str, object]:
        """Serialize the System to a specification dictionary.

        Returns:
            Dictionary containing schema_version and all 5 axis configs.
        """
        ...

    @classmethod
    def from_spec(cls, spec: dict[str, object]) -> System[TS, TG, TD, TC, TU]:
        """Reconstruct a System from a specification dictionary.

        Args:
            spec: Dictionary with schema_version and 5 axis configs.

        Returns:
            A composed System instance.
        """
        ...


# ============================================================
# SystemConfig: Validated Composition of 6-D Ontology
# ============================================================


# Family-specific tolerances for validation
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


@dataclass(frozen=True, slots=True)
class SystemConfig:
    """Validated composition of 6-D ontology — single source of truth for a system.

    Composes the six orthogonal axes (S ⊗ G ⊗ D ⊗ M ⊗ C ⊗ U) and provides
    validated, cross-validated access. The 6th axis (M = Plasticity) enables
    meta-dynamics and fast plastic state evolution.
    """

    substrate: SubstrateConfig
    geometry: GeometryConfig
    dynamics: StateDynamicsConfig
    plasticity: PlasticityConfig
    credit: CreditAssignmentConfig
    update: ParameterUpdateConfig

    def __init__(
        self,
        substrate: SubstrateConfig,
        geometry: GeometryConfig,
        dynamics: StateDynamicsConfig,
        credit: CreditAssignmentConfig,
        update: ParameterUpdateConfig,
        plasticity: PlasticityConfig | None = None,
    ):
        object.__setattr__(self, "substrate", substrate)
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "dynamics", dynamics)
        object.__setattr__(self, "credit", credit)
        object.__setattr__(self, "update", update)
        object.__setattr__(
            self,
            "plasticity",
            plasticity if plasticity is not None else PlasticityConfig.null(),
        )

    def validate(self) -> None:  # ruff: ignore[complex-structure, too-many-branches]
        """Cross-axis validation (hard constraints only).

        Raises:
            ValueError: If configuration violates hard compatibility constraints.
        """
        # Recurrent geometry requires energy-based dynamics
        if self.geometry.topology_type in ("recurrent", "recurrent_attractor"):  # ruff: ignore[literal-membership, collapsible-if]
            if self.dynamics.dynamics_type != "energy_minimization":
                raise ValueError(
                    f"Recurrent geometry (topology_type={self.geometry.topology_type!r}) "
                    f"requires energy_minimization dynamics, got {self.dynamics.dynamics_type!r}"
                )

        # Thermodynamic contrast credit requires energy-based dynamics
        if self.credit.credit_type in ("thermodynamic_contrast", "equilibrium"):  # ruff: ignore[literal-membership, collapsible-if]
            if self.dynamics.dynamics_type != "energy_minimization":
                raise ValueError(
                    f"Thermodynamic contrast credit (credit_type={self.credit.credit_type!r}) "
                    f"requires energy_minimization dynamics, got {self.dynamics.dynamics_type!r}"
                )

        # Spiking dynamics requires temporal trace or STDP credit
        if self.dynamics.dynamics_type == "spike_integration":  # ruff: ignore[collapsible-if]
            if self.credit.credit_type not in (  # ruff: ignore[literal-membership]
                "temporal_trace",
                "spiking",
                "target_inversion",
                "target_prop",
            ):
                raise ValueError(
                    f"Spike integration dynamics requires temporal trace or target inversion credit, "
                    f"got {self.credit.credit_type!r}"
                )

        # Tile mesh geometry requires compatible dynamics
        if self.geometry.topology_type in ("tile_mesh", "tile"):  # ruff: ignore[literal-membership, collapsible-if]
            if self.dynamics.dynamics_type not in (  # ruff: ignore[literal-membership]
                "energy_minimization",
                "instantaneous",
            ):
                raise ValueError(
                    f"Tile mesh geometry requires energy_minimization or instantaneous dynamics, "
                    f"got {self.dynamics.dynamics_type!r}"
                )

        # Beta matching: StateDynamics.beta should match CreditAssignment.beta for energy-based systems
        if self.dynamics.dynamics_type == "energy_minimization":  # ruff: ignore[collapsible-if]
            if abs(self.dynamics.beta - self.credit.beta) > 1e-6:
                # Soft constraint: warn but don't fail
                warnings.warn(
                    f"Beta mismatch: dynamics.beta={self.dynamics.beta} != credit.beta={self.credit.beta}. "
                    f"This may cause incorrect gradient scaling in EqProp.",
                    UserWarning,
                    stacklevel=2,
                )

        # ============================================================
        # Substrate-Dynamics Compatibility Constraints
        # ============================================================

        # Neuromorphic substrate requires spike integration or energy minimization dynamics
        # (instantaneous pass-through doesn't capture neuromorphic temporal dynamics)
        if self.substrate.precision == "float16" and self.substrate.sparsity > 0.9:  # ruff: ignore[collapsible-if]
            # Likely neuromorphic substrate
            if self.dynamics.dynamics_type not in (  # ruff: ignore[literal-membership]
                "spike_integration",
                "energy_minimization",
                "diffusion",
            ):
                raise ValueError(
                    f"Neuromorphic substrate (precision={self.substrate.precision}, "
                    f"sparsity={self.substrate.sparsity}) requires temporal dynamics "
                    f"(spike_integration, energy_minimization, or diffusion), "
                    f"got {self.dynamics.dynamics_type!r}"
                )

        # Analog substrate with noise requires dynamics that support noise injection
        if self.substrate.precision == "float32" and self.substrate.noise_level > 0.0:  # ruff: ignore[collapsible-if]
            if self.dynamics.dynamics_type == "instantaneous":
                # Instantaneous dynamics doesn't use substrate noise during settling
                # (only single forward pass). Warn but don't fail.
                warnings.warn(
                    f"Analog substrate with noise_level={self.substrate.noise_level} "
                    f"used with instantaneous dynamics. Noise only applied once at input. "
                    f"Consider energy_minimization or diffusion dynamics for continuous noise injection.",
                    UserWarning,
                    stacklevel=2,
                )

        # Complex substrate requires compatible credit assignment
        # Complex/holomorphic networks work best with thermodynamic contrast
        # (phase-sensitive gradients) or backprop (Wirtinger calculus)
        if self.substrate.precision == "float32" and getattr(  # ruff: ignore[collapsible-if]
            self.substrate, "_complex_emulated", False
        ):
            # This is a complex substrate (emulated via real/imag channels)
            if self.credit.credit_type not in (  # ruff: ignore[literal-membership]
                "thermodynamic_contrast",
                "equilibrium",
                "gradient",
                "backprop",
            ):
                warnings.warn(
                    f"Complex substrate used with {self.credit.credit_type!r} credit. "
                    f"Best results with thermodynamic_contrast (holomorphic EqProp) "
                    f"or gradient (holomorphic backprop).",
                    UserWarning,
                    stacklevel=2,
                )

        # Quantum substrate requires compatible dynamics
        # Quantum circuits need energy-based or instantaneous dynamics
        if self.substrate.precision == "complex64":
            if self.dynamics.dynamics_type not in (  # ruff: ignore[literal-membership]
                "energy_minimization",
                "instantaneous",
                "diffusion",
            ):
                raise ValueError(
                    f"Quantum substrate requires energy_minimization, instantaneous, "
                    f"or diffusion dynamics, got {self.dynamics.dynamics_type!r}"
                )

            # Quantum substrate with thermodynamic contrast needs matching beta
            if self.credit.credit_type in ("thermodynamic_contrast", "equilibrium"):  # ruff: ignore[literal-membership, collapsible-if]
                if abs(self.dynamics.beta - self.credit.beta) > 1e-6:
                    warnings.warn(
                        f"Quantum substrate with thermodynamic contrast: "
                        f"beta mismatch dynamics.beta={self.dynamics.beta} != credit.beta={self.credit.beta}. "
                        f"Phase-sensitive gradients require matched beta.",
                        UserWarning,
                        stacklevel=2,
                    )

        # Sparse substrate requires compatible update rule
        # Sparse weights need updates that preserve sparsity structure
        if self.substrate.sparsity > 0.5:  # ruff: ignore[collapsible-if]
            if self.update.update_type == "riemannian_orthogonal":
                warnings.warn(
                    f"Sparse substrate (sparsity={self.substrate.sparsity}) with "
                    f"RiemannianOrthogonalUpdate may densify weights. "
                    f"Consider SpectralConstrainedUpdate or ElasticConsolidationUpdate.",
                    UserWarning,
                    stacklevel=2,
                )

        # Ternary substrate requires compatible credit assignment
        # Ternary quantization works best with equilibrium/thermodynamic contrast
        # (contrastive learning naturally handles weight quantization)
        if (  # ruff: ignore[collapsible-if]
            self.substrate.precision == "float32"
            and self.substrate.sparsity == 0.0
            and self.substrate.weight_bounds == (-1.0, 1.0)
        ):
            # Heuristic: likely ternary substrate (sparsity emerges from thresholding)
            if self.credit.credit_type not in (  # ruff: ignore[literal-membership]
                "thermodynamic_contrast",
                "equilibrium",
                "gradient",
                "backprop",
            ):
                warnings.warn(
                    f"Ternary-like substrate used with {self.credit.credit_type!r} credit. "
                    f"Best results with thermodynamic_contrast (Ternary EqProp) "
                    f"or gradient (Ternary backprop with STE).",
                    UserWarning,
                    stacklevel=2,
                )

        # Diffusion dynamics requires noise-aware substrate
        if self.dynamics.dynamics_type == "diffusion":  # ruff: ignore[collapsible-if]
            if self.substrate.noise_level == 0.0:
                warnings.warn(
                    "Diffusion dynamics (Langevin) requires substrate noise_level > 0 "
                    "for proper sampling. Consider setting noise_level on substrate.",
                    UserWarning,
                    stacklevel=2,
                )

        # Predictive settling dynamics requires compatible credit
        # PC uses local errors, works with thermodynamic contrast or local goodness
        if self.dynamics.dynamics_type == "predictive_settling":  # ruff: ignore[collapsible-if]
            if self.credit.credit_type not in (  # ruff: ignore[literal-membership]
                "thermodynamic_contrast",
                "equilibrium",
                "local_goodness",
                "forward_only",
            ):
                raise ValueError(
                    f"Predictive settling dynamics requires thermodynamic_contrast, "
                    f"local_goodness, or forward_only credit, got {self.credit.credit_type!r}"
                )

        # Energy minimization with momentum requires compatible update
        if (
            self.dynamics.dynamics_type == "energy_minimization"
            and self.dynamics.momentum > 0.0
        ) and self.update.update_type == "riemannian_orthogonal":
            warnings.warn(
                f"EnergyMinimizationDynamics with momentum={self.dynamics.momentum} "
                f"combined with RiemannianOrthogonalUpdate may cause instability. "
                f"Consider EuclideanUpdate with momentum.",
                UserWarning,
                stacklevel=2,
            )

        # Geometry-Substrate constraints
        # Spatial lattice / neuromorphic geometry requires neuromorphic substrate
        if self.geometry.topology_type in ("spatial_lattice", "neuromorphic", "fabric"):  # ruff: ignore[literal-membership, collapsible-if]
            if not (
                self.substrate.precision == "float16" and self.substrate.sparsity > 0.9
            ):
                warnings.warn(
                    f"Spatial/neuromorphic geometry ({self.geometry.topology_type}) "
                    f"works best with neuromorphic substrate (float16, high sparsity). "
                    f"Current substrate: precision={self.substrate.precision}, "
                    f"sparsity={self.substrate.sparsity}",
                    UserWarning,
                    stacklevel=2,
                )

        # Tile mesh geometry with sparse substrate
        if (
            self.geometry.topology_type in ("tile_mesh", "tile")  # ruff: ignore[literal-membership]
            and self.substrate.sparsity > 0.5
        ):
            warnings.warn(
                f"Tile mesh geometry with sparse substrate (sparsity={self.substrate.sparsity}) "
                f"may benefit from structured sparsity (N:M or block) for efficient matmul.",
                UserWarning,
                stacklevel=2,
            )

    @classmethod
    def valid_combinations(cls) -> list[dict[str, str]]:  # ruff: ignore[complex-structure, too-many-branches]
        """Return all valid 6-D coordinate combinations for AutoScientist.

        Returns:
            List of dicts, each representing a valid combination of
            substrate, geometry, dynamics, plasticity, credit, update.
            These are the coordinates that pass cross-axis validation.
        """
        # Core valid combinations derived from validation rules
        combinations = []

        # Substrate types
        substrates = [
            {
                "type": "digital",
                "precision": "float32",
                "noise_level": 0.0,
                "sparsity": 0.0,
            },
            {
                "type": "memristive",
                "precision": "float32",
                "noise_level": 0.01,
                "sparsity": 0.0,
            },
            {
                "type": "neuromorphic",
                "precision": "float16",
                "noise_level": 0.0,
                "sparsity": 0.95,
            },
            {
                "type": "optical",
                "precision": "float32",
                "noise_level": 0.0,
                "sparsity": 0.0,
            },
            {
                "type": "quantum",
                "precision": "complex64",
                "noise_level": 0.0,
                "sparsity": 0.0,
            },
            {
                "type": "sparse",
                "precision": "float32",
                "noise_level": 0.0,
                "sparsity": 0.8,
            },
            {
                "type": "ternary",
                "precision": "float32",
                "noise_level": 0.0,
                "sparsity": 0.0,
            },
        ]

        # Geometry types
        geometries = [
            {
                "topology_type": "feedforward",
                "input_dim": 784,
                "output_dim": 10,
                "hidden_dims": [256, 128],
            },
            {
                "topology_type": "recurrent",
                "input_dim": 784,
                "output_dim": 10,
                "hidden_dims": [256],
            },
            {
                "topology_type": "tile_mesh",
                "input_dim": 784,
                "output_dim": 10,
                "num_layers": 4,
                "neurons_per_tile": 64,
                "tiles_per_layer": 4,
            },
        ]

        # Dynamics types
        dynamics_options = [
            {"dynamics_type": "energy_minimization", "max_steps": 20, "beta": 0.5},
            {"dynamics_type": "predictive_settling", "max_steps": 20, "beta": 0.5},
            {"dynamics_type": "spike_integration", "max_steps": 50, "beta": 0.5},
            {"dynamics_type": "instantaneous", "max_steps": 1, "beta": 0.5},
            {"dynamics_type": "diffusion", "max_steps": 100, "beta": 0.5},
        ]

        # Plasticity types
        plasticities = [
            {"type": "null"},
            {"type": "routing", "gate_dim": 64},
            {
                "type": "fast_weights",
                "fast_weight_dim": 512,
                "decay": 0.9,
                "learning_rate": 0.1,
            },
            {"type": "substrate_coupled"},
        ]

        # Credit types
        credits = [
            {"credit_type": "thermodynamic_contrast", "beta": 0.5},
            {"credit_type": "random_projections", "beta": 0.5},
            {"credit_type": "local_goodness", "beta": 0.5},
            {"credit_type": "temporal_trace", "beta": 0.5},
            {"credit_type": "target_inversion", "beta": 0.5},
            {"credit_type": "gradient", "beta": 0.5},
        ]

        # Update types
        updates = [
            {"update_type": "euclidean", "step_size": 0.01},
            {"update_type": "riemannian_orthogonal", "step_size": 0.01},
            {"update_type": "spectral_constrained", "step_size": 0.01},
            {"update_type": "natural_gradient", "step_size": 0.01},
            {"update_type": "elastic_consolidation", "step_size": 0.01},
        ]

        # Generate combinations and validate
        for sub in substrates:  # ruff: ignore[too-many-nested-blocks]
            for geom in geometries:
                for dyn in dynamics_options:
                    for plas in plasticities:
                        for cred in credits:
                            for upd in updates:
                                coord = {
                                    "substrate": sub,
                                    "geometry": geom,
                                    "dynamics": dyn,
                                    "plasticity": plas,
                                    "credit": cred,
                                    "update": upd,
                                }
                                # Quick validation: skip known invalid combos
                                # Recurrent geometry requires energy_minimization
                                if geom["topology_type"] in (  # ruff: ignore[literal-membership, collapsible-if]
                                    "recurrent",
                                    "recurrent_attractor",
                                ):
                                    if dyn["dynamics_type"] != "energy_minimization":
                                        continue
                                # Thermodynamic contrast requires energy_minimization
                                if cred["credit_type"] in (  # ruff: ignore[literal-membership, collapsible-if]
                                    "thermodynamic_contrast",
                                    "equilibrium",
                                ):
                                    if dyn["dynamics_type"] != "energy_minimization":
                                        continue
                                # Spike integration requires temporal trace or target inversion credit
                                if dyn["dynamics_type"] == "spike_integration":  # ruff: ignore[collapsible-if]
                                    if cred["credit_type"] not in (  # ruff: ignore[literal-membership]
                                        "temporal_trace",
                                        "target_inversion",
                                        "target_prop",
                                    ):
                                        continue
                                # Tile mesh requires compatible dynamics
                                if geom["topology_type"] in ("tile_mesh", "tile"):  # ruff: ignore[literal-membership, collapsible-if]
                                    if dyn["dynamics_type"] not in (  # ruff: ignore[literal-membership]
                                        "energy_minimization",
                                        "instantaneous",
                                    ):
                                        continue
                                # Quantum substrate requires compatible dynamics
                                if sub["precision"] == "complex64":  # ruff: ignore[collapsible-if]
                                    if dyn["dynamics_type"] not in (  # ruff: ignore[literal-membership]
                                        "energy_minimization",
                                        "instantaneous",
                                        "diffusion",
                                    ):
                                        continue
                                # Predictive settling requires compatible credit
                                if dyn["dynamics_type"] == "predictive_settling":  # ruff: ignore[collapsible-if]
                                    if cred["credit_type"] not in (  # ruff: ignore[literal-membership]
                                        "thermodynamic_contrast",
                                        "equilibrium",
                                        "local_goodness",
                                        "forward_only",
                                    ):
                                        continue

                                combinations.append(coord)

        return combinations


# ============================================================
# ModelAdapter: Wrap existing models as System compositions
# ============================================================


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

    def __init__(self, model: nn.Module, metadata: ComponentMetadata | None = None):
        self.model = model
        self._metadata = metadata

    def _get_family_tolerances(self) -> tuple[float, float]:
        """Get family-specific tolerances based on model metadata."""
        if self._metadata and self._metadata.family:
            family = self._metadata.family.lower()
            # Check for exact match first
            if family in FAMILY_TOLERANCES:
                return FAMILY_TOLERANCES[family]
            # Check for partial matches
            for key, tol in FAMILY_TOLERANCES.items():
                if key != "default" and key in family:
                    return tol
        return FAMILY_TOLERANCES["default"]

    def to_system(
        self,
    ) -> System[Substrate, Geometry, StateDynamics, CreditAssignment, ParameterUpdate]:
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

    def _infer_substrate(self) -> Substrate:
        # Priority 1: Registry metadata compute_profile
        substrate = self._infer_substrate_from_compute_profile()
        if substrate is not None:
            return substrate

        # Priority 2: Model attributes
        substrate = self._infer_substrate_from_backend()
        if substrate is not None:
            return substrate

        # Priority 3: Family tag heuristics
        substrate = self._infer_substrate_from_family()
        if substrate is not None:
            return substrate

        return DigitalSubstrate(
            SubstrateConfig(
                precision="float32",
                noise_level=0.0,
                weight_bounds=None,
                sparsity=0.0,
                device="cpu",
            )
        )

    def _infer_substrate_from_compute_profile(self) -> Substrate | None:
        if not (self._metadata and self._metadata.compute_profile):
            return None
        profile = self._metadata.compute_profile
        if profile == ComputeProfile.ANALOG:
            return AnalogSubstrate(
                SubstrateConfig(
                    precision="float32",
                    noise_level=0.0,
                    weight_bounds=None,
                    sparsity=0.0,
                    device="cpu",
                )
            )
        if profile == ComputeProfile.OPTICAL:
            return OpticalSubstrate(
                SubstrateConfig(
                    precision="float32",
                    noise_level=0.0,
                    weight_bounds=None,
                    sparsity=0.0,
                    device="cpu",
                )
            )
        if profile == ComputeProfile.MEMRISTOR:
            return MemristiveSubstrate(
                SubstrateConfig(
                    precision="float32",
                    noise_level=0.0,
                    weight_bounds=None,
                    sparsity=0.0,
                    device="cpu",
                )
            )
        if profile == ComputeProfile.NEUROMORPHIC:
            return NeuromorphicSubstrate(
                SubstrateConfig(
                    precision="float32",
                    noise_level=0.0,
                    weight_bounds=None,
                    sparsity=0.0,
                    device="cpu",
                )
            )
        return None

    def _infer_substrate_from_backend(self) -> Substrate | None:
        # Check model attributes for backend hints
        if hasattr(self.model, "backend"):
            backend = getattr(self.model, "backend", "").lower()
            if "analog" in backend:
                return AnalogSubstrate(SubstrateConfig.analog())
            if "memrist" in backend:
                return MemristiveSubstrate(SubstrateConfig.memristive())
            if "neuromorph" in backend:
                return NeuromorphicSubstrate(SubstrateConfig.neuromorphic())
            if "optical" in backend or "photonic" in backend:
                return OpticalSubstrate(SubstrateConfig.optical())
            if "quantum" in backend:
                return QuantumSubstrate(SubstrateConfig.quantum())
        return None

    def _infer_substrate_from_family(self) -> Substrate | None:
        if not (self._metadata and self._metadata.family):
            return None
        family = self._metadata.family.lower()
        if "spiking" in family or "snn" in family or "stdp" in family:
            return NeuromorphicSubstrate(SubstrateConfig.neuromorphic())
        if "tile" in family or "equitile" in family:
            return DigitalSubstrate(SubstrateConfig.digital())
        return None

    def _infer_geometry(self) -> Geometry:
        # Simplified: return FeedforwardGeometry
        return FeedforwardGeometry(
            GeometryConfig.feedforward(
                input_dim=784, output_dim=10, hidden_dims=(256, 128)
            )
        )

    def _infer_dynamics(self) -> StateDynamics:
        # Simplified: return InstantaneousDynamics
        return InstantaneousDynamics(StateDynamicsConfig.instantaneous())

    def _infer_credit(self) -> CreditAssignment:
        # Simplified: return GradientCredit
        return GradientCredit(CreditAssignmentConfig.gradient())

    def _infer_update(self) -> ParameterUpdate:
        # Simplified: return EuclideanUpdate
        return EuclideanUpdate(ParameterUpdateConfig.euclidean())

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
        # Use family-specific tolerances if not explicitly provided
        if rtol is None or atol is None:
            family_rtol, family_atol = self._get_family_tolerances()
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
        else:
            legacy_metrics = self._standard_metrics(x, y)

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

    def _standard_metrics(self, x: Tensor, y: Tensor) -> dict[str, object]:
        """Loss/accuracy for models without a ``train_step`` method."""
        from computronium.core.losses import compute_loss

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            loss = compute_loss(nn.CrossEntropyLoss(), logits, y)
        if was_training:
            self.model.train()
        acc = (logits.argmax(-1) == y).float().mean().item()
        return {"loss": float(loss.item()), "accuracy": acc}

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
                if isinstance(legacy_val, (int, float)) and isinstance(
                    system_val, (int, float)
                ):
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
        substrate: Substrate,
        geometry: Geometry,
        dynamics: StateDynamics,
        credit: CreditAssignment,
        update: ParameterUpdate,
        model: nn.Module,
    ):
        self.substrate = substrate
        self.geometry = geometry
        self.dynamics = dynamics
        self.credit = credit
        self.update = update
        self._model = model
        self._optimizer: torch.optim.Optimizer | None = None

    def train_step(self, x: Tensor, y: Tensor) -> dict[str, float]:
        model_train_step = cast(
            "Callable[[Tensor, Tensor], dict[str, float]] | None",
            getattr(self._model, "train_step", None),
        )
        if model_train_step is not None:
            return model_train_step(x, y)
        from computronium.core.trainer import bptt_step

        if self._optimizer is None:
            self._optimizer = torch.optim.SGD(
                self._model.parameters(), lr=self.update.config.step_size
            )
        out = bptt_step(self._model, self._optimizer, x, y)
        logits: Tensor = out["logits"]  # type: ignore[assignment]
        acc = (logits.argmax(-1) == y).float().mean().item()
        loss_raw = out["loss"]
        if not isinstance(loss_raw, int | float | Tensor):
            raise TypeError(f"bptt_step returned non-numeric loss: {type(loss_raw)!r}")
        return {"loss": float(loss_raw), "accuracy": acc}

    def forward(self, x: Tensor) -> Tensor:
        return self._model(x)

    def to_spec(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "substrate": self.substrate.config.__dict__,
            "geometry": self.geometry.config.__dict__,
            "dynamics": self.dynamics.config.__dict__,
            "credit": self.credit.config.__dict__,
            "update": self.update.config.__dict__,
        }

    @classmethod
    def from_spec(cls, spec: dict[str, object]) -> System[TS, TG, TD, TC, TU]:
        raise NotImplementedError("Cannot reconstruct adapted system from spec")


def substrate_from_config(config: SubstrateConfig) -> Substrate:  # ruff: ignore[too-many-return-statements]
    """Factory function to instantiate substrate from config."""
    from computronium.ontology.substrate import SubstrateType

    match config.substrate_type:
        case SubstrateType.DIGITAL:
            from computronium.ontology.substrate import DigitalSubstrate

            return DigitalSubstrate(config)
        case SubstrateType.ANALOG:
            from computronium.ontology.substrate import AnalogSubstrate

            return AnalogSubstrate(config)
        case SubstrateType.MEMRISTIVE:
            from computronium.ontology.substrate import MemristiveSubstrate

            return MemristiveSubstrate(config)
        case SubstrateType.NEUROMORPHIC:
            from computronium.ontology.substrate import NeuromorphicSubstrate

            return NeuromorphicSubstrate(config)
        case SubstrateType.OPTICAL:
            from computronium.ontology.substrate import OpticalSubstrate

            return OpticalSubstrate(config)
        case SubstrateType.QUANTUM:
            from computronium.ontology.substrate import QuantumSubstrate

            return QuantumSubstrate(config)
        case SubstrateType.SPARSE:
            from computronium.ontology.substrate import SparseSubstrate

            return SparseSubstrate(config)
        case SubstrateType.TERNARY:
            from computronium.ontology.substrate import TernarySubstrate

            return TernarySubstrate(config)
        case _:
            # Complex substrate uses DIGITAL type with special config
            if config.precision == "float32" and getattr(
                config, "_complex_emulated", False
            ):
                from computronium.ontology.substrate import ComplexSubstrate

                return ComplexSubstrate(config)
            from computronium.ontology.substrate import DigitalSubstrate

            return DigitalSubstrate(config)
