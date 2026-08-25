"""5-Dimensional Physico-Computational Ontology for Bioplausible Systems.

This module defines the five orthogonal axes (S x G x D x C x U) that compose
any computronium neural network. The tensor product of these primitives
replaces the flat registry of 111+ hardcoded model permutations with a
generative, mathematically rigorous composition engine.

Ontology Layers:
    1. Substrate (S) - Physical state space constraints (precision, noise, sparsity)
    2. Geometry (G) - Topology & routing (spatial arrangement of nodes)
    3. StateDynamics (D) - Forward evolution & settling (how activations evolve)
    4. CreditAssignment (C) - Error routing & pseudo-gradients (learning signal)
    5. ParameterUpdate (U) - Optimization rule (how pseudo-gradients become ΔW)

Each layer is a Protocol enabling structural typing and zero-cost abstraction.
The composing System[TS, TG, TD, TC, TU] uses PEP 695 generics for full
type safety: invalid compositions are caught at type-check time.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

    from computronium.config.experiment import ExperimentConfig

import torch
from torch import Tensor, nn

from computronium.core.joint.transition import PlasticityConfig
from computronium.core.registry import ComponentMetadata, ComputeProfile, LocalityLevel

if TYPE_CHECKING:
    from computronium.core.substrates.complex_substrate import ComplexSubstrate
    from computronium.core.substrates.sparse_substrate import SparseSubstrate
    from computronium.core.substrates.ternary_substrate import TernarySubstrate
from computronium.core.tile.topology import TileGraph

__all__ = [
    "FAMILY_TOLERANCES",
    "AnalogSubstrate",
    "BackpropCredit",
    "ComplexSubstrate",
    "CreditAssignment",
    "CreditAssignmentConfig",
    "DiffusionDynamics",
    "DigitalSubstrate",
    "ElasticConsolidationUpdate",
    "EnergyMinimizationDynamics",
    "EuclideanUpdate",
    "FeedforwardGeometry",
    "Geometry",
    "GeometryConfig",
    "HomeostaticCredit",
    "InstantaneousDynamics",
    "LazyStateDynamics",
    "LocalGoodnessCredit",
    "MemristiveSubstrate",
    "ModelAdapter",
    "NaturalGradientUpdate",
    "NeuromorphicSubstrate",
    "NoisySubstrate",
    "OpticalSubstrate",
    "ParameterUpdate",
    "ParameterUpdateConfig",
    "PlasticityConfig",
    "PredictiveSettlingDynamics",
    "QuantizedSubstrate",
    "QuantumSubstrate",
    "RandomProjectionsCredit",
    "RecurrentGeometry",
    "RiemannianOrthogonalUpdate",
    "SparseSubstrate",
    "SpectralConstrainedUpdate",
    "SpikeIntegrationDynamics",
    "StateDynamics",
    "StateDynamicsConfig",
    "Substrate",
    "SubstrateConfig",
    "System",
    "SystemConfig",
    "SystemState",
    "TargetInversionCredit",
    "TemporalTraceCredit",
    "TernarySubstrate",
    "ThermodynamicContrast",
]


# ============================================================
# Configuration Dataclasses (immutable, slotted)
# ============================================================


@dataclass(frozen=True, slots=True)
class SubstrateConfig:
    """Configuration for a physical substrate.

    Attributes:
        precision: Numeric precision ("float32", "float16", "bfloat16",
            "int8", "int4", "binary")
        noise_level: Standard deviation of additive state noise
        weight_bounds: Optional (min, max) tuple for weight clamping
        sparsity: Target sparsity ratio [0, 1]
        device: Target device ("cpu", "cuda", "mps", "fpga", "analog",
            "optical")
    """

    precision: str
    noise_level: float
    weight_bounds: tuple[float, float] | None
    sparsity: float
    device: str

    @classmethod
    def digital(
        cls,
        *,
        precision: str = "float32",
        noise_level: float = 0.0,
        weight_bounds: tuple[float, float] | None = None,
        sparsity: float = 0.0,
        device: str = "cpu",
    ) -> SubstrateConfig:
        return cls(
            precision=precision,
            noise_level=noise_level,
            weight_bounds=weight_bounds,
            sparsity=sparsity,
            device=device,
        )

    @classmethod
    def analog(
        cls,
        *,
        noise_level: float = 0.1,
        device: str = "cpu",
    ) -> SubstrateConfig:
        return cls(
            precision="float32",
            noise_level=noise_level,
            weight_bounds=(-1.0, 1.0),
            sparsity=0.0,
            device=device,
        )

    @classmethod
    def memristive(
        cls,
        *,
        noise_level: float = 0.05,
        device: str = "cpu",
    ) -> SubstrateConfig:
        return cls(
            precision="int8",
            noise_level=noise_level,
            weight_bounds=(0.0, 1.0),
            sparsity=0.1,
            device=device,
        )

    @classmethod
    def neuromorphic(
        cls,
        *,
        noise_level: float = 0.01,
        device: str = "cpu",
    ) -> SubstrateConfig:
        return cls(
            precision="float16",
            noise_level=noise_level,
            sparsity=0.95,
            weight_bounds=None,
            device=device,
        )

    @classmethod
    def optical(
        cls,
        *,
        noise_level: float = 0.01,
        device: str = "cpu",
    ) -> SubstrateConfig:
        return cls(
            precision="float32",
            noise_level=noise_level,
            weight_bounds=None,
            sparsity=0.0,
            device=device,
        )

    @classmethod
    def quantum(
        cls,
        *,
        noise_level: float = 0.02,
        device: str = "cpu",
    ) -> SubstrateConfig:
        return cls(
            precision="complex64",
            noise_level=noise_level,
            weight_bounds=None,
            sparsity=0.0,
            device=device,
        )

    @classmethod
    def complex(
        cls,
        *,
        noise_level: float = 0.0,
        weight_bounds: tuple[float, float] | None = (-1.0, 1.0),
        sparsity: float = 0.0,
        device: str = "cpu",
    ) -> SubstrateConfig:
        """Complex-valued substrate for holomorphic networks.

        Uses float32 emulation (real/imag channels) for efficient GPU execution
        with Triton-accelerated complex ops (matmul, tanh, conjugate transpose).
        """
        return cls(
            precision="float32",  # Emulated complex via real/imag channels
            noise_level=noise_level,
            weight_bounds=weight_bounds,
            sparsity=sparsity,
            device=device,
        )

    @classmethod
    def sparse(
        cls,
        *,
        sparsity: float = 0.5,
        noise_level: float = 0.0,
        precision: str = "float32",
        weight_bounds: tuple[float, float] | None = (-1.0, 1.0),
        device: str = "cpu",
    ) -> SubstrateConfig:
        """Sparse substrate with dynamic sparsity masks.

        Supports unstructured, N:M structured, block, and channel-wise sparsity
        with efficient sparse matmul where available.
        """
        return cls(
            precision=precision,
            noise_level=noise_level,
            weight_bounds=weight_bounds,
            sparsity=sparsity,
            device=device,
        )

    @classmethod
    def ternary(
        cls,
        *,
        noise_level: float = 0.0,
        weight_bounds: tuple[float, float] | None = (-1.0, 1.0),
        device: str = "cpu",
    ) -> SubstrateConfig:
        """Ternary substrate with STE-based quantization.

        Weights quantized to {-α, 0, +α} with Straight-Through Estimator
        for gradient backpropagation through the quantization function.
        """
        return cls(
            precision="float32",  # Latent weights stay float32
            noise_level=noise_level,
            weight_bounds=weight_bounds,
            sparsity=0.0,  # Sparsity emerges from thresholding
            device=device,
        )


@dataclass(frozen=True, slots=True)
class GeometryConfig:
    """Configuration for network geometry/topology.

    Attributes:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dims: List of hidden layer dimensions
        num_layers: Number of layers (alternative to hidden_dims)
        topology_type: "feedforward", "recurrent", "tile_mesh",
            "neuromorphic", "spatial_lattice"
        connectivity: Optional adjacency specification
        recurrent_weight: Optional recurrent weight matrix (for recurrent topology)
        init_scale: Weight initialization scale for recurrent weights
    """

    input_dim: int
    output_dim: int
    hidden_dims: tuple[int, ...]
    num_layers: int
    topology_type: str
    connectivity: dict | None
    recurrent_weight: list[list[float]] | None
    init_scale: float = 0.1

    @classmethod
    def feedforward(
        cls,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...],
        init_scale: float = 0.1,
    ) -> GeometryConfig:
        return cls(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            num_layers=len(hidden_dims),
            topology_type="feedforward",
            connectivity=None,
            recurrent_weight=None,
            init_scale=init_scale,
        )

    @classmethod
    def recurrent(
        cls,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...],
        init_scale: float = 0.1,
    ) -> GeometryConfig:
        return cls(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            num_layers=len(hidden_dims),
            topology_type="recurrent",
            connectivity=None,
            recurrent_weight=None,
            init_scale=init_scale,
        )

    @classmethod
    def tile_mesh(
        cls,
        *,
        input_dim: int,
        output_dim: int,
        num_layers: int,
        neurons_per_tile: int,
        tiles_per_layer: int,
        init_scale: float = 0.1,
    ) -> GeometryConfig:
        return cls(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=(),
            num_layers=num_layers,
            topology_type="tile_mesh",
            connectivity=None,
            recurrent_weight=None,
            init_scale=init_scale,
        )


@dataclass(frozen=True, slots=True)
class StateDynamicsConfig:
    """Configuration for state dynamics/settling.

    Attributes:
        dynamics_type: "energy_minimization", "predictive_settling",
            "spike_integration", "instantaneous", "diffusion"
        max_steps: Maximum settling iterations
        convergence_threshold: Early stopping threshold
        convergence_start: Step to start checking convergence
        step_size: Learning rate for state updates
        beta: Nudge strength for energy-based methods
        momentum: Momentum coefficient for heavy-ball dynamics (energy_minimization)
        track_free_energy_per_iter: Record free energy at each iteration
            for Control-Lyapunov analysis
        gradient_checkpointing: Use gradient checkpointing to trade compute
            for memory during settling (energy_minimization only)
    """

    dynamics_type: str
    max_steps: int
    convergence_threshold: float
    convergence_start: int
    step_size: float
    beta: float
    momentum: float
    track_free_energy_per_iter: bool
    gradient_checkpointing: bool = False

    @classmethod
    def energy_minimization(
        cls,
        *,
        max_steps: int = 30,
        convergence_threshold: float = 1e-4,
        convergence_start: int = 5,
        step_size: float = 0.1,
        beta: float = 0.5,
        momentum: float = 0.0,
        track_free_energy_per_iter: bool = False,
        gradient_checkpointing: bool = False,
    ) -> StateDynamicsConfig:
        return cls(
            dynamics_type="energy_minimization",
            max_steps=max_steps,
            convergence_threshold=convergence_threshold,
            convergence_start=convergence_start,
            step_size=step_size,
            beta=beta,
            momentum=momentum,
            track_free_energy_per_iter=track_free_energy_per_iter,
            gradient_checkpointing=gradient_checkpointing,
        )

    @classmethod
    def predictive_settling(
        cls,
        *,
        max_steps: int = 30,
        convergence_threshold: float = 1e-4,
        convergence_start: int = 5,
        step_size: float = 0.1,
        beta: float = 0.5,
        momentum: float = 0.0,
        track_free_energy_per_iter: bool = False,
    ) -> StateDynamicsConfig:
        return cls(
            dynamics_type="predictive_settling",
            max_steps=max_steps,
            convergence_threshold=convergence_threshold,
            convergence_start=convergence_start,
            step_size=step_size,
            beta=beta,
            momentum=momentum,
            track_free_energy_per_iter=track_free_energy_per_iter,
        )

    @classmethod
    def spike_integration(
        cls,
        *,
        max_steps: int = 30,
        convergence_threshold: float = 1e-4,
        convergence_start: int = 5,
        step_size: float = 0.1,
        beta: float = 0.5,
        momentum: float = 0.0,
        track_free_energy_per_iter: bool = False,
    ) -> StateDynamicsConfig:
        return cls(
            dynamics_type="spike_integration",
            max_steps=max_steps,
            convergence_threshold=convergence_threshold,
            convergence_start=convergence_start,
            step_size=step_size,
            beta=beta,
            momentum=momentum,
            track_free_energy_per_iter=track_free_energy_per_iter,
        )

    @classmethod
    def instantaneous(
        cls,
        *,
        max_steps: int = 1,
        convergence_threshold: float = 1e-4,
        convergence_start: int = 1,
        step_size: float = 0.1,
        beta: float = 0.1,
        momentum: float = 0.0,
        track_free_energy_per_iter: bool = False,
    ) -> StateDynamicsConfig:
        return cls(
            dynamics_type="instantaneous",
            max_steps=max_steps,
            convergence_threshold=convergence_threshold,
            convergence_start=convergence_start,
            step_size=step_size,
            beta=beta,
            momentum=momentum,
            track_free_energy_per_iter=track_free_energy_per_iter,
        )

    @classmethod
    def diffusion(
        cls,
        *,
        max_steps: int = 30,
        convergence_threshold: float = 1e-4,
        convergence_start: int = 5,
        step_size: float = 0.1,
        beta: float = 0.5,
        momentum: float = 0.0,
        track_free_energy_per_iter: bool = False,
    ) -> StateDynamicsConfig:
        """Diffusion-based dynamics for continuous-time settling.

        Implements the dynamics: dh/dt = -∇E(h) + noise
        where E is the energy function and noise models diffusion.
        """
        return cls(
            dynamics_type="diffusion",
            max_steps=max_steps,
            convergence_threshold=convergence_threshold,
            convergence_start=convergence_start,
            step_size=step_size,
            beta=beta,
            momentum=momentum,
            track_free_energy_per_iter=track_free_energy_per_iter,
        )


@dataclass(frozen=True, slots=True)
class CreditAssignmentConfig:
    """Configuration for credit assignment.

    Attributes:
        credit_type: "thermodynamic_contrast", "random_projections",
            "local_goodness", "temporal_trace", "target_inversion"
        beta: Nudge strength (for thermodynamic contrast)
        feedback_matrix: Optional fixed feedback matrix
            (for random projections)
        local_objective: Layer-local loss type (for local goodness)
        orthogonal_init: Initialize feedback matrices with orthogonal weights
        feedback_scale: Scaling factor for feedback matrices
    """

    credit_type: str
    beta: float
    feedback_matrix: Tensor | None
    local_objective: str
    orthogonal_init: bool
    feedback_scale: float

    @classmethod
    def thermodynamic_contrast(
        cls,
        *,
        beta: float = 0.5,
        feedback_matrix: Tensor | None = None,
        local_objective: str = "mse",
        orthogonal_init: bool = False,
        feedback_scale: float = 0.01,
    ) -> CreditAssignmentConfig:
        return cls(
            credit_type="thermodynamic_contrast",
            beta=beta,
            feedback_matrix=feedback_matrix,
            local_objective=local_objective,
            orthogonal_init=orthogonal_init,
            feedback_scale=feedback_scale,
        )

    @classmethod
    def random_projections(
        cls,
        *,
        beta: float = 0.5,
        feedback_matrix: Tensor | None = None,
        local_objective: str = "mse",
        orthogonal_init: bool = False,
        feedback_scale: float = 0.01,
    ) -> CreditAssignmentConfig:
        return cls(
            credit_type="random_projections",
            beta=beta,
            feedback_matrix=feedback_matrix,
            local_objective=local_objective,
            orthogonal_init=orthogonal_init,
            feedback_scale=feedback_scale,
        )

    @classmethod
    def local_goodness(
        cls,
        *,
        beta: float = 0.5,
        feedback_matrix: Tensor | None = None,
        local_objective: str = "mse",
        orthogonal_init: bool = False,
        feedback_scale: float = 0.01,
    ) -> CreditAssignmentConfig:
        return cls(
            credit_type="local_goodness",
            beta=beta,
            feedback_matrix=feedback_matrix,
            local_objective=local_objective,
            orthogonal_init=orthogonal_init,
            feedback_scale=feedback_scale,
        )

    @classmethod
    def temporal_trace(
        cls,
        *,
        beta: float = 0.5,
        feedback_matrix: Tensor | None = None,
        local_objective: str = "mse",
        orthogonal_init: bool = False,
        feedback_scale: float = 0.01,
    ) -> CreditAssignmentConfig:
        return cls(
            credit_type="temporal_trace",
            beta=beta,
            feedback_matrix=feedback_matrix,
            local_objective=local_objective,
            orthogonal_init=orthogonal_init,
            feedback_scale=feedback_scale,
        )

    @classmethod
    def target_inversion(
        cls,
        *,
        beta: float = 0.5,
        feedback_matrix: Tensor | None = None,
        local_objective: str = "mse",
        orthogonal_init: bool = False,
        feedback_scale: float = 0.01,
    ) -> CreditAssignmentConfig:
        return cls(
            credit_type="target_inversion",
            beta=beta,
            feedback_matrix=feedback_matrix,
            local_objective=local_objective,
            orthogonal_init=orthogonal_init,
            feedback_scale=feedback_scale,
        )

    @classmethod
    def gradient(
        cls,
        *,
        beta: float = 0.5,
        feedback_matrix: Tensor | None = None,
        local_objective: str = "mse",
        orthogonal_init: bool = False,
        feedback_scale: float = 0.01,
    ) -> CreditAssignmentConfig:
        return cls(
            credit_type="gradient",
            beta=beta,
            feedback_matrix=feedback_matrix,
            local_objective=local_objective,
            orthogonal_init=orthogonal_init,
            feedback_scale=feedback_scale,
        )


@dataclass(frozen=True, slots=True)
class ParameterUpdateConfig:
    """Configuration for parameter update rule.

    Attributes:
        update_type: "riemannian_orthogonal", "spectral_constrained",
            "natural_gradient", "elastic_consolidation", "euclidean"
        step_size: Learning rate
        momentum: Momentum coefficient (for euclidean)
        ortho_steps: Newton-Schulz iterations (for riemannian)
        spectral_norm: Target spectral norm (for spectral_constrained)
        fisher_damping: Damping for natural gradient
        ewc_lambda: EWC regularization strength
    """

    update_type: str
    step_size: float
    momentum: float
    ortho_steps: int
    spectral_norm: float
    fisher_damping: float
    ewc_lambda: float

    @classmethod
    def euclidean(
        cls,
        *,
        step_size: float = 0.01,
        momentum: float = 0.9,
        ortho_steps: int = 5,
        spectral_norm: float = 1.0,
        fisher_damping: float = 1e-3,
        ewc_lambda: float = 1000.0,
    ) -> ParameterUpdateConfig:
        return cls(
            update_type="euclidean",
            step_size=step_size,
            momentum=momentum,
            ortho_steps=ortho_steps,
            spectral_norm=spectral_norm,
            fisher_damping=fisher_damping,
            ewc_lambda=ewc_lambda,
        )

    @classmethod
    def riemannian_orthogonal(
        cls,
        *,
        step_size: float = 0.01,
        momentum: float = 0.9,
        ortho_steps: int = 5,
        spectral_norm: float = 1.0,
        fisher_damping: float = 1e-3,
        ewc_lambda: float = 1000.0,
    ) -> ParameterUpdateConfig:
        return cls(
            update_type="riemannian_orthogonal",
            step_size=step_size,
            momentum=momentum,
            ortho_steps=ortho_steps,
            spectral_norm=spectral_norm,
            fisher_damping=fisher_damping,
            ewc_lambda=ewc_lambda,
        )

    @classmethod
    def spectral_constrained(
        cls,
        *,
        step_size: float = 0.01,
        momentum: float = 0.9,
        ortho_steps: int = 5,
        spectral_norm: float = 1.0,
        fisher_damping: float = 1e-3,
        ewc_lambda: float = 1000.0,
    ) -> ParameterUpdateConfig:
        return cls(
            update_type="spectral_constrained",
            step_size=step_size,
            momentum=momentum,
            ortho_steps=ortho_steps,
            spectral_norm=spectral_norm,
            fisher_damping=fisher_damping,
            ewc_lambda=ewc_lambda,
        )

    @classmethod
    def natural_gradient(
        cls,
        *,
        step_size: float = 0.01,
        momentum: float = 0.9,
        ortho_steps: int = 5,
        spectral_norm: float = 1.0,
        fisher_damping: float = 1e-3,
        ewc_lambda: float = 1000.0,
    ) -> ParameterUpdateConfig:
        return cls(
            update_type="natural_gradient",
            step_size=step_size,
            momentum=momentum,
            ortho_steps=ortho_steps,
            spectral_norm=spectral_norm,
            fisher_damping=fisher_damping,
            ewc_lambda=ewc_lambda,
        )

    @classmethod
    def elastic_consolidation(
        cls,
        *,
        step_size: float = 0.01,
        momentum: float = 0.9,
        ortho_steps: int = 5,
        spectral_norm: float = 1.0,
        fisher_damping: float = 1e-3,
        ewc_lambda: float = 1000.0,
    ) -> ParameterUpdateConfig:
        return cls(
            update_type="elastic_consolidation",
            step_size=step_size,
            momentum=momentum,
            ortho_steps=ortho_steps,
            spectral_norm=spectral_norm,
            fisher_damping=fisher_damping,
            ewc_lambda=ewc_lambda,
        )


# ============================================================
# State Container
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
# Layer 1: Substrate (The Physical State Space)
# ============================================================


@runtime_checkable
class Substrate(Protocol):
    """Physical substrate constraints on weights and activations.

    Defines the physical medium: precision, noise profiles, locality constraints,
    and hardware-specific forward/weight update operators.

    Every computronium system runs on a substrate. The substrate injects
    physically accurate noise, enforces weight constraints (e.g. positivity,
    bounded conductance), and provides the forward operator that the Geometry
    layer routes through.
    """

    config: SubstrateConfig

    @abstractmethod
    def quantize_weights(self, w: Tensor) -> Tensor:
        """Apply substrate-specific weight quantization/clamping."""
        ...

    @abstractmethod
    def inject_state_noise(self, s: Tensor) -> Tensor:
        """Inject substrate-appropriate noise into state tensor."""
        ...

    @abstractmethod
    def get_forward_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        """Return the substrate's forward operator: (input, weights) -> output.

        This operator encodes the physics of the substrate (e.g. Kirchhoff's
        laws for crossbars, phase interference for photonic, vesicle release
        for biological).
        """
        ...

    @abstractmethod
    def get_weight_update_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        """Return the substrate's weight update operator.

        (pseudo_grad, current_w) -> ΔW. Encodes how the substrate physically
        modifies weights (e.g. pulse-based conductance change for memristors,
        phase shift for photonic).
        """
        ...

    @abstractmethod
    def initial_state(self, x: Tensor) -> Tensor:
        """Create initial state for given input on this substrate."""
        ...


# ============================================================
# Layer 2: Geometry (The Topology & Routing)
# ============================================================


@runtime_checkable
class Geometry(Protocol):
    """Spatial arrangement and message-passing topology.

    Defines the graph structure: nodes (computational units), edges (connections),
    and the routing protocol for forward/backward passes. The Geometry owns
    the parameters (weights) and exposes them to the substrate's operators.

    Key responsibility: route activations through the topology using the
    substrate's forward operator.
    """

    config: GeometryConfig

    @property
    @abstractmethod
    def params(self) -> dict[str, Tensor]:
        """Return all learnable parameters as a name -> tensor mapping."""
        ...

    @abstractmethod
    def forward(self, x: Tensor, substrate: Substrate) -> Tensor:
        """Route input through the topology using substrate's forward operator.

        Args:
            x: Input tensor
            substrate: The substrate providing the forward operator

        Returns:
            Output tensor after routing through the geometry
        """
        ...

    @abstractmethod
    def route(self, activations: Tensor) -> Tensor:
        """Route activations through the topology (single step).

        Used by StateDynamics for iterative settling.
        """
        ...

    @abstractmethod
    def update_params(self, new_params: dict[str, Tensor]) -> None:
        """Update geometry parameters in-place from ParameterUpdate output."""
        ...

    @abstractmethod
    def transition_modules(self) -> list[nn.Module]:
        """Return modules in forward order for TransitionGraph protocol."""
        ...

    def forward_with_intermediates(
        self, x: Tensor, substrate: Substrate
    ) -> list[Tensor]:
        """Return intermediate activations for each layer (optional).

        Default implementation returns only the final output.
        Override for geometries that support layer-wise inspection
        (e.g., FeedforwardGeometry for predictive coding).

        Args:
            x: Input tensor
            substrate: The substrate providing the forward operator

        Returns:
            List of activations [input, layer1_out, layer2_out, ..., output]
        """
        out = self.forward(x, substrate)
        return [x, out]


# ============================================================
# Layer 3: StateDynamics (Forward Evolution & Settling)
# ============================================================


@runtime_checkable
class StateDynamics(Protocol):
    """How the network's activations evolve over time (the forward pass).

    Encapsulates the differential equation or iterative map governing state
    evolution: energy minimization (EqProp), predictive settling (PC), spike
    integration (LIF), or instantaneous pass (backprop/FF).

    The StateDynamics uses Geometry.route() and Substrate operators to evolve
    the state. It produces free and nudged states for CreditAssignment.
    """

    config: StateDynamicsConfig

    @abstractmethod
    def settle(
        self,
        state: SystemState,
        geometry: Geometry,
        substrate: Substrate,
        target: Tensor | None = None,
    ) -> SystemState:
        """Settle the network to a fixed point (or run single pass).

        Args:
            state: Current system state (contains input x)
            geometry: Network topology for routing
            substrate: Physical substrate for operators/noise
            target: Optional target for nudged phase

        Returns:
            Updated state with settled activations in state.activations
        """
        ...

    @abstractmethod
    def compute_energy(self, state: SystemState, geometry: Geometry) -> Tensor:
        """Compute the energy of the current state.

        For energy-based dynamics (EqProp, Hopfield, PC), this is the
        Lyapunov function. For non-energy dynamics, returns a proxy.
        """
        ...


# ============================================================
# Layer 4: CreditAssignment (Error Routing & Pseudo-Gradients)
# ============================================================


@runtime_checkable
class CreditAssignment(Protocol):
    """How the network computes the direction of learning (pseudo-gradient).

    Uses only locally available signals to compute a gradient-like update
    direction. The five canonical types:
    - ThermodynamicContrast: (nudged - free) / beta (EqProp)
    - RandomProjections: Fixed/adaptive random feedback matrices (FA, DFA)
    - LocalGoodness: Layer-local contrastive objectives (FF, PEPITA)
    - TemporalTrace: Spike-timing correlations (STDP)
    - TargetInversion: Propagating local targets (Target Prop)

    Output is a list of pseudo-gradients, one per layer/module, matching
    the Geometry's parameter structure.
    """

    config: CreditAssignmentConfig

    @abstractmethod
    def compute_pseudo_gradient(
        self,
        free_state: SystemState,
        nudged_state: SystemState,
        loss: Tensor,
        geometry: Geometry,
    ) -> list[Tensor]:
        """Compute pseudo-gradients from free and nudged states.

        Args:
            free_state: State after free-phase settling
            nudged_state: State after nudged-phase settling
            loss: Task loss at nudged state
            geometry: Network topology (provides layer structure)

        Returns:
            List of pseudo-gradient tensors, one per learnable layer
        """
        ...

    # DEFAULT METHOD — non-breaking, only overridden by LocalGoodness/TargetInversion
    def surrogate_objective(
        self,
        free_state: SystemState,
        nudged_state: SystemState,
        geometry: Geometry,
    ) -> Tensor:
        """Compute layer-local surrogate loss for gradient checking.

        Only LocalGoodnessCredit and TargetInversionCredit override this.
        Others raise NotImplementedError.
        """
        raise NotImplementedError(
            "Surrogate objective not defined for this credit rule"
        )


# ============================================================
# Layer 5: ParameterUpdate (The Optimization Rule)
# ============================================================


@runtime_checkable
class ParameterUpdate(Protocol):
    """How pseudo-gradients translate into physical weight changes (ΔW).

    Maps the pseudo-gradient tensor (from CreditAssignment) to actual
    parameter deltas. The five canonical types:
    - RiemannianOrthogonal: Muon-style orthogonal updates
    - SpectralConstrained: Lipschitz-bounded updates
    - NaturalGradient: Fisher-information geometry updates
    - ElasticConsolidation: EWC-style importance-weighted updates
    - Euclidean: Standard SGD/Adam in flat space

    The update is applied to the Geometry's parameters.
    """

    config: ParameterUpdateConfig

    @abstractmethod
    def step(
        self,
        params: dict[str, Tensor],
        pseudo_grads: list[Tensor],
        geometry: Geometry,
    ) -> dict[str, Tensor]:
        """Compute parameter updates from pseudo-gradients.

        Args:
            params: Current parameters (name -> tensor)
            pseudo_grads: Pseudo-gradients from CreditAssignment
            geometry: Network topology (for layer-wise adaptation)

        Returns:
            Updated parameters (name -> tensor)
        """
        ...


# ============================================================
# Composing System: The 5-Layer Tensor Product
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
        """Execute one training step through the 5-layer pipeline."""
        state = SystemState(x=x, y=y)

        # 1. Substrate + Geometry: Forward pass (initial state)
        state.activations = self.geometry.forward(x, self.substrate)
        if state.activations is not None:
            state.activations = self.substrate.inject_state_noise(state.activations)

        # 2. StateDynamics: Free phase settling
        free_state = self.dynamics.settle(
            state, self.geometry, self.substrate, target=None
        )
        free_state.energy = self.dynamics.compute_energy(free_state, self.geometry)

        # 3. StateDynamics: Nudged phase settling
        nudged_state = self.dynamics.settle(
            state, self.geometry, self.substrate, target=y
        )
        nudged_state.energy = self.dynamics.compute_energy(nudged_state, self.geometry)
        nudged_state.loss = self._compute_loss(nudged_state, y)

        # 4. CreditAssignment: Compute pseudo-gradients
        pseudo_grads = self.credit.compute_pseudo_gradient(
            free_state, nudged_state, nudged_state.loss, self.geometry
        )

        # 5. ParameterUpdate: Apply updates
        new_params = self.update.step(self.geometry.params, pseudo_grads, self.geometry)
        self.geometry.update_params(new_params)

        return {
            "loss": nudged_state.loss.item() if nudged_state.loss is not None else 0.0,
            "energy": free_state.energy.item()
            if free_state.energy is not None
            else 0.0,
            "accuracy": free_state.metrics.get("accuracy", 0.0),
        }

    def _compute_loss(self, state: SystemState, y: Tensor) -> Tensor:  # ruff: ignore[no-self-use]
        """Compute task loss from final state."""
        acts = state.activations
        if acts is None:
            return torch.tensor(0.0)
        logits = acts[-1] if isinstance(acts, list) else acts
        return torch.nn.functional.cross_entropy(logits, y)

    def forward(self, x: Tensor) -> Tensor:
        """Inference forward pass (free phase only, no weight updates)."""
        state = SystemState(x=x)
        state.activations = self.geometry.forward(x, self.substrate)
        if state.activations is not None:
            state.activations = self.substrate.inject_state_noise(state.activations)
        state = self.dynamics.settle(state, self.geometry, self.substrate, target=None)
        acts = state.activations
        if acts is None:
            return torch.empty(0)
        if isinstance(acts, list):
            return acts[-1]
        return acts

    def to_spec(self) -> dict:
        """Serialize the System to a specification dictionary.

        Returns:
            Dictionary containing schema_version and all 5 axis configs.
        """
        ...

    @classmethod
    def from_spec(cls, spec: dict) -> System:
        """Reconstruct a System from a specification dictionary.

        Args:
            spec: Dictionary with schema_version and 5 axis configs.

        Returns:
            A composed System instance.
        """
        ...


# ============================================================
# SystemConfig: Validated Composition of 5-D Ontology
# ============================================================


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

    def validate(self) -> None:
        """Cross-axis validation (hard constraints only).

        Raises:
            ValueError: If configuration violates hard compatibility constraints.
        """
        # Recurrent geometry requires energy-based dynamics
        if self.geometry.topology_type in ("recurrent", "recurrent_attractor"):
            if self.dynamics.dynamics_type != "energy_minimization":
                raise ValueError(
                    f"Recurrent geometry (topology_type={self.geometry.topology_type!r}) "
                    f"requires energy_minimization dynamics, got {self.dynamics.dynamics_type!r}"
                )

        # Thermodynamic contrast credit requires energy-based dynamics
        if self.credit.credit_type in ("thermodynamic_contrast", "equilibrium"):
            if self.dynamics.dynamics_type != "energy_minimization":
                raise ValueError(
                    f"Thermodynamic contrast credit (credit_type={self.credit.credit_type!r}) "
                    f"requires energy_minimization dynamics, got {self.dynamics.dynamics_type!r}"
                )

        # Spiking dynamics requires temporal trace or STDP credit
        if self.dynamics.dynamics_type == "spike_integration":
            if self.credit.credit_type not in (
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
        if self.geometry.topology_type in ("tile_mesh", "tile"):
            if self.dynamics.dynamics_type not in (
                "energy_minimization",
                "instantaneous",
            ):
                raise ValueError(
                    f"Tile mesh geometry requires energy_minimization or instantaneous dynamics, "
                    f"got {self.dynamics.dynamics_type!r}"
                )

        # Beta matching: StateDynamics.beta should match CreditAssignment.beta for energy-based systems
        if self.dynamics.dynamics_type == "energy_minimization":
            if abs(self.dynamics.beta - self.credit.beta) > 1e-6:
                # Soft constraint: warn but don't fail
                import warnings

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
        if self.substrate.precision == "float16" and self.substrate.sparsity > 0.9:
            # Likely neuromorphic substrate
            if self.dynamics.dynamics_type not in (
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
        if self.substrate.precision == "float32" and self.substrate.noise_level > 0.0:
            if self.dynamics.dynamics_type == "instantaneous":
                # Instantaneous dynamics doesn't use substrate noise during settling
                # (only single forward pass). Warn but don't fail.
                import warnings

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
        if self.substrate.precision == "float32" and getattr(
            self.substrate, "_complex_emulated", False
        ):
            # This is a complex substrate (emulated via real/imag channels)
            if self.credit.credit_type not in (
                "thermodynamic_contrast",
                "equilibrium",
                "gradient",
                "backprop",
            ):
                import warnings

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
            if self.dynamics.dynamics_type not in (
                "energy_minimization",
                "instantaneous",
                "diffusion",
            ):
                raise ValueError(
                    f"Quantum substrate requires energy_minimization, instantaneous, "
                    f"or diffusion dynamics, got {self.dynamics.dynamics_type!r}"
                )

            # Quantum substrate with thermodynamic contrast needs matching beta
            if self.credit.credit_type in ("thermodynamic_contrast", "equilibrium"):
                if abs(self.dynamics.beta - self.credit.beta) > 1e-6:
                    import warnings

                    warnings.warn(
                        f"Quantum substrate with thermodynamic contrast: "
                        f"beta mismatch dynamics.beta={self.dynamics.beta} != credit.beta={self.credit.beta}. "
                        f"Phase-sensitive gradients require matched beta.",
                        UserWarning,
                        stacklevel=2,
                    )

        # Sparse substrate requires compatible update rule
        # Sparse weights need updates that preserve sparsity structure
        if self.substrate.sparsity > 0.5:
            if self.update.update_type == "riemannian_orthogonal":
                import warnings

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
        if (
            self.substrate.precision == "float32"
            and self.substrate.sparsity == 0.0
            and self.substrate.weight_bounds == (-1.0, 1.0)
        ):
            # Heuristic: likely ternary substrate (sparsity emerges from thresholding)
            if self.credit.credit_type not in (
                "thermodynamic_contrast",
                "equilibrium",
                "gradient",
                "backprop",
            ):
                import warnings

                warnings.warn(
                    f"Ternary-like substrate used with {self.credit.credit_type!r} credit. "
                    f"Best results with thermodynamic_contrast (Ternary EqProp) "
                    f"or gradient (Ternary backprop with STE).",
                    UserWarning,
                    stacklevel=2,
                )

        # Diffusion dynamics requires noise-aware substrate
        if self.dynamics.dynamics_type == "diffusion":
            if self.substrate.noise_level == 0.0:
                import warnings

                warnings.warn(
                    "Diffusion dynamics (Langevin) requires substrate noise_level > 0 "
                    "for proper sampling. Consider setting noise_level on substrate.",
                    UserWarning,
                    stacklevel=2,
                )

        # Predictive settling dynamics requires compatible credit
        # PC uses local errors, works with thermodynamic contrast or local goodness
        if self.dynamics.dynamics_type == "predictive_settling":
            if self.credit.credit_type not in (
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
            import warnings

            warnings.warn(
                f"EnergyMinimizationDynamics with momentum={self.dynamics.momentum} "
                f"combined with RiemannianOrthogonalUpdate may cause instability. "
                f"Consider EuclideanUpdate with momentum.",
                UserWarning,
                stacklevel=2,
            )

        # Geometry-Substrate constraints
        # Spatial lattice / neuromorphic geometry requires neuromorphic substrate
        if self.geometry.topology_type in ("spatial_lattice", "neuromorphic", "fabric"):
            if not (
                self.substrate.precision == "float16" and self.substrate.sparsity > 0.9
            ):
                import warnings

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
            self.geometry.topology_type in ("tile_mesh", "tile")
            and self.substrate.sparsity > 0.5
        ):
            import warnings

            warnings.warn(
                f"Tile mesh geometry with sparse substrate (sparsity={self.substrate.sparsity}) "
                f"may benefit from structured sparsity (N:M or block) for efficient matmul.",
                UserWarning,
                stacklevel=2,
            )

    @classmethod
    def from_experiment(cls, exp: ExperimentConfig) -> SystemConfig:
        """Build from unified ExperimentConfig — single entry point.

        Args:
            exp: Unified experiment configuration.

        Returns:
            Validated SystemConfig composed from experiment's ontology and model config.
        """

        ont = exp.ontology
        model = exp.model

        # Build substrate config
        if ont.substrate is not None:
            substrate = ont.substrate
        else:
            # Map substrate_type to factory method
            substrate_map = {
                "digital": SubstrateConfig.digital,
                "analog": SubstrateConfig.analog,
                "memristive": SubstrateConfig.memristive,
                "neuromorphic": SubstrateConfig.neuromorphic,
                "optical": SubstrateConfig.optical,
                "quantum": SubstrateConfig.quantum,
                "complex": SubstrateConfig.complex,
                "sparse": SubstrateConfig.sparse,
                "ternary": SubstrateConfig.ternary,
            }
            substrate_factory = substrate_map.get(
                ont.substrate_type, SubstrateConfig.digital
            )
            substrate = substrate_factory(precision=ont.substrate_precision)

        # Build geometry config
        if ont.geometry is not None:
            geometry = ont.geometry
        else:
            hidden_dims = ont.hidden_dims or tuple(model.hidden_dims)
            topology_map = {
                "feedforward": GeometryConfig.feedforward,
                "recurrent": GeometryConfig.recurrent,
                "recurrent_attractor": GeometryConfig.recurrent,
                "tile_mesh": GeometryConfig.tile_mesh,
                "tile": GeometryConfig.tile_mesh,
            }
            geometry_factory = topology_map.get(
                ont.topology_type, GeometryConfig.feedforward
            )

            if ont.topology_type in ("tile_mesh", "tile"):
                geometry = geometry_factory(
                    input_dim=model.input_dim,
                    output_dim=model.output_dim,
                    num_layers=model.num_layers,
                    neurons_per_tile=model.neurons_per_tile,
                    tiles_per_layer=model.tiles_per_layer,
                )
            else:
                geometry = geometry_factory(
                    input_dim=model.input_dim,
                    output_dim=model.output_dim,
                    hidden_dims=hidden_dims,
                )

        # Build dynamics config
        if ont.dynamics is not None:
            dynamics = ont.dynamics
        else:
            dynamics_map = {
                "energy_minimization": StateDynamicsConfig.energy_minimization,
                "predictive_settling": StateDynamicsConfig.predictive_settling,
                "spike_integration": StateDynamicsConfig.spike_integration,
                "instantaneous": StateDynamicsConfig.instantaneous,
                "diffusion": StateDynamicsConfig.diffusion,
            }
            dynamics_factory = dynamics_map.get(
                ont.dynamics_type, StateDynamicsConfig.instantaneous
            )
            dynamics = dynamics_factory(max_steps=ont.max_steps, beta=ont.beta)

        # Build credit config
        if ont.credit is not None:
            credit = ont.credit
        else:
            credit_map = {
                "thermodynamic_contrast": CreditAssignmentConfig.thermodynamic_contrast,
                "equilibrium": CreditAssignmentConfig.thermodynamic_contrast,
                "random_projections": CreditAssignmentConfig.random_projections,
                "feedback_alignment": CreditAssignmentConfig.random_projections,
                "local_goodness": CreditAssignmentConfig.local_goodness,
                "forward_only": CreditAssignmentConfig.local_goodness,
                "temporal_trace": CreditAssignmentConfig.temporal_trace,
                "spiking": CreditAssignmentConfig.temporal_trace,
                "target_inversion": CreditAssignmentConfig.target_inversion,
                "target_prop": CreditAssignmentConfig.target_inversion,
                "gradient": CreditAssignmentConfig.gradient,
                "backprop": CreditAssignmentConfig.gradient,
            }
            credit_factory = credit_map.get(
                ont.credit_type, CreditAssignmentConfig.gradient
            )
            credit = credit_factory(beta=ont.beta)

        # Build update config
        if ont.update is not None:
            update = ont.update
        else:
            update_map = {
                "riemannian_orthogonal": ParameterUpdateConfig.riemannian_orthogonal,
                "muon": ParameterUpdateConfig.riemannian_orthogonal,
                "spectral_constrained": ParameterUpdateConfig.spectral_constrained,
                "spectral": ParameterUpdateConfig.spectral_constrained,
                "natural_gradient": ParameterUpdateConfig.natural_gradient,
                "fisher": ParameterUpdateConfig.natural_gradient,
                "elastic_consolidation": ParameterUpdateConfig.elastic_consolidation,
                "ewc": ParameterUpdateConfig.elastic_consolidation,
                "euclidean": ParameterUpdateConfig.euclidean,
            }
        update_factory = update_map.get(
            ont.update_type, ParameterUpdateConfig.euclidean
        )
        update = update_factory(step_size=ont.step_size)

        # Build plasticity config (6th axis)
        if hasattr(ont, "plasticity") and ont.plasticity is not None:
            plasticity = ont.plasticity
        else:
            # Default to NullPlasticity for backward compatibility
            plasticity = PlasticityConfig.null()

        # Create and validate
        sys_config = cls(
            substrate=substrate,
            geometry=geometry,
            dynamics=dynamics,
            plasticity=plasticity,
            credit=credit,
            update=update,
        )
        sys_config.validate()
        return sys_config

    @classmethod
    def valid_combinations(cls) -> list[dict]:
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
        for sub in substrates:
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
                                if geom["topology_type"] in (
                                    "recurrent",
                                    "recurrent_attractor",
                                ):
                                    if dyn["dynamics_type"] != "energy_minimization":
                                        continue
                                # Thermodynamic contrast requires energy_minimization
                                if cred["credit_type"] in (
                                    "thermodynamic_contrast",
                                    "equilibrium",
                                ):
                                    if dyn["dynamics_type"] != "energy_minimization":
                                        continue
                                # Spike integration requires temporal trace or target inversion credit
                                if dyn["dynamics_type"] == "spike_integration":
                                    if cred["credit_type"] not in (
                                        "temporal_trace",
                                        "target_inversion",
                                        "target_prop",
                                    ):
                                        continue
                                # Tile mesh requires compatible dynamics
                                if geom["topology_type"] in ("tile_mesh", "tile"):
                                    if dyn["dynamics_type"] not in (
                                        "energy_minimization",
                                        "instantaneous",
                                    ):
                                        continue
                                # Quantum substrate requires compatible dynamics
                                if sub["precision"] == "complex64":
                                    if dyn["dynamics_type"] not in (
                                        "energy_minimization",
                                        "instantaneous",
                                        "diffusion",
                                    ):
                                        continue
                                # Predictive settling requires compatible credit
                                if dyn["dynamics_type"] == "predictive_settling":
                                    if cred["credit_type"] not in (
                                        "thermodynamic_contrast",
                                        "equilibrium",
                                        "local_goodness",
                                        "forward_only",
                                    ):
                                        continue

                                combinations.append(coord)

        return combinations


# ============================================================
# Default/Reference Implementations
# ============================================================


class DigitalSubstrate:
    """Reference substrate: infinite precision, continuous time, no noise."""

    def __init__(self, config: SubstrateConfig | None = None):
        self.config = config or SubstrateConfig.digital()

    def _to_precision(self, tensor: Tensor) -> Tensor:
        """Convert tensor to the configured precision."""
        precision = self.config.precision
        if precision == "float32":
            return tensor.to(torch.float32)
        elif precision == "float16":
            return tensor.to(torch.float16)
        elif precision == "bfloat16":
            return tensor.to(torch.bfloat16)
        elif precision == "int8":
            return tensor.to(torch.int8)
        elif precision == "int4":
            # int4 not natively supported, use int8
            return tensor.to(torch.int8)
        elif precision == "binary":
            return tensor.to(torch.bool)
        return tensor

    def quantize_weights(self, w: Tensor) -> Tensor:
        return self._to_precision(w)

    def inject_state_noise(self, s: Tensor) -> Tensor:
        return self._to_precision(s)

    def get_forward_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        def forward(x: Tensor, w: Tensor) -> Tensor:
            x = self._to_precision(x)
            w = self._to_precision(w)
            return self._to_precision(x @ w.T)

        return forward

    def get_weight_update_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        def update(grad: Tensor, w: Tensor) -> Tensor:
            grad = self._to_precision(grad)
            return self._to_precision(grad)

        return update

    def initial_state(self, x: Tensor) -> Tensor:
        return self._to_precision(x)


class FeedforwardGeometry(nn.Module):
    """Standard feedforward DAG topology (MLP, CNN)."""

    _layers: nn.ModuleList

    def __init__(
        self,
        config: GeometryConfig,
        layers: nn.ModuleList | list[nn.Module] | None = None,
    ):
        super().__init__()
        self.config = config
        self._layers = nn.ModuleList(layers) if layers else nn.ModuleList()
        if not self._layers and config.hidden_dims:
            self._build_layers()

    def _build_layers(self) -> None:
        dims = (
            self.config.input_dim,
            *self.config.hidden_dims,
            self.config.output_dim,
        )
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self._layers = nn.ModuleList(layers)
        self._set_param_names()

    def _set_param_names(self) -> None:
        """Set _param_name attribute on weight tensors for substrate keying."""
        for i, layer in enumerate(self._layers):
            if isinstance(layer, nn.Linear):
                layer.weight._param_name = f"layer_{i}_weight"
                if layer.bias is not None:
                    layer.bias._param_name = f"layer_{i}_bias"

    @property
    def params(self) -> dict[str, Tensor]:
        return dict(self._layers.named_parameters())  # type: ignore[return-value]

    def forward(self, x: Tensor, substrate: Substrate | None = None) -> Tensor:
        if substrate is None:
            substrate = DigitalSubstrate()
        op = substrate.get_forward_operator()
        h = x
        for layer in self._layers:
            if isinstance(layer, nn.Linear):
                h = op(h, layer.weight)
                if layer.bias is not None:
                    h += layer.bias
            else:
                h = layer(h)
        return h

    def forward_with_intermediates(
        self, x: Tensor, substrate: Substrate | None = None
    ) -> list[Tensor]:
        """Forward pass returning intermediate activations for each layer.

        Returns:
            List of activations [input, layer1_out, layer2_out, ..., output]
            where layer outputs are after activation functions (ReLU).
        """
        if substrate is None:
            substrate = DigitalSubstrate()
        op = substrate.get_forward_operator()
        h = x
        acts = [h]  # Include input
        for layer in self._layers:
            if isinstance(layer, nn.Linear):
                h = op(h, layer.weight)
                if layer.bias is not None:
                    h += layer.bias
            else:
                h = layer(h)
                # Add after activation functions (ReLU, etc.)
                acts.append(h)
        # Add final output if last layer was Linear (no trailing activation)
        if self._layers and isinstance(self._layers[-1], nn.Linear):
            acts.append(h)
        return acts

    def route(self, activations: Tensor) -> Tensor:
        """Single-step routing through the geometry (for settling dynamics)."""
        h = activations
        for layer in self._layers:
            if isinstance(layer, nn.Linear):
                h = h @ layer.weight.T  # ruff: ignore[non-augmented-assignment]
                if layer.bias is not None:
                    h += layer.bias
            else:
                h = layer(h)
        return h

    def update_params(self, new_params: dict[str, Tensor]) -> None:
        for name, param in new_params.items():
            parts = name.split(".")
            if len(parts) >= 2 and parts[0].isdigit():
                layer_idx = int(parts[0])
                param_name = ".".join(parts[1:])
                if layer_idx < len(self._layers) and hasattr(
                    self._layers[layer_idx], param_name
                ):
                    getattr(self._layers[layer_idx], param_name).data.copy_(param)
            else:
                for layer in self._layers:
                    if hasattr(layer, name):
                        getattr(layer, name).data.copy_(param)

    def transition_modules(self) -> list[nn.Module]:
        return [m for m in self._layers if isinstance(m, nn.Linear)]


class RecurrentGeometry(nn.Module):
    """Recurrent attractor topology (Hopfield, EqProp MLPs).

    The hidden state is recurrently connected: h_{t+1} = f(h_t, x).
    """

    _layers: nn.ModuleList
    _recurrent_weight: nn.Parameter | None

    def __init__(
        self,
        config: GeometryConfig,
        layers: nn.ModuleList | None = None,
        hidden_dim: int | None = None,
        recurrent_weight: Tensor | None = None,
    ):
        super().__init__()
        self.config = config
        self._layers = layers or nn.ModuleList()
        self._recurrent_weight = None
        if not self._layers and config.hidden_dims:
            self._build_layers()
        if recurrent_weight is not None:
            self._recurrent_weight = nn.Parameter(recurrent_weight)
        elif hidden_dim is not None and self._recurrent_weight is None:
            # For EqProp, initialize recurrent weight to small random values
            # so the nudge can propagate backwards through the network
            # (zero init prevents gradient flow to hidden layers)
            self._recurrent_weight = nn.Parameter(
                torch.randn(hidden_dim, hidden_dim) * config.init_scale * 0.1
            )

    def _build_layers(self) -> None:
        # For recurrent: input -> hidden (with recurrent), hidden -> output
        dims = (
            self.config.input_dim,
            *self.config.hidden_dims,
            self.config.output_dim,
        )
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self._layers = nn.ModuleList(layers)
        # Add recurrent weight for the last hidden layer
        if len(self.config.hidden_dims) > 0 and self._recurrent_weight is None:
            hidden_dim = self.config.hidden_dims[-1]
            if self.config.recurrent_weight is not None:
                self._recurrent_weight = nn.Parameter(
                    torch.tensor(self.config.recurrent_weight)
                )
            else:
                # Small random initialization for EqProp
                self._recurrent_weight = nn.Parameter(
                    torch.randn(hidden_dim, hidden_dim) * self.config.init_scale * 0.1
                )
        self._set_param_names()

    def _set_param_names(self) -> None:
        """Set _param_name attribute on weight tensors for substrate keying."""
        for i, layer in enumerate(self._layers):
            if isinstance(layer, nn.Linear):
                layer.weight._param_name = f"layer_{i}_weight"
                if layer.bias is not None:
                    layer.bias._param_name = f"layer_{i}_bias"
        if self._recurrent_weight is not None:
            self._recurrent_weight._param_name = "recurrent_weight"

    @property
    def params(self) -> dict[str, Tensor]:
        params = dict(self._layers.named_parameters())
        if self._recurrent_weight is not None:
            params["recurrent_weight"] = self._recurrent_weight
        return params  # type: ignore[return-value]

    def forward(self, x: Tensor, substrate: Substrate | None = None) -> Tensor:
        """Full forward pass with recurrence (single step)."""
        if substrate is None:
            substrate = DigitalSubstrate()
        op = substrate.get_forward_operator()
        h = x
        for i, layer in enumerate(self._layers):
            if isinstance(layer, nn.Linear):
                h = op(h, layer.weight)
                if layer.bias is not None:
                    h += layer.bias
            else:
                h = layer(h)
            # Apply recurrent connection after each hidden layer (except output)
            if self._recurrent_weight is not None and i < len(self._layers) - 2:
                h += op(h, self._recurrent_weight)
        return h

    def route(self, activations: Tensor) -> Tensor:
        """Single recurrent step: h_{t+1} = f(h_t) = activation(W_rec @ h_t).

        Expects activations to be the hidden state (batch_size x hidden_dim).
        """
        h = activations
        if self._recurrent_weight is not None:
            # Hidden state should match recurrent weight dimensions
            if h.shape[-1] == self._recurrent_weight.shape[0]:
                h @= self._recurrent_weight.T
            else:
                # Activations are output dim; we can't apply recurrent weight
                # This happens when route is called on output instead of hidden state
                pass
        return h

    def update_params(self, new_params: dict[str, Tensor]) -> None:
        for name, param in new_params.items():
            if name == "recurrent_weight" and self._recurrent_weight is not None:
                self._recurrent_weight.data.copy_(param)
            else:
                parts = name.split(".")
                if len(parts) >= 2 and parts[0].isdigit():
                    layer_idx = int(parts[0])
                    param_name = ".".join(parts[1:])
                    if layer_idx < len(self._layers) and hasattr(
                        self._layers[layer_idx], param_name
                    ):
                        getattr(self._layers[layer_idx], param_name).data.copy_(param)
                else:
                    for layer in self._layers:
                        if hasattr(layer, name):
                            getattr(layer, name).data.copy_(param)

    def transition_modules(self) -> list[nn.Module]:
        return [m for m in self._layers if isinstance(m, nn.Linear)]

    def forward_with_intermediates(
        self, x: Tensor, substrate: Substrate | None = None
    ) -> list[Tensor]:
        """Forward pass returning intermediate activations for each layer."""
        if substrate is None:
            substrate = DigitalSubstrate()
        op = substrate.get_forward_operator()
        h = x
        acts = [h]
        for i, layer in enumerate(self._layers):
            if isinstance(layer, nn.Linear):
                h = op(h, layer.weight)
                if layer.bias is not None:
                    h += layer.bias
            else:
                h = layer(h)
                # Add after activation functions
                acts.append(h)
            # Apply recurrent connection after each hidden layer (except output)
            if self._recurrent_weight is not None and i < len(self._layers) - 2:
                h += op(h, self._recurrent_weight)
        # Add final output if last layer was Linear (no trailing activation)
        if isinstance(self._layers[-1], nn.Linear):
            acts.append(h)
        return acts


class TileGeometry(nn.Module):
    """TileNet mesh topology: modular independent tiles with local boundaries and asynchronous routing.

    Implements the Geometry protocol for the TileNet architecture. The topology consists of
    a layered tile graph where each tile is a computational unit with its own neurons,
    and tiles within a layer route activations in parallel. Inter-layer connections
    follow a dense or configurable adjacency pattern.

    Key properties:
    - Tile-local computation (each tile has its own weight matrix)
    - Layer-wise parallel routing
    - Configurable intra/inter-layer connectivity
    - Support for skip connections
    - Asynchronous boundary conditions (MoE-style gating)
    """

    _tile_weights: nn.ParameterDict
    _tile_biases: nn.ParameterDict
    _graph: TileGraph
    _input_projection: nn.Linear
    _output_projection: nn.Linear

    def __init__(
        self,
        config: GeometryConfig,
        tile_graph: TileGraph | None = None,
        neurons_per_tile: int = 48,
        tiles_per_layer: int = 4,
        use_skip_connections: bool = False,
    ):
        super().__init__()
        self.config = config
        self._tile_weights = nn.ParameterDict()
        self._tile_biases = nn.ParameterDict()

        if tile_graph is not None:
            self._graph = tile_graph
        else:
            self._graph = TileGraph()
            self._graph.build_layered(
                input_dim=config.input_dim,
                output_dim=config.output_dim,
                neurons_per_tile=neurons_per_tile,
                num_hidden_layers=max(config.num_layers - 2, 1),
                tiles_per_layer=tiles_per_layer,
                use_skip_connections=use_skip_connections,
            )

        self._build_projections()
        self._build_tile_params()
        self._set_projection_param_names()

    def _build_projections(self) -> None:
        """Build input/output projections between raw IO and tile-state space."""
        input_neurons = sum(
            self._graph.tiles[tid].neurons for tid in self._graph.input_tile_ids
        )
        output_neurons = sum(
            self._graph.tiles[tid].neurons for tid in self._graph.output_tile_ids
        )
        self._input_projection = nn.Linear(
            self.config.input_dim, input_neurons, bias=True
        )
        self._output_projection = nn.Linear(
            output_neurons, self.config.output_dim, bias=True
        )

    def _build_tile_params(self) -> None:
        """Per-edge incoming weights and per-tile biases."""
        import math

        for tid, tile in self._graph.tiles.items():
            if tile.is_input:
                continue
            bias = nn.Parameter(torch.zeros(tile.neurons))
            bias._param_name = f"tile_bias_{tid}"
            self._tile_biases[str(tid)] = bias
            for src_id in tile.bwd_neighbors:
                src = self._graph.tiles[src_id]
                bound = 1.0 / math.sqrt(src.neurons) if src.neurons > 0 else 0.0
                w = torch.empty(tile.neurons, src.neurons).uniform_(-bound, bound)
                param = nn.Parameter(w)
                param._param_name = f"tile_weight_{src_id}_{tid}"
                self._tile_weights[f"{src_id}_{tid}"] = param

    def _set_projection_param_names(self) -> None:
        """Set _param_name on input/output projection weights."""
        if self._input_projection is not None:
            self._input_projection.weight._param_name = "input_proj_weight"
            if self._input_projection.bias is not None:
                self._input_projection.bias._param_name = "input_proj_bias"
        if self._output_projection is not None:
            self._output_projection.weight._param_name = "output_proj_weight"
            if self._output_projection.bias is not None:
                self._output_projection.bias._param_name = "output_proj_bias"

    @staticmethod
    def _weight_key(src_id: int, dst_id: int) -> str:
        return f"{src_id}_{dst_id}"

    @property
    def params(self) -> dict[str, Tensor]:
        params = {}
        if self._input_projection is not None:
            params.update({
                f"input_proj.{k}": v
                for k, v in self._input_projection.named_parameters()
            })
        if self._output_projection is not None:
            params.update({
                f"output_proj.{k}": v
                for k, v in self._output_projection.named_parameters()
            })
        params.update({f"tile_bias.{k}": v for k, v in self._tile_biases.items()})
        params.update({f"tile_weight.{k}": v for k, v in self._tile_weights.items()})
        return params

    def forward(self, x: Tensor, substrate: Substrate | None = None) -> Tensor:
        """Route input through the tile mesh using substrate's forward operator."""
        if substrate is None:
            substrate = DigitalSubstrate()
        op = substrate.get_forward_operator()

        # Project input to tile space
        h = self._input_projection(x)

        # Set input tile activities
        offset = 0
        for tid in self._graph.input_tile_ids:
            n = self._graph.tiles[tid].neurons
            self._graph.tiles[tid].activity = h[:, offset : offset + n]
            offset += n

        # Forward propagate through layers (skip input layer)
        for layer_tiles in self._graph.layer_ids[1:]:
            for tid in layer_tiles:
                tile = self._graph.tiles[tid]
                # Compute weighted sum of incoming activities + bias
                acc: Tensor | None = None
                for src_id in tile.bwd_neighbors:
                    src_act = self._graph.tiles[src_id].activity
                    if src_act is None:
                        continue
                    w = self._tile_weights[self._weight_key(src_id, tid)]
                    contrib = op(src_act, w)
                    acc = contrib if acc is None else acc + contrib
                if acc is not None:
                    acc += (
                        self
                        ._tile_biases[str(tid)]
                        .unsqueeze(0)
                        .expand(acc.shape[0], -1)
                    )
                    tile.activity = acc
                    tile.prediction = acc

        # Collect output tile activities
        out_acts: list[Tensor] = []
        for tid in self._graph.output_tile_ids:
            act = self._graph.tiles[tid].activity
            if act is not None:
                out_acts.append(act)

        if not out_acts:
            return torch.empty(x.shape[0], self.config.output_dim, device=x.device)

        h = torch.cat(out_acts, dim=1)
        return self._output_projection(h)

    def route(self, activations: Tensor) -> Tensor:
        """Single-step routing through the tile mesh (for settling dynamics).

        This is used by StateDynamics for iterative settling. It expects
        activations to already be in tile space (i.e., after input projection).
        """
        # For settling, we assume activations represent the current tile activities
        # We need to distribute them to the appropriate tiles and route one step
        # This is a simplified version - in practice, the StateDynamics would
        # maintain the tile activities directly
        return self._route_flat(activations)

    def _route_flat(self, flat_activations: Tensor) -> Tensor:
        """Route flat concatenated tile activities through one step."""
        # Distribute flat activations to tiles
        self._set_tile_activities_from_flat(flat_activations)

        # One propagation step through all non-input tiles
        for layer_tiles in self._graph.layer_ids[1:]:
            for tid in layer_tiles:
                tile = self._graph.tiles[tid]
                acc: Tensor | None = None
                for src_id in tile.bwd_neighbors:
                    src_act = self._graph.tiles[src_id].activity
                    if src_act is None:
                        continue
                    w = self._tile_weights[self._weight_key(src_id, tid)]
                    contrib = src_act @ w.T
                    acc = contrib if acc is None else acc + contrib
                if acc is not None:
                    acc += (
                        self
                        ._tile_biases[str(tid)]
                        .unsqueeze(0)
                        .expand(acc.shape[0], -1)
                    )
                    tile.activity = acc
                    tile.prediction = acc

        # Collect and flatten
        acts: list[Tensor] = []
        for layer_tiles in self._graph.layer_ids:
            for tid in layer_tiles:
                act = self._graph.tiles[tid].activity
                if act is not None:
                    acts.append(act)
        return (
            torch.cat(acts, dim=1)
            if acts
            else torch.empty(
                flat_activations.shape[0], 0, device=flat_activations.device
            )
        )

    def _set_tile_activities_from_flat(self, flat_activations: Tensor) -> None:
        """Distribute flat concatenated activations to individual tiles."""
        offset = 0
        for layer_tiles in self._graph.layer_ids:
            for tid in layer_tiles:
                n = self._graph.tiles[tid].neurons
                if offset + n <= flat_activations.shape[1]:
                    self._graph.tiles[tid].activity = flat_activations[
                        :, offset : offset + n
                    ]
                    offset += n

    def _get_flat_activities(self) -> Tensor:
        """Collect all tile activities as a flat concatenated tensor."""
        acts: list[Tensor] = []
        for layer_tiles in self._graph.layer_ids:
            for tid in layer_tiles:
                act = self._graph.tiles[tid].activity
                if act is not None:
                    acts.append(act)
        return torch.cat(acts, dim=1) if acts else torch.empty(1, 0)

    def update_params(self, new_params: dict[str, Tensor]) -> None:
        """Update geometry parameters in-place from ParameterUpdate output."""
        for name, param in new_params.items():
            if name.startswith("input_proj.") and self._input_projection is not None:
                pname = name.replace("input_proj.", "")
                if hasattr(self._input_projection, pname):
                    getattr(self._input_projection, pname).data.copy_(param)
            elif (
                name.startswith("output_proj.") and self._output_projection is not None
            ):
                pname = name.replace("output_proj.", "")
                if hasattr(self._output_projection, pname):
                    getattr(self._output_projection, pname).data.copy_(param)
            elif name.startswith("tile_bias."):
                key = name.replace("tile_bias.", "")
                if key in self._tile_biases:
                    self._tile_biases[key].data.copy_(param)
            elif name.startswith("tile_weight."):
                key = name.replace("tile_weight.", "")
                if key in self._tile_weights:
                    self._tile_weights[key].data.copy_(param)
            # Try direct match for backward compatibility
            elif name in self._tile_weights:
                self._tile_weights[name].data.copy_(param)
            elif name in self._tile_biases:
                self._tile_biases[name].data.copy_(param)

    def _validate_shapes(self) -> None:
        """Validate that projection dimensions match tile graph structure.

        Checks that input/output projection dimensions match the sum of
        input/output tile neurons. Raises ValueError if mismatch detected.
        """
        # Validate input projection
        input_neurons = sum(
            self._graph.tiles[tid].neurons for tid in self._graph.input_tile_ids
        )
        in_feats = self._input_projection.in_features
        out_feats = self._input_projection.out_features
        if in_feats != self.config.input_dim:
            msg = f"Input projection in_features ({in_feats}) != config.input_dim ({self.config.input_dim})"
            raise ValueError(msg)
        if out_feats != input_neurons:
            msg = f"Input projection out_features ({out_feats}) != sum of input tile neurons ({input_neurons})"
            raise ValueError(msg)

        # Validate output projection
        output_neurons = sum(
            self._graph.tiles[tid].neurons for tid in self._graph.output_tile_ids
        )
        out_in_feats = self._output_projection.in_features
        out_out_feats = self._output_projection.out_features
        if out_in_feats != output_neurons:
            msg = f"Output projection in_features ({out_in_feats}) != sum of output tile neurons ({output_neurons})"
            raise ValueError(msg)
        if out_out_feats != self.config.output_dim:
            msg = f"Output projection out_features ({out_out_feats}) != config.output_dim ({self.config.output_dim})"
            raise ValueError(msg)

    def transition_modules(self) -> list[nn.Module]:
        """Return modules in forward order for TransitionGraph protocol."""
        modules = []
        if self._input_projection is not None:
            modules.append(self._input_projection)
        # Tile weights are Parameters, not Modules
        if self._output_projection is not None:
            modules.append(self._output_projection)
        return modules

    def get_boundary_tiles(self, device_map: dict[int, int]) -> dict[int, list[int]]:
        """Identify boundary tiles that connect to different devices.

        For P2P/distributed training, this identifies tiles that need
        cross-device communication.
        """
        return self._graph.get_boundary_tiles(device_map)

    def forward_with_intermediates(
        self, x: Tensor, substrate: Substrate | None = None
    ) -> list[Tensor]:
        """Forward pass returning intermediate activations for each layer."""
        if substrate is None:
            substrate = DigitalSubstrate()
        op = substrate.get_forward_operator()

        # Project input to tile space
        h = self._input_projection(x)
        acts = [h]  # Input projection output

        # Set input tile activities
        offset = 0
        for tid in self._graph.input_tile_ids:
            n = self._graph.tiles[tid].neurons
            self._graph.tiles[tid].activity = h[:, offset : offset + n]
            offset += n

        # Forward propagate through layers (skip input layer)
        for layer_tiles in self._graph.layer_ids[1:]:
            for tid in layer_tiles:
                tile = self._graph.tiles[tid]
                acc: Tensor | None = None
                for src_id in tile.bwd_neighbors:
                    src_act = self._graph.tiles[src_id].activity
                    if src_act is None:
                        continue
                    w = self._tile_weights[self._weight_key(src_id, tid)]
                    contrib = op(src_act, w)
                    acc = contrib if acc is None else acc + contrib
                if acc is not None:
                    acc += (
                        self
                        ._tile_biases[str(tid)]
                        .unsqueeze(0)
                        .expand(acc.shape[0], -1)
                    )
                    tile.activity = acc
                    tile.prediction = acc

        # Collect output tile activities
        out_acts: list[Tensor] = []
        for tid in self._graph.output_tile_ids:
            act = self._graph.tiles[tid].activity
            if act is not None:
                out_acts.append(act)

        if out_acts:
            h = torch.cat(out_acts, dim=1)
            out = self._output_projection(h)
            acts.append(out)

        return acts


class InstantaneousDynamics:
    """Single-pass feedforward (Backprop, Forward-Forward)."""

    def __init__(self, config: StateDynamicsConfig | None = None):
        self.config = config or StateDynamicsConfig.instantaneous()

    def settle(  # ruff: ignore[no-self-use]
        self,
        state: SystemState,
        geometry: Geometry,
        substrate: Substrate,
        target: Tensor | None = None,
    ) -> SystemState:
        # Single forward pass - no settling
        if state.x is not None:
            acts = geometry.forward_with_intermediates(state.x, substrate)
        else:
            acts = [state.activations] if state.activations is not None else []
        if target is None:
            state.free_state = acts
        else:
            state.nudged_state = acts
        state.activations = acts
        return state

    def compute_energy(self, state: SystemState, geometry: Geometry) -> Tensor:  # ruff: ignore[no-self-use]
        # Proxy: negative log-likelihood for instantaneous pass
        if state.loss is not None:
            return torch.as_tensor(state.loss)
        return torch.tensor(0.0)


class ThermodynamicContrast:
    """Equilibrium Propagation credit assignment: (nudged - free) / beta.

    For feedforward networks, computes parameter pseudo-gradients using the
    contrastive Hebbian rule: ΔW = (free_pre @ free_post - nudged_pre @ nudged_post) / β
    """

    def __init__(self, config: CreditAssignmentConfig | None = None):
        self.config = config or CreditAssignmentConfig.thermodynamic_contrast()

    def compute_pseudo_gradient(
        self,
        free_state: SystemState,
        nudged_state: SystemState,
        loss: Tensor,
        geometry: Geometry,
    ) -> list[Tensor]:
        if free_state.activations is None or nudged_state.activations is None:
            return []

        free_acts = (
            free_state.activations
            if isinstance(free_state.activations, list)
            else [free_state.activations]
        )
        nudged_acts = (
            nudged_state.activations
            if isinstance(nudged_state.activations, list)
            else [nudged_state.activations]
        )

        # Get parameter names to match gradients
        param_names = list(geometry.params.keys())
        weight_names = [
            n for n in param_names if "weight" in n and geometry.params[n].ndim == 2
        ]

        grads = []
        # Compute contrastive Hebbian gradients for each weight matrix
        # Assume activations are ordered: [input, hidden1, hidden2, ..., output]
        n_layers = len(free_acts) - 1
        for l in range(n_layers):
            if l < len(weight_names):
                # Free phase correlation: free_pre^T @ free_post
                free_pre = free_acts[l]  # (batch, in_dim)
                free_post = free_acts[l + 1]  # (batch, out_dim)
                free_corr = free_pre.T @ free_post  # (in_dim, out_dim)

                # Nudged phase correlation
                nudged_pre = nudged_acts[l]
                nudged_post = nudged_acts[l + 1]
                nudged_corr = nudged_pre.T @ nudged_post

                # Contrastive gradient: (free - nudged) / β
                contrast = (
                    (free_corr - nudged_corr) / self.config.beta / free_pre.shape[0]
                )
                # Transpose to match weight shape (out_dim, in_dim)
                grads.append(contrast.T)

        return grads

    def surrogate_objective(
        self,
        free_state: SystemState,
        nudged_state: SystemState,
        geometry: Geometry,
    ) -> Tensor:
        """Surrogate objective for EqProp: negative free energy difference."""
        if free_state.energy is not None and nudged_state.energy is not None:
            return nudged_state.energy - free_state.energy
        return torch.tensor(0.0)


# Alias for backwards compatibility with README/docs
ThermodynamicContrastCredit = ThermodynamicContrast


class EuclideanUpdate:
    """Standard Euclidean SGD/Adam update."""

    def __init__(self, config: ParameterUpdateConfig | None = None):
        self.config = config or ParameterUpdateConfig.euclidean()
        self._momentum_buffers: dict[str, Tensor] = {}

    def step(
        self,
        params: dict[str, Tensor],
        pseudo_grads: list[Tensor],
        geometry: Geometry,  # ruff: ignore[unused-method-argument]
    ) -> dict[str, Tensor]:
        updated = {}

        grad_idx = 0
        for name, param in params.items():
            if grad_idx < len(pseudo_grads):
                grad = pseudo_grads[grad_idx]
                # Only apply gradient if shapes match
                if grad.shape == param.shape:
                    if self.config.momentum > 0:
                        buf = self._momentum_buffers.get(name, torch.zeros_like(param))
                        buf.mul_(self.config.momentum).add_(grad)
                        self._momentum_buffers[name] = buf
                        updated[name] = param - self.config.step_size * buf
                    else:
                        updated[name] = param - self.config.step_size * grad
                    grad_idx += 1
                else:
                    # Shape mismatch - skip this parameter
                    updated[name] = param
            else:
                updated[name] = param

        return updated


# ============================================================
# Adapter: Wrap existing models as System compositions
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

    def to_system(self) -> System:
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
        if not hasattr(self.model, "backend"):
            return None
        backend = getattr(self.model, "backend", None)
        if not isinstance(backend, str):
            return None
        backend_map = {
            "quantized": QuantizedSubstrate,
            "noisy": NoisySubstrate,
            "optical": OpticalSubstrate,
            "crossbar": MemristiveSubstrate,
            "quantum": QuantumSubstrate,
        }
        cls = backend_map.get(backend)
        return (
            cls(
                SubstrateConfig(
                    precision="float32",
                    noise_level=0.0,
                    weight_bounds=None,
                    sparsity=0.0,
                    device="cpu",
                )
            )
            if cls
            else None
        )

    def _infer_substrate_from_family(self) -> Substrate | None:
        if not (self._metadata and self._metadata.family):
            return None
        family = self._metadata.family.lower()
        family_map = [
            (("quantized", "ternary"), QuantizedSubstrate),
            (("noisy",), NoisySubstrate),
            (("optical", "photonic"), OpticalSubstrate),
            (("crossbar", "memristive"), MemristiveSubstrate),
            (("quantum",), QuantumSubstrate),
        ]
        for keys, cls in family_map:
            if any(k in family for k in keys):
                return cls(
                    SubstrateConfig(
                        precision="float32",
                        noise_level=0.0,
                        weight_bounds=None,
                        sparsity=0.0,
                        device="cpu",
                    )
                )
        return None

    def _infer_geometry(self) -> Geometry:
        # Priority 1: Check for specific topology types
        geometry: Geometry | None = None
        if hasattr(self.model, "topology_type"):
            topo_map = {
                "recurrent": self._make_recurrent_geometry,
                "recurrent_attractor": self._make_recurrent_geometry,
                "tile_mesh": self._make_tile_geometry,
                "tile": self._make_tile_geometry,
                "neuromorphic": self._make_neuromorphic_geometry,
                "fabric": self._make_neuromorphic_geometry,
                "spatial_lattice": self._make_spatial_geometry,
                "3d": self._make_spatial_geometry,
            }
            topology_type = getattr(self.model, "topology_type", None)
            if isinstance(topology_type, str):
                maker = topo_map.get(topology_type)
                if maker:
                    geometry = maker()

        # Priority 2: Registry metadata family
        if geometry is None and self._metadata and self._metadata.family:
            family = self._metadata.family.lower()
            if "tile" in family:
                geometry = self._make_tile_geometry()
            elif any(k in family for k in ("cube", "neural_cube")):
                geometry = self._make_spatial_geometry()
            elif any(k in family for k in ("graph", "fabric")):
                geometry = self._make_neuromorphic_geometry()
            elif any(k in family for k in ("recurrent", "equilibrium", "eqprop", "ep")):
                geometry = self._make_recurrent_geometry()

        # Priority 3: Model attributes - check for TileAlgorithm
        if (
            geometry is None
            and hasattr(self.model, "graph")
            and hasattr(self.model, "config")
        ):
            # Check if it's a TileAlgorithm or similar tile-based model
            config = getattr(self.model, "config", None)
            if config is not None and hasattr(config, "neurons_per_tile"):
                geometry = self._make_tile_geometry()

        # Priority 4: Model attributes - check for transition_modules
        if geometry is None and hasattr(self.model, "transition_modules"):
            layers = self.model.transition_modules()  # type: ignore[attr-defined]
            geometry = FeedforwardGeometry(
                GeometryConfig(
                    input_dim=self._infer_input_dim(),
                    output_dim=self._infer_output_dim(),
                    hidden_dims=self._infer_hidden_dims(),
                    num_layers=len(self._infer_hidden_dims()),
                    topology_type="feedforward",
                    connectivity=None,
                    recurrent_weight=None,
                ),
                layers=layers,
            )

        if geometry is None:
            geometry = FeedforwardGeometry(
                GeometryConfig(
                    input_dim=self._infer_input_dim(),
                    output_dim=self._infer_output_dim(),
                    hidden_dims=self._infer_hidden_dims(),
                    num_layers=len(self._infer_hidden_dims()),
                    topology_type="feedforward",
                    connectivity=None,
                    recurrent_weight=None,
                )
            )

        return geometry

    def _infer_input_dim(self) -> int:
        if hasattr(self.model, "input_dim"):
            return int(self.model.input_dim)
        if isinstance(self.model, nn.Sequential):
            for layer in self.model:
                if isinstance(layer, nn.Linear):
                    return layer.in_features
        return 0

    def _infer_output_dim(self) -> int:
        if hasattr(self.model, "output_dim"):
            return int(self.model.output_dim)
        if isinstance(self.model, nn.Sequential):
            for layer in reversed(self.model):
                if isinstance(layer, nn.Linear):
                    return layer.out_features
        return 0

    def _infer_hidden_dims(self) -> tuple[int, ...]:
        if hasattr(self.model, "hidden_dims"):
            hidden_dims = self.model.hidden_dims
            if isinstance(hidden_dims, (list, tuple)):
                return tuple(int(d) for d in hidden_dims)
        if hasattr(self.model, "hidden_dim"):
            hidden_dim = self.model.hidden_dim
            if isinstance(hidden_dim, int):
                return (hidden_dim,)
        if hasattr(self.model, "config"):
            config = getattr(self.model, "config", None)
            if (
                config is not None
                and not isinstance(config, Tensor)
                and hasattr(config, "hidden_dims")
            ):
                hidden_dims = config.hidden_dims
                if isinstance(hidden_dims, (list, tuple)):
                    return tuple(int(d) for d in hidden_dims)

        # Try to infer from nn.Sequential or ModuleList
        if isinstance(self.model, nn.Sequential):
            dims = []
            for layer in self.model:
                if isinstance(layer, nn.Linear):
                    if not dims:
                        dims.append(layer.in_features)
                    dims.append(layer.out_features)
            if len(dims) >= 3:  # input, hidden..., output
                return tuple(dims[1:-1])  # Exclude input and output
        return ()

    def _make_recurrent_geometry(self) -> RecurrentGeometry:
        return RecurrentGeometry(
            GeometryConfig(
                input_dim=self._infer_input_dim(),
                output_dim=self._infer_output_dim(),
                hidden_dims=self._infer_hidden_dims(),
                num_layers=len(self._infer_hidden_dims()),
                topology_type="recurrent",
                connectivity=None,
                recurrent_weight=None,
            ),
            hidden_dim=self._infer_hidden_dims()[-1]
            if self._infer_hidden_dims()
            else None,
        )

    def _make_tile_geometry(self) -> Geometry:
        # Use the full TileGeometry implementation
        # Extract tile-specific config from model
        config = getattr(self.model, "config", None)
        if config is not None:
            neurons_per_tile = getattr(config, "neurons_per_tile", 48)
            tiles_per_layer = getattr(config, "tiles_per_layer", 4)
            use_skip = getattr(config, "use_skip_connections", False)
        else:
            neurons_per_tile = getattr(self.model, "neurons_per_tile", 48)
            tiles_per_layer = getattr(self.model, "tiles_per_layer", 4)
            use_skip = getattr(self.model, "use_skip_connections", False)

        # Try to get the tile graph from the model if it has one
        tile_graph = getattr(self.model, "graph", None)

        return TileGeometry(
            GeometryConfig(
                input_dim=getattr(self.model, "input_dim", 0),
                output_dim=getattr(self.model, "output_dim", 0),
                hidden_dims=self._infer_hidden_dims(),
                num_layers=len(self._infer_hidden_dims()),
                topology_type="tile_mesh",
                connectivity=None,
                recurrent_weight=None,
            ),
            tile_graph=tile_graph,
            neurons_per_tile=neurons_per_tile,
            tiles_per_layer=tiles_per_layer,
            use_skip_connections=use_skip,
        )

    def _make_neuromorphic_geometry(self) -> Geometry:
        return FeedforwardGeometry(
            GeometryConfig(
                input_dim=getattr(self.model, "input_dim", 0),
                output_dim=getattr(self.model, "output_dim", 0),
                hidden_dims=self._infer_hidden_dims(),
                num_layers=len(self._infer_hidden_dims()),
                topology_type="neuromorphic",
                connectivity=None,
                recurrent_weight=None,
            )
        )

    def _make_spatial_geometry(self) -> Geometry:
        return FeedforwardGeometry(
            GeometryConfig(
                input_dim=getattr(self.model, "input_dim", 0),
                output_dim=getattr(self.model, "output_dim", 0),
                hidden_dims=self._infer_hidden_dims(),
                num_layers=len(self._infer_hidden_dims()),
                topology_type="spatial_lattice",
                connectivity=None,
                recurrent_weight=None,
            )
        )

    def _infer_dynamics(self) -> StateDynamics:
        # Priority 1: Registry metadata (most reliable)
        if self._metadata and self._metadata.family:
            family = self._metadata.family.lower()
            dynamics = self._dynamics_from_family(family)
            if dynamics is not None:
                return dynamics

        # Priority 2: Model attributes
        if hasattr(self.model, "gradient_method"):
            method = getattr(self.model, "gradient_method", None)
            if isinstance(method, str):
                dynamics = self._dynamics_from_gradient_method(method)
                if dynamics is not None:
                    return dynamics

        if hasattr(self.model, "max_steps") and getattr(self.model, "max_steps", 1) > 1:
            max_steps = getattr(self.model, "max_steps", 30)
            if not isinstance(max_steps, int):
                max_steps = 30
            return EnergyMinimizationDynamics(
                StateDynamicsConfig(
                    dynamics_type="energy_minimization",
                    max_steps=max_steps,
                    convergence_threshold=1e-4,
                    convergence_start=5,
                    step_size=0.1,
                    beta=getattr(self.model, "beta", 0.5),
                    track_free_energy_per_iter=False,
                )
            )

        # Priority 3: Locality level
        if self._metadata and self._metadata.locality_level:
            dynamics = self._dynamics_from_locality(self._metadata.locality_level)
            if dynamics is not None:
                return dynamics

        return InstantaneousDynamics(StateDynamicsConfig.instantaneous())

    def _dynamics_from_family(self, family: str) -> StateDynamics | None:  # ruff: ignore[no-self-use]
        equilibrium_keys = ("equilibrium", "eqprop", "ep", "chl")
        if any(k in family for k in equilibrium_keys):
            return EnergyMinimizationDynamics(StateDynamicsConfig.energy_minimization())
        if any(k in family for k in ("predictive", "pc")):
            return PredictiveSettlingDynamics(StateDynamicsConfig.predictive_settling())
        if any(k in family for k in ("spiking", "stdp", "snn")):
            return SpikeIntegrationDynamics(
                StateDynamicsConfig(
                    dynamics_type="spike_integration",
                    max_steps=30,
                    convergence_threshold=1e-4,
                    convergence_start=5,
                    step_size=0.1,
                    beta=0.5,
                    track_free_energy_per_iter=False,
                )
            )
        forward_keys = (
            "forward_only",
            "ff",
            "pepita",
            "fa",
            "feedback",
            "dfa",
            "target",
            "target_prop",
            "tp",
        )
        if any(k in family for k in forward_keys):
            return InstantaneousDynamics(StateDynamicsConfig.instantaneous())
        return None

    def _dynamics_from_gradient_method(self, method: str) -> StateDynamics | None:
        if method == "equilibrium":
            return EnergyMinimizationDynamics(
                StateDynamicsConfig.energy_minimization(
                    max_steps=getattr(self.model, "max_steps", 30),
                    beta=getattr(self.model, "beta", 0.5),
                )
            )
        if method == "predictive_coding":
            return PredictiveSettlingDynamics(StateDynamicsConfig.predictive_settling())
        if method in {"spiking", "stdp"}:
            return SpikeIntegrationDynamics(StateDynamicsConfig.spike_integration())
        return None

    def _dynamics_from_locality(self, locality: LocalityLevel) -> StateDynamics | None:  # ruff: ignore[no-self-use]
        if locality == LocalityLevel.EQUILIBRIUM:
            return EnergyMinimizationDynamics(StateDynamicsConfig.energy_minimization())
        if locality == LocalityLevel.FORWARD_ONLY:
            return InstantaneousDynamics(StateDynamicsConfig.instantaneous())
        if locality == LocalityLevel.LOCAL:
            return SpikeIntegrationDynamics(StateDynamicsConfig.spike_integration())
        return None

    def _infer_credit(self) -> CreditAssignment:
        # Priority 1: Registry metadata
        if self._metadata and self._metadata.credit_assignment_type:
            credit_type = self._metadata.credit_assignment_type
            credit = self._credit_from_type(credit_type, with_config=True)
            if credit is not None:
                return credit

        # Priority 2: Model attributes
        if hasattr(self.model, "credit_assignment_type"):
            credit_type = getattr(self.model, "credit_assignment_type", None)
            if isinstance(credit_type, str):
                credit = self._credit_from_type(credit_type, with_config=False)
                if credit is not None:
                    return credit

        # Priority 3: Family
        if self._metadata and self._metadata.family:
            family = self._metadata.family.lower()
            credit = self._credit_from_family(family)
            if credit is not None:
                return credit

        return BackpropCredit(
            CreditAssignmentConfig(
                credit_type="gradient",
                beta=0.5,
                feedback_matrix=None,
                local_objective="mse",
                orthogonal_init=False,
                feedback_scale=0.01,
            )
        )

    def _credit_from_type(  # ruff: ignore[no-self-use]
        self, credit_type: str, with_config: bool
    ) -> CreditAssignment | None:
        if credit_type == "equilibrium":
            if with_config:
                return ThermodynamicContrast(
                    CreditAssignmentConfig(
                        credit_type="thermodynamic_contrast",
                        beta=0.5,
                        feedback_matrix=None,
                        local_objective="mse",
                        orthogonal_init=False,
                        feedback_scale=0.01,
                    )
                )
            return ThermodynamicContrast(
                CreditAssignmentConfig(
                    credit_type="thermodynamic_contrast",
                    beta=0.5,
                    feedback_matrix=None,
                    local_objective="mse",
                    orthogonal_init=False,
                    feedback_scale=0.01,
                )
            )
        credit_map: dict[str, type[CreditAssignment]] = {
            "random_projections": RandomProjectionsCredit,
            "feedback_alignment": RandomProjectionsCredit,
            "local_goodness": LocalGoodnessCredit,
            "temporal_trace": TemporalTraceCredit,
            "target_inversion": TargetInversionCredit,
            "gradient": BackpropCredit,
            "backpropagation": BackpropCredit,
        }
        cls = credit_map.get(credit_type)
        return (
            cls(
                CreditAssignmentConfig(
                    credit_type=credit_type,
                    beta=0.5,
                    feedback_matrix=None,
                    local_objective="mse",
                    orthogonal_init=False,
                    feedback_scale=0.01,
                )
            )
            if cls
            else None
        )

    def _credit_from_family(self, family: str) -> CreditAssignment | None:  # ruff: ignore[no-self-use]
        family_map: list[tuple[tuple[str, ...], type[CreditAssignment]]] = [
            (("eqprop", "equilibrium", "ep"), ThermodynamicContrast),
            (("fa", "feedback", "dfa"), RandomProjectionsCredit),
            (("hebbian", "chl"), ThermodynamicContrast),
            (("forward_only", "ff", "pepita"), LocalGoodnessCredit),
            (("spiking", "stdp"), TemporalTraceCredit),
            (("target_prop", "tp"), TargetInversionCredit),
        ]
        for keys, cls in family_map:
            if any(k in family for k in keys):
                return cls(
                    CreditAssignmentConfig(
                        credit_type="thermodynamic_contrast"
                        if cls is ThermodynamicContrast
                        else "random_projections",
                        beta=0.5,
                        feedback_matrix=None,
                        local_objective="mse",
                        orthogonal_init=False,
                        feedback_scale=0.01,
                    )
                )
        return None

    def _infer_update(self) -> ParameterUpdate:
        # Priority 1: Registry metadata tags
        update: ParameterUpdate | None = None
        if self._metadata and self._metadata.tags:
            tags = {t.lower() for t in self._metadata.tags}
            if tags & {"muon", "riemannian"}:
                update = RiemannianOrthogonalUpdate(
                    ParameterUpdateConfig(
                        update_type="riemannian_orthogonal",
                        step_size=0.01,
                        momentum=0.9,
                        ortho_steps=5,
                        spectral_norm=1.0,
                        fisher_damping=1e-3,
                        ewc_lambda=1000.0,
                    )
                )
            elif "spectral" in tags:
                update = SpectralConstrainedUpdate(
                    ParameterUpdateConfig(
                        update_type="spectral_constrained",
                        step_size=0.01,
                        momentum=0.9,
                        ortho_steps=5,
                        spectral_norm=1.0,
                        fisher_damping=1e-3,
                        ewc_lambda=1000.0,
                    )
                )
            elif tags & {"fisher", "natural"}:
                update = NaturalGradientUpdate(
                    ParameterUpdateConfig(
                        update_type="natural_gradient",
                        step_size=0.01,
                        momentum=0.9,
                        ortho_steps=5,
                        spectral_norm=1.0,
                        fisher_damping=1e-3,
                        ewc_lambda=1000.0,
                    )
                )
            elif tags & {"ewc", "elastic"}:
                update = ElasticConsolidationUpdate(
                    ParameterUpdateConfig(
                        update_type="elastic_consolidation",
                        step_size=0.01,
                        momentum=0.9,
                        ortho_steps=5,
                        spectral_norm=1.0,
                        fisher_damping=1e-3,
                        ewc_lambda=1000.0,
                    )
                )

        # Priority 2: Family
        if update is None and self._metadata and self._metadata.family:
            family = self._metadata.family.lower()
            if any(k in family for k in ("muon", "mep")):
                update = RiemannianOrthogonalUpdate(
                    ParameterUpdateConfig(
                        update_type="riemannian_orthogonal",
                        step_size=0.01,
                        momentum=0.9,
                        ortho_steps=5,
                        spectral_norm=1.0,
                        fisher_damping=1e-3,
                        ewc_lambda=1000.0,
                    )
                )
            elif "fisher" in family:
                update = NaturalGradientUpdate(
                    ParameterUpdateConfig(
                        update_type="natural_gradient",
                        step_size=0.01,
                        momentum=0.9,
                        ortho_steps=5,
                        spectral_norm=1.0,
                        fisher_damping=1e-3,
                        ewc_lambda=1000.0,
                    )
                )
            elif "ewc" in family:
                update = ElasticConsolidationUpdate(
                    ParameterUpdateConfig(
                        update_type="elastic_consolidation",
                        step_size=0.01,
                        momentum=0.9,
                        ortho_steps=5,
                        spectral_norm=1.0,
                        fisher_damping=1e-3,
                        ewc_lambda=1000.0,
                    )
                )

        return (
            update
            if update is not None
            else EuclideanUpdate(
                ParameterUpdateConfig(
                    update_type="euclidean",
                    step_size=0.01,
                    momentum=0.9,
                    ortho_steps=5,
                    spectral_norm=1.0,
                    fisher_damping=1e-3,
                    ewc_lambda=1000.0,
                )
            )
        )

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
        if hasattr(self.model, "train_step"):
            try:
                legacy_result = self.model.train_step(x, y)
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


# Additional substrate implementations
class AnalogSubstrate(DigitalSubstrate):
    """Analog compute substrate with continuous values and noise."""

    def __init__(self, config: SubstrateConfig | None = None):
        super().__init__(config or SubstrateConfig.analog())

    def inject_state_noise(self, s: Tensor) -> Tensor:
        """Add analog noise to state."""
        s = super().inject_state_noise(s)
        noise = torch.randn_like(s) * self.config.noise_level
        return self._to_precision(s + noise)


class MemristiveSubstrate(DigitalSubstrate):
    """Memristive crossbar: conductance matrices, bounded precision, IR-drop noise.

    Models the physical behavior of memristor crossbar arrays:
    - Conductance-based weights (positive, bounded)
    - IR-drop effects on column/row lines
    - Non-linear I-V characteristics
    - Stochastic switching and conductance drift
    - Pulse-based weight updates
    """

    def __init__(self, config: SubstrateConfig | None = None):
        super().__init__(config or SubstrateConfig.memristive())
        # Physical parameters
        self._ron = 1e3  # Low resistance state (ohms)
        self._roff = 1e6  # High resistance state (ohms)
        self._vth = 0.2  # Threshold voltage (V)
        self._wire_resistance = 10.0  # Wire resistance per segment (ohms)
        self._read_voltage = 0.1  # Read voltage (V)
        self._write_voltage = 1.0  # Write voltage (V)
        self._pulse_width = 100e-9  # Pulse width (s)
        self._drift_coefficient = 0.01  # Conductance drift per cycle

    def quantize_weights(self, w: Tensor) -> Tensor:
        """Quantize weights to memristor conductance levels (positive, bounded)."""
        # Map from [-1, 1] to [G_min, G_max] conductance range
        w = torch.clamp(w, -1.0, 1.0)
        # Convert to conductance: G = G_min + (G_max - G_min) * (w + 1) / 2
        g_min = 1.0 / self._roff
        g_max = 1.0 / self._ron
        conductance = g_min + (g_max - g_min) * (w + 1.0) / 2.0
        # Quantize to discrete levels (e.g., 8-bit)
        scale = (g_max - g_min) / 255.0
        return ((conductance - g_min) / scale).round().clamp(0, 255) * scale + g_min

    def get_forward_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        """Crossbar forward operator: I = V * G with IR-drop approximation."""

        def crossbar_forward(x: Tensor, w: Tensor) -> Tensor:
            # x: input voltages (batch_size, n_inputs)
            # w: conductance matrix (n_outputs, n_inputs)
            # Output: currents (batch_size, n_outputs)
            # Simplified IR-drop: voltage drop along columns

            x = self._to_precision(x)
            w = self._to_precision(w)

            # Compute ideal currents
            currents = x @ w.T  # (batch, n_outputs)

            # IR-drop approximation: voltage drop proportional to column current
            # This is a simplified model - real IR-drop requires solving linear system
            column_currents = currents.abs().sum(dim=0)  # Total current per column
            ir_drop = column_currents * self._wire_resistance  # Voltage drop per column
            # Effective voltage at each input is reduced by IR drop
            effective_voltage = self._read_voltage - ir_drop.unsqueeze(0)
            effective_voltage = torch.clamp(effective_voltage, min=0.0)
            # Scale currents by effective voltage
            currents = currents * (effective_voltage / self._read_voltage)

            return self._to_precision(currents)

        return crossbar_forward

    def get_weight_update_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        """Pulse-based memristor weight update operator."""

        def memristor_update(pseudo_grad: Tensor, current_w: Tensor) -> Tensor:
            # Pulse-based update: conductance change proportional to pulse amplitude/duration
            # This simulates the stochastic nature of memristor switching
            update_magnitude = pseudo_grad.abs().mean()
            # Probabilistic switching based on voltage
            switch_prob = torch.sigmoid(update_magnitude * 10.0 - 5.0)
            mask = torch.rand_like(pseudo_grad) < switch_prob
            # Conductance increases for positive pulses, decreases for negative
            delta = mask * pseudo_grad.sign() * self._pulse_width * 0.01
            new_conductance = current_w + delta
            # Clamp to physical bounds
            g_min = 1.0 / self._roff
            g_max = 1.0 / self._ron
            return new_conductance.clamp(g_min, g_max)

        return memristor_update


class NeuromorphicSubstrate(DigitalSubstrate):
    """Event-driven neuromorphic: sparse spikes, asynchronous.

    Models event-driven neuromorphic hardware (Loihi, TrueNorth, SpiNNaker):
    - Sparse spike-based communication (AER - Address Event Representation)
    - Leaky integrate-and-fire neuron dynamics
    - Synaptic delay queues
    - Asynchronous event processing
    - Spike-timing-dependent plasticity (STDP) support
    """

    def __init__(self, config: SubstrateConfig | None = None):
        super().__init__(config or SubstrateConfig.neuromorphic())
        # Neuromorphic parameters
        self._tau_mem = 20.0  # Membrane time constant (ms)
        self._tau_syn = 5.0  # Synaptic time constant (ms)
        self._v_thresh = 1.0  # Spike threshold
        self._v_reset = 0.0  # Reset potential
        self._refractory = 2.0  # Refractory period (ms)
        self._dt = 1.0  # Simulation timestep (ms)

    def inject_state_noise(self, s: Tensor) -> Tensor:
        """Spike dropout and thermal noise."""
        s = super().inject_state_noise(s)
        # Spike dropout (sparse activity)
        spike_mask = torch.rand_like(s) > self.config.sparsity
        # Thermal noise
        thermal_noise = torch.randn_like(s) * self.config.noise_level
        return self._to_precision(s * spike_mask.float() + thermal_noise)

    def get_forward_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        """Neuromorphic forward operator: spike-based convolution."""

        def neuromorphic_forward(x: Tensor, w: Tensor) -> Tensor:
            # x: spike trains (batch, time, n_inputs) or rates (batch, n_inputs)
            # w: synaptic weights (n_outputs, n_inputs)
            # For rate-coded inputs, use weighted sum with synaptic filtering
            x = self._to_precision(x)
            w = self._to_precision(w)
            if x.ndim == 3:
                # Spike train input: convolve with synaptic kernel
                # Simplified: average firing rate over time window
                rates = x.mean(dim=1)  # (batch, n_inputs)
            else:
                rates = x
            return self._to_precision(rates @ w.T)

        return neuromorphic_forward

    def get_weight_update_operator(self) -> Callable[[Tensor, Tensor], Tensor]:  # ruff: ignore[no-self-use]
        """STDP-based weight update operator."""

        def stdp_update(pseudo_grad: Tensor, current_w: Tensor) -> Tensor:
            # STDP: pre-before-post -> LTP, post-before-pre -> LTD
            # Simplified: use pseudo-gradient sign as correlation proxy
            ltp_mask = pseudo_grad > 0
            ltd_mask = pseudo_grad < 0
            delta = torch.zeros_like(current_w)
            delta[ltp_mask] += 0.01  # Potentiation
            delta[ltd_mask] -= 0.01  # Depression
            # Weight dependence (soft bounds)
            delta *= 1.0 - current_w.abs()
            return current_w + delta

        return stdp_update


class OpticalSubstrate(DigitalSubstrate):
    """Photonic substrate: phase/amplitude encoding, coherent interference.

    Models photonic neural networks (MZI meshes, coherent Ising machines):
    - Phase and amplitude encoding of information
    - Coherent interference for matrix multiplication
    - Thermal crosstalk and phase noise
    - MZI (Mach-Zehnder Interferometer) mesh calibration
    - Optical non-linearities (optional)
    """

    def __init__(self, config: SubstrateConfig | None = None):
        super().__init__(config or SubstrateConfig.optical())
        # Photonic parameters
        self._wavelength = 1550e-9  # Wavelength (m)
        self._phase_noise_std = 0.01  # Phase noise (rad)
        self._thermal_crosstalk = 0.02  # Thermal crosstalk coefficient
        self._insertion_loss = 0.1  # dB per MZI
        self._phase_shifter_range = 2 * torch.pi  # Full 2π range

    def get_forward_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        """Photonic forward operator: coherent interference in MZI mesh."""

        def photonic_forward(x: Tensor, w: Tensor) -> Tensor:
            # x: input field amplitudes (batch, n_inputs) - complex
            # w: MZI mesh parameters (phases) (n_outputs, n_inputs, 2) - internal, external
            # For real-valued weights, interpret as phase shifts
            # MZI transfer matrix: cos(phi/2) for bar, i*sin(phi/2) for cross
            n_outputs = w.shape[0]

            x = self._to_precision(x)
            w = self._to_precision(w)

            # Convert real weights to phases (assuming [-1,1] -> [-π, π])
            phases = w * torch.pi

            # Add phase noise
            phase_noise = torch.randn_like(phases) * self._phase_noise_std
            noisy_phases = phases + phase_noise

            # MZI mesh matrix multiplication (simplified)
            # Each MZI implements: [cos(φ/2)  i*sin(φ/2); i*sin(φ/2)  cos(φ/2)]
            # For real inputs/outputs, we use the real part of the transfer
            # This is a simplified rectangular mesh model
            cos_half = torch.cos(noisy_phases / 2)
            sin_half = torch.sin(noisy_phases / 2)

            # Apply insertion loss
            loss_factor = 10 ** (-self._insertion_loss / 20)
            cos_half *= loss_factor
            sin_half *= loss_factor

            # Add thermal crosstalk (correlated noise across nearby MZIs)
            if self._thermal_crosstalk > 0:
                crosstalk = (
                    torch.randn(n_outputs, 1, device=x.device) * self._thermal_crosstalk
                )
                cos_half += crosstalk
                sin_half += crosstalk

            # Build effective weight matrix
            # For simplicity, use cos(φ) as effective real weight
            effective_w = cos_half

            return self._to_precision(x @ effective_w.T)

        return photonic_forward

    def get_weight_update_operator(self) -> Callable[[Tensor, Tensor], Tensor]:  # ruff: ignore[no-self-use]
        """Phase shifter based weight update."""

        def phase_update(pseudo_grad: Tensor, current_w: Tensor) -> Tensor:
            # Update phases based on gradient
            # Map gradient to phase shift (wrapped to [-π, π])
            phase_shift = pseudo_grad * 0.1
            new_phases = (current_w * torch.pi + phase_shift) % (
                2 * torch.pi
            ) - torch.pi
            return new_phases / torch.pi  # Back to [-1, 1]

        return phase_update


class QuantumSubstrate(DigitalSubstrate):
    """Quantum substrate: parameterized unitary gates.

    Models quantum neural networks (variational quantum circuits):
    - Parameterized unitary gates (RY, RZ, CNOT layers)
    - Noise channels (depolarizing, amplitude damping)
    - Barren plateau mitigation strategies
    - Measurement-based readout
    """

    def __init__(self, config: SubstrateConfig | None = None):
        super().__init__(config or SubstrateConfig.quantum())
        # Quantum parameters
        self._n_qubits = 8
        self._depolarizing_prob = 0.01
        self._amplitude_damping_prob = 0.005
        self._n_layers = 3

    def get_forward_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        """Quantum circuit evaluation (simplified classical simulation)."""

        def quantum_forward(x: Tensor, w: Tensor) -> Tensor:
            # x: classical input (batch, n_features)
            # w: circuit parameters or weight matrix
            # Simplified: if w is 2D, treat as weight matrix; if 1D, treat as circuit params

            x = self._to_precision(x)
            w = self._to_precision(w)

            if w.ndim == 2:
                # Treat as weight matrix: x @ w.T
                out = x @ w.T
            else:
                # Amplitude encoding: normalize and embed in quantum state
                x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)

                # Simplified variational circuit: alternating RY/RZ rotations + entanglement
                n_params = w.shape[0]
                state = x_norm  # Initial state

                # Layer of RY rotations
                for i in range(min(self._n_qubits, n_params)):
                    if i < state.shape[1]:
                        angle = w[i]
                        state[:, i] = state[:, i] * torch.cos(angle / 2)  # Simplified

                # Entanglement layer (CNOTs)
                # Simplified: pairwise mixing
                for i in range(0, min(self._n_qubits - 1, state.shape[1] - 1), 2):
                    if i + 1 < state.shape[1]:
                        state[:, i], state[:, i + 1] = state[:, i + 1], state[:, i]

                # Measurement: Z expectation values
                out = (state**2).sum(dim=-1, keepdim=True)

            # Add quantum noise
            noise = torch.randn_like(out) * self.config.noise_level
            return self._to_precision(out + noise)

        return quantum_forward

    def get_weight_update_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        """Parameter shift rule for quantum gradients.

        Classical simulation of parameter-shift rule for 1-qubit rotation.
        For each parameter θ: ∇f(θ) ≈ [f(θ+π/2) - f(θ-π/2)] / 2
        pseudo_grad is the "target direction"; we return the parameter-shift estimate.
        """

        def parameter_shift_update(pseudo_grad: Tensor, current_w: Tensor) -> Tensor:
            # Classical simulation of parameter-shift rule for 1-qubit rotation
            # For each parameter θ: ∇f(θ) ≈ [f(θ+π/2) - f(θ-π/2)] / 2
            # pseudo_grad is the "target direction"; we return the parameter-shift estimate
            # Simplified: assume current_w encodes rotation angles; shift each by ±π/2
            shifted_plus = self._evaluate_circuit(current_w + math.pi / 2)
            shifted_minus = self._evaluate_circuit(current_w - math.pi / 2)
            param_shift_grad = (shifted_plus - shifted_minus) / 2
            # Use a default step size since SubstrateConfig doesn't have one
            step_size = getattr(self.config, "step_size", 0.01)
            return current_w - step_size * param_shift_grad

        return parameter_shift_update

    def _evaluate_circuit(self, params: Tensor) -> Tensor:
        """Classical simulation of parameterized quantum circuit.

        Minimal: 1 qubit, RY(θ), measure <Z>
        """

        # <Z> = cos(θ) for RY(θ)|0>
        return torch.cos(params)


# Additional substrate implementations (kept for backward compatibility)
class QuantizedSubstrate(DigitalSubstrate):
    def __init__(self, config: SubstrateConfig | None = None):
        super().__init__(config or SubstrateConfig.digital(precision="int8"))

    def quantize_weights(self, w: Tensor) -> Tensor:  # ruff: ignore[no-self-use]
        scale = w.abs().max() / 127
        return (w / scale).round().clamp(-128, 127) * scale


class NoisySubstrate(DigitalSubstrate):
    def __init__(self, config: SubstrateConfig | None = None):
        super().__init__(config or SubstrateConfig.digital(noise_level=0.05))

    def inject_state_noise(self, s: Tensor) -> Tensor:
        noise = torch.randn_like(s) * self.config.noise_level
        return s + noise


# Additional dynamics implementations
class EnergyMinimizationDynamics:
    """Energy-based settling (EqProp, Hopfield, CHL).

    Supports heavy-ball momentum for accelerated convergence.
    Implements the contrastive nudged phase: output layer receives
    beta * (target - output) during the nudged phase.

    Gradient checkpointing trades compute for memory by recomputing
    intermediate activations during backward pass.

    Free energy tracking enables Control-Lyapunov analysis of the
    thermodynamic contrast between free and nudged phases.
    """

    def __init__(self, config: StateDynamicsConfig | None = None):
        self.config = config or StateDynamicsConfig.energy_minimization()
        self._velocity: list[Tensor] | None = None
        self._free_energy_history: list[float] | None = None

    def _compute_energy(self, all_acts: list[Tensor], geometry: Geometry) -> float:
        """Compute Hopfield energy for the current state.
        
        E = 0.5 * sum(h_i^2) - sum_{i,j} W_{ij} h_i h_j - sum_i b_i h_i
        For ReLU networks with symmetric weights approximation.
        """
        if not all_acts or len(all_acts) < 2:
            return 0.0

        # Use hidden + output layers (skip input)
        acts = all_acts[1:]  # [hidden1, hidden2, ..., output]
        device = acts[0].device
        total_energy = torch.tensor(0.0, device=device)

        # Extract weight matrices from geometry params
        # Params are named like "0.weight", "2.weight", etc. (layer indices)
        # Exclude recurrent_weight which is not a feedforward weight
        params = geometry.params
        weight_names = [
            n for n in params
            if "weight" in n and params[n].ndim == 2 and not n.startswith("recurrent")
        ]
        # Sort by layer index (e.g., "0.weight" -> 0, "2.weight" -> 2)
        weight_names.sort(key=lambda x: int(x.split("_")[1]) if "_" in x and x.split("_")[1].isdigit() else 0)
        
        if not weight_names:
            return 0.0

        num_hidden = len(acts) - 1  # Number of hidden layers (excluding output)
        num_ff_weights = len(weight_names)  # Number of feedforward weight matrices

        # For each hidden/output layer, compute energy contribution
        # E = 0.5 * ||h||^2 - h^T * W * h_prev (for each layer)
        for i in range(len(acts)):
            h = acts[i]
            # 0.5 * ||h||^2
            total_energy = total_energy + 0.5 * (h**2).sum()

        # Subtract interaction terms: h^T * W * h_prev
        for i in range(len(acts)):
            h = acts[i]
            if i < num_hidden:
                # Hidden layer i: weight index i (0, 1, ..., num_hidden-1)
                weight_idx = i
                h_prev = acts[i - 1] if i > 0 else all_acts[0]
            else:
                # Output layer: last feedforward weight (hidden -> output)
                weight_idx = num_ff_weights - 1
                h_prev = acts[i - 1]  # Last hidden layer
            
            if weight_idx < num_ff_weights:
                W = params[weight_names[weight_idx]]
                # h^T * W * h_prev -> sum over batch, then mean
                # h: (batch, dim_i), W: (dim_i, dim_{i-1}), h_prev: (batch, dim_{i-1})
                # h @ W: (batch, dim_{i-1}) @ h_prev.T: (dim_{i-1}, batch) -> (batch, batch)
                interaction = (h @ W @ h_prev.T).trace()
                total_energy = total_energy - interaction

        # Subtract bias terms (exclude recurrent bias if any)
        bias_names = [
            n for n in params
            if "bias" in n and params[n].ndim == 1 and not n.startswith("recurrent")
        ]
        bias_names.sort(key=lambda x: int(x.split("_")[1]) if "_" in x and x.split("_")[1].isdigit() else 0)
        num_ff_biases = len(bias_names)
        for i in range(len(acts)):
            h = acts[i]
            if i < num_hidden:
                bias_idx = i
            else:
                bias_idx = num_ff_biases - 1
            if bias_idx < num_ff_biases:
                b = params[bias_names[bias_idx]]
                total_energy = total_energy - (h @ b).sum()

        # Return mean per sample
        batch_size = acts[0].size(0)
        return (total_energy / batch_size).item()

    def _settle_step(
        self,
        all_acts: list[Tensor],
        geometry: Geometry,
        beta: float,
        target: Tensor | None,
        velocity: list[Tensor] | None,
        momentum: float,
    ) -> tuple[list[Tensor], list[Tensor] | None]:
        """One settling step, pure function for gradient checkpointing."""
        num_hidden = len(all_acts) - 2
        new_acts = [all_acts[0]]  # Input layer is fixed
        new_velocity = None

        if momentum > 0 and velocity is not None:
            new_velocity = []

        # Update each hidden layer
        for i in range(num_hidden):
            layer = geometry._layers[i * 2]  # Linear layer at even indices
            pre = layer(all_acts[i])  # Bottom-up from previous layer

            # Recurrent term (for RecurrentGeometry, last hidden layer only)
            if hasattr(geometry, '_recurrent_weight') and geometry._recurrent_weight is not None:
                if i == num_hidden - 1:  # Last hidden layer has recurrent connection
                    pre = pre + all_acts[i + 1] @ geometry._recurrent_weight.T

            # Top-down drive from layer above
            next_layer = geometry._layers[(i + 1) * 2]  # Next Linear layer
            top_down = all_acts[i + 2] @ next_layer.weight

            total = pre + top_down

            # Apply momentum (heavy-ball)
            if momentum > 0 and velocity is not None:
                total = momentum * velocity[i] + total
                new_velocity.append(total.detach().clone())

            # Apply activation (ReLU)
            if i < len(geometry._layers) - 2 and isinstance(geometry._layers[i * 2 + 1], nn.Module):
                act_fn = geometry._layers[i * 2 + 1]
                h_new = act_fn(total)
            else:
                h_new = total

            new_acts.append(h_new)

        # Output layer
        out_layer = geometry._layers[-1] if isinstance(geometry._layers[-1], nn.Linear) else None
        if out_layer is not None:
            out = out_layer(new_acts[-1])
            # Apply nudging to output layer during nudged phase
            if beta > 0 and target is not None:
                # Convert target to one-hot if needed
                if target.dim() == 1:
                    target_oh = torch.zeros_like(out)
                    target_oh.scatter_(1, target.unsqueeze(1), 1.0)
                else:
                    target_oh = target
                out = out + beta * (target_oh - out)

            new_acts.append(out)
        else:
            new_acts.append(all_acts[-1])

        return new_acts, new_velocity

    def settle(
        self,
        state: SystemState,
        geometry: Geometry,
        substrate: Substrate,
        target: Tensor | None = None,
    ) -> SystemState:
        # Run settling iterations for full multi-layer EqProp dynamics
        # Implements the same dynamics as legacy EquilibriumMLP.forward_dynamics

        if state.x is None:
            return state

        # Get initial activations with intermediates for credit assignment
        all_acts = geometry.forward_with_intermediates(state.x, substrate)
        if not all_acts:
            return state

        # Number of hidden layers (excluding input and output)
        # all_acts = [input, hidden1, hidden2, ..., output]
        num_hidden = len(all_acts) - 2

        # Initialize velocity for momentum (per hidden layer)
        if self.config.momentum > 0:
            batch_size = all_acts[0].size(0)
            self._velocity = [
                torch.zeros_like(all_acts[i + 1]) for i in range(num_hidden)
            ]
        else:
            self._velocity = None

        # Initialize free energy history if tracking enabled
        if self.config.track_free_energy_per_iter:
            self._free_energy_history = []
            # Track initial energy
            self._free_energy_history.append(self._compute_energy(all_acts, geometry))
        else:
            self._free_energy_history = None

        beta = self.config.beta if target is not None else 0.0
        momentum = self.config.momentum

        # Use gradient checkpointing if enabled
        if self.config.gradient_checkpointing:
            import torch.utils.checkpoint as checkpoint

            for step in range(self.config.max_steps):
                # Checkpoint the step function
                all_acts, self._velocity = checkpoint.checkpoint(
                    self._settle_step,
                    all_acts,
                    geometry,
                    beta,
                    target,
                    self._velocity,
                    momentum,
                    use_reentrant=False,
                )

                # Track free energy if enabled
                if self._free_energy_history is not None:
                    self._free_energy_history.append(self._compute_energy(all_acts, geometry))

                # Check convergence (can't checkpoint this as it's not differentiable)
                if step >= self.config.convergence_start:
                    delta = torch.dist(all_acts[-1], all_acts[-1], p=float("inf")).item()
                    if delta < self.config.convergence_threshold:
                        break
        else:
            # Original non-checkpointed path
            for step in range(self.config.max_steps):
                new_acts = [all_acts[0]]  # Input layer is fixed
                new_velocity = None

                if momentum > 0 and self._velocity is not None:
                    new_velocity = []

                # Update each hidden layer
                for i in range(num_hidden):
                    layer = geometry._layers[i * 2]  # Linear layer at even indices
                    pre = layer(all_acts[i])  # Bottom-up from previous layer

                    # Recurrent term (for RecurrentGeometry, last hidden layer only)
                    if hasattr(geometry, '_recurrent_weight') and geometry._recurrent_weight is not None:
                        if i == num_hidden - 1:  # Last hidden layer has recurrent connection
                            pre = pre + all_acts[i + 1] @ geometry._recurrent_weight.T

                    # Top-down drive from layer above
                    next_layer = geometry._layers[(i + 1) * 2]  # Next Linear layer
                    top_down = all_acts[i + 2] @ next_layer.weight

                    total = pre + top_down

                    # Apply momentum (heavy-ball)
                    if momentum > 0 and self._velocity is not None:
                        total = momentum * self._velocity[i] + total
                        new_velocity.append(total.detach().clone())

                    # Apply activation (ReLU)
                    if i < len(geometry._layers) - 2 and isinstance(geometry._layers[i * 2 + 1], nn.Module):
                        act_fn = geometry._layers[i * 2 + 1]
                        h_new = act_fn(total)
                    else:
                        h_new = total

                    new_acts.append(h_new)

                # Output layer
                out_layer = geometry._layers[-1] if isinstance(geometry._layers[-1], nn.Linear) else None
                if out_layer is not None:
                    out = out_layer(new_acts[-1])
                    # Apply nudging to output layer during nudged phase
                    if beta > 0 and target is not None:
                        # Convert target to one-hot if needed
                        if target.dim() == 1:
                            target_oh = torch.zeros_like(out)
                            target_oh.scatter_(1, target.unsqueeze(1), 1.0)
                        else:
                            target_oh = target
                        out = out + beta * (target_oh - out)

                    new_acts.append(out)
                else:
                    new_acts.append(all_acts[-1])

                # Track free energy if enabled
                if self._free_energy_history is not None:
                    self._free_energy_history.append(self._compute_energy(new_acts, geometry))

                # Check convergence
                if step >= self.config.convergence_start:
                    delta = torch.dist(new_acts[-1], all_acts[-1], p=float("inf")).item()
                    if delta < self.config.convergence_threshold:
                        all_acts = new_acts
                        break
                all_acts = new_acts
                if new_velocity is not None:
                    self._velocity = new_velocity

        if target is None:
            state.free_state = all_acts
        else:
            state.nudged_state = all_acts
        state.activations = all_acts
        return state

    def compute_energy(self, state: SystemState, geometry: Geometry) -> Tensor:
        """Compute free energy (Hopfield energy) for the current state."""
        acts = state.free_state
        if acts is None:
            acts = state.nudged_state
        if acts is None:
            acts = state.activations
        if acts is None:
            return torch.tensor(0.0)
        if isinstance(acts, list):
            # Use hidden + output layers
            energy_val = self._compute_energy(acts, geometry)
            return torch.tensor(energy_val, device=acts[0].device if acts else "cpu")
        return (acts**2).mean()

    def get_free_energy_history(self) -> list[float] | None:
        """Return the free energy history tracked during settling.

        Returns None if tracking was not enabled (track_free_energy_per_iter=False).
        """
        return self._free_energy_history


class PredictiveSettlingDynamics:
    """Predictive Coding: hierarchical prediction error minimization.

    Implements the Predictive Coding (PC) / Rao-Ballard framework:
    - Each layer has representation units (r) and error units (e)
    - Top-down predictions: μ_l = f(W_l * r_{l+1})
    - Bottom-up errors: e_l = r_l - μ_l
    - Dynamics: τ * dr/dt = -r + f(W * r) + W^T * e (precision-weighted)
    - Free energy: F = Σ ||e_l||^2 / (2 * precision_l)

    The settling process minimizes free energy through:
    1. Update error units: e_l = r_l - μ_l (prediction errors)
    2. Update representation units: r_l ← r_l + η * (W^T * e_{l-1} - e_l)
    3. Repeat until convergence
    """

    def __init__(self, config: StateDynamicsConfig | None = None):
        self.config = config or StateDynamicsConfig.predictive_settling()
        self._precision: list[Tensor] | None = (
            None  # Layer-wise precision (inverse variance)
        )
        self._free_energy_history: list[float] | None = None

    def _init_precision(self, n_layers: int, device: torch.device) -> None:
        """Initialize precision parameters for each layer."""
        if self._precision is None:
            self._precision = [torch.ones(1, device=device) for _ in range(n_layers)]

    def _get_layer_activations(
        self, state: SystemState, geometry: Geometry, substrate: Substrate
    ) -> list[Tensor]:
        """Extract per-layer activations from state.

        If state.activations is a single tensor (output only), we compute
        intermediate activations by running a forward pass through the geometry.
        """
        acts = state.activations
        if acts is None:
            return []
        if isinstance(acts, list):
            # If it's already a list from a previous call, assume it's filtered
            return acts
        # Single tensor - compute intermediate activations using geometry protocol method
        if state.x is not None:
            all_acts = geometry.forward_with_intermediates(state.x, substrate)
            return all_acts
        # Fallback: return single output
        return [acts]

    def _extract_weights(
        self, geometry: Geometry, n_layers: int, device: torch.device
    ) -> list[Tensor]:
        """Extract weight matrices from geometry."""
        weights = []
        if hasattr(geometry, "_layers"):
            layers = getattr(geometry, "_layers", None)
            if layers is not None and hasattr(layers, "__iter__"):
                for layer in layers:  # type: ignore[attr-defined]
                    if isinstance(layer, torch.nn.Linear):
                        weights.append(layer.weight)
        elif hasattr(geometry, "params"):
            params = geometry.params
            weights = [
                p for name, p in params.items() if "weight" in name and p.ndim == 2
            ]

        if len(weights) != n_layers - 1:
            # Fallback: create identity-like weights
            weights = [
                torch.eye(20, 20, device=device)  # Default size
                for _ in range(n_layers - 1)
            ]
        return weights

    def settle(
        self,
        state: SystemState,
        geometry: Geometry,
        substrate: Substrate,
        target: Tensor | None = None,
    ) -> SystemState:
        """Settle the network to minimize free energy (predictive coding)."""
        # Get current layer activations
        layer_acts = self._get_layer_activations(state, geometry, substrate)
        if not layer_acts:
            return state

        n_layers = len(layer_acts)
        device = layer_acts[0].device
        self._init_precision(n_layers, device)

        # Extract weight matrices from geometry
        weights = self._extract_weights(geometry, n_layers, device)

        if len(weights) != n_layers - 1:
            # Fallback for missing weights - create compatible shapes
            weights = []
            for i in range(n_layers - 1):
                out_dim = layer_acts[i + 1].shape[-1]
                in_dim = layer_acts[i].shape[-1]
                weights.append(torch.eye(out_dim, in_dim, device=device))

        # Initialize free energy tracking if enabled
        if self.config.track_free_energy_per_iter:
            self._free_energy_history = []
        else:
            self._free_energy_history = None

        # Predictive coding settling iterations
        for step in range(self.config.max_steps):
            # Store previous activations for convergence check
            prev_acts = [a.clone() for a in layer_acts]

            # Compute top-down predictions and bottom-up errors
            # e_l = r_l - g(W_l * r_{l+1}) for l=0..n-2
            # e_{n-1} = target - r_{n-1} (nudged) or -r_{n-1} (free)
            errors = []
            for l in range(n_layers - 1):
                # W_l has shape (d_{l+1}, d_l) for forward pass r_l -> r_{l+1}
                # For prediction r_l <- r_{l+1}, we need W_l^T: (d_l, d_{l+1})
                # But linear() expects weight (out, in), so we use W_l.T for (d_l, d_{l+1})
                pred = torch.nn.functional.linear(layer_acts[l + 1], weights[l].T)
                pred = torch.nn.functional.relu(pred)
                error = layer_acts[l] - pred
                errors.append(error)

            # Output layer error
            if target is not None:
                # Convert target to one-hot if needed
                if target.dim() == 1:
                    # Class indices -> one-hot
                    target_onehot = torch.nn.functional.one_hot(
                        target, num_classes=layer_acts[-1].shape[-1]
                    ).float()
                else:
                    target_onehot = target
                errors.append(target_onehot - layer_acts[-1])
            else:
                errors.append(-layer_acts[-1])  # Free phase: minimize output activity

            # Update representation units
            # r_l ← r_l + η * (W_l^T * e_{l+1} - e_l) for l=1..n-2 (hidden layers only)
            # r_{n-1} ← r_{n-1} + η * (-e_{n-1}) (output layer)
            # Input layer (l=0) is clamped to data and NOT updated
            new_acts = [layer_acts[0]]  # Keep input layer unchanged
            for l in range(1, n_layers):
                if l == n_layers - 1:
                    # Output layer: only local error
                    update = -errors[-1]
                else:
                    # Hidden layers: W_l^T * e_{l+1} - e_l
                    # weights[l] is (d_{l+1}, d_l), so weights[l].T is (d_l, d_{l+1})
                    # errors[l+1] is (batch, d_{l+1})
                    # We need: (d_l, d_{l+1}) @ (d_{l+1}, batch) -> (d_l, batch) -> (batch, d_l)
                    top_down = (weights[l].T @ errors[l + 1].T).T
                    update = top_down - errors[l]

                if self._precision is not None and l < len(self._precision):
                    update = update * self._precision[l]

                new_act = layer_acts[l] + self.config.step_size * update
                new_acts.append(new_act)

            layer_acts = new_acts
            layer_acts = [substrate.inject_state_noise(a) for a in layer_acts]

            # Track free energy if enabled
            if (
                self.config.track_free_energy_per_iter
                and self._free_energy_history is not None
            ):
                # Compute free energy at this step
                total_energy = torch.tensor(0.0, device=device)
                for l in range(n_layers - 1):
                    pred = torch.nn.functional.linear(layer_acts[l + 1], weights[l].T)
                    pred = torch.nn.functional.relu(pred)
                    error = layer_acts[l] - pred
                    precision = (
                        self._precision[l]
                        if self._precision is not None and l < len(self._precision)
                        else torch.ones(1, device=device)
                    )
                    precision_scalar = precision.squeeze()
                    total_energy = total_energy + (error**2).sum() / (
                        2 * precision_scalar
                    )
                self._free_energy_history.append(total_energy.item())

            # Check convergence
            if step >= self.config.convergence_start:
                max_delta = max(
                    torch.dist(new, old, p=float("inf")).item()
                    for new, old in zip(layer_acts, prev_acts)
                )
                if max_delta < self.config.convergence_threshold:
                    break

        # Update state
        if target is None:
            state.free_state = layer_acts
        else:
            state.nudged_state = layer_acts
        state.activations = layer_acts
        return state

    def compute_energy(self, state: SystemState, geometry: Geometry) -> Tensor:
        """Compute free energy (variational bound on negative log likelihood)."""
        # Need substrate reference for intermediate activations
        if hasattr(self, "substrate_ref"):
            pass  # Already set by settle
        layer_acts = (
            state.free_state if state.free_state is not None else state.activations
        )
        if not layer_acts or not isinstance(layer_acts, list):
            # state.activations could be Tensor or list[Tensor], get device from first tensor
            if state.activations is not None:
                if isinstance(state.activations, list):
                    device = state.activations[0].device if state.activations else "cpu"
                else:
                    device = state.activations.device
            else:
                device = "cpu"
            return torch.tensor(0.0, device=device)
        if len(layer_acts) == 0:
            return torch.tensor(0.0, device="cpu")

        # At this point layer_acts is guaranteed to be list[Tensor] with at least one element
        acts: list[Tensor] = layer_acts
        n_layers = len(acts)
        device = acts[0].device
        self._init_precision(n_layers, device)

        weights = self._extract_weights(geometry, n_layers, device)
        if len(weights) != n_layers - 1:
            weights = [
                torch.eye(
                    layer_acts[i + 1].shape[-1], layer_acts[i].shape[-1], device=device
                )
                for i in range(n_layers - 1)
            ]

        total_energy = torch.tensor(0.0, device=device)
        for l in range(n_layers - 1):
            pred = torch.nn.functional.linear(layer_acts[l + 1], weights[l].T)
            pred = torch.nn.functional.relu(pred)
            error = layer_acts[l] - pred
            precision = (
                self._precision[l]
                if self._precision is not None and l < len(self._precision)
                else torch.ones(1, device=device)
            )
            precision_scalar = precision.squeeze()
            total_energy = total_energy + (error**2).sum() / (2 * precision_scalar)

        return total_energy

    def get_free_energy_history(self) -> list[float] | None:
        """Return the free energy history tracked during settling.

        Returns None if tracking was not enabled (track_free_energy_per_iter=False).
        """
        return self._free_energy_history


class SpikeIntegrationDynamics:
    """Spiking dynamics: membrane potential integration and thresholding.

    Implements Leaky Integrate-and-Fire (LIF) neuron dynamics for feedforward
    and recurrent architectures. Supports multi-layer networks by iterating
    through geometry layers.
    """

    def __init__(self, config: StateDynamicsConfig | None = None):
        self.config = config or StateDynamicsConfig.spike_integration()

    def settle(
        self,
        state: SystemState,
        geometry: Geometry,
        substrate: Substrate,  # ruff: ignore[unused-method-argument]
        target: Tensor | None = None,  # ruff: ignore[unused-method-argument]
    ) -> SystemState:
        # Simplified LIF dynamics for feedforward networks
        if state.x is None:
            return state

        # Get initial activations with intermediates
        all_acts = geometry.forward_with_intermediates(state.x, substrate)
        if not all_acts:
            return state

        # all_acts = [input, hidden1, hidden2, ..., output]
        # We'll simulate spiking dynamics on hidden layers
        num_layers = len(all_acts) - 1  # Exclude input
        
        spike_counts_per_layer = []
        
        for step in range(self.config.max_steps):
            # Forward pass with spiking at each layer
            h = all_acts[0]  # Input layer (no spiking)
            new_acts = [h]
            
            for i in range(num_layers):
                # Get weight matrix for this layer
                if hasattr(geometry, '_layers'):
                    layer_idx = i * 2
                    if layer_idx < len(geometry._layers) and isinstance(geometry._layers[layer_idx], nn.Linear):
                        layer = geometry._layers[layer_idx]
                        h = layer(h)
                    else:
                        # Fallback: use route
                        h = geometry.route(h)
                else:
                    h = geometry.route(h)
                
                # Apply activation if present
                if hasattr(geometry, '_layers'):
                    act_idx = i * 2 + 1
                    if act_idx < len(geometry._layers):
                        act_layer = geometry._layers[act_idx]
                        if isinstance(act_layer, nn.Module) and not isinstance(act_layer, nn.Linear):
                            h = act_layer(h)
                
                # Spike thresholding (LIF)
                spikes = (h > 1.0).float()
                spike_counts_per_layer.append(spikes.sum(dim=0))  # Per neuron spike count
                h = torch.where(h > 1.0, torch.zeros_like(h), h)  # Reset after spike
                
                new_acts.append(h)
            
            # Check convergence on output layer
            if step >= self.config.convergence_start:
                delta = torch.dist(new_acts[-1], all_acts[-1], p=float("inf")).item()
                if delta < self.config.convergence_threshold:
                    all_acts = new_acts
                    break
            
            all_acts = new_acts

        state.activations = all_acts
        state.spike_counts = spike_counts_per_layer
        if state.metrics is not None:
            state.metrics["spike_counts"] = spike_counts_per_layer
        return state

    def compute_energy(self, state: SystemState, geometry: Geometry) -> Tensor:  # ruff: ignore[no-self-use]
        return torch.tensor(0.0)


class LazyStateDynamics:
    """Lazy per-step state dynamics (Lazy EqProp variant).

    Computes activations on-demand during settling, deferring computation
    until the energy/contrastive step requires them. This implements the
    "lazy" evaluation strategy from LazyEqProp where intermediate states
    are materialized only when needed for the pseudo-gradient computation.

    The dynamics mimics EquilibriumMLP.forward_dynamics but with lazy
    activation caching — useful for memory-constrained substrates.
    """

    def __init__(self, config: StateDynamicsConfig | None = None):
        self.config = config or StateDynamicsConfig.energy_minimization()
        self._activation_cache: dict[int, Tensor] = {}

    def settle(
        self,
        state: SystemState,
        geometry: Geometry,
        substrate: Substrate,
        target: Tensor | None = None,
    ) -> SystemState:
        """Settle with lazy activation computation."""
        h = state.activations
        if h is None:
            return state
        if isinstance(h, list):
            h = h[-1]  # Use last layer for single-tensor routing

        # Lazy evaluation: only compute when actually settling
        for step in range(self.config.max_steps):
            h_new = geometry.route(h)
            h_new = substrate.inject_state_noise(h_new)

            # Cache activations lazily at convergence check points
            if step >= self.config.convergence_start:
                self._activation_cache[step] = h_new.clone()

            if step >= self.config.convergence_start:
                delta = torch.dist(h_new, h, p=float("inf")).item()
                if delta < self.config.convergence_threshold:
                    h = h_new
                    break
            h = h_new

        if target is None:
            state.free_state = h
        else:
            state.nudged_state = h
        state.activations = h
        return state

    def compute_energy(self, state: SystemState, geometry: Geometry) -> Tensor:
        """Compute energy using cached activations if available."""
        acts = state.free_state if state.free_state is not None else state.activations
        if acts is None:
            return torch.tensor(0.0)
        if isinstance(acts, list):
            acts = acts[-1]
        return (acts**2).mean()

    def get_cached_activations(self) -> dict[int, Tensor]:
        """Return lazily cached activations for inspection."""
        return self._activation_cache

    def clear_cache(self) -> None:
        """Clear the lazy activation cache."""
        self._activation_cache.clear()


class DiffusionDynamics:
    """Diffusion-based state dynamics for continuous-time settling.

    Implements the stochastic differential equation:
        dh/dt = -∇E(h) + √(2D) ξ(t)

    where E is the energy function, D is the diffusion coefficient (noise),
    and ξ(t) is white noise. This models Langevin dynamics for sampling
    from the Boltzmann distribution p(h) ∝ exp(-E(h)/D).

    The discrete-time update (Euler-Maruyama):
        h_{t+1} = h_t - step_size * ∇E(h_t) + √(2 * step_size * D) * N(0, I)

    For energy-based models, ∇E(h) ≈ h - f(h) where f is the geometry route.
    """

    def __init__(self, config: StateDynamicsConfig | None = None):
        self.config = config or StateDynamicsConfig.diffusion()
        self._diffusion_coeff = 1.0  # D in the SDE

    def settle(
        self,
        state: SystemState,
        geometry: Geometry,
        substrate: Substrate,
        target: Tensor | None = None,
    ) -> SystemState:
        h = state.activations
        if h is None:
            return state
        if isinstance(h, list):
            h = h[-1]  # Use last layer for single-tensor routing

        step_size = self.config.step_size
        diffusion_coeff = self._diffusion_coeff

        for step in range(self.config.max_steps):
            # Deterministic drift: -∇E(h) ≈ h - geometry.route(h)
            h_drift = geometry.route(h)
            h_drift = substrate.inject_state_noise(h_drift)
            drift = h_drift - h

            # Stochastic diffusion term: √(2 * step_size * D) * N(0, I)
            noise_scale = math.sqrt(2.0 * step_size * diffusion_coeff)
            diffusion = torch.randn_like(h) * noise_scale

            # Euler-Maruyama update
            h_new = h + step_size * drift + diffusion

            if step >= self.config.convergence_start:
                delta = torch.dist(h_new, h, p=float("inf")).item()
                if delta < self.config.convergence_threshold:
                    h = h_new
                    break
            h = h_new

        if target is None:
            state.free_state = h
        else:
            state.nudged_state = h
        state.activations = h
        return state

    def compute_energy(self, state: SystemState, geometry: Geometry) -> Tensor:
        """Compute energy of the current state."""
        acts = state.free_state if state.free_state is not None else state.activations
        if acts is None:
            acts = state.nudged_state
        if acts is None:
            return torch.tensor(0.0)
        if isinstance(acts, list):
            acts = acts[-1]
        # Energy: 0.5 * ||h - f(h)||^2 (fixed point energy)
        with torch.no_grad():
            h_pred = geometry.route(acts)
        return 0.5 * ((acts - h_pred) ** 2).mean()


# Additional credit assignment implementations
class RandomProjectionsCredit:
    """Feedback Alignment / Direct Feedback Alignment.

    Implements Feedback Alignment (FA) and Direct Feedback Alignment (DFA):
    - FA: Fixed random feedback matrices B_l for each layer
    - DFA: Single fixed random matrix from output to all hidden layers

    The pseudo-gradient for layer l is: e_l = B_l @ e_{l+1} * f'(h_l)
    where e_{l+1} is the error from the layer above.
    """

    def __init__(self, config: CreditAssignmentConfig | None = None):
        self.config = config or CreditAssignmentConfig.random_projections()
        self._feedback_weights: dict[str, Tensor] | None = None
        self._is_dfa = (
            config is not None and config.credit_type == "direct_feedback_alignment"
        )

    def _init_feedback_weights(self, geometry: Geometry, device: torch.device) -> None:
        """Initialize fixed random feedback matrices."""
        if self._feedback_weights is not None:
            return

        self._feedback_weights = {}
        hidden_dims = geometry.config.hidden_dims
        out_dim = geometry.config.output_dim

        # Use config parameters for initialization
        scale = self.config.feedback_scale
        use_orthogonal = self.config.orthogonal_init

        if self._is_dfa:
            # DFA: Single feedback matrix from output to all hidden layers
            for i, h_dim in enumerate(hidden_dims):
                if use_orthogonal:
                    # Orthogonal initialization: random matrix then QR decomposition
                    if h_dim >= out_dim:
                        mat = torch.randn(h_dim, out_dim, device=device)
                        q, _ = torch.linalg.qr(mat)
                        self._feedback_weights[f"layer_{i}"] = q[:, :out_dim] * scale
                    else:
                        mat = torch.randn(out_dim, h_dim, device=device)
                        q, _ = torch.linalg.qr(mat)
                        self._feedback_weights[f"layer_{i}"] = q[:h_dim, :].T * scale
                else:
                    self._feedback_weights[f"layer_{i}"] = (
                        torch.randn(h_dim, out_dim, device=device) * scale
                    )
        else:
            # FA: Chain of feedback matrices between adjacent layers
            prev_dim = out_dim
            for i, h_dim in enumerate(reversed(hidden_dims)):
                # Feedback from layer i+1 to layer i
                if use_orthogonal:
                    if h_dim >= prev_dim:
                        mat = torch.randn(h_dim, prev_dim, device=device)
                        q, _ = torch.linalg.qr(mat)
                        self._feedback_weights[f"layer_{len(hidden_dims) - 1 - i}"] = (
                            q[:, :prev_dim] * scale
                        )
                    else:
                        mat = torch.randn(prev_dim, h_dim, device=device)
                        q, _ = torch.linalg.qr(mat)
                        self._feedback_weights[f"layer_{len(hidden_dims) - 1 - i}"] = (
                            q[:h_dim, :].T * scale
                        )
                else:
                    self._feedback_weights[f"layer_{len(hidden_dims) - 1 - i}"] = (
                        torch.randn(h_dim, prev_dim, device=device) * scale
                    )
                prev_dim = h_dim

    def compute_pseudo_gradient(
        self,
        free_state: SystemState,
        nudged_state: SystemState,
        loss: Tensor,
        geometry: Geometry,
    ) -> list[Tensor]:
        """Compute pseudo-gradients using fixed random feedback matrices.

        For FA/DFA, we compute the output error and propagate it backward
        through fixed random matrices instead of the weight transposes.
        """
        device = next(iter(geometry.params.values())).device
        self._init_feedback_weights(geometry, device)

        # Type narrowing: _feedback_weights is now initialized
        assert self._feedback_weights is not None
        feedback_weights = self._feedback_weights

        # Get layer activations from nudged state
        acts = nudged_state.activations
        if acts is None:
            return []

        if not isinstance(acts, list):
            # Single layer - no hidden layers to propagate to
            return []

        # Type narrowing: acts is now list[Tensor]
        acts_list: list[Tensor] = acts

        # Compute output error (dL/dy)
        # For classification with cross-entropy: output_error = softmax(logits) - one_hot(y)
        logits = acts_list[-1]
        if logits.dim() == 2 and logits.shape[-1] > 1:
            # Multi-class classification
            probs = torch.softmax(logits, dim=-1)
            y = free_state.y
            if y is not None:
                target = torch.zeros_like(probs)
                target.scatter_(-1, y.unsqueeze(-1), 1.0)
                output_error = probs - target
            else:
                output_error = probs
        else:
            # Binary or regression
            y_val = free_state.y
            if y_val is not None:
                output_error = torch.sigmoid(logits) - y_val.float().unsqueeze(-1)
            else:
                output_error = torch.sigmoid(logits)

        # Backward pass through fixed feedback matrices
        grads = []
        hidden_dims = geometry.config.hidden_dims

        if self._is_dfa:
            # DFA: Direct feedback from output to all hidden layers
            for i in range(len(hidden_dims)):
                fb = feedback_weights[f"layer_{i}"]
                # e_l = B_l @ e_out
                layer_error = output_error @ fb.T  # (batch, h_dim)
                # Apply derivative of activation (assuming ReLU)
                # Hidden layer i corresponds to acts_list[i + 1] (input is acts_list[0])
                act = acts_list[i + 1]
                layer_error = layer_error * (act > 0).float()  # type: ignore[attr-defined]  # ReLU derivative
                # Pseudo-gradient: layer_error.T @ act_{l-1}
                if i == 0:
                    # First hidden layer: input is x
                    pre_act = free_state.x if free_state.x is not None else acts_list[0]
                else:
                    pre_act = acts_list[i]
                if pre_act is not None:
                    grad = layer_error.T @ pre_act  # (h_dim, in_dim)
                    grads.append(grad.T)  # (out_dim, in_dim) to match weight shape
        else:
            # FA: Layer-wise feedback
            error = output_error
            for i in reversed(range(len(hidden_dims))):
                fb = feedback_weights[f"layer_{i}"]
                # e_l = B_l @ e_{l+1}
                layer_error = error @ fb.T  # (batch, h_dim)
                # Apply derivative of activation
                # Hidden layer i corresponds to acts_list[i + 1]
                act = acts_list[i + 1]
                layer_error = layer_error * (act > 0).float()  # type: ignore[attr-defined]
                # Pseudo-gradient
                pre_act = (
                    acts_list[i]
                    if i > 0
                    else (free_state.x if free_state.x is not None else acts_list[0])
                )
                if pre_act is not None:
                    grad = layer_error.T @ pre_act
                    grads.insert(0, grad.T)
                error = layer_error  # Propagate to next lower layer

        return grads

    def surrogate_objective(
        self,
        free_state: SystemState,
        nudged_state: SystemState,
        geometry: Geometry,
    ) -> Tensor:
        """Surrogate objective for FA: output error norm."""
        if nudged_state.loss is not None:
            return nudged_state.loss
        return torch.tensor(0.0)


class LocalGoodnessCredit:
    """Forward-Forward / PEPITA: layer-local contrastive objectives."""

    def __init__(self, config: CreditAssignmentConfig | None = None):
        self.config = config or CreditAssignmentConfig.local_goodness()

    def compute_pseudo_gradient(  # ruff: ignore[no-self-use]
        self,
        free_state: SystemState,
        nudged_state: SystemState,
        loss: Tensor,
        geometry: Geometry,
    ) -> list[Tensor]:
        # Layer-local goodness gradients
        grads = []
        acts = free_state.activations
        if isinstance(acts, list):
            for act in acts[1:]:  # Skip input
                # Positive pass: maximize goodness
                # Negative pass: minimize goodness
                pos_grad = act * (1 - torch.sigmoid(act))
                grads.append(pos_grad)
        return grads

    def surrogate_objective(
        self,
        free_state: SystemState,
        nudged_state: SystemState,
        geometry: Geometry,
    ) -> Tensor:
        """Layer-local goodness: sum of σ(h)^2 for positive pass, minimize for negative."""
        acts = free_state.activations
        if isinstance(acts, list):
            total_goodness = sum((torch.sigmoid(act) ** 2).sum() for act in acts[1:])
            return total_goodness
        return torch.tensor(0.0)


class BackpropCredit:
    """Standard backpropagation (global credit assignment)."""

    def __init__(self, config: CreditAssignmentConfig | None = None):
        self.config = config or CreditAssignmentConfig.gradient()

    def compute_pseudo_gradient(  # ruff: ignore[no-self-use]
        self,
        free_state: SystemState,
        nudged_state: SystemState,
        loss: Tensor,
        geometry: Geometry,
    ) -> list[Tensor]:
        # Backprop computes true gradients via autograd
        # This is a placeholder - actual implementation uses autograd
        params = list(geometry.params.values())
        if params and loss is not None:
            grads = torch.autograd.grad(
                loss, params, create_graph=False, allow_unused=True
            )
            return [g for g in grads if g is not None]
        return []

    def surrogate_objective(
        self,
        free_state: SystemState,
        nudged_state: SystemState,
        geometry: Geometry,
    ) -> Tensor:
        """Surrogate objective for Backprop: task loss."""
        if nudged_state.loss is not None:
            return nudged_state.loss
        return torch.tensor(0.0)


class TemporalTraceCredit:
    """STDP: spike-timing-dependent correlation.

    Implements spike-timing-dependent plasticity (STDP) for credit assignment:
    - Records pre- and post-synaptic spike times per layer
    - Computes STDP window function for weight updates
    - Causal (pre before post) -> LTP (potentiation)
    - Anti-causal (post before pre) -> LTD (depression)
    """

    def __init__(self, config: CreditAssignmentConfig | None = None):
        self.config = config or CreditAssignmentConfig.temporal_trace()
        self._pre_spike_times: dict[int, Tensor] = {}  # layer -> spike times
        self._post_spike_times: dict[int, Tensor] = {}
        # STDP parameters
        self._a_plus = 0.01  # LTP amplitude
        self._a_minus = 0.01  # LTD amplitude
        self._tau_plus = 20.0  # LTP time constant (ms)
        self._tau_minus = 20.0  # LTD time constant (ms)

    def record_spikes(
        self, layer_idx: int, pre_spikes: Tensor, post_spikes: Tensor
    ) -> None:
        """Call from SpikeIntegrationDynamics during settling."""
        self._pre_spike_times[layer_idx] = pre_spikes
        self._post_spike_times[layer_idx] = post_spikes

    def compute_stdp_window(
        self, pre_spikes: Tensor, post_spikes: Tensor, dt: Tensor
    ) -> Tensor:
        """Return weight change per Δt bin. Exponential STDP window.

        Δt = post - pre; causal (Δt>0) => LTP; anti-causal (Δt<0) => LTD
        W(Δt) = A_plus * exp(-Δt/τ_plus) for Δt>0; -A_minus * exp(Δt/τ_minus) for Δt<0
        """
        # pre_spikes: (n_pre, n_spikes_pre), post_spikes: (n_post, n_spikes_post)
        # Compute all pairwise Δt
        delta_t = post_spikes.unsqueeze(1) - pre_spikes.unsqueeze(0)  # (n_post, n_pre)
        # Apply exponential STDP window
        ltp_mask = delta_t > 0
        ltd_mask = delta_t < 0
        window = torch.zeros_like(delta_t)
        window[ltp_mask] = self._a_plus * torch.exp(-delta_t[ltp_mask] / self._tau_plus)
        window[ltd_mask] = -self._a_minus * torch.exp(
            delta_t[ltd_mask] / self._tau_minus
        )
        return window

    def compute_pseudo_gradient(
        self,
        free_state: SystemState,
        nudged_state: SystemState,
        loss: Tensor,
        geometry: Geometry,
    ) -> list[Tensor]:
        """Compute STDP-based pseudo-gradients from recorded spikes."""
        grads = []
        params = geometry.params
        weight_names = [n for n in params if "weight" in n and params[n].ndim == 2]

        for layer_idx, weight_name in enumerate(weight_names):
            if (
                layer_idx not in self._pre_spike_times
                or layer_idx not in self._post_spike_times
            ):
                # No spike data for this layer, return zero gradient
                weight = params[weight_name]
                grads.append(torch.zeros_like(weight))
                continue

            pre_spikes = self._pre_spike_times[layer_idx]
            post_spikes = self._post_spike_times[layer_idx]

            # Compute STDP window
            # For efficiency, use a representative dt range
            dt_range = torch.linspace(-50, 50, 101)  # -50 to 50 ms
            window = self.compute_stdp_window(pre_spikes, post_spikes, dt_range)

            # Average window across spike pairs to get weight change per connection
            # Shape: (n_post, n_pre) -> average over pre/post populations
            if window.numel() > 0:
                avg_window = window.mean()  # Scalar average
                weight = params[weight_name]
                # Scale by average STDP window
                grad = torch.full_like(weight, avg_window.item())
                grads.append(grad)
            else:
                grads.append(torch.zeros_like(params[weight_name]))

        return grads

    def surrogate_objective(
        self,
        free_state: SystemState,
        nudged_state: SystemState,
        geometry: Geometry,
    ) -> Tensor:
        """Surrogate objective for STDP: negative correlation (maximize causal)."""
        total_correlation = torch.tensor(0.0)
        for layer_idx in self._pre_spike_times:
            if layer_idx in self._post_spike_times:
                pre = self._pre_spike_times[layer_idx]
                post = self._post_spike_times[layer_idx]
                if pre.numel() > 0 and post.numel() > 0:
                    # Simple correlation proxy
                    total_correlation += pre.float().mean() * post.float().mean()
        return (
            -total_correlation
        )  # Minimize negative correlation = maximize correlation


class TargetInversionCredit:
    """Target Propagation: propagate local targets backward."""

    def __init__(self, config: CreditAssignmentConfig | None = None):
        self.config = config or CreditAssignmentConfig.target_inversion()

    def compute_pseudo_gradient(  # ruff: ignore[no-self-use]
        self,
        free_state: SystemState,
        nudged_state: SystemState,
        loss: Tensor,
        geometry: Geometry,
    ) -> list[Tensor]:
        # Target propagation uses learned inverse maps
        return []

    def surrogate_objective(
        self,
        free_state: SystemState,
        nudged_state: SystemState,
        geometry: Geometry,
    ) -> Tensor:
        """Local target MSE: ‖h_l - target_l‖^2 per layer."""
        acts = free_state.activations
        if isinstance(acts, list):
            # For target prop, surrogate is MSE between free and nudged activations per layer
            free_acts = acts
            nudged_acts = (
                nudged_state.activations
                if isinstance(nudged_state.activations, list)
                else []
            )
            total_mse = torch.tensor(
                0.0, device=free_acts[0].device if free_acts else "cpu"
            )
            for fa, na in zip(
                free_acts[1:], nudged_acts[1:] if len(nudged_acts) > 1 else []
            ):
                if fa.shape == na.shape:
                    total_mse += torch.nn.functional.mse_loss(fa, na)
            return total_mse
        return torch.tensor(0.0)


class HomeostaticCredit:
    """Homeostatic credit assignment with dynamic Lipschitz scaling.

    Implements the homeostatic mechanism from HomeostaticEqProp:
    - Monitors per-layer velocity (state change rate)
    - Estimates layer-wise Lipschitz constants
    - Applies braking (scale down) when velocity exceeds threshold or Lipschitz > target
    - Applies boosting (scale up) when velocity is too low and Lipschitz < target

    This provides autonomous stability without external hyperparameter tuning.
    """

    def __init__(self, config: CreditAssignmentConfig | None = None):
        self.config = config or CreditAssignmentConfig(
            credit_type="homeostatic",
            beta=0.5,
            feedback_matrix=None,
            local_objective="mse",
            orthogonal_init=False,
            feedback_scale=0.01,
        )
        self._layer_scales: dict[int, float] = {}
        self._last_velocities: dict[int, float] = {}
        self._homeostasis_history: list[dict] = []

        # Homeostatic parameters
        self._alpha = 0.5
        self._target_lipschitz = 0.95
        self._velocity_threshold_high = 0.1
        self._velocity_threshold_low = 0.01
        self._adaptation_rate = 0.01

    def _estimate_layer_lipschitz(self, layer_idx: int, geometry: Geometry) -> float:
        """Estimate Lipschitz constant for a layer using power iteration."""
        params = geometry.params
        weight_names = [n for n in params if "weight" in n and params[n].ndim == 2]

        if layer_idx >= len(weight_names):
            return 1.0

        weight_name = weight_names[layer_idx]
        scale = self._layer_scales.get(layer_idx, 1.0)
        W = params[weight_name] * scale

        with torch.no_grad():
            u = torch.randn(W.shape[1], device=W.device)
            u = torch.nn.functional.normalize(u, dim=0)
            for _ in range(3):
                v = torch.nn.functional.normalize(W @ u, dim=0)
                u = torch.nn.functional.normalize(W.T @ v, dim=0)
            sigma = torch.norm(W @ u).item()
        return sigma

    def compute_pseudo_gradient(
        self,
        free_state: SystemState,
        nudged_state: SystemState,
        loss: Tensor,
        geometry: Geometry,
    ) -> list[Tensor]:
        """Compute homeostatic pseudo-gradients.

        The pseudo-gradient is the standard contrastive gradient scaled
        by the homeostatic layer scales (which adapt based on velocity/Lipschitz).
        """
        if free_state.activations is None or nudged_state.activations is None:
            return []

        free_acts = (
            free_state.activations
            if isinstance(free_state.activations, list)
            else [free_state.activations]
        )
        nudged_acts = (
            nudged_state.activations
            if isinstance(nudged_state.activations, list)
            else [nudged_state.activations]
        )

        param_names = list(geometry.params.keys())
        weight_names = [
            n for n in param_names if "weight" in n and geometry.params[n].ndim == 2
        ]

        grads = []
        n_layers = len(free_acts) - 1

        for layer_idx in range(n_layers):
            if layer_idx < len(weight_names):
                # Standard contrastive Hebbian gradient
                free_pre = free_acts[layer_idx]
                free_post = free_acts[layer_idx + 1]
                free_corr = free_pre.T @ free_post

                nudged_pre = nudged_acts[layer_idx]
                nudged_post = nudged_acts[layer_idx + 1]
                nudged_corr = nudged_pre.T @ nudged_post

                contrast = (free_corr - nudged_corr) / free_pre.shape[0]
                grad = contrast.T

                # Apply homeostatic scaling
                scale = self._layer_scales.get(layer_idx, 1.0)
                grads.append(grad * scale)

        # Track velocities for homeostasis (difference between nudged and free)
        if free_state.activations is not None and nudged_state.activations is not None:
            for layer_idx in range(n_layers):
                if layer_idx < len(free_acts) - 1 and layer_idx < len(nudged_acts) - 1:
                    free_h = free_acts[layer_idx + 1]
                    nudged_h = nudged_acts[layer_idx + 1]
                    velocity = torch.mean(torch.abs(nudged_h - free_h)).item()
                    self._last_velocities[layer_idx] = velocity

        return grads

    def apply_homeostasis(self, geometry: Geometry) -> dict:
        """Apply homeostatic adaptation based on tracked velocities.

        Call this after computing pseudo-gradients to update layer scales.
        """
        brake_total = 0.0
        boost_total = 0.0
        layers_braked = 0
        layers_boosted = 0

        for layer_idx, velocity in self._last_velocities.items():
            current_L = self._estimate_layer_lipschitz(layer_idx, geometry)

            if velocity > self._velocity_threshold_high or current_L > (
                self._target_lipschitz + 0.1
            ):
                error_v = max(0, velocity - self._velocity_threshold_high)
                error_l = max(0, current_L - self._target_lipschitz)
                error = error_v + error_l

                factor = 1.0 - (self._adaptation_rate * (1.0 + 10.0 * error))
                factor = max(0.5, factor)

                self._layer_scales[layer_idx] = (
                    self._layer_scales.get(layer_idx, 1.0) * factor
                )
                brake_total += 1.0 - factor
                layers_braked += 1

            elif velocity < self._velocity_threshold_low:
                current_L = self._estimate_layer_lipschitz(layer_idx, geometry)
                if current_L < self._target_lipschitz:
                    error = self._velocity_threshold_low - velocity
                    factor = 1.0 + (self._adaptation_rate * (1.0 + 5.0 * error))
                    factor = min(1.5, factor)

                    self._layer_scales[layer_idx] = (
                        self._layer_scales.get(layer_idx, 1.0) * factor
                    )
                    boost_total += factor - 1.0
                    layers_boosted += 1

        # Clamp scales
        for k in self._layer_scales:
            self._layer_scales[k] = max(0.1, min(3.0, self._layer_scales[k]))

        avg_velocity = (
            sum(self._last_velocities.values()) / len(self._last_velocities)
            if self._last_velocities
            else 0.0
        )
        avg_lipschitz = (
            sum(
                self._estimate_layer_lipschitz(i, geometry)
                for i in range(len(self._layer_scales))
            )
            / len(self._layer_scales)
            if self._layer_scales
            else 0.0
        )

        metrics = {
            "avg_velocity": avg_velocity,
            "lipschitz_estimate": avg_lipschitz,
            "brake_applied": brake_total,
            "boost_applied": boost_total,
            "layers_braked": layers_braked,
            "layers_boosted": layers_boosted,
            "layer_scales": dict(self._layer_scales),
        }

        self._homeostasis_history.append(metrics)
        return metrics

    def surrogate_objective(
        self,
        free_state: SystemState,
        nudged_state: SystemState,
        geometry: Geometry,
    ) -> Tensor:
        """Surrogate objective: negative energy difference with homeostatic penalty."""
        if free_state.energy is not None and nudged_state.energy is not None:
            base_obj = nudged_state.energy - free_state.energy
            # Add homeostatic regularization
            reg = sum((s - 1.0) ** 2 for s in self._layer_scales.values())
            return base_obj + 0.01 * reg
        return torch.tensor(0.0)

    def get_stability_report(self) -> str:
        """Generate a stability report."""
        if not self._layer_scales:
            return "No layers tracked yet"

        max_L = max(
            self._estimate_layer_lipschitz(i, None) if i < 10 else 0.0
            for i in self._layer_scales
        )
        status = "STABLE" if max_L < 1.0 else "UNSTABLE"

        lines = [
            f"Max Lipschitz: {max_L:.4f} {status}",
            f"Layer Scales: {[f'{s:.3f}' for s in self._layer_scales.values()]}",
        ]
        if self._homeostasis_history:
            last = self._homeostasis_history[-1]
            lines.append(
                f"Last Action: {last['layers_braked']} braked, {last['layers_boosted']} boosted"
            )

        return "\n".join(lines)


# Additional parameter update implementations
class RiemannianOrthogonalUpdate:
    """Muon: Riemannian optimization on Stiefel manifold."""

    def __init__(self, config: ParameterUpdateConfig | None = None):
        self.config = config or ParameterUpdateConfig.riemannian_orthogonal()

    def _newton_schulz(self, g: Tensor, steps: int = 20) -> Tensor:  # ruff: ignore[no-self-use]
        """Compute orthogonal projection via Newton-Schulz iteration.

        Newton-Schulz computes the polar factor (orthogonal part) of a matrix.
        Uses standard iteration: x_{k+1} = 0.5 * x_k * (3I - x_k^T x_k)
        Requires ~15-20 steps for 1e-5 accuracy.
        For non-square matrices, uses SVD-based orthogonalization.
        """
        # Handle non-square matrices via SVD
        m, n = g.shape
        if m != n:
            # For non-square, use SVD: U @ Vh gives orthogonal projection
            # torch.linalg.svd returns (U, S, Vh) where Vh = V^T
            u, _, vh = torch.linalg.svd(g, full_matrices=False)
            return u @ vh

        # Normalize by spectral norm (max singular value) for convergence
        _, s, _ = torch.linalg.svd(g, full_matrices=False)
        spectral_norm = s[0]
        if spectral_norm > 0:
            x = g / spectral_norm
        else:
            x = g

        eye = torch.eye(n, device=x.device, dtype=x.dtype)
        for _ in range(steps):
            xtx = x.T @ x
            x = 0.5 * x @ (3 * eye - xtx)
        return x

    def step(
        self,
        params: dict[str, Tensor],
        pseudo_grads: list[Tensor],
        geometry: Geometry,  # ruff: ignore[unused-method-argument]
    ) -> dict[str, Tensor]:
        updated = {}
        for i, (name, param) in enumerate(params.items()):
            if i < len(pseudo_grads) and pseudo_grads[i] is not None:
                grad = pseudo_grads[i]
                if grad.ndim >= 2 and grad.shape[0] == grad.shape[1]:  # ruff: ignore[magic-value-comparison]
                    # Only orthogonalize square gradients
                    grad = self._newton_schulz(grad, self.config.ortho_steps)
                updated[name] = param - self.config.step_size * grad
            else:
                updated[name] = param
        return updated


class SpectralConstrainedUpdate:
    """Spectral norm constrained updates for stability."""

    def __init__(self, config: ParameterUpdateConfig | None = None):
        self.config = config or ParameterUpdateConfig.spectral_constrained()

    def step(
        self,
        params: dict[str, Tensor],
        pseudo_grads: list[Tensor],
        geometry: Geometry,  # ruff: ignore[unused-method-argument]
    ) -> dict[str, Tensor]:
        updated = {}
        min_ndim_for_svd = 2
        for i, (name, param) in enumerate(params.items()):
            if i < len(pseudo_grads) and pseudo_grads[i] is not None:
                grad = pseudo_grads[i]
                # Project gradient to satisfy spectral constraint
                if grad.ndim >= min_ndim_for_svd:
                    u, s, v = torch.linalg.svd(grad, full_matrices=False)
                    s = torch.clamp(s, max=self.config.spectral_norm)
                    grad = u @ torch.diag(s) @ v
                updated[name] = param - self.config.step_size * grad
            else:
                updated[name] = param
        return updated


class NaturalGradientUpdate:
    """Natural gradient using Fisher information geometry."""

    def __init__(self, config: ParameterUpdateConfig | None = None):
        self.config = config or ParameterUpdateConfig.natural_gradient()
        self._fisher: dict[str, Tensor] = {}

    def step(
        self,
        params: dict[str, Tensor],
        pseudo_grads: list[Tensor],
        geometry: Geometry,  # ruff: ignore[unused-method-argument]
    ) -> dict[str, Tensor]:
        updated = {}
        for i, (name, param) in enumerate(params.items()):
            if i < len(pseudo_grads) and pseudo_grads[i] is not None:
                grad = pseudo_grads[i]
                # Accumulate Fisher (diagonal approximation)
                if name not in self._fisher:
                    self._fisher[name] = torch.zeros_like(param)
                self._fisher[name].mul_(0.99).addcmul_(grad, grad, value=0.01)
                # Natural gradient: F^-1 * g
                nat_grad = grad / (
                    self._fisher[name].sqrt() + self.config.fisher_damping
                )
                updated[name] = param - self.config.step_size * nat_grad
            else:
                updated[name] = param
        return updated


class ElasticConsolidationUpdate:
    """Elastic Weight Consolidation for continual learning."""

    def __init__(self, config: ParameterUpdateConfig | None = None):
        self.config = config or ParameterUpdateConfig.elastic_consolidation()
        self._importance: dict[str, Tensor] = {}
        self._old_params: dict[str, Tensor] = {}

    def step(
        self,
        params: dict[str, Tensor],
        pseudo_grads: list[Tensor],
        geometry: Geometry,  # ruff: ignore[unused-method-argument]
    ) -> dict[str, Tensor]:
        updated = {}
        for i, (name, param) in enumerate(params.items()):
            if i < len(pseudo_grads) and pseudo_grads[i] is not None:
                grad = pseudo_grads[i]
                # EWC penalty: lambda * importance * (param - old_param)
                ewc_penalty = torch.zeros_like(param)
                if name in self._importance and name in self._old_params:
                    ewc_penalty = (
                        self.config.ewc_lambda
                        * self._importance[name]
                        * (param - self._old_params[name])
                    )
                updated[name] = param - self.config.step_size * (grad + ewc_penalty)
            else:
                updated[name] = param
        return updated

    def consolidate(self, params: dict[str, Tensor], fisher: dict[str, Tensor]) -> None:
        """Call after task completion to consolidate weights."""
        self._importance = fisher
        self._old_params = {k: v.clone() for k, v in params.items()}


# ============================================================
# Internal: Adapted System for wrapping existing models
# ============================================================


class _AdaptedSystem:
    """Internal adapter that wraps an existing model as a System.

    Delegates train_step/forward to the wrapped model while exposing
    the 5-D ontology interface for AutoScientist queries.
    """

    substrate: Substrate
    geometry: Geometry
    dynamics: StateDynamics
    credit: CreditAssignment
    update: ParameterUpdate
    _model: nn.Module

    def __init__(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
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

    def train_step(self, x: Tensor, y: Tensor) -> dict[str, float]:
        # Try legacy model's train_step first
        if hasattr(self._model, "train_step"):
            result = self._model.train_step(x, y)  # type: ignore[attr-defined]
            if result is not None:
                return result
            # Legacy model returned None (e.g., EqProp single-hidden implicit path)
            # Fall back to ontology 5-layer pipeline
        # Ontology 5-layer pipeline fallback
        state = SystemState(x=x, y=y)
        # 1. Substrate + Geometry: Forward pass
        state.activations = self.geometry.forward(x, self.substrate)
        if state.activations is not None:
            state.activations = self.substrate.inject_state_noise(state.activations)
        # 2. StateDynamics: Free phase settling
        free_state = self.dynamics.settle(
            state, self.geometry, self.substrate, target=None
        )
        free_state.energy = self.dynamics.compute_energy(free_state, self.geometry)
        # 3. StateDynamics: Nudged phase settling
        nudged_state = self.dynamics.settle(
            state, self.geometry, self.substrate, target=y
        )
        nudged_state.energy = self.dynamics.compute_energy(nudged_state, self.geometry)
        nudged_state.loss = self._compute_loss(nudged_state, y)
        # 4. CreditAssignment: Compute pseudo-gradients
        pseudo_grads = self.credit.compute_pseudo_gradient(
            free_state, nudged_state, nudged_state.loss, self.geometry
        )
        # 5. ParameterUpdate: Apply updates
        new_params = self.update.step(self.geometry.params, pseudo_grads, self.geometry)
        self.geometry.update_params(new_params)
        return {
            "loss": nudged_state.loss.item() if nudged_state.loss is not None else 0.0,
            "energy": free_state.energy.item()
            if free_state.energy is not None
            else 0.0,
            "accuracy": free_state.metrics.get("accuracy", 0.0),
        }

    def forward(self, x: Tensor) -> Tensor:
        return self._model(x)  # type: ignore[operator]

    def _compute_loss(self, state: SystemState, y: Tensor) -> Tensor:  # ruff: ignore[no-self-use]
        """Compute task loss from final state (required by System protocol)."""
        acts = state.activations
        if acts is None:
            return torch.tensor(0.0)
        logits = acts[-1] if isinstance(acts, list) else acts
        return torch.nn.functional.cross_entropy(logits, y)

    @property
    def model(self) -> nn.Module:
        return self._model
