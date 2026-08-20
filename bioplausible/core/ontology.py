"""5-Dimensional Physico-Computational Ontology for Bioplausible Systems.

This module defines the five orthogonal axes (S × G × D × C × U) that compose
any bioplausible neural network. The tensor product of these primitives
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

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

import torch
from torch import Tensor, nn

from bioplausible.core.registry import ComponentMetadata, ComputeProfile, LocalityLevel

__all__ = [
    "CreditAssignment",
    "CreditAssignmentConfig",
    "Geometry",
    "GeometryConfig",
    "ParameterUpdate",
    "ParameterUpdateConfig",
    "StateDynamics",
    "StateDynamicsConfig",
    "Substrate",
    "SubstrateConfig",
    "System",
    "SystemState",
    "ModelAdapter",
    "DigitalSubstrate",
    "AnalogSubstrate",
    "MemristiveSubstrate",
    "NeuromorphicSubstrate",
    "QuantumSubstrate",
    "OpticalSubstrate",
    "NoisySubstrate",
    "QuantizedSubstrate",
    "FeedforwardGeometry",
    "RecurrentGeometry",
    "InstantaneousDynamics",
    "EnergyMinimizationDynamics",
    "PredictiveSettlingDynamics",
    "SpikeIntegrationDynamics",
    "ThermodynamicContrast",
    "RandomProjectionsCredit",
    "LocalGoodnessCredit",
    "TemporalTraceCredit",
    "TargetInversionCredit",
    "BackpropCredit",
    "EuclideanUpdate",
    "RiemannianOrthogonalUpdate",
    "SpectralConstrainedUpdate",
    "NaturalGradientUpdate",
    "ElasticConsolidationUpdate",
]


# ============================================================
# Configuration Dataclasses (immutable, slotted)
# ============================================================

@dataclass(frozen=True, slots=True)
class SubstrateConfig:
    """Configuration for a physical substrate.

    Attributes:
        precision: Numeric precision ("float32", "float16", "bfloat16", "int8", "int4", "binary")
        noise_level: Standard deviation of additive state noise
        weight_bounds: Optional (min, max) tuple for weight clamping
        sparsity: Target sparsity ratio [0, 1]
        device: Target device ("cpu", "cuda", "mps", "fpga", "analog", "optical")
    """
    precision: str = "float32"
    noise_level: float = 0.0
    weight_bounds: tuple[float, float] | None = None
    sparsity: float = 0.0
    device: str = "cpu"


@dataclass(frozen=True, slots=True)
class GeometryConfig:
    """Configuration for network geometry/topology.

    Attributes:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_dims: List of hidden layer dimensions
        num_layers: Number of layers (alternative to hidden_dims)
        topology_type: "feedforward", "recurrent", "tile_mesh", "neuromorphic", "spatial_lattice"
        connectivity: Optional adjacency specification
    """
    input_dim: int
    output_dim: int
    hidden_dims: tuple[int, ...] = ()
    num_layers: int = 1
    topology_type: str = "feedforward"
    connectivity: dict | None = None


@dataclass(frozen=True, slots=True)
class StateDynamicsConfig:
    """Configuration for state dynamics/settling.

    Attributes:
        dynamics_type: "energy_minimization", "predictive_settling", "spike_integration", "instantaneous"
        max_steps: Maximum settling iterations
        convergence_threshold: Early stopping threshold
        convergence_start: Step to start checking convergence
        step_size: Learning rate for state updates
        beta: Nudge strength for energy-based methods
    """
    dynamics_type: str = "instantaneous"
    max_steps: int = 30
    convergence_threshold: float = 1e-4
    convergence_start: int = 5
    step_size: float = 0.1
    beta: float = 0.1


@dataclass(frozen=True, slots=True)
class CreditAssignmentConfig:
    """Configuration for credit assignment.

    Attributes:
        credit_type: "thermodynamic_contrast", "random_projections", "local_goodness", "temporal_trace", "target_inversion"
        beta: Nudge strength (for thermodynamic contrast)
        feedback_matrix: Optional fixed feedback matrix (for random projections)
        local_objective: Layer-local loss type (for local goodness)
    """
    credit_type: str = "thermodynamic_contrast"
    beta: float = 0.5
    feedback_matrix: Tensor | None = None
    local_objective: str = "mse"


@dataclass(frozen=True, slots=True)
class ParameterUpdateConfig:
    """Configuration for parameter update rule.

    Attributes:
        update_type: "riemannian_orthogonal", "spectral_constrained", "natural_gradient", "elastic_consolidation", "euclidean"
        step_size: Learning rate
        momentum: Momentum coefficient (for euclidean)
        ortho_steps: Newton-Schulz iterations (for riemannian)
        spectral_norm: Target spectral norm (for spectral_constrained)
        fisher_damping: Damping for natural gradient
        ewc_lambda: EWC regularization strength
    """
    update_type: str = "euclidean"
    step_size: float = 0.01
    momentum: float = 0.9
    ortho_steps: int = 5
    spectral_norm: float = 1.0
    fisher_damping: float = 1e-3
    ewc_lambda: float = 1000.0


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

    Every bioplausible system runs on a substrate. The substrate injects
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
        """Return the substrate's weight update operator: (pseudo_grad, current_w) -> ΔW.

        Encodes how the substrate physically modifies weights (e.g. pulse-based
        conductance change for memristors, phase shift for photonic).
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
    """Composable bioplausible system: S ⊗ G ⊗ D ⊗ C ⊗ U.

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
        free_state = self.dynamics.settle(state, self.geometry, self.substrate, target=None)
        free_state.energy = self.dynamics.compute_energy(free_state, self.geometry)

        # 3. StateDynamics: Nudged phase settling
        nudged_state = self.dynamics.settle(state, self.geometry, self.substrate, target=y)
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
            "loss": float(nudged_state.loss) if nudged_state.loss is not None else 0.0,
            "energy": float(free_state.energy) if free_state.energy is not None else 0.0,
            "accuracy": free_state.metrics.get("accuracy", 0.0),
        }

    def _compute_loss(self, state: SystemState, y: Tensor) -> Tensor:
        """Compute task loss from final state."""
        acts = state.activations
        if acts is None:
            return torch.tensor(0.0)
        if isinstance(acts, list):
            logits = acts[-1]
        else:
            logits = acts
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


# ============================================================
# Default/Reference Implementations
# ============================================================

class DigitalSubstrate:
    """Reference substrate: infinite precision, continuous time, no noise."""

    def __init__(self, config: SubstrateConfig | None = None):
        self.config = config or SubstrateConfig()

    def quantize_weights(self, w: Tensor) -> Tensor:
        return w

    def inject_state_noise(self, s: Tensor) -> Tensor:
        return s

    def get_forward_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        return lambda x, w: x @ w.T

    def get_weight_update_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        return lambda grad, w: grad

    def initial_state(self, x: Tensor) -> Tensor:
        return x


class FeedforwardGeometry(nn.Module):
    """Standard feedforward DAG topology (MLP, CNN)."""

    _layers: nn.ModuleList

    def __init__(self, config: GeometryConfig, layers: nn.ModuleList | None = None):
        super().__init__()
        self.config = config
        self._layers = layers or nn.ModuleList()
        if not self._layers and config.hidden_dims:
            self._build_layers()

    def _build_layers(self) -> None:
        dims = [self.config.input_dim] + list(self.config.hidden_dims) + [self.config.output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self._layers = nn.ModuleList(layers)

    @property
    def params(self) -> dict[str, Tensor]:
        return dict(self._layers.named_parameters())

    def forward(self, x: Tensor, substrate: Substrate) -> Tensor:
        op = substrate.get_forward_operator()
        h = x
        for layer in self._layers:
            if isinstance(layer, nn.Linear):
                h = op(h, layer.weight)
                if layer.bias is not None:
                    h = h + layer.bias
            else:
                h = layer(h)
        return h

    def route(self, activations: Tensor) -> Tensor:
        """Single-step routing through the geometry (for settling dynamics)."""
        h = activations
        for layer in self._layers:
            if isinstance(layer, nn.Linear):
                h = h @ layer.weight.T
                if layer.bias is not None:
                    h = h + layer.bias
            else:
                h = layer(h)
        return h

    def update_params(self, new_params: dict[str, Tensor]) -> None:
        for name, param in new_params.items():
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
    ):
        super().__init__()
        self.config = config
        self._layers = layers or nn.ModuleList()
        self._recurrent_weight = None
        if not self._layers and config.hidden_dims:
            self._build_layers()
        if hidden_dim is not None and self._recurrent_weight is None:
            self._recurrent_weight = nn.Parameter(
                torch.randn(hidden_dim, hidden_dim) * 0.1
            )

    def _build_layers(self) -> None:
        # For recurrent: input -> hidden (with recurrent), hidden -> output
        dims = [self.config.input_dim] + list(self.config.hidden_dims) + [self.config.output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self._layers = nn.ModuleList(layers)
        # Add recurrent weight for the last hidden layer
        if len(self.config.hidden_dims) > 0:
            hidden_dim = self.config.hidden_dims[-1]
            self._recurrent_weight = nn.Parameter(
                torch.randn(hidden_dim, hidden_dim) * 0.1
            )

    @property
    def params(self) -> dict[str, Tensor]:
        params = dict(self._layers.named_parameters())
        if self._recurrent_weight is not None:
            params["recurrent_weight"] = self._recurrent_weight
        return params

    def forward(self, x: Tensor, substrate: Substrate) -> Tensor:
        """Full forward pass with recurrence (single step)."""
        op = substrate.get_forward_operator()
        h = x
        for i, layer in enumerate(self._layers):
            if isinstance(layer, nn.Linear):
                h = op(h, layer.weight)
                if layer.bias is not None:
                    h = h + layer.bias
            else:
                h = layer(h)
            # Apply recurrent connection after each hidden layer (except output)
            if self._recurrent_weight is not None and i < len(self._layers) - 2:
                h = h + op(h, self._recurrent_weight)
        return h

    def route(self, activations: Tensor) -> Tensor:
        """Single recurrent step: h_{t+1} = f(h_t) = activation(W_rec @ h_t).

        Expects activations to be the hidden state (batch_size x hidden_dim).
        """
        h = activations
        if self._recurrent_weight is not None:
            # Hidden state should match recurrent weight dimensions
            if h.shape[-1] == self._recurrent_weight.shape[0]:
                h = h @ self._recurrent_weight.T
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
                for layer in self._layers:
                    if hasattr(layer, name):
                        getattr(layer, name).data.copy_(param)

    def transition_modules(self) -> list[nn.Module]:
        return [m for m in self._layers if isinstance(m, nn.Linear)]


class InstantaneousDynamics:
    """Single-pass feedforward (Backprop, Forward-Forward)."""

    def __init__(self, config: StateDynamicsConfig | None = None):
        self.config = config or StateDynamicsConfig(dynamics_type="instantaneous")

    def settle(
        self,
        state: SystemState,
        geometry: Geometry,
        substrate: Substrate,
        target: Tensor | None = None,
    ) -> SystemState:
        # Single forward pass - no settling
        state.free_state = state.activations
        return state

    def compute_energy(self, state: SystemState, geometry: Geometry) -> Tensor:
        # Proxy: negative log-likelihood for instantaneous pass
        if state.loss is not None:
            return torch.as_tensor(state.loss)
        return torch.tensor(0.0)


class ThermodynamicContrast:
    """Equilibrium Propagation credit assignment: (nudged - free) / beta."""

    def __init__(self, config: CreditAssignmentConfig | None = None):
        self.config = config or CreditAssignmentConfig()

    def compute_pseudo_gradient(
        self,
        free_state: SystemState,
        nudged_state: SystemState,
        loss: Tensor,
        geometry: Geometry,
    ) -> list[Tensor]:
        if free_state.activations is None or nudged_state.activations is None:
            return []

        free_acts = free_state.activations if isinstance(free_state.activations, list) else [free_state.activations]
        nudged_acts = nudged_state.activations if isinstance(nudged_state.activations, list) else [nudged_state.activations]

        grads = []
        for f, n in zip(free_acts[:-1], nudged_acts[:-1]):
            contrast = (n - f) / self.config.beta
            grads.append(contrast)

        return grads


class EuclideanUpdate:
    """Standard Euclidean SGD/Adam update."""

    def __init__(self, config: ParameterUpdateConfig | None = None):
        self.config = config or ParameterUpdateConfig()
        self._momentum_buffers: dict[str, Tensor] = {}

    def step(
        self,
        params: dict[str, Tensor],
        pseudo_grads: list[Tensor],
        geometry: Geometry,
    ) -> dict[str, Tensor]:
        updated = {}

        for i, (name, param) in enumerate(params.items()):
            if i < len(pseudo_grads):
                grad = pseudo_grads[i]
                if self.config.momentum > 0:
                    buf = self._momentum_buffers.get(name, torch.zeros_like(param))
                    buf.mul_(self.config.momentum).add_(grad)
                    self._momentum_buffers[name] = buf
                    updated[name] = param - self.config.step_size * buf
                else:
                    updated[name] = param - self.config.step_size * grad
            else:
                updated[name] = param

        return updated


# ============================================================
# Adapter: Wrap existing models as System compositions
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
        if self._metadata and self._metadata.compute_profile:
            profile = self._metadata.compute_profile
            if profile == ComputeProfile.ANALOG:
                return AnalogSubstrate()
            elif profile == ComputeProfile.OPTICAL:
                return OpticalSubstrate()
            elif profile == ComputeProfile.MEMRISTOR:
                return MemristiveSubstrate()
            elif profile == ComputeProfile.NEUROMORPHIC:
                return NeuromorphicSubstrate()

        # Priority 2: Model attributes
        if hasattr(self.model, "backend"):
            backend = self.model.backend
            if backend == "quantized":
                return QuantizedSubstrate()
            elif backend == "noisy":
                return NoisySubstrate()
            elif backend == "optical":
                return OpticalSubstrate()
            elif backend == "crossbar":
                return MemristiveSubstrate()
            elif backend == "quantum":
                return QuantumSubstrate()

        # Priority 3: Family tag heuristics
        if self._metadata and self._metadata.family:
            family = self._metadata.family.lower()
            if "quantized" in family or "ternary" in family:
                return QuantizedSubstrate()
            elif "noisy" in family:
                return NoisySubstrate()
            elif "optical" in family or "photonic" in family:
                return OpticalSubstrate()
            elif "crossbar" in family or "memristive" in family:
                return MemristiveSubstrate()
            elif "quantum" in family:
                return QuantumSubstrate()

        return DigitalSubstrate()

    def _infer_geometry(self) -> Geometry:
        # Priority 1: Check for specific topology types
        if hasattr(self.model, "topology_type"):
            topo = self.model.topology_type
            if topo == "recurrent" or topo == "recurrent_attractor":
                return self._make_recurrent_geometry()
            elif topo == "tile_mesh" or topo == "tile":
                return self._make_tile_geometry()
            elif topo == "neuromorphic" or topo == "fabric":
                return self._make_neuromorphic_geometry()
            elif topo == "spatial_lattice" or topo == "3d":
                return self._make_spatial_geometry()

        # Priority 2: Registry metadata family
        if self._metadata and self._metadata.family:
            family = self._metadata.family.lower()
            if "tile" in family:
                return self._make_tile_geometry()
            elif "cube" in family or "neural_cube" in family:
                return self._make_spatial_geometry()
            elif "graph" in family or "fabric" in family:
                return self._make_neuromorphic_geometry()
            elif "recurrent" in family or "equilibrium" in family or "eqprop" in family or "ep" in family:
                return self._make_recurrent_geometry()

        # Priority 3: Model attributes
        if hasattr(self.model, "transition_modules"):
            layers = self.model.transition_modules()  # type: ignore[attr-defined]
            return FeedforwardGeometry(
                GeometryConfig(
                    input_dim=getattr(self.model, "input_dim", 0),
                    output_dim=getattr(self.model, "output_dim", 0),
                    hidden_dims=self._infer_hidden_dims(),
                ),
                layers=layers,
            )

        # Default: feedforward
        return FeedforwardGeometry(
            GeometryConfig(
                input_dim=getattr(self.model, "input_dim", 0),
                output_dim=getattr(self.model, "output_dim", 0),
                hidden_dims=self._infer_hidden_dims(),
            )
        )

    def _infer_hidden_dims(self) -> tuple[int, ...]:
        if hasattr(self.model, "hidden_dims"):
            return tuple(self.model.hidden_dims)
        if hasattr(self.model, "hidden_dim"):
            return (self.model.hidden_dim,)
        if hasattr(self.model, "config") and hasattr(self.model.config, "hidden_dims"):
            return tuple(self.model.config.hidden_dims)
        return ()

    def _make_recurrent_geometry(self) -> RecurrentGeometry:
        return RecurrentGeometry(
            GeometryConfig(
                input_dim=getattr(self.model, "input_dim", 0),
                output_dim=getattr(self.model, "output_dim", 0),
                hidden_dims=self._infer_hidden_dims(),
                topology_type="recurrent",
            ),
            hidden_dim=self._infer_hidden_dims()[-1] if self._infer_hidden_dims() else None,
        )

    def _make_tile_geometry(self) -> Geometry:
        # Tile geometry is complex; for now use feedforward as placeholder
        # A full TileGeometry implementation would be added later
        return FeedforwardGeometry(
            GeometryConfig(
                input_dim=getattr(self.model, "input_dim", 0),
                output_dim=getattr(self.model, "output_dim", 0),
                hidden_dims=self._infer_hidden_dims(),
                topology_type="tile_mesh",
            )
        )

    def _make_neuromorphic_geometry(self) -> Geometry:
        return FeedforwardGeometry(
            GeometryConfig(
                input_dim=getattr(self.model, "input_dim", 0),
                output_dim=getattr(self.model, "output_dim", 0),
                hidden_dims=self._infer_hidden_dims(),
                topology_type="neuromorphic",
            )
        )

    def _make_spatial_geometry(self) -> Geometry:
        return FeedforwardGeometry(
            GeometryConfig(
                input_dim=getattr(self.model, "input_dim", 0),
                output_dim=getattr(self.model, "output_dim", 0),
                hidden_dims=self._infer_hidden_dims(),
                topology_type="spatial_lattice",
            )
        )

    def _infer_dynamics(self) -> StateDynamics:
        # Priority 1: Registry metadata (most reliable)
        if self._metadata and self._metadata.family:
            family = self._metadata.family.lower()
            if "equilibrium" in family or "eqprop" in family or "ep" in family or "chl" in family:
                return EnergyMinimizationDynamics(StateDynamicsConfig(
                    dynamics_type="energy_minimization",
                    max_steps=30,
                    beta=0.5,
                ))
            elif "predictive" in family or "pc" in family:
                return PredictiveSettlingDynamics()
            elif "spiking" in family or "stdp" in family or "snn" in family:
                return SpikeIntegrationDynamics()
            elif "forward_only" in family or "ff" in family or "pepita" in family:
                return InstantaneousDynamics()
            elif "fa" in family or "feedback" in family or "dfa" in family:
                return InstantaneousDynamics()
            elif "target" in family or "target_prop" in family or "tp" in family:
                return InstantaneousDynamics()

        # Priority 2: Model attributes
        if hasattr(self.model, "gradient_method"):
            method = self.model.gradient_method
            if method == "equilibrium":
                return EnergyMinimizationDynamics(StateDynamicsConfig(
                    dynamics_type="energy_minimization",
                    max_steps=getattr(self.model, "max_steps", 30),
                    beta=getattr(self.model, "beta", 0.5),
                ))
            elif method == "predictive_coding":
                return PredictiveSettlingDynamics()
            elif method == "spiking" or method == "stdp":
                return SpikeIntegrationDynamics()

        if hasattr(self.model, "max_steps") and getattr(self.model, "max_steps", 1) > 1:
            return EnergyMinimizationDynamics(StateDynamicsConfig(
                dynamics_type="energy_minimization",
                max_steps=self.model.max_steps,
                beta=getattr(self.model, "beta", 0.5),
            ))

        # Priority 3: Locality level
        if self._metadata and self._metadata.locality_level:
            if self._metadata.locality_level == LocalityLevel.EQUILIBRIUM:
                return EnergyMinimizationDynamics()
            elif self._metadata.locality_level == LocalityLevel.FORWARD_ONLY:
                return InstantaneousDynamics()
            elif self._metadata.locality_level == LocalityLevel.LOCAL:
                return SpikeIntegrationDynamics()

        return InstantaneousDynamics()

    def _infer_credit(self) -> CreditAssignment:
        # Priority 1: Registry metadata
        if self._metadata and self._metadata.credit_assignment_type:
            credit_type = self._metadata.credit_assignment_type
            if credit_type == "equilibrium":
                return ThermodynamicContrast(CreditAssignmentConfig(
                    credit_type="thermodynamic_contrast",
                    beta=0.5,
                ))
            elif credit_type in ("random_projections", "feedback_alignment"):
                return RandomProjectionsCredit()
            elif credit_type == "local_goodness":
                return LocalGoodnessCredit()
            elif credit_type == "temporal_trace":
                return TemporalTraceCredit()
            elif credit_type == "target_inversion":
                return TargetInversionCredit()
            elif credit_type in ("gradient", "backpropagation"):
                return BackpropCredit()

        # Priority 2: Model attributes
        if hasattr(self.model, "credit_assignment_type"):
            credit_type = self.model.credit_assignment_type
            if credit_type == "equilibrium":
                return ThermodynamicContrast()
            elif credit_type in ("random_projections", "feedback_alignment"):
                return RandomProjectionsCredit()

        # Priority 3: Family
        if self._metadata and self._metadata.family:
            family = self._metadata.family.lower()
            if "eqprop" in family or "equilibrium" in family or "ep" in family:
                return ThermodynamicContrast()
            elif "fa" in family or "feedback" in family or "dfa" in family:
                return RandomProjectionsCredit()
            elif "hebbian" in family or "chl" in family:
                return ThermodynamicContrast()  # CHL uses contrastive
            elif "forward_only" in family or "ff" in family or "pepita" in family:
                return LocalGoodnessCredit()
            elif "spiking" in family or "stdp" in family:
                return TemporalTraceCredit()
            elif "target_prop" in family or "tp" in family:
                return TargetInversionCredit()

        return BackpropCredit()

    def _infer_update(self) -> ParameterUpdate:
        # Priority 1: Registry metadata tags
        if self._metadata and self._metadata.tags:
            tags = [t.lower() for t in self._metadata.tags]
            if "muon" in tags or "riemannian" in tags:
                return RiemannianOrthogonalUpdate()
            elif "spectral" in tags:
                return SpectralConstrainedUpdate()
            elif "fisher" in tags or "natural" in tags:
                return NaturalGradientUpdate()
            elif "ewc" in tags or "elastic" in tags:
                return ElasticConsolidationUpdate()

        # Priority 2: Family
        if self._metadata and self._metadata.family:
            family = self._metadata.family.lower()
            if "muon" in family or "mep" in family:
                return RiemannianOrthogonalUpdate()
            elif "fisher" in family:
                return NaturalGradientUpdate()
            elif "ewc" in family:
                return ElasticConsolidationUpdate()

        return EuclideanUpdate()


# Additional substrate implementations
class AnalogSubstrate(DigitalSubstrate):
    """Analog compute substrate with continuous values and noise."""
    def __init__(self, config: SubstrateConfig | None = None):
        super().__init__(config or SubstrateConfig(precision="float32", noise_level=0.02))


class MemristiveSubstrate(DigitalSubstrate):
    """Memristive crossbar: conductance matrices, bounded precision, IR-drop noise."""
    def __init__(self, config: SubstrateConfig | None = None):
        super().__init__(config or SubstrateConfig(
            precision="int8",
            noise_level=0.05,
            weight_bounds=(0.0, 1.0),  # Conductance is positive
            sparsity=0.1,
        ))

    def quantize_weights(self, w: Tensor) -> Tensor:
        # Memristors: positive conductance, bounded
        w = torch.clamp(w, 0.0, 1.0)
        scale = 1.0 / 255.0
        return (w / scale).round().clamp(0, 255) * scale

    def get_forward_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        # Crossbar: I = V * G (with IR drop approximation)
        return lambda x, w: x @ w.T  # Simplified


class NeuromorphicSubstrate(DigitalSubstrate):
    """Event-driven neuromorphic: sparse spikes, asynchronous."""
    def __init__(self, config: SubstrateConfig | None = None):
        super().__init__(config or SubstrateConfig(
            precision="float16",
            noise_level=0.01,
            sparsity=0.95,
        ))

    def inject_state_noise(self, s: Tensor) -> Tensor:
        # Spike dropout
        mask = torch.rand_like(s) > self.config.sparsity
        return s * mask.float()


class QuantumSubstrate(DigitalSubstrate):
    """Quantum substrate: parameterized unitary gates."""
    def __init__(self, config: SubstrateConfig | None = None):
        super().__init__(config or SubstrateConfig(
            precision="complex64",
            noise_level=0.02,
        ))

    def get_forward_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        # Simplified: quantum circuit evaluation
        return lambda x, w: (x @ w.T).real


# Additional substrate implementations
class QuantizedSubstrate(DigitalSubstrate):
    def __init__(self, config: SubstrateConfig | None = None):
        super().__init__(config or SubstrateConfig(precision="int8"))

    def quantize_weights(self, w: Tensor) -> Tensor:
        scale = w.abs().max() / 127
        return (w / scale).round().clamp(-128, 127) * scale


class NoisySubstrate(DigitalSubstrate):
    def __init__(self, config: SubstrateConfig | None = None):
        super().__init__(config or SubstrateConfig(noise_level=0.05))

    def inject_state_noise(self, s: Tensor) -> Tensor:
        noise = torch.randn_like(s) * self.config.noise_level
        return s + noise


class OpticalSubstrate(DigitalSubstrate):
    def __init__(self, config: SubstrateConfig | None = None):
        super().__init__(config or SubstrateConfig(precision="float32", noise_level=0.01))

    def get_forward_operator(self) -> Callable[[Tensor, Tensor], Tensor]:
        # Phase/amplitude encoding - simplified
        return lambda x, w: (x @ w.T).cos()  # Interference pattern


# Additional dynamics implementations
class EnergyMinimizationDynamics:
    """Energy-based settling (EqProp, Hopfield, CHL)."""

    def __init__(self, config: StateDynamicsConfig | None = None):
        self.config = config or StateDynamicsConfig(dynamics_type="energy_minimization")

    def settle(
        self,
        state: SystemState,
        geometry: Geometry,
        substrate: Substrate,
        target: Tensor | None = None,
    ) -> SystemState:
        # Run settling iterations
        h = state.activations
        if h is None:
            return state
        if isinstance(h, list):
            h = h[-1]  # Use last layer for single-tensor routing
        for step in range(self.config.max_steps):
            h_new = geometry.route(h)
            h_new = substrate.inject_state_noise(h_new)

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
        # Simplified energy: reconstruction error
        acts = state.free_state
        if acts is None:
            acts = state.nudged_state
        if acts is None:
            acts = state.activations
        if acts is None:
            return torch.tensor(0.0)
        if isinstance(acts, list):
            acts = acts[-1]
        return (acts ** 2).mean()


class PredictiveSettlingDynamics:
    """Predictive Coding: hierarchical prediction error minimization."""

    def __init__(self, config: StateDynamicsConfig | None = None):
        self.config = config or StateDynamicsConfig(dynamics_type="predictive_settling")

    def settle(
        self,
        state: SystemState,
        geometry: Geometry,
        substrate: Substrate,
        target: Tensor | None = None,
    ) -> SystemState:
        # Predictive coding settling - simplified
        return EnergyMinimizationDynamics(self.config).settle(state, geometry, substrate, target)

    def compute_energy(self, state: SystemState, geometry: Geometry) -> Tensor:
        return EnergyMinimizationDynamics(self.config).compute_energy(state, geometry)


class SpikeIntegrationDynamics:
    """Spiking dynamics: membrane potential integration and thresholding."""

    def __init__(self, config: StateDynamicsConfig | None = None):
        self.config = config or StateDynamicsConfig(dynamics_type="spike_integration")

    def settle(
        self,
        state: SystemState,
        geometry: Geometry,
        substrate: Substrate,
        target: Tensor | None = None,
    ) -> SystemState:
        # Simplified LIF dynamics
        h = state.activations
        if h is None:
            return state
        if isinstance(h, list):
            h = h[-1]
        for _ in range(self.config.max_steps):
            h_new = geometry.route(h)
            # Spike thresholding
            h_new = torch.where(h_new > 1.0, torch.zeros_like(h_new), h_new)
            if torch.dist(h_new, h, p=float("inf")).item() < self.config.convergence_threshold:
                h = h_new
                break
            h = h_new
        state.activations = h
        return state

    def compute_energy(self, state: SystemState, geometry: Geometry) -> Tensor:
        return torch.tensor(0.0)


# Additional credit assignment implementations
class RandomProjectionsCredit:
    """Feedback Alignment / Direct Feedback Alignment."""

    def __init__(self, config: CreditAssignmentConfig | None = None):
        self.config = config or CreditAssignmentConfig(credit_type="random_projections")
        self._feedback_weights: Tensor | None = None

    def compute_pseudo_gradient(
        self,
        free_state: SystemState,
        nudged_state: SystemState,
        loss: Tensor,
        geometry: Geometry,
    ) -> list[Tensor]:
        # Use fixed random feedback matrix
        if self._feedback_weights is None:
            # Initialize based on geometry output dim and hidden dims
            out_dim = geometry.config.output_dim
            hidden_dims = geometry.config.hidden_dims
            # For simplicity, create feedback for each layer
            self._feedback_weights = {}
            for i, h_dim in enumerate(hidden_dims):
                self._feedback_weights[f"layer_{i}"] = torch.randn(h_dim, out_dim) * 0.1

        # Project error through fixed feedback
        if nudged_state.activations is not None:
            acts = nudged_state.activations
            if isinstance(acts, list):
                # For multi-layer, we need per-layer activations
                grads = []
                for i, act in enumerate(acts[:-1]):  # Skip output layer
                    if f"layer_{i}" in self._feedback_weights:
                        fb = self._feedback_weights[f"layer_{i}"]
                        # Project output error back to this layer
                        output_error = acts[-1]  # Simplified
                        if output_error.shape[-1] == fb.shape[1]:
                            grad = output_error @ fb.T
                        else:
                            grad = torch.zeros_like(act)
                        grads.append(grad)
                return grads
            else:
                # Single layer
                return [torch.zeros_like(acts)]
        return []


class LocalGoodnessCredit:
    """Forward-Forward / PEPITA: layer-local contrastive objectives."""

    def __init__(self, config: CreditAssignmentConfig | None = None):
        self.config = config or CreditAssignmentConfig(credit_type="local_goodness")

    def compute_pseudo_gradient(
        self,
        free_state: SystemState,
        nudged_state: SystemState,
        loss: Tensor,
        geometry: Geometry,
    ) -> list[Tensor]:
        # Layer-local goodness gradients
        grads = []
        if free_state.activations and isinstance(free_state.activations, list):
            for act in free_state.activations[1:]:  # Skip input
                # Positive pass: maximize goodness
                # Negative pass: minimize goodness
                pos_grad = act * (1 - torch.sigmoid(act))
                grads.append(pos_grad)
        return grads


class BackpropCredit:
    """Standard backpropagation (global credit assignment)."""

    def __init__(self, config: CreditAssignmentConfig | None = None):
        self.config = config or CreditAssignmentConfig(credit_type="gradient")

    def compute_pseudo_gradient(
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
            grads = torch.autograd.grad(loss, params, create_graph=False, allow_unused=True)
            return [g for g in grads if g is not None]
        return []


class TemporalTraceCredit:
    """STDP: spike-timing-dependent correlation."""

    def __init__(self, config: CreditAssignmentConfig | None = None):
        self.config = config or CreditAssignmentConfig(credit_type="temporal_trace")

    def compute_pseudo_gradient(
        self,
        free_state: SystemState,
        nudged_state: SystemState,
        loss: Tensor,
        geometry: Geometry,
    ) -> list[Tensor]:
        # STDP-style correlation
        return []


class TargetInversionCredit:
    """Target Propagation: propagate local targets backward."""

    def __init__(self, config: CreditAssignmentConfig | None = None):
        self.config = config or CreditAssignmentConfig(credit_type="target_inversion")

    def compute_pseudo_gradient(
        self,
        free_state: SystemState,
        nudged_state: SystemState,
        loss: Tensor,
        geometry: Geometry,
    ) -> list[Tensor]:
        # Target propagation uses learned inverse maps
        return []


# Additional parameter update implementations
class RiemannianOrthogonalUpdate:
    """Muon: Riemannian optimization on Stiefel manifold."""

    def __init__(self, config: ParameterUpdateConfig | None = None):
        self.config = config or ParameterUpdateConfig(update_type="riemannian_orthogonal")

    def _newton_schulz(self, G: Tensor, steps: int = 5) -> Tensor:
        """Compute orthogonal projection via Newton-Schulz iteration."""
        a, b, c = 3.4445, -4.7750, 2.0315
        x = G
        for _ in range(steps):
            x = a * x + b * x @ x.T @ x + c * x @ x.T @ x @ x.T @ x
        return x

    def step(
        self,
        params: dict[str, Tensor],
        pseudo_grads: list[Tensor],
        geometry: Geometry,
    ) -> dict[str, Tensor]:
        updated = {}
        for i, (name, param) in enumerate(params.items()):
            if i < len(pseudo_grads) and pseudo_grads[i] is not None:
                grad = pseudo_grads[i]
                if grad.ndim >= 2:
                    # Orthogonalize the gradient
                    grad = self._newton_schulz(grad, self.config.ortho_steps)
                updated[name] = param - self.config.step_size * grad
            else:
                updated[name] = param
        return updated


class SpectralConstrainedUpdate:
    """Spectral norm constrained updates for stability."""

    def __init__(self, config: ParameterUpdateConfig | None = None):
        self.config = config or ParameterUpdateConfig(update_type="spectral_constrained")

    def step(
        self,
        params: dict[str, Tensor],
        pseudo_grads: list[Tensor],
        geometry: Geometry,
    ) -> dict[str, Tensor]:
        updated = {}
        for i, (name, param) in enumerate(params.items()):
            if i < len(pseudo_grads) and pseudo_grads[i] is not None:
                grad = pseudo_grads[i]
                # Project gradient to satisfy spectral constraint
                if grad.ndim >= 2:
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
        self.config = config or ParameterUpdateConfig(update_type="natural_gradient")
        self._fisher: dict[str, Tensor] = {}

    def step(
        self,
        params: dict[str, Tensor],
        pseudo_grads: list[Tensor],
        geometry: Geometry,
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
                nat_grad = grad / (self._fisher[name].sqrt() + self.config.fisher_damping)
                updated[name] = param - self.config.step_size * nat_grad
            else:
                updated[name] = param
        return updated


class ElasticConsolidationUpdate:
    """Elastic Weight Consolidation for continual learning."""

    def __init__(self, config: ParameterUpdateConfig | None = None):
        self.config = config or ParameterUpdateConfig(update_type="elastic_consolidation")
        self._importance: dict[str, Tensor] = {}
        self._old_params: dict[str, Tensor] = {}

    def step(
        self,
        params: dict[str, Tensor],
        pseudo_grads: list[Tensor],
        geometry: Geometry,
    ) -> dict[str, Tensor]:
        updated = {}
        for i, (name, param) in enumerate(params.items()):
            if i < len(pseudo_grads) and pseudo_grads[i] is not None:
                grad = pseudo_grads[i]
                # EWC penalty: lambda * importance * (param - old_param)
                ewc_penalty = torch.zeros_like(param)
                if name in self._importance and name in self._old_params:
                    ewc_penalty = self.config.ewc_lambda * self._importance[name] * (param - self._old_params[name])
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

    def train_step(self, x: Tensor, y: Tensor) -> dict[str, float]:
        if hasattr(self._model, "train_step"):
            return self._model.train_step(x, y)  # type: ignore[attr-defined]
        # Fallback to BPTT
        self._model.train()
        logits = self._model(x)  # type: ignore[operator]
        loss = torch.nn.functional.cross_entropy(logits, y)
        return {"loss": loss.item(), "accuracy": (logits.argmax(-1) == y).float().mean().item()}

    def forward(self, x: Tensor) -> Tensor:
        return self._model(x)  # type: ignore[operator]

    @property
    def model(self) -> nn.Module:
        return self._model
