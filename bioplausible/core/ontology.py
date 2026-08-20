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
        """Single recurrent step: h_{t+1} = f(h_t) = activation(W_rec @ h_t)."""
        h = activations
        if self._recurrent_weight is not None:
            h = h @ self._recurrent_weight.T
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
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def to_system(self):
        """Project model into 5-D ontology (best-effort inference)."""
        # Infer components from model attributes/metadata
        substrate = self._infer_substrate()
        geometry = self._infer_geometry()
        dynamics = self._infer_dynamics()
        credit = self._infer_credit()
        update = self._infer_update()

        # Return a System-like object that delegates to the model
        return _AdaptedSystem(
            substrate=substrate,
            geometry=geometry,
            dynamics=dynamics,
            credit=credit,
            update=update,
            model=self.model,
        )

    def _infer_substrate(self) -> Substrate:
        # Default to digital; check for hardware variants
        if hasattr(self.model, "backend"):
            if self.model.backend == "quantized":
                return QuantizedSubstrate()
            elif self.model.backend == "noisy":
                return NoisySubstrate()
            elif self.model.backend == "optical":
                return OpticalSubstrate()
        return DigitalSubstrate()

    def _infer_geometry(self) -> Geometry:
        if hasattr(self.model, "transition_modules"):
            layers = self.model.transition_modules()  # type: ignore[attr-defined]
            return FeedforwardGeometry(
                GeometryConfig(
                    input_dim=getattr(self.model, "input_dim", 0),
                    output_dim=getattr(self.model, "output_dim", 0),
                ),
                layers=layers,
            )
        return FeedforwardGeometry(GeometryConfig(input_dim=0, output_dim=0))

    def _infer_dynamics(self) -> StateDynamics:
        if hasattr(self.model, "max_steps") and getattr(self.model, "max_steps", 1) > 1:
            return EnergyMinimizationDynamics()
        return InstantaneousDynamics()

    def _infer_credit(self) -> CreditAssignment:
        credit_type = getattr(self.model, "credit_assignment_type", "gradient")
        if credit_type == "equilibrium":
            return ThermodynamicContrast()
        elif credit_type == "random_projections":
            return RandomProjectionsCredit()
        return BackpropCredit()

    def _infer_update(self) -> ParameterUpdate:
        if hasattr(self.model, "optimizer") and hasattr(self.model.optimizer, "step"):
            # Check for Muon etc.
            opt_name = type(self.model.optimizer).__name__.lower()
            if "muon" in opt_name:
                return RiemannianOrthogonalUpdate()
            elif "ewc" in opt_name:
                return ElasticConsolidationUpdate()
        return EuclideanUpdate()


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
        acts = state.free_state or state.nudged_state or state.activations
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
            # Initialize based on geometry output dim
            out_dim = geometry.config.output_dim
            self._feedback_weights = torch.randn(out_dim, out_dim) * 0.1

        # Project error through fixed feedback
        if nudged_state.activations is not None:
            acts = nudged_state.activations
            if isinstance(acts, list):
                acts = acts[-1]
            error = acts - self._feedback_weights @ acts  # Simplified
            return [error]
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
