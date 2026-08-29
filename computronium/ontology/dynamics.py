"""Layer 3: StateDynamics — Forward Evolution & Settling."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import math
import torch
from torch import Tensor
from torch import Tensor

if TYPE_CHECKING:
    from computronium.ontology.geometry import Geometry
    from computronium.ontology.substrate import Substrate
    from computronium.state import CompositeState


# ============================================================
# StateDynamics Configuration
# ============================================================


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


# ============================================================
# StateDynamics Protocol
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
        state: CompositeState,
        geometry: Geometry,
        substrate: Substrate,
        target: Tensor | None = None,
    ) -> CompositeState:
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
    def compute_energy(self, state: CompositeState, geometry: Geometry) -> Tensor:
        """Compute the energy of the current state.

        For energy-based dynamics (EqProp, Hopfield, PC), this is the
        Lyapunov function. For non-energy dynamics, returns a proxy.
        """
        ...


# ============================================================
# EnergyMinimizationDynamics: Full EqProp Implementation
# ============================================================


def _compute_hopfield_energy(all_acts: list[Tensor], geometry: Geometry) -> Tensor:
    """Compute Hopfield energy for the current state.

    E = 0.5 * sum(h_i^2) - sum_{i,j} W_{ij} h_i h_j - sum_i b_i h_i
    For ReLU networks with symmetric weights approximation.
    """
    if not all_acts or len(all_acts) < 2:
        return torch.tensor(0.0, device=all_acts[0].device if all_acts else "cpu")

    # Use hidden + output layers (skip input)
    acts = all_acts[1:]  # [hidden1, hidden2, ..., output]
    device = acts[0].device
    total_energy = torch.tensor(0.0, device=device)

    # Extract weight matrices from geometry params
    params = geometry.params
    weight_names = [
        n
        for n in params
        if "weight" in n and params[n].ndim == 2 and not n.startswith("recurrent")
    ]
    # Sort by layer index (e.g., "0.weight" -> 0, "2.weight" -> 2)
    weight_names.sort(
        key=lambda x: (
            int(x.split("_")[1]) if "_" in x and x.split("_")[1].isdigit() else 0
        )
    )

    if not weight_names:
        return torch.tensor(0.0, device=device)

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
            # For linear network (no hidden layers), h_prev should be input
            if num_hidden == 0:
                h_prev = all_acts[0]
            else:
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
        n
        for n in params
        if "bias" in n and params[n].ndim == 1 and not n.startswith("recurrent")
    ]
    bias_names.sort(
        key=lambda x: (
            int(x.split("_")[1]) if "_" in x and x.split("_")[1].isdigit() else 0
        )
    )
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
    return total_energy / batch_size


def _settle_step(
    all_acts: list[Tensor],
    geometry: Geometry,
    beta: float,
    target: Tensor | None,
    velocity: list[Tensor] | None,
    momentum: float,
    step_size: float,
    config: StateDynamicsConfig,
) -> tuple[list[Tensor], list[Tensor] | None]:
    """One settling step, pure function for gradient checkpointing."""
    num_hidden = len(all_acts) - 2
    new_acts = [all_acts[0]]  # Input layer is fixed
    new_velocity: list[Tensor] | None = (
        [] if momentum > 0 and velocity is not None else None
    )

    layers = _layer_stack(geometry)
    if layers is None:
        raise TypeError("Energy-based settling requires a layered geometry")
    recurrent_weight = _recurrent_weight(geometry)
    linears = [m for m in layers if isinstance(m, torch.nn.Linear)]
    activations = [m for m in layers if not isinstance(m, torch.nn.Linear)]

    # Update each hidden layer
    for i in range(num_hidden):
        pre = linears[i](all_acts[i])  # Bottom-up from previous layer

        # Recurrent term (for RecurrentGeometry, last hidden layer only)
        if recurrent_weight is not None and i == num_hidden - 1:
            pre = pre + all_acts[i + 1] @ recurrent_weight.T

        # Top-down drive from layer above
        top_down = all_acts[i + 2] @ linears[i + 1].weight

        total = pre + top_down

        # Apply momentum (heavy-ball)
        if velocity is not None and new_velocity is not None:
            total = momentum * velocity[i] + total
            new_velocity.append(total.detach().clone())

        # Apply activation (ReLU), then relax toward the target state
        # (h_{t+1} = (1-η)·h_t + η·f(...) keeps the bidirectional settle
        # loop contractive for gain ≈ 1 topologies)
        target_h = activations[i](total) if i < len(activations) else total
        h_new = all_acts[i + 1] + step_size * (target_h - all_acts[i + 1])

        new_acts.append(h_new)

    # Output layer
    out_layer = linears[-1] if linears else None
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


def _layer_stack(geometry: Geometry) -> torch.nn.ModuleList | None:
    """Return the geometry's ordered module stack if it is layer-based."""
    layers = getattr(geometry, "_layers", None)
    return layers if isinstance(layers, torch.nn.ModuleList) else None


def _recurrent_weight(geometry: Geometry) -> Tensor | None:
    """Return the geometry's recurrent weight matrix if present."""
    weight = getattr(geometry, "_recurrent_weight", None)
    return weight if isinstance(weight, Tensor) else None


class EnergyMinimizationDynamics:
    """Energy-based settling (Equilibrium Propagation, Hopfield, CHL).

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

    def settle(
        self,
        state: CompositeState,
        geometry: Geometry,
        substrate: Substrate,
        target: Tensor | None = None,
    ) -> CompositeState:
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
            self._free_energy_history.append(
                _compute_hopfield_energy(all_acts, geometry).item()
            )
        else:
            self._free_energy_history = None

        beta = self.config.beta if target is not None else 0.0
        momentum = self.config.momentum

        # Auto-detect gradient checkpointing: never on CPU (pure overhead),
        # on GPU enable only if explicitly set OR if VRAM would be exceeded.
        device = all_acts[0].device
        use_checkpointing = self.config.gradient_checkpointing
        if use_checkpointing and device.type == "cpu":
            # Never checkpoint on CPU - it's pure compute overhead with no memory benefit
            use_checkpointing = False
        elif use_checkpointing is False and device.type == "cuda":
            # Auto-enable if model + activations would exceed ~80% of available VRAM
            try:
                free_vram, _ = torch.cuda.mem_get_info(device)
                # Estimate: params + optimizer state + activations (max_steps * layers * batch * hidden)
                total_params = sum(
                    p.numel() for p in geometry.params.values() if p.requires_grad
                )
                hidden_size = (
                    all_acts[1].numel() // all_acts[1].shape[0]
                )  # per-sample hidden dim
                batch_size = all_acts[0].shape[0]
                # Rough estimate: activations for all settle steps (each step has ~layers activations)
                est_activation_mem = (
                    self.config.max_steps
                    * (len(_layer_stack(geometry) or ()))
                    * batch_size
                    * hidden_size
                    * 4  # fp32 bytes
                )
                est_total = (
                    total_params * 4 * 3
                ) + est_activation_mem  # params + optimizer + activations
                if est_total > free_vram * 0.8:
                    use_checkpointing = True
            except Exception:
                pass  # Fall back to config value

        # Use gradient checkpointing if enabled
        if use_checkpointing:
            from torch.utils import checkpoint

            for _step in range(self.config.max_steps):
                prev_output = all_acts[-1].detach()
                # Checkpoint the step function
                all_acts, self._velocity = checkpoint.checkpoint(
                    _settle_step,
                    all_acts,
                    geometry,
                    beta,
                    target,
                    self._velocity,
                    momentum,
                    self.config.step_size,
                    self.config,
                    use_reentrant=False,
                )

                # Track free energy if enabled
                if self._free_energy_history is not None:
                    self._free_energy_history.append(
                        _compute_hopfield_energy(all_acts, geometry).item()
                    )

                # Check convergence (can't checkpoint this as it's not differentiable)
                if _step >= self.config.convergence_start:
                    delta = torch.dist(all_acts[-1], prev_output, p=float("inf")).item()
                    if delta < self.config.convergence_threshold:
                        break
        else:
            # Original non-checkpointed path
            for step in range(self.config.max_steps):
                new_acts, new_velocity = _settle_step(
                    all_acts,
                    geometry,
                    beta,
                    target,
                    self._velocity,
                    momentum,
                    self.config.step_size,
                    self.config,
                )
                if new_velocity is not None:
                    self._velocity = new_velocity

                # Track free energy if enabled
                if self._free_energy_history is not None:
                    self._free_energy_history.append(
                        _compute_hopfield_energy(new_acts, geometry).item()
                    )

                # Check convergence
                if step >= self.config.convergence_start:
                    delta = torch.dist(
                        new_acts[-1], all_acts[-1], p=float("inf")
                    ).item()
                    if delta < self.config.convergence_threshold:
                        all_acts = new_acts
                        break
                all_acts = new_acts

        if target is None:
            state.free_state = all_acts
        else:
            state.nudged_state = all_acts
        state.activations = all_acts
        return state

    def compute_energy(self, state: CompositeState, geometry: Geometry) -> Tensor:
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
            energy_val = _compute_hopfield_energy(acts, geometry)
            return energy_val
        return (acts**2).mean()

    def get_free_energy_history(self) -> list[float] | None:
        """Return the free energy history tracked during settling.

        Returns None if tracking was not enabled (track_free_energy_per_iter=False).
        """
        return self._free_energy_history


# ============================================================
# Other Default/Reference StateDynamics Implementations
# ============================================================


class PredictiveSettlingDynamics:
    """Predictive coding settling (Rao & Ballard, Whittington & Bogacz).

    Minimizes prediction error via iterative inference.
    """

    def __init__(self, config: StateDynamicsConfig | None = None):
        self.config = config or StateDynamicsConfig.predictive_settling()

    def settle(
        self,
        state: CompositeState,
        geometry: Geometry,
        substrate: Substrate,
        target: Tensor | None = None,
    ) -> CompositeState:
        # Simplified predictive settling
        if state.x is not None:
            x = state.x
        else:
            raise ValueError("State must contain input 'x'")

        h = substrate.initial_state(x)
        op = substrate.get_forward_operator()

        for step in range(self.config.max_steps):
            # Predictive coding update
            prediction = geometry.route(h)
            error = x - prediction
            h = h + self.config.step_size * op(
                error, geometry.params.get("weight", torch.eye(h.shape[-1]))
            )

        new_activity = {**state.activity, "x": x, "output": h}
        new_state = CompositeState(
            activity=new_activity,
            plastic=state.plastic,
            substrate=state.substrate,
        )

        if target is None:
            new_state.free_state = [h]
        else:
            new_state.nudged_state = [h]

        return new_state

    def compute_energy(self, state: CompositeState, geometry: Geometry) -> Tensor:
        if state.activations is not None:
            acts = (
                state.activations
                if isinstance(state.activations, list)
                else [state.activations]
            )
            h = acts[-1] if acts else torch.zeros(1)
        else:
            h = state.activity.get("output", torch.zeros(1))
        return h.pow(2).sum()


class SpikeIntegrationDynamics:
    """Spiking neuron integration (LIF, AdEx)."""

    def __init__(self, config: StateDynamicsConfig | None = None):
        self.config = config or StateDynamicsConfig.spike_integration()

    def settle(
        self,
        state: CompositeState,
        geometry: Geometry,
        substrate: Substrate,
        target: Tensor | None = None,
    ) -> CompositeState:
        if state.x is not None:
            x = state.x
        else:
            raise ValueError("State must contain input 'x'")

        h = substrate.initial_state(x)
        # op = substrate.get_forward_operator()  # Unused in this simplified implementation

        for _step in range(self.config.max_steps):
            # LIF dynamics: tau * dh/dt = -h + I_syn
            I_syn = geometry.route(h)
            h = h + self.config.step_size * (-h + I_syn)

        new_activity = {**state.activity, "x": x, "output": h}
        new_state = CompositeState(
            activity=new_activity,
            plastic=state.plastic,
            substrate=state.substrate,
        )

        if target is None:
            new_state.free_state = [h]
        else:
            new_state.nudged_state = [h]

        return new_state

    def compute_energy(self, state: CompositeState, geometry: Geometry) -> Tensor:
        if state.activations is not None:
            acts = (
                state.activations
                if isinstance(state.activations, list)
                else [state.activations]
            )
            h = acts[-1] if acts else torch.zeros(1)
        else:
            h = state.activity.get("output", torch.zeros(1))
        return h.pow(2).sum()


class InstantaneousDynamics:
    """Single-pass feedforward (Backprop, Forward-Forward)."""

    def __init__(self, config: StateDynamicsConfig | None = None):
        self.config = config or StateDynamicsConfig.instantaneous()

    def settle(
        self,
        state: CompositeState,
        geometry: Geometry,
        substrate: Substrate,
        target: Tensor | None = None,
    ) -> CompositeState:
        # Single forward pass - no settling
        if state.x is not None:
            acts = geometry.forward_with_intermediates(state.x, substrate)
        else:
            acts = state.activations if state.activations is not None else []
        if target is None:
            state.free_state = acts
        else:
            state.nudged_state = acts
        state.activations = acts
        return state

    def compute_energy(self, state: CompositeState, geometry: Geometry) -> Tensor:
        # Proxy: negative log-likelihood for instantaneous pass
        if state.loss is not None:
            return torch.as_tensor(state.loss)
        return torch.tensor(0.0)


class DiffusionDynamics:
    """Langevin/diffusion dynamics for continuous-time settling."""

    def __init__(self, config: StateDynamicsConfig | None = None):
        self.config = config or StateDynamicsConfig.diffusion()

    def settle(
        self,
        state: CompositeState,
        geometry: Geometry,
        substrate: Substrate,
        target: Tensor | None = None,
    ) -> CompositeState:
        if state.x is not None:
            x = state.x
        else:
            raise ValueError("State must contain input 'x'")

        h = substrate.initial_state(x)
        # op = substrate.get_forward_operator()  # Unused in this simplified implementation

        for _step in range(self.config.max_steps):
            # Langevin dynamics: dh = -∇E dt + sqrt(2*D) dW
            energy_grad = torch.autograd.grad(
                self.compute_energy_from_state(h, geometry, substrate), h
            )[0]
            noise = torch.randn_like(h) * math.sqrt(2 * self.config.step_size)
            with torch.no_grad():
                h = h - self.config.step_size * energy_grad + noise

        new_activity = {**state.activity, "x": x, "output": h}
        new_state = CompositeState(
            activity=new_activity,
            plastic=state.plastic,
            substrate=state.substrate,
        )

        if target is None:
            new_state.free_state = [h]
        else:
            new_state.nudged_state = [h]

        return new_state

    def compute_energy_from_state(
        self, h: Tensor, geometry: Geometry, substrate: Substrate
    ) -> Tensor:
        energy = h.pow(2).sum()
        return energy

    def compute_energy(self, state: CompositeState, geometry: Geometry) -> Tensor:
        if state.activations is not None:
            acts = (
                state.activations
                if isinstance(state.activations, list)
                else [state.activations]
            )
            h = acts[-1] if acts else torch.zeros(1)
        else:
            h = state.activity.get("output", torch.zeros(1))
        # Use a default substrate for energy computation if not available
        substrate_obj = state.substrate.get("substrate")
        if substrate_obj is None:
            from computronium.ontology.substrate import (
                DigitalSubstrate,
                SubstrateConfig,
            )

            substrate = DigitalSubstrate(SubstrateConfig.digital())
        else:
            substrate = substrate_obj  # type: ignore[assignment]
        return self.compute_energy_from_state(h, geometry, substrate)


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
        state: CompositeState,
        geometry: Geometry,
        substrate: Substrate,
        target: Tensor | None = None,
    ) -> CompositeState:
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

        new_activity = {**state.activity, "output": h}
        new_state = CompositeState(
            activity=new_activity,
            plastic=state.plastic,
            substrate=state.substrate,
        )

        if target is None:
            new_state.free_state = [h]
        else:
            new_state.nudged_state = [h]
        new_state.activations = [h]

        return new_state

    def compute_energy(self, state: CompositeState, geometry: Geometry) -> Tensor:
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
