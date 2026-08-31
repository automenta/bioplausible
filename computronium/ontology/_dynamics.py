"""Layer 3: StateDynamics — Forward Evolution & Settling."""

from __future__ import annotations

import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
from torch import Tensor

if TYPE_CHECKING:
    from computronium.ontology.geometry import Geometry
    from computronium.ontology.substrate import Substrate
    from computronium.state import CompositeState

from computronium.ontology._settle_kernel import (
    SubstrateSettleKernel,
    extract_layered_params,
)
from computronium.ontology.geometry import _layer_stack

# ============================================================
# State type detection helpers (duck typing for SystemState + CompositeState)
# ============================================================


def _is_composite_state(state: object) -> bool:
    """Check if state is a CompositeState (has activity/plastic/substrate dicts)."""
    return hasattr(state, "activity") and isinstance(
        getattr(state, "activity", None), dict
    )


def _get_state_x(state: object) -> Tensor | None:
    """Get input x from either SystemState or CompositeState."""
    return getattr(state, "x", None)


def _get_state_activations(state: object) -> list[Tensor] | Tensor | None:
    """Get activations from either SystemState or CompositeState."""
    return getattr(state, "activations", None)


def _get_state_free_state(state: object) -> list[Tensor] | Tensor | None:
    """Get free_state from either SystemState or CompositeState."""
    return getattr(state, "free_state", None)


def _get_state_nudged_state(state: object) -> list[Tensor] | Tensor | None:
    """Get nudged_state from either SystemState or CompositeState."""
    return getattr(state, "nudged_state", None)


def _get_state_loss(state: object) -> Tensor | float | None:
    """Get loss from either SystemState or CompositeState."""
    return getattr(state, "loss", None)


def _get_state_metrics(state: object) -> dict[str, float] | None:
    """Get metrics from either SystemState or CompositeState."""
    return getattr(state, "metrics", None)


def _set_state_x(state: object, value: Tensor | None) -> None:
    """Set input x on either SystemState or CompositeState."""
    if value is None:
        if hasattr(state, "activity") and isinstance(state.activity, dict):
            state.activity.pop("x", None)
        else:
            state.x = None
    elif hasattr(state, "activity") and isinstance(state.activity, dict):
        state.activity["x"] = value
    else:
        state.x = value


def _set_state_activations(state: object, value: list[Tensor] | Tensor | None) -> None:
    """Set activations on either SystemState or CompositeState."""
    if value is None:
        if hasattr(state, "activity") and isinstance(state.activity, dict):
            state.activity.pop("activations", None)
        else:
            state.activations = None
    elif hasattr(state, "activity") and isinstance(state.activity, dict):
        state.activity["activations"] = value
    else:
        state.activations = value


def _set_state_free_state(state: object, value: list[Tensor] | Tensor | None) -> None:
    """Set free_state on either SystemState or CompositeState."""
    if value is None:
        if hasattr(state, "activity") and isinstance(state.activity, dict):
            state.activity.pop("free_state", None)
        else:
            state.free_state = None
    elif hasattr(state, "activity") and isinstance(state.activity, dict):
        state.activity["free_state"] = value
    else:
        state.free_state = value


def _set_state_nudged_state(state: object, value: list[Tensor] | Tensor | None) -> None:
    """Set nudged_state on either SystemState or CompositeState."""
    if value is None:
        if hasattr(state, "activity") and isinstance(state.activity, dict):
            state.activity.pop("nudged_state", None)
        else:
            state.nudged_state = None
    elif hasattr(state, "activity") and isinstance(state.activity, dict):
        state.activity["nudged_state"] = value
    else:
        state.nudged_state = value


def _set_state_loss(state: object, value: Tensor | float | None) -> None:
    """Set loss on either SystemState or CompositeState."""
    if value is None:
        if hasattr(state, "activity") and isinstance(state.activity, dict):
            state.activity.pop("loss", None)
        else:
            state.loss = None
    elif hasattr(state, "activity") and isinstance(state.activity, dict):
        state.activity["loss"] = value
    else:
        state.loss = value


def _set_state_metrics(state: object, value: dict[str, float] | None) -> None:
    """Set metrics on either SystemState or CompositeState."""
    if value is None:
        if hasattr(state, "activity") and isinstance(state.activity, dict):
            state.activity.pop("metrics", None)
        else:
            state.metrics = None
    elif hasattr(state, "activity") and isinstance(state.activity, dict):
        state.activity["metrics"] = value
    else:
        state.metrics = value


def _get_state_activity(state: object) -> dict | None:
    """Get activity dict from CompositeState, or None for SystemState."""
    if hasattr(state, "activity") and isinstance(state.activity, dict):
        return state.activity
    return None


def _create_output_state(
    state: object,
    *,
    x: Tensor | None = None,
    output: Tensor | None = None,
    free_state: list[Tensor] | Tensor | None = None,
    nudged_state: list[Tensor] | Tensor | None = None,
    activations: list[Tensor] | Tensor | None = None,
    spike_counts: list[Tensor] | None = None,
) -> object:
    """Create a new state of the same type with updated fields."""
    if _is_composite_state(state):
        from computronium.state import CompositeState

        activity = dict(state.activity)
        if x is not None:
            activity["x"] = x
        if output is not None:
            activity["output"] = output
        if free_state is not None:
            activity["free_state"] = free_state
        if nudged_state is not None:
            activity["nudged_state"] = nudged_state
        if activations is not None:
            activity["activations"] = activations
        if spike_counts is not None:
            activity["spike_counts"] = spike_counts
        return CompositeState(
            activity=activity,
            plastic=state.plastic,
            substrate=state.substrate,
        )
    else:
        # SystemState - create new instance with updated fields
        from computronium.ontology.system import SystemState

        return SystemState(
            x=x if x is not None else getattr(state, "x", None),
            y=getattr(state, "y", None),
            activations=activations
            if activations is not None
            else getattr(state, "activations", None),
            free_state=free_state
            if free_state is not None
            else getattr(state, "free_state", None),
            nudged_state=nudged_state
            if nudged_state is not None
            else getattr(state, "nudged_state", None),
            pseudo_gradients=getattr(state, "pseudo_gradients", None),
            energy=getattr(state, "energy", None),
            loss=getattr(state, "loss", None),
            metrics=dict(getattr(state, "metrics", {}) or {}),
            spike_counts=spike_counts
            if spike_counts is not None
            else getattr(state, "spike_counts", None),
        )


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

        # Extract layered params and construct substrate-native settle kernel
        params = extract_layered_params(geometry)
        if params is None:
            raise TypeError("Energy-based settling requires a layered geometry")
        kernel = SubstrateSettleKernel(
            substrate=substrate,
            params=params,
            step_size=self.config.step_size,
            momentum=self.config.momentum,
        )

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

        # Kernel step function for checkpointing
        def _kernel_step(
            acts: list[Tensor],
            beta_: float,
            target_: Tensor | None,
            velocity_: list[Tensor] | None,
        ) -> tuple[list[Tensor], list[Tensor] | None]:
            return kernel.step(acts, beta_, target_, velocity_)

        # Use gradient checkpointing if enabled
        if use_checkpointing:
            from torch.utils import checkpoint

            for _step in range(self.config.max_steps):
                prev_output = all_acts[-1].detach()
                # Checkpoint the kernel step function
                all_acts, self._velocity = checkpoint.checkpoint(
                    _kernel_step,
                    all_acts,
                    beta,
                    target,
                    self._velocity,
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
            # Non-checkpointed path
            for step in range(self.config.max_steps):
                new_acts, new_velocity = kernel.step(
                    all_acts, beta, target, self._velocity
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
        # Predictive coding settling: minimize prediction error
        x = _get_state_x(state)
        if x is None:
            raise ValueError("State must contain input 'x'")

        # Initialize free energy history if tracking enabled
        if self.config.track_free_energy_per_iter:
            self._free_energy_history: list[float] | None = []
        else:
            self._free_energy_history = None

        # For TileGeometry, the forward pass already does the complete tile mesh
        # propagation. Use forward_with_intermediates to get all layer activations.
        if hasattr(geometry, "_graph"):
            # TileGeometry or similar mesh-based geometry
            acts = geometry.forward_with_intermediates(x, substrate)
            if not acts:
                return state
            # Track initial free energy
            if self.config.track_free_energy_per_iter:
                # Free energy in predictive coding = sum of squared prediction errors
                fe = sum((a - geometry.route(a)).pow(2).sum().item() for a in acts[:-1])
                self._free_energy_history.append(fe)
            # Last activation is the output
            h = acts[-1]
            new_state = _create_output_state(
                state,
                x=x,
                output=h,
                free_state=acts if target is None else None,
                nudged_state=acts if target is not None else None,
                activations=acts,
            )
            return new_state

        # Standard predictive coding settling for recurrent/feedforward geometries
        h = substrate.initial_state(x)
        op = substrate.get_forward_operator()

        for _step in range(self.config.max_steps):
            # Predictive coding update
            prediction = geometry.route(h)
            # Ensure prediction matches input dimension for shape-safe error computation
            if prediction.shape[-1] != h.shape[-1]:
                if prediction.shape[-1] >= h.shape[-1]:
                    prediction = prediction[..., : h.shape[-1]]
                else:
                    pad_size = h.shape[-1] - prediction.shape[-1]
                    prediction = torch.nn.functional.pad(prediction, (0, pad_size)).to(
                        prediction.device
                    )
            error = x - prediction
            h = h + self.config.step_size * op(
                error,
                geometry.params.get("weight", torch.eye(h.shape[-1], device=h.device)),
            )
            # Track free energy per iteration
            if self.config.track_free_energy_per_iter:
                fe = error.pow(2).sum().item()
                self._free_energy_history.append(fe)

        new_state = _create_output_state(
            state,
            x=x,
            output=h,
            free_state=[h] if target is None else None,
            nudged_state=[h] if target is not None else None,
            activations=[h],
        )

        return new_state

    def get_free_energy_history(self) -> list[float] | None:
        """Return the free energy history tracked during settling.

        Returns None if tracking was not enabled (track_free_energy_per_iter=False).
        """
        return self._free_energy_history

    def compute_energy(self, state: CompositeState, geometry: Geometry) -> Tensor:
        acts = _get_state_activations(state)
        if acts is not None:
            acts = acts if isinstance(acts, list) else [acts]
            h = acts[-1] if acts else torch.zeros(1)
        else:
            activity = _get_state_activity(state)
            h = activity.get("output", torch.zeros(1)) if activity else torch.zeros(1)
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
        x = _get_state_x(state)
        if x is None:
            raise ValueError("State must contain input 'x'")

        h = substrate.initial_state(x)

        spike_counts = []
        threshold = 1.0  # Spike threshold

        for _step in range(self.config.max_steps):
            # LIF dynamics: tau * dh/dt = -h + I_syn
            I_syn = geometry.route(h)
            h = h + self.config.step_size * (-h + I_syn)
            # Count spikes: neurons where membrane potential crosses threshold
            spikes = (h > threshold).float()
            spike_counts.append(spikes.sum(dim=1))  # [batch]
            # Reset spiking neurons
            h = torch.where(h > threshold, torch.zeros_like(h), h)

        new_state = _create_output_state(
            state,
            x=x,
            output=h,
            free_state=[h] if target is None else None,
            nudged_state=[h] if target is not None else None,
            activations=[h],
            spike_counts=spike_counts,
        )

        return new_state

    def compute_energy(self, state: CompositeState, geometry: Geometry) -> Tensor:
        acts = _get_state_activations(state)
        if acts is not None:
            acts = acts if isinstance(acts, list) else [acts]
            h = acts[-1] if acts else torch.zeros(1)
        else:
            activity = _get_state_activity(state)
            h = activity.get("output", torch.zeros(1)) if activity else torch.zeros(1)
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
        x = _get_state_x(state)
        if x is None:
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

        new_state = _create_output_state(
            state,
            x=x,
            output=h,
            free_state=[h] if target is None else None,
            nudged_state=[h] if target is not None else None,
            activations=[h],
        )

        return new_state

    def compute_energy_from_state(
        self, h: Tensor, geometry: Geometry, substrate: Substrate
    ) -> Tensor:
        energy = h.pow(2).sum()
        return energy

    def compute_energy(self, state: CompositeState, geometry: Geometry) -> Tensor:
        acts = _get_state_activations(state)
        if acts is not None:
            acts = acts if isinstance(acts, list) else [acts]
            h = acts[-1] if acts else torch.zeros(1)
        else:
            activity = _get_state_activity(state)
            h = activity.get("output", torch.zeros(1)) if activity else torch.zeros(1)
        # Use a default substrate for energy computation if not available
        substrate_obj = (
            state.substrate.get("substrate") if _is_composite_state(state) else None
        )
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
        h = _get_state_activations(state)
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

        new_state = _create_output_state(
            state,
            output=h,
            free_state=[h] if target is None else None,
            nudged_state=[h] if target is not None else None,
            activations=[h],
        )

        return new_state

    def compute_energy(self, state: CompositeState, geometry: Geometry) -> Tensor:
        """Compute energy using cached activations if available."""
        acts = _get_state_free_state(state)
        if acts is None:
            acts = _get_state_activations(state)
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
