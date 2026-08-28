"""Layer 3: StateDynamics — Forward Evolution & Settling."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
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
# Default/Reference StateDynamics Implementations
# ============================================================


def _learnable_weight_names(params: dict[str, Tensor]) -> list[str]:
    """Parameter names that receive pseudo-gradients (2-D weight matrices)."""
    return [n for n, p in params.items() if "weight" in n and p.ndim == 2]


class EnergyMinimizationDynamics:
    """Energy-based settling (Equilibrium Propagation, Hopfield, PC).

    Minimizes an energy function via gradient descent on state variables.
    """

    def __init__(self, config: StateDynamicsConfig | None = None):
        self.config = config or StateDynamicsConfig.energy_minimization()

    def settle(
        self,
        state: CompositeState,
        geometry: Geometry,
        substrate: Substrate,
        target: Tensor | None = None,
    ) -> CompositeState:
        """Settle to fixed point via gradient descent on energy."""
        # Initialize state
        if state.x is not None:
            x = state.x
        else:
            raise ValueError("State must contain input 'x'")

        # Get initial state from substrate
        h = substrate.initial_state(x)

        # Run settling iterations
        for step in range(self.config.max_steps):
            # Compute energy gradient
            h.requires_grad_(True)
            energy = self.compute_energy_from_state(h, geometry, substrate)
            grad = torch.autograd.grad(energy, h, create_graph=False)[0]

            # Gradient descent step
            with torch.no_grad():
                if self.config.momentum > 0:
                    if not hasattr(self, "_momentum_buffer"):
                        self._momentum_buffer = torch.zeros_like(h)
                    self._momentum_buffer.mul_(self.config.momentum).add_(grad)
                    h = h - self.config.step_size * self._momentum_buffer
                else:
                    h = h - self.config.step_size * grad

            # Check convergence
            if step >= self.config.convergence_start:
                if grad.norm() < self.config.convergence_threshold:
                    break

        # Store settled state
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
        """Compute energy from current state."""
        # Energy = sum of squared errors + regularization
        energy = h.pow(2).sum()
        return energy

    def compute_energy(self, state: CompositeState, geometry: Geometry) -> Tensor:
        """Compute the energy of the current state."""
        if state.activations is not None:
            acts = state.activations if isinstance(state.activations, list) else [state.activations]
            h = acts[-1] if acts else torch.zeros(1)
        else:
            h = state.activity.get("output", torch.zeros(1))
        return self.compute_energy_from_state(h, geometry, state.substrate.get("substrate", Substrate()))


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
            h = h + self.config.step_size * op(error, geometry.params.get("weight", torch.eye(h.shape[-1])))

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
            acts = state.activations if isinstance(state.activations, list) else [state.activations]
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
        op = substrate.get_forward_operator()

        for step in range(self.config.max_steps):
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
            acts = state.activations if isinstance(state.activations, list) else [state.activations]
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
        op = substrate.get_forward_operator()

        for step in range(self.config.max_steps):
            # Langevin dynamics: dh = -∇E dt + sqrt(2*D) dW
            energy_grad = torch.autograd.grad(
                self.compute_energy_from_state(h, geometry, substrate), h
            )[0]
            noise = torch.randn_like(h) * (2 * self.config.step_size).sqrt()
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
            acts = state.activations if isinstance(state.activations, list) else [state.activations]
            h = acts[-1] if acts else torch.zeros(1)
        else:
            h = state.activity.get("output", torch.zeros(1))
        return self.compute_energy_from_state(h, geometry, state.substrate.get("substrate", Substrate()))