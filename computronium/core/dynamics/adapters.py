"""Cross-Dynamics Adapters.

Enables translation between different state dynamics paradigms,
allowing compositions that mix dynamics types or distill one
dynamics type into another.
"""

from __future__ import annotations

import torch
from torch import Tensor

from computronium.core.ontology import (
    EnergyMinimizationDynamics,
    Geometry,
    InstantaneousDynamics,
    LazyStateDynamics,
    PredictiveSettlingDynamics,
    SpikeIntegrationDynamics,
    StateDynamics,
    StateDynamicsConfig,
    Substrate,
    SystemState,
)

# ============================================================
# Base Adapter Class
# ============================================================


class DynamicsAdapter:
    """Base class for cross-dynamics adapters.

    Wraps a source dynamics and emulates target dynamics behavior.
    """

    def __init__(
        self,
        source_dynamics: StateDynamics,
        target_config: StateDynamicsConfig | None = None,
    ):
        self._source = source_dynamics
        self._target_config = target_config or source_dynamics.config
        self.config = self._target_config

    @property
    def source_dynamics(self) -> StateDynamics:
        return self._source

    @property
    def target_config(self) -> StateDynamicsConfig:
        return self._target_config

    def settle(
        self,
        state: SystemState,
        geometry: Geometry,
        substrate: Substrate,
        target: Tensor | None = None,
    ) -> SystemState:
        return self._source.settle(state, geometry, substrate, target)

    def compute_energy(self, state: SystemState, geometry: Geometry) -> Tensor:
        return self._source.compute_energy(state, geometry)


# ============================================================
# EnergyMinimization -> Instantaneous (Equilibrium Distillation)
# ============================================================


class EnergyToInstantaneousAdapter(DynamicsAdapter):
    """Distill equilibrium dynamics to single-pass feedforward.

    Learns a feedforward network that approximates the settled
    equilibrium state of an energy-based system.

    Source: EnergyMinimizationDynamics (iterative settling)
    Target: InstantaneousDynamics (single forward pass)
    """

    def __init__(
        self,
        source_dynamics: EnergyMinimizationDynamics,
        target_config: StateDynamicsConfig | None = None,
        *,
        distillation_steps: int = 1000,
        distillation_lr: float = 1e-3,
    ):
        target_config = target_config or StateDynamicsConfig.instantaneous()
        super().__init__(source_dynamics, target_config)
        self._distillation_steps = distillation_steps
        self._distillation_lr = distillation_lr
        self._distilled_model: torch.nn.Module | None = None

    def distill(
        self,
        geometry: Geometry,
        substrate: Substrate,
        input_samples: Tensor,
    ) -> torch.nn.Module:
        """Distill equilibrium dynamics into a feedforward network.

        Args:
            geometry: Network geometry
            substrate: Physical substrate
            input_samples: Sample inputs for distillation [N, input_dim]

        Returns:
            Feedforward module approximating equilibrium
        """
        # Create a simple MLP to predict equilibrium state
        input_dim = geometry.config.input_dim
        output_dim = geometry.config.output_dim
        hidden_dims = geometry.config.hidden_dims

        layers = []
        dims = (input_dim, *hidden_dims, output_dim)
        for i in range(len(dims) - 1):
            layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(torch.nn.ReLU())
        distilled = torch.nn.Sequential(*layers)

        optimizer = torch.optim.Adam(distilled.parameters(), lr=self._distillation_lr)
        self._source.config.track_free_energy_per_iter = True

        for step in range(self._distillation_steps):
            optimizer.zero_grad()
            # Get equilibrium states for batch
            batch_size = min(32, input_samples.shape[0])
            idx = torch.randperm(input_samples.shape[0])[:batch_size]
            x_batch = input_samples[idx]

            # Run source dynamics to get equilibrium
            state = SystemState(x=x_batch)
            state.activations = geometry.forward(x_batch, substrate)
            state = self._source.settle(state, geometry, substrate, target=None)
            eq_state = state.activations

            # Predict with distilled model
            pred = distilled(x_batch)

            # Match final layer output
            if isinstance(eq_state, list):
                eq_state = eq_state[-1]

            loss = torch.nn.functional.mse_loss(pred, eq_state)
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"Distillation step {step}: loss = {loss.item():.6f}")

        self._distilled_model = distilled
        return distilled

    def settle(
        self,
        state: SystemState,
        geometry: Geometry,
        substrate: Substrate,
        target: Tensor | None = None,
    ) -> SystemState:
        """Single-pass forward using distilled model."""
        if self._distilled_model is None:
            # Fall back to single-step energy minimization
            self._source.config.max_steps = 1
            return self._source.settle(state, geometry, substrate, target)

        x = state.x
        if x is None:
            return state

        with torch.no_grad():
            pred = self._distilled_model(x)

        state.activations = pred
        state.free_state = pred
        return state


# ============================================================
# SpikeIntegration -> Instantaneous (Rate-Coded Surrogate)
# ============================================================


class SpikeToInstantaneousAdapter(DynamicsAdapter):
    """Surrogate gradient through spikes for rate-coded inference.

    Uses surrogate gradients (fast sigmoid, piecewise linear) to
    backpropagate through spiking non-linearities, enabling
    training of SNNs with standard backprop.

    Source: SpikeIntegrationDynamics (LIF spikes)
    Target: InstantaneousDynamics (rate-coded with surrogate grad)
    """

    def __init__(
        self,
        source_dynamics: SpikeIntegrationDynamics,
        target_config: StateDynamicsConfig | None = None,
        *,
        surrogate_type: str = "fast_sigmoid",  # "fast_sigmoid", "piecewise", "gaussian"
        beta: float = 10.0,
        time_steps: int = 100,
    ):
        target_config = target_config or StateDynamicsConfig.instantaneous()
        super().__init__(source_dynamics, target_config)
        self._surrogate_type = surrogate_type
        self._beta = beta
        self._time_steps = time_steps

    def _surrogate_gradient(self, v: Tensor) -> Tensor:
        """Surrogate gradient for spiking threshold."""
        if self._surrogate_type == "fast_sigmoid":
            return self._beta / (1 + self._beta * v.abs()) ** 2
        if self._surrogate_type == "piecewise":
            return (v.abs() < 1.0 / self._beta).float() * self._beta
        if self._surrogate_type == "gaussian":
            return torch.exp(-0.5 * (self._beta * v) ** 2) * self._beta
        return torch.ones_like(v)

    def settle(
        self,
        state: SystemState,
        geometry: Geometry,
        substrate: Substrate,
        _target: Tensor | None = None,
    ) -> SystemState:
        """Rate-coded forward with surrogate gradient support."""
        h = state.activations
        if h is None:
            return state
        if isinstance(h, list):
            h = h[-1]

        # Rate-coded single pass (no spike simulation)
        h_new = geometry.route(h)
        h_new = substrate.inject_state_noise(h_new)

        # Apply surrogate gradient for backprop compatibility
        # This enables gradient flow through the "spiking" non-linearity
        surrogate = self._surrogate_gradient(h_new)
        h_new = h_new * surrogate.detach() + h_new * (1 - surrogate.detach())

        state.activations = h_new
        state.free_state = h_new
        return state


# ============================================================
# LazyStateDynamics -> EnergyMinimization (On-Demand to Full Settle)
# ============================================================


class LazyToEnergyAdapter(DynamicsAdapter):
    """Convert lazy (event-driven) dynamics to full energy minimization.

    Materializes all intermediate activations that lazy dynamics
    would compute on-demand, enabling full equilibrium analysis.

    Source: LazyStateDynamics (on-demand activation)
    Target: EnergyMinimizationDynamics (full settling)
    """

    def __init__(
        self,
        source_dynamics: LazyStateDynamics,
        target_config: StateDynamicsConfig | None = None,
    ):
        target_config = target_config or StateDynamicsConfig.energy_minimization()
        super().__init__(source_dynamics, target_config)

    def settle(
        self,
        state: SystemState,
        geometry: Geometry,
        substrate: Substrate,
        target: Tensor | None = None,
    ) -> SystemState:
        """Run full settling, materializing all lazy activations."""
        # Use source's lazy settle but force full materialization
        state = self._source.settle(state, geometry, substrate, target)

        # Materialize all cached activations
        cached = self._source.get_cached_activations()
        if cached and state.activations is not None:
            # Use the last cached activation as the full settled state
            last_step = max(cached.keys())
            state.activations = cached[last_step]
            if target is None:
                state.free_state = cached[last_step]
            else:
                state.nudged_state = cached[last_step]

        return state


# ============================================================
# PredictiveSettling -> EnergyMinimization (Free Energy to Equilibrium)
# ============================================================


class PredictiveToEnergyAdapter(DynamicsAdapter):
    """Map Predictive Coding free energy to EqProp equilibrium energy.

    Translates the hierarchical prediction error minimization of PC
    into the contrastive energy minimization of EqProp.

    Source: PredictiveSettlingDynamics (free energy)
    Target: EnergyMinimizationDynamics (equilibrium energy)
    """

    def __init__(
        self,
        source_dynamics: PredictiveSettlingDynamics,
        target_config: StateDynamicsConfig | None = None,
    ):
        target_config = target_config or StateDynamicsConfig.energy_minimization()
        super().__init__(source_dynamics, target_config)

    def settle(
        self,
        state: SystemState,
        geometry: Geometry,
        substrate: Substrate,
        target: Tensor | None = None,
    ) -> SystemState:
        """Run predictive settling, then compute equivalent equilibrium energy."""
        # Run full predictive coding settling
        state = self._source.settle(state, geometry, substrate, target)

        # The free energy from PC corresponds to equilibrium energy
        # For compatibility, store both
        if hasattr(self._source, "_free_energy_history"):
            state.metrics = state.metrics or {}
            state.metrics["free_energy_history"] = self._source._free_energy_history

        return state

    def compute_energy(self, state: SystemState, geometry: Geometry) -> Tensor:
        """Compute energy as free energy from predictive coding."""
        return self._source.compute_energy(state, geometry)


# ============================================================
# Instantaneous -> EnergyMinimization (Feedforward to Equilibrium)
# ============================================================


class InstantaneousToEnergyAdapter(DynamicsAdapter):
    """Wrap instantaneous dynamics as single-step energy minimization.

    Allows feedforward/backprop models to participate in energy-based
    compositions by treating the forward pass as a single energy step.

    Source: InstantaneousDynamics (single pass)
    Target: EnergyMinimizationDynamics (max_steps=1)
    """

    def __init__(
        self,
        source_dynamics: InstantaneousDynamics,
        target_config: StateDynamicsConfig | None = None,
    ):
        target_config = target_config or StateDynamicsConfig.energy_minimization(
            max_steps=1
        )
        super().__init__(source_dynamics, target_config)

    def compute_energy(self, state: SystemState, geometry: Geometry) -> Tensor:
        """Energy proxy: negative log-likelihood for feedforward."""
        return self._source.compute_energy(state, geometry)


# ============================================================
# Factory Function
# ============================================================


def create_dynamics_adapter(
    source_type: str,
    target_type: str,
    source_dynamics: StateDynamics,
    target_config: StateDynamicsConfig | None = None,
    **kwargs,
) -> DynamicsAdapter:
    """Factory for cross-dynamics adapters.

    Args:
        source_type: Source dynamics ("energy_minimization", "spike_integration",
                     "lazy", "predictive_settling", "instantaneous", "diffusion")
        target_type: Target dynamics
        source_dynamics: Source dynamics instance
        target_config: Optional target config
        **kwargs: Adapter-specific parameters

    Returns:
        Configured DynamicsAdapter instance
    """
    adapter_map: dict[tuple[str, str], type[DynamicsAdapter]] = {
        ("energy_minimization", "instantaneous"): EnergyToInstantaneousAdapter,
        ("spike_integration", "instantaneous"): SpikeToInstantaneousAdapter,
        ("lazy", "energy_minimization"): LazyToEnergyAdapter,
        ("predictive_settling", "energy_minimization"): PredictiveToEnergyAdapter,
        ("instantaneous", "energy_minimization"): InstantaneousToEnergyAdapter,
    }

    key = (source_type, target_type)
    if key not in adapter_map:
        available = list(adapter_map.keys())
        msg = f"No adapter for {source_type} -> {target_type}. Available: {available}"
        raise ValueError(msg)

    adapter_class = adapter_map[key]
    return adapter_class(source_dynamics, target_config, **kwargs)


__all__ = [
    "DynamicsAdapter",
    "EnergyToInstantaneousAdapter",
    "InstantaneousToEnergyAdapter",
    "LazyToEnergyAdapter",
    "PredictiveToEnergyAdapter",
    "SpikeToInstantaneousAdapter",
    "create_dynamics_adapter",
]
