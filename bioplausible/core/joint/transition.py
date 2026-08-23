"""Joint Transition Protocol: CoupledTransition + NullPlasticity.

The linchpin of the 6-D joint architecture.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
from torch import Tensor

from bioplausible.core.joint.state import CompositeState
if TYPE_CHECKING:
    from bioplausible.core.joint.context import SystemContext

__all__ = [
    "CoupledTransition",
    "NullPlasticity",
    "PlasticityConfig",
    "PlasticityPrimitive",
]


# ============================================================
# PlasticityConfig: 6th Axis Configuration
# ============================================================


@dataclass(frozen=True, slots=True)
class PlasticityConfig:
    """Configuration for the Plasticity/MetaDynamics axis (M).

    Attributes:
        plasticity_type: Type of plasticity primitive.
        plastic_state_dims: Dimensions for plastic state variables (ψ).
        consolidation_config: Configuration for episode-boundary consolidation.
    """

    plasticity_type: str = "null"
    plastic_state_dims: dict[str, int] | None = None
    consolidation_config: dict | None = None

    @classmethod
    def null(cls) -> "PlasticityConfig":
        """Null plasticity configuration (Zero-Extension Theorem)."""
        return cls(plasticity_type="null")

    @classmethod
    def routing(cls, gate_dim: int = 64, **kwargs) -> "PlasticityConfig":
        """Routing plasticity: state-dependent pathway gating."""
        return cls(
            plasticity_type="routing",
            plastic_state_dims={"gate_logits": gate_dim, "active_routes": gate_dim},
            consolidation_config=kwargs,
        )

    @classmethod
    def fast_weights(cls, fast_weight_dim: int = 512, **kwargs) -> "PlasticityConfig":
        """Fast weight plasticity: episode-local associative memory."""
        return cls(
            plasticity_type="fast_weights",
            plastic_state_dims={"fast_weights": fast_weight_dim},
            consolidation_config=kwargs,
        )

    @classmethod
    def substrate_coupled(cls, **kwargs) -> "PlasticityConfig":
        """Substrate-coupled plasticity: reuse substrate adapters as physical plasticity."""
        return cls(
            plasticity_type="substrate_coupled",
            plastic_state_dims=None,  # ψ ≡ σ or tightly coupled
            consolidation_config=kwargs,
        )

    @classmethod
    def rule_state(cls, num_operators: int = 8, **kwargs) -> "PlasticityConfig":
        """Rule state plasticity (Z3): operator selection via ψ."""
        return cls(
            plasticity_type="rule_state",
            plastic_state_dims={"operator_logits": num_operators},
            consolidation_config=kwargs,
        )


# ============================================================
# PlasticityPrimitive: Protocol for plasticity laws
# ============================================================


@runtime_checkable
class PlasticityPrimitive(Protocol):
    """Protocol for plasticity dynamics: ψ_{t+1} = P(ψ_t, z_t; θ, G, S).

    Plasticity receives the full joint state and returns updated plastic state.
    It does NOT modify θ directly — only consolidation at episode boundaries
    can promote consolidatable ψ to θ.
    """

    config: PlasticityConfig

    @abstractmethod
    def step(
        self,
        psi: dict[str, Tensor],
        z: "CompositeState",
        context: "SystemContext",
    ) -> dict[str, Tensor]:
        """Compute next plastic state.

        Args:
            psi: Current plastic state ψ_t
            z: Full joint state (activity, plastic, substrate)
            context: Immutable system context (θ, geometry, substrate, configs)

        Returns:
            Updated plastic state ψ_{t+1}
        """
        ...

    @abstractmethod
    def initial_psi(self, context: "SystemContext") -> dict[str, Tensor]:
        """Create initial plastic state for a new episode."""
        ...


# ============================================================
# NullPlasticity: Zero-Extension Theorem (ψ_{t+1} = ψ_t)
# ============================================================


class NullPlasticity:
    """Null plasticity: ψ_{t+1} = ψ_t — Joint system with M=Null ≡ 5-D system.

    This is the compatibility slice that makes every 5-D system a valid
    6-D coordinate with M=Null. The Zero-Extension Theorem guarantees
    behavioral equivalence.
    """

    config = PlasticityConfig.null()

    def step(
        self,
        psi: dict[str, Tensor],
        z: "CompositeState",
        context: "SystemContext",
    ) -> dict[str, Tensor]:
        """Null plasticity: plastic state unchanged."""
        return psi

    def initial_psi(self, context: "SystemContext", batch_size: int = 1) -> dict[str, Tensor]:
        """Null plasticity has no plastic state."""
        return {}


# ============================================================
# CoupledTransition Protocol: The Linchpin
# ============================================================


@runtime_checkable
class CoupledTransition(Protocol):
    """Joint dynamical system transition: z_{t+1} = F_θ(z_t; G, S, M).

    The core protocol for the 6-D joint architecture. Executes one step
    of the coupled system, including:
    1. Activity evolution (StateDynamics)
    2. Plasticity projection (PlasticityPrimitive)
    3. Substrate physics (Substrate)
    """

    @abstractmethod
    def step(
        self,
        z: "CompositeState",
        context: "SystemContext",
    ) -> "CompositeState":
        """Execute one step of the joint dynamical system.

        Args:
            z: Current joint state z_t = (x_t, ψ_t, σ_t)
            context: Immutable context containing θ, geometry, substrate, configs

        Returns:
            Next joint state z_{t+1}
        """
        ...


# ============================================================
# Legacy Wrapper: 5-D System as Joint Transition with M=Null
# ============================================================


class LegacyDynamicsAsCoupledTransition:
    """Wraps existing 5-D System as a joint transition with ψ={}, σ={}, M=Null.

    Enables zero-cost compatibility: all existing 5-D compositions remain
    valid as NullPlasticity slices of the joint system.
    """

    def __init__(self, system: "System") -> None:
        from bioplausible.core.ontology import System

        self.system = system
        self._null_plasticity = NullPlasticity()

    def step(
        self,
        z: "CompositeState",
        context: "SystemContext",
    ) -> "CompositeState":
        """Execute one step using the wrapped 5-D system."""
        # Extract input from activity
        x = z.activity.get("x")
        if x is None:
            raise ValueError("CompositeState.activity must contain 'x' key for input")

        # Run 5-D system train_step (includes free + nudged phase)
        # Note: This is a simplification; full integration would interleave
        # plasticity steps within the settling loop.
        y = z.activity.get("y")
        if y is None:
            # Inference mode
            out = self.system.forward(x)
            new_activity = {"x": x, "output": out}
            new_plastic = {}
            new_substrate = {}
        else:
            # Training mode - use train_step
            metrics = self.system.train_step(x, y)
            # Get updated geometry params as theta
            new_activity = {"x": x, "output": metrics}
            new_plastic = {}
            new_substrate = {}

        return CompositeState(
            activity=new_activity,
            plastic=new_plastic,
            substrate=new_substrate,
        )


# Import System for type annotation (at bottom to avoid circular)
if TYPE_CHECKING:
    from bioplausible.core.ontology import System