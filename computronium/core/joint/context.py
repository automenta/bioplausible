"""Joint System Context: Immutable context for the coupled transition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from computronium.core.joint.transition import PlasticityConfig
    from computronium.ontology import (
        CreditAssignmentConfig,
        Geometry,
        GeometryConfig,
        ParameterUpdateConfig,
        StateDynamicsConfig,
        Substrate,
        SubstrateConfig,
    )
    from computronium.state import StateRegistry

__all__ = ["SystemContext"]


@dataclass(frozen=True, slots=True)
class SystemContext:
    """Immutable context for the joint transition.

    Contains all fixed parameters and configurations that do not change
    during an episode. The joint transition F_θ(z; G, S) uses this context
    for geometry, substrate physics, and configuration.

    Attributes:
        theta: Persistent parameters (immutable intra-episode, requires_grad=True).
        geometry: Network topology and routing (immutable).
        substrate: Physical substrate providing forward/update operators.
        substrate_config: Substrate configuration.
        geometry_config: Geometry configuration.
        dynamics_config: State dynamics configuration.
        credit_config: Credit assignment configuration.
        update_config: Parameter update configuration.
        plasticity_config: Plasticity configuration (6th axis).
        registry: State variable lifecycle registry.
    """

    theta: Mapping[str, Tensor]
    geometry: Geometry
    substrate: Substrate
    substrate_config: SubstrateConfig
    geometry_config: GeometryConfig
    dynamics_config: StateDynamicsConfig
    credit_config: CreditAssignmentConfig
    update_config: ParameterUpdateConfig
    plasticity_config: PlasticityConfig
    registry: StateRegistry

    def __post_init__(self) -> None:
        # Validate that all theta tensors require grad
        for name, param in self.theta.items():
            if not param.requires_grad:
                raise ValueError(
                    f"theta[{name!r}] must have requires_grad=True. "
                    f"Persistent parameters must be differentiable."
                )

    @property
    def device(self) -> torch.device:
        """Get the device from the first theta parameter."""
        if not self.theta:
            return torch.device("cpu")
        return next(iter(self.theta.values())).device

    def with_updated_theta(self, new_theta: Mapping[str, Tensor]) -> SystemContext:
        """Create a new context with updated theta (for episode boundary consolidation)."""
        return SystemContext(
            theta=new_theta,
            geometry=self.geometry,
            substrate=self.substrate,
            substrate_config=self.substrate_config,
            geometry_config=self.geometry_config,
            dynamics_config=self.dynamics_config,
            credit_config=self.credit_config,
            update_config=self.update_config,
            plasticity_config=self.plasticity_config,
            registry=self.registry,
        )
