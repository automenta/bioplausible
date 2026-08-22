"""Joint Architecture: 6-D Composable Bioplausible Systems.

This package implements the joint dynamical system runtime that extends
the 5-D ontology (S ⊗ G ⊗ D ⊗ C ⊗ U) with a 6th Plasticity axis (M).

The joint system state: z = (x, ψ, σ) evolves as:
    z_{t+1} = F_θ(z_t; G, S, M)

Where M is the plasticity/meta-dynamics primitive.
"""

from __future__ import annotations

from bioplausible.core.joint.consolidation import ConsolidationConfig, consolidate
from bioplausible.core.joint.context import SystemContext
from bioplausible.core.joint.state import (
    CompositeState,
    JointTrajectoryRecorder,
    StateRegistry,
    StateVariable,
)
from bioplausible.core.joint.transition import (
    CoupledTransition,
    LegacyDynamicsAsCoupledTransition,
    NullPlasticity,
    PlasticityConfig,
    PlasticityPrimitive,
)
from bioplausible.core.joint.trajectory import JointTrajectory

__all__ = [
    # State
    "StateVariable",
    "StateRegistry",
    "CompositeState",
    "JointTrajectoryRecorder",
    # Context
    "SystemContext",
    # Transition
    "CoupledTransition",
    "PlasticityPrimitive",
    "PlasticityConfig",
    "NullPlasticity",
    "LegacyDynamicsAsCoupledTransition",
    # Trajectory
    "JointTrajectory",
    # Consolidation
    "ConsolidationConfig",
    "consolidate",
]