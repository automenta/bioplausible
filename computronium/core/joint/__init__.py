"""Joint Architecture: 6-D Composable Bioplausible Systems.

This package implements the joint dynamical system runtime that extends
the 5-D ontology (S ⊗ G ⊗ D ⊗ C ⊗ U) with a 6th Plasticity axis (M).

The joint system state: z = (x, ψ, σ) evolves as:
    z_{t+1} = F_θ(z_t; G, S, M)

Where M is the plasticity/meta-dynamics primitive.
"""

from __future__ import annotations

from computronium.core.joint.consolidation import ConsolidationConfig, consolidate
from computronium.core.joint.transition import (
    CoupledTransition,
    NullPlasticity,
    PlasticityConfig,
    PlasticityPrimitive,
)
from computronium.state import (
    CompositeState,
    StateRegistry,
    StateVariable,
    SystemContext,
)

__all__ = [  # ruff: ignore[unsorted-dunder-all]
    # State
    "StateVariable",
    "StateRegistry",
    "CompositeState",
    # Context
    "SystemContext",
    # Transition
    "CoupledTransition",
    "PlasticityPrimitive",
    "PlasticityConfig",
    "NullPlasticity",
    # Consolidation
    "ConsolidationConfig",
    "consolidate",
]
