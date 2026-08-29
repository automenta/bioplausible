"""Plasticity Primitives (M-axis) - 5-D Ontology Extension.

Plasticity defines how learning rules adapt the system's parameters.
This module re-exports plasticity primitives from their implementation locations
to provide a unified ontology import surface.
"""

# Re-export from computronium.state (core joint primitives)
from computronium.state import (
    NullPlasticity,
    PlasticityConfig,
    PlasticityPrimitive,
    TransitionFn,
)

# Re-export from computronium.core.plasticity (implementations)
from computronium.core.plasticity.fast_weights import FastWeightPlasticity
from computronium.core.plasticity.routing import RoutingPlasticity
from computronium.core.plasticity.rule_state import RuleStatePlasticity
from computronium.core.plasticity.substrate_coupled import SubstrateCoupledPlasticity

__all__ = [
    # Core primitives from state
    "NullPlasticity",
    "PlasticityConfig",
    "PlasticityPrimitive",
    "TransitionFn",
    # Implementations from core.plasticity
    "FastWeightPlasticity",
    "RoutingPlasticity",
    "RuleStatePlasticity",
    "SubstrateCoupledPlasticity",
]