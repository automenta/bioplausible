"""
Zoo Propagators Package

Learning rules / propagators registered with the unified registry.

This package also re-exports model-side implementations for propagators
that require model-level control (e.g., ForwardForwardNet, PEPITA,
DifferenceTargetProp, FabricPCGraphPCN, PredictiveCodingHybrid).
See bioplausible/__init__.py for the two-tier architecture overview.

Some propagator names (ff, pepita, target_prop, difference_target_prop,
predictive_coding) are not registered directly as propagators because they
require model-level control. The Registry provides cross-references to the
model-side implementations via its error messages when these are queried.
"""

from bioplausible.core.registry import register_propagator

# Re-export model-side implementations (the "model side" of the two-tier architecture).
# See bioplausible/__init__.py docstring for the architectural rationale.
from bioplausible.zoo.models.forward_only import PEPITA, ForwardForwardNet
from bioplausible.zoo.models.predictive_coding import (
    FabricPCGraphPCN,
    PredictiveCodingHybrid,
)
from bioplausible.zoo.models.target_prop import DifferenceTargetProp

from . import (
    backprop,  # ruff: ignore[unused-import]
    base,  # ruff: ignore[unused-import]
    eqprop,  # ruff: ignore[unused-import]
    fa,  # ruff: ignore[unused-import]
    hebbian,  # ruff: ignore[unused-import]
    mep,  # ruff: ignore[unused-import]
    spiking,  # ruff: ignore[unused-import]
)

__all__ = [
    "register_propagator",
    # Model-side re-exports (original names)
    "ForwardForwardNet",
    "PEPITA",
    "DifferenceTargetProp",
    "FabricPCGraphPCN",
    "PredictiveCodingHybrid",
    "backprop",
    "base",
    "eqprop",
    "fa",
    "hebbian",
    "mep",
    "spiking",
]
