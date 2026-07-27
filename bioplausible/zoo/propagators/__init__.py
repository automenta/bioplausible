"""
Zoo Propagators Package

Learning rules / propagators registered with the unified registry.

This package also re-exports model-side implementations for propagators
that require model-level control (e.g., ForwardForwardNet, PEPITA,
DifferenceTargetProp, FabricPCGraphPCN, PredictiveCodingHybrid).
See bioplausible/__init__.py for the two-tier architecture overview.
"""

from bioplausible.core.registry import register_propagator

from . import (
    backprop,  # noqa: F401
    base,  # noqa: F401
    eqprop,  # noqa: F401
    fa,  # noqa: F401
    forward_only,  # noqa: F401
    hebbian,  # noqa: F401
    mep,  # noqa: F401
    predictive_coding,  # noqa: F401
    spiking,  # noqa: F401
    target_prop,  # noqa: F401
)

# Import stub classes with distinct names so they don't shadow model-side
from .forward_only import FF as FFStub, PEPITA as PEPITAStub  # noqa: F401
from .target_prop import TargetProp as TargetPropStub, DifferenceTargetProp as DTPStub  # noqa: F401
from .predictive_coding import PCN as PCNStub  # noqa: F401

# Re-export model-side implementations (the "model side" of the two-tier architecture).
# See bioplausible/__init__.py docstring for the architectural rationale.
from bioplausible.zoo.models.forward_only import ForwardForwardNet, PEPITA
from bioplausible.zoo.models.target_prop import DifferenceTargetProp
from bioplausible.zoo.models.predictive_coding import (
    FabricPCGraphPCN,
    PredictiveCodingHybrid,
)

__all__ = [
    "register_propagator",
    # Stub classes (propagator side) - distinct names
    "FFStub",
    "PEPITAStub",
    "TargetPropStub",
    "DTPStub",
    "PCNStub",
    # Model-side re-exports (original names)
    "ForwardForwardNet",
    "PEPITA",
    "DifferenceTargetProp",
    "FabricPCGraphPCN",
    "PredictiveCodingHybrid",
]
