"""Substrate package."""

from computronium.ontology._substrate import (
    AnalogSubstrate,
    ComplexSubstrate,
    DigitalSubstrate,
    MemristiveSubstrate,
    NeuromorphicSubstrate,
    NoisySubstrate,
    OpticalSubstrate,
    QuantizedSubstrate,
    QuantumSubstrate,
    SparseSubstrate,
    Substrate,
    SubstrateConfig,
    SubstrateType,
    TernarySubstrate,
)
from computronium.ontology.substrate.factory import substrate_from_config

__all__ = [
    "AnalogSubstrate",
    "ComplexSubstrate",
    "DigitalSubstrate",
    "MemristiveSubstrate",
    "NeuromorphicSubstrate",
    "NoisySubstrate",
    "OpticalSubstrate",
    "QuantizedSubstrate",
    "QuantumSubstrate",
    "SparseSubstrate",
    "Substrate",
    "SubstrateConfig",
    "SubstrateType",
    "TernarySubstrate",
    "substrate_from_config",
]
