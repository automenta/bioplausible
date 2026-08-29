"""Substrate package."""

from computronium.ontology._substrate import (
    AnalogSubstrate,
    ComplexSubstrate,
    DigitalSubstrate,
    MemristiveSubstrate,
    NeuromorphicSubstrate,
    NoisySubstrate,
    OpticalSubstrate,
    QuantumSubstrate,
    QuantizedSubstrate,
    SparseSubstrate,
    Substrate,
    SubstrateConfig,
    SubstrateType,
    TernarySubstrate,
)
from computronium.ontology.substrate.factory import substrate_from_config

__all__ = [
    "SubstrateType",
    "SubstrateConfig",
    "Substrate",
    "DigitalSubstrate",
    "AnalogSubstrate",
    "MemristiveSubstrate",
    "NeuromorphicSubstrate",
    "OpticalSubstrate",
    "QuantumSubstrate",
    "SparseSubstrate",
    "TernarySubstrate",
    "ComplexSubstrate",
    "NoisySubstrate",
    "QuantizedSubstrate",
    "substrate_from_config",
]