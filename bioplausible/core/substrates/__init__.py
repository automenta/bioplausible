"""Substrate adapters for cross-substrate emulation."""

from bioplausible.core.substrates.adapters import (
    ComplexToOpticalAdapter,
    DigitalToAnalogAdapter,
    DigitalToComplexAdapter,
    DigitalToMemristiveAdapter,
    DigitalToNeuromorphicAdapter,
    DigitalToQuantumAdapter,
    DigitalToSparseAdapter,
    DigitalToTernaryAdapter,
    SubstrateAdapter,
    create_substrate_adapter,
)

__all__ = [
    "SubstrateAdapter",
    "DigitalToComplexAdapter",
    "ComplexToOpticalAdapter",
    "DigitalToTernaryAdapter",
    "DigitalToSparseAdapter",
    "DigitalToNeuromorphicAdapter",
    "DigitalToQuantumAdapter",
    "DigitalToMemristiveAdapter",
    "DigitalToAnalogAdapter",
    "create_substrate_adapter",
]