"""Substrate adapters for cross-substrate emulation."""

from computronium.core.substrates.adapters import (
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
    "ComplexToOpticalAdapter",
    "DigitalToAnalogAdapter",
    "DigitalToComplexAdapter",
    "DigitalToMemristiveAdapter",
    "DigitalToNeuromorphicAdapter",
    "DigitalToQuantumAdapter",
    "DigitalToSparseAdapter",
    "DigitalToTernaryAdapter",
    "SubstrateAdapter",
    "create_substrate_adapter",
]
