"""ModelAdapter decomposition for Strangler Fig migration.

This package splits the monolithic ModelAdapter into composable parts:

- adapter.py: Main facade coordinating all inferrers
- inference.py: Protocols and implementations for each ontology axis
- registry.py: Metadata extraction from ComponentMetadata
- heuristics.py: Fallback inference for legacy models

Usage:
    from computronium.ontology.adapter import ModelAdapter

    adapter = ModelAdapter(model, metadata)
    system = adapter.to_system()
    result = adapter.validate(x, y)
"""

from computronium.ontology.adapter.adapter import (
    AdapterConfig,
    ModelAdapter,
    SubstrateInferer,
    GeometryInferer,
    DynamicsInferer,
    CreditInferer,
    UpdateInferer,
)

__all__ = [
    "AdapterConfig",
    "ModelAdapter",
    "SubstrateInferer",
    "GeometryInferer",
    "DynamicsInferer",
    "CreditInferer",
    "UpdateInferer",
]