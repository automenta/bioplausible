"""TileAlgorithm factory registry with @tile_algorithm decorator.

This module provides a decorator for registering TileAlgorithm factory methods
with metadata, enabling config-driven composition via TileAlgorithm.from_config().
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class TileAlgorithmMetadata:
    """Metadata for a registered TileAlgorithm factory variant."""

    name: str
    algorithm: str
    mode: str
    description: str
    default_beta: float | None
    requires_beta: bool
    credit_assignment_type: str
    locality_level: str
    bio_plausibility_score: float
    tags: list[str] = field(default_factory=list)


# Registry of TileAlgorithm factory variants
_TILE_ALGORITHM_REGISTRY: dict[str, TileAlgorithmMetadata] = {}
_TILE_ALGORITHM_FACTORIES: dict[str, Callable] = {}


def tile_algorithm(
    name: str,
    *,
    algorithm: str,
    mode: str,
    description: str,
    default_beta: float | None = None,
    requires_beta: bool = False,
    credit_assignment_type: str = "equilibrium",
    locality_level: str = "equilibrium",
    bio_plausibility_score: float = 0.9,
    tags: list[str] | None = None,
) -> Callable:
    """Decorator to register a TileAlgorithm factory method.

    Args:
        name: Unique name for this variant (e.g., "ep", "fa", "hebbian")
        algorithm: Algorithm identifier used in TileAlgorithmConfig.algorithm
        mode: Training mode used in TileAlgorithmConfig.mode
        description: Human-readable description
        default_beta: Default beta value (None if not used by this variant)
        requires_beta: Whether this variant requires beta parameter
        credit_assignment_type: Credit assignment type for registry queries
        locality_level: Locality level for registry queries
        bio_plausibility_score: Bio-plausibility score (0.0-1.0)
        tags: Additional tags for discovery

    Returns:
        Decorator function that registers the factory method.
    """

    def decorator(func: Callable) -> Callable:
        metadata = TileAlgorithmMetadata(
            name=name,
            algorithm=algorithm,
            mode=mode,
            description=description,
            default_beta=default_beta,
            requires_beta=requires_beta,
            credit_assignment_type=credit_assignment_type,
            locality_level=locality_level,
            bio_plausibility_score=bio_plausibility_score,
            tags=tags or [],
        )
        _TILE_ALGORITHM_REGISTRY[name] = metadata
        _TILE_ALGORITHM_FACTORIES[name] = func
        return func

    return decorator


def get_tile_algorithm_metadata(name: str) -> TileAlgorithmMetadata:
    """Get metadata for a registered TileAlgorithm variant."""
    if name not in _TILE_ALGORITHM_REGISTRY:
        raise ValueError(
            f"Unknown TileAlgorithm variant: {name}. "
            f"Available: {list(_TILE_ALGORITHM_REGISTRY.keys())}"
        )
    return _TILE_ALGORITHM_REGISTRY[name]


def get_tile_algorithm_factory(name: str) -> Callable:
    """Get factory function for a registered TileAlgorithm variant."""
    if name not in _TILE_ALGORITHM_FACTORIES:
        raise ValueError(
            f"Unknown TileAlgorithm variant: {name}. "
            f"Available: {list(_TILE_ALGORITHM_FACTORIES.keys())}"
        )
    return _TILE_ALGORITHM_FACTORIES[name]


def list_tile_algorithms() -> list[str]:
    """List all registered TileAlgorithm variant names."""
    return list(_TILE_ALGORITHM_REGISTRY.keys())


def list_tile_algorithms_with_metadata() -> dict[str, TileAlgorithmMetadata]:
    """Get all registered TileAlgorithm variants with their metadata."""
    return dict(_TILE_ALGORITHM_REGISTRY)


__all__ = [
    "TileAlgorithmMetadata",
    "tile_algorithm",
    "get_tile_algorithm_metadata",
    "get_tile_algorithm_factory",
    "list_tile_algorithms",
    "list_tile_algorithms_with_metadata",
]
