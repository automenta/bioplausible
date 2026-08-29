"""Config factory protocol for unified configuration handling."""

from typing import Protocol

# Use string forward references to avoid circular imports
SubstrateConfig = "computronium.ontology.substrate.SubstrateConfig"
GeometryConfig = "computronium.ontology.geometry.GeometryConfig"
StateDynamicsConfig = "computronium.ontology.dynamics.StateDynamicsConfig"
CreditAssignmentConfig = "computronium.ontology.credit.CreditAssignmentConfig"
ParameterUpdateConfig = "computronium.ontology.update.ParameterUpdateConfig"


class ConfigFactory(Protocol):
    """Protocol for config classes with standard serialization methods."""

    def to_spec(self) -> dict: ...

    @classmethod
    def from_spec(cls, spec: dict): ...

    def validate(self) -> None: ...


# Re-export common config types for convenience
__all__ = [
    "ConfigFactory",
    "SubstrateConfig",
    "GeometryConfig",
    "StateDynamicsConfig",
    "CreditAssignmentConfig",
    "ParameterUpdateConfig",
]
