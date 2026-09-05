"""Layer 3: StateDynamics — Forward Evolution & Settling."""

from typing import TYPE_CHECKING, Final, cast

from computronium.ontology.dynamics._dynamics import (
    DiffusionDynamics,
    EnergyMinimizationDynamics,
    ErrorPredictiveCodingDynamics,
    InstantaneousDynamics,
    LazyStateDynamics,
    PredictiveSettlingDynamics,
    SpikeIntegrationDynamics,
    StateDynamics,
    StateDynamicsConfig,
)

DYNAMICS_REGISTRY: Final[dict[str, type[StateDynamics]]] = {
    "energy_minimization": EnergyMinimizationDynamics,
    "predictive_settling": PredictiveSettlingDynamics,
    "error_predictive_coding": ErrorPredictiveCodingDynamics,
    "spike_integration": SpikeIntegrationDynamics,
    "instantaneous": InstantaneousDynamics,
    "diffusion": DiffusionDynamics,
    "lazy": LazyStateDynamics,
}


if TYPE_CHECKING:
    from collections.abc import Callable


def dynamics_from_config(config: StateDynamicsConfig) -> StateDynamics:
    """Instantiate the registered StateDynamics for a config's ``dynamics_type``."""
    cls = DYNAMICS_REGISTRY.get(config.dynamics_type.lower())
    if cls is None:
        raise ValueError(f"Unknown dynamics_type: {config.dynamics_type!r}")
    # The registry stores concrete implementations behind the runtime-checkable
    # Protocol; their common constructor shape is (config).
    factory = cast("Callable[[StateDynamicsConfig], StateDynamics]", cls)
    return factory(config)


__all__ = [
    "DYNAMICS_REGISTRY",
    "DiffusionDynamics",
    "EnergyMinimizationDynamics",
    "ErrorPredictiveCodingDynamics",
    "InstantaneousDynamics",
    "LazyStateDynamics",
    "PredictiveSettlingDynamics",
    "SpikeIntegrationDynamics",
    "StateDynamics",
    "StateDynamicsConfig",
    "dynamics_from_config",
]
