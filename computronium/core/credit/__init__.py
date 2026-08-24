"""Credit adapters for cross-credit translation."""

from computronium.core.credit.adapters import (
    BackpropToThermodynamicAdapter,
    CreditAdapter,
    LocalGoodnessToThermodynamicAdapter,
    RandomProjectionsToThermodynamicAdapter,
    TargetInversionToThermodynamicAdapter,
    TemporalTraceToThermodynamicAdapter,
    ThermodynamicToBackpropAdapter,
    ThermodynamicToHomeostaticAdapter,
    create_credit_adapter,
)

__all__ = [
    "BackpropToThermodynamicAdapter",
    "CreditAdapter",
    "LocalGoodnessToThermodynamicAdapter",
    "RandomProjectionsToThermodynamicAdapter",
    "TargetInversionToThermodynamicAdapter",
    "TemporalTraceToThermodynamicAdapter",
    "ThermodynamicToBackpropAdapter",
    "ThermodynamicToHomeostaticAdapter",
    "create_credit_adapter",
]
