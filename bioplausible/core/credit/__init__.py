"""Credit adapters for cross-credit translation."""

from bioplausible.core.credit.adapters import (
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
