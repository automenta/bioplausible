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
    "CreditAdapter",
    "ThermodynamicToBackpropAdapter",
    "RandomProjectionsToThermodynamicAdapter",
    "LocalGoodnessToThermodynamicAdapter",
    "ThermodynamicToHomeostaticAdapter",
    "TemporalTraceToThermodynamicAdapter",
    "TargetInversionToThermodynamicAdapter",
    "BackpropToThermodynamicAdapter",
    "create_credit_adapter",
]