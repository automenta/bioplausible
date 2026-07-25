"""
Muon/Dion optimizers.

Re-exports the pure update strategies (MuonUpdate, DionUpdate, PlainUpdate,
FisherUpdate) so they are discoverable under ``zoo.optimizers``.  Actual
registration with the ``Registry`` is done in ``zoo.mep._registration``.
"""

from bioplausible.zoo.mep.optimizers import (
    DionUpdate,
    FisherUpdate,
    MuonUpdate,
    PlainUpdate,
)

__all__ = [
    "DionUpdate",
    "FisherUpdate",
    "MuonUpdate",
    "PlainUpdate",
]
