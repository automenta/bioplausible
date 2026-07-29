"""
MEP (Muon Equilibrium Propagation) presets registered as propagators.

Re-exports the MEP composite presets (smep, sdmep, local_ep, natural_ep,
muon_backprop) so they are discoverable under ``zoo.propagators``.
Actual registration with the ``Registry`` is done in
``zoo.mep._registration``.
"""

from bioplausible.zoo.mep.presets import (
    local_ep,
    muon_backprop,
    natural_ep,
    sdmep,
    smep,
    smep_fast,
)

__all__ = [
    "local_ep",
    "muon_backprop",
    "natural_ep",
    "sdmep",
    "smep",
    "smep_fast",
]
