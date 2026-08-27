"""
MEP: Muon Equilibrium Propagation

A biologically plausible deep learning framework using Equilibrium Propagation
with geometry-aware updates (Muon orthogonalization, Dion low-rank, spectral constraints).

Quick Start:
    from computronium.zoo.mep.presets import smep, sdmep, muon_backprop

    # SMEP with EP
    optimizer = smep(model.parameters(), model=model, mode='ep')
    optimizer.step(x=x, target=y)

    # Muon with backprop (drop-in SGD replacement)
    optimizer = muon_backprop(model.parameters())
    optimizer.step()

See NICHES.md for optimizer selection guide.
"""

from computronium.core.registry import (
    LocalityLevel,
    register_param_update,
    register_propagator,
)

# Trigger registry registration (side-effect import)
from . import _registration
from .optimizers import (
    BackpropGradient,
    CompositeOptimizer,
    DionUpdate,
    EnergyFunction,
    EPGradient,
    ErrorFeedback,
    FisherUpdate,
    LocalEPGradient,
    MuonUpdate,
    NaturalGradient,
    NoConstraint,
    NoFeedback,
    PlainUpdate,
    Settler,
    SpectralConstraint,
)
from .presets import local_ep, muon_backprop, natural_ep, sdmep, smep, smep_fast

__version__ = "0.3.0"
__all__ = [
    "BackpropGradient",
    "CompositeOptimizer",
    "DionUpdate",
    "EPGradient",
    "EnergyFunction",
    "ErrorFeedback",
    "FisherUpdate",
    "LocalEPGradient",
    "LocalityLevel",
    "MuonUpdate",
    "NaturalGradient",
    "NoConstraint",
    "NoFeedback",
    "PlainUpdate",
    "Settler",
    "SpectralConstraint",
    "_registration",
    "local_ep",
    "muon_backprop",
    "natural_ep",
    "register_param_update",
    "register_propagator",
    "sdmep",
    "smep",
    "smep_fast",
]
