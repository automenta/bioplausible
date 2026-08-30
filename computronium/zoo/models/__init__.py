"""
Zoo Models Package

All models registered with the unified registry system.

Only native compositions and thin tile wrappers remain.
Legacy modules (eqprop/, fa.py, backprop.py, hebbian.py, predictive_coding.py,
mep.py, o1memory.py, spiking.py, target_prop.py, forward_only.py, wrappers.py,
base.py, transitions.py) have been removed.
"""

from computronium.core.registry import LocalityLevel, Registry, register_model

# Import native models to trigger registration
from computronium.models import native as native_models  # noqa: F401

from . import deployments, tile_fa, tile_lm, tile_models

__all__: list[str] = [
    "LocalityLevel",
    "Registry",
    "deployments",
    "native_models",
    "register_model",
    "tile_fa",
    "tile_lm",
    "tile_models",
]
