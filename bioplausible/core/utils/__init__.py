"""Core utilities package."""

from bioplausible.core.utils.activations import (
    ActivationName,
    approx_spectral_norm,
    cross_entropy,
    get_activation,
    get_backend,
    softmax,
    spectral_normalize,
    to_numpy,
)
from bioplausible.core.utils.device import get_device, get_optimal_backend
from bioplausible.core.utils.logging import get_logger
from bioplausible.core.utils.seeds import set_all_seeds

__all__ = [
    # activations
    "ActivationName",
    "approx_spectral_norm",
    "cross_entropy",
    "get_activation",
    "get_backend",
    "softmax",
    "spectral_normalize",
    "to_numpy",
    # device
    "get_device",
    "get_optimal_backend",
    # logging
    "get_logger",
    # seeds
    "set_all_seeds",
]