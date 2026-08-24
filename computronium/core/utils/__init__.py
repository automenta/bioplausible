"""Core utilities package."""

from computronium.core.logging import get_logger
from computronium.core.utils.activations import (
    ActivationName,
    approx_spectral_norm,
    cross_entropy,
    get_activation,
    get_backend,
    softmax,
    spectral_normalize,
    to_numpy,
)
from computronium.core.utils.device import get_device, get_optimal_backend
from computronium.core.utils.optimizer import OptimizerConfig, create_optimizer
from computronium.core.utils.reproducibility import (
    EnvironmentInfo,
    ReproducibilityConfig,
    ReproducibilityTracker,
    ReproducibleConfig,
    create_tracker,
    set_reproducible_mode,
)
from computronium.core.utils.seeds import set_all_seeds

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
    # optimizer factory (REFACTOR.md §2.3)
    "OptimizerConfig",
    "create_optimizer",
    # reproducibility
    "EnvironmentInfo",
    "ReproducibilityConfig",
    "ReproducibilityTracker",
    "ReproducibleConfig",
    "create_tracker",
    "set_reproducible_mode",
    # seeds
    "set_all_seeds",
]
