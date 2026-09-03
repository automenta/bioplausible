"""Standard PyTorch optimizers as PARAM_UPDATE registry components.

First-class ontology-side registrations so standard optimizers are available
without the deprecated zoo package.
"""

from torch.optim import (
    SGD as TorchSGD,  # ruff: ignore[constant-imported-as-non-constant]
)
from torch.optim import Adam as TorchAdam
from torch.optim import AdamW as TorchAdamW
from torch.optim import RMSprop as TorchRMSprop

from computronium.core.registry import register_param_update

__all__ = ["SGD", "Adam", "AdamW", "RMSprop"]


@register_param_update("sgd", family="optimizer")
class SGD(TorchSGD):
    """SGD optimizer wrapper."""


@register_param_update("adam", family="optimizer")
class Adam(TorchAdam):
    """Adam optimizer wrapper."""


@register_param_update("adamw", family="optimizer")
class AdamW(TorchAdamW):
    """AdamW optimizer wrapper."""


@register_param_update("rmsprop", family="optimizer")
class RMSprop(TorchRMSprop):
    """RMSprop optimizer wrapper."""
