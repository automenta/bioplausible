"""
Standard PyTorch optimizers.
"""

from torch.optim import SGD as TorchSGD
from torch.optim import Adam as TorchAdam
from torch.optim import AdamW as TorchAdamW

from computronium.core.registry import register_param_update

__all__ = [
    "SGD",
    "Adam",
    "AdamW",
]


@register_param_update("sgd", family="optimizer")
class SGD(TorchSGD):
    """SGD optimizer wrapper."""


@register_param_update("adam", family="optimizer")
class Adam(TorchAdam):
    """Adam optimizer wrapper."""


@register_param_update("adamw", family="optimizer")
class AdamW(TorchAdamW):
    """AdamW optimizer wrapper."""
