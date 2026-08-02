"""
Standard PyTorch optimizers.
"""

from torch.optim import SGD as TorchSGD
from torch.optim import Adam as TorchAdam
from torch.optim import AdamW as TorchAdamW

from bioplausible.core.registry import register_optimizer

__all__ = [
    "SGD",
    "Adam",
    "AdamW",
]


@register_optimizer("sgd", family="optimizer")
class SGD(TorchSGD):
    """SGD optimizer wrapper."""


@register_optimizer("adam", family="optimizer")
class Adam(TorchAdam):
    """Adam optimizer wrapper."""


@register_optimizer("adamw", family="optimizer")
class AdamW(TorchAdamW):
    """AdamW optimizer wrapper."""
