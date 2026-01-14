from .looped_mlp import LoopedMLP, BackpropMLP
from .conv_eqprop import ConvEqProp
from .transformer_eqprop import TransformerEqProp
from .modern_conv_eqprop import ModernConvEqProp, SimpleConvEqProp
from .eqprop_base import EqPropModel
from .adaptive_fa import AdaptiveFA
from .chl import ContrastiveHebbianLearning, CHLAutoencoder

__all__ = [
    'LoopedMLP',
    'BackpropMLP',
    'ConvEqProp',
    'TransformerEqProp',
    'ModernConvEqProp',
    'SimpleConvEqProp',
    'EqPropModel',
    'AdaptiveFA',
    'ContrastiveHebbianLearning',
    'CHLAutoencoder',
]
