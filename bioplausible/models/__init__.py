from .looped_mlp import LoopedMLP, BackpropMLP
from .conv_eqprop import ConvEqProp
from .transformer_eqprop import TransformerEqProp
from .modern_conv_eqprop import ModernConvEqProp, SimpleConvEqProp
from .eqprop_base import EqPropModel
from .adaptive_fa import AdaptiveFA
from .eq_align import EquilibriumAlignment
from .chl import ContrastiveHebbianLearning, CHLAutoencoder
from .lazy_eqprop import LazyEqProp
from .ternary import TernaryEqProp
from .eqprop_diffusion import EqPropDiffusion
from .neural_cube import NeuralCube
from .feedback_alignment import FeedbackAlignmentEqProp
from .dfa_eqprop import DirectFeedbackAlignmentEqProp
from .temporal_resonance import TemporalResonanceEqProp
from .homeostatic import HomeostaticEqProp
from .hebbian_chain import DeepHebbianChain
from .causal_transformer_eqprop import CausalTransformerEqProp
from .backprop_transformer_lm import BackpropTransformerLM

from ..lm_models import (
    FullEqPropLM,
    EqPropAttentionOnlyLM,
    RecurrentEqPropLM,
    HybridEqPropLM,
    LoopedMLPForLM,
    EQPROP_LM_REGISTRY
)

__all__ = [
    'LoopedMLP',
    'BackpropMLP',
    'ConvEqProp',
    'TransformerEqProp',
    'ModernConvEqProp',
    'SimpleConvEqProp',
    'EqPropModel',
    'AdaptiveFA',
    'EquilibriumAlignment',
    'ContrastiveHebbianLearning',
    'CHLAutoencoder',
    'LazyEqProp',
    'TernaryEqProp',
    'EqPropDiffusion',
    'NeuralCube',
    'FeedbackAlignmentEqProp',
    'DirectFeedbackAlignmentEqProp',
    'TemporalResonanceEqProp',
    'HomeostaticEqProp',
    'DeepHebbianChain',
    'CausalTransformerEqProp',
    'BackpropTransformerLM',
    'FullEqPropLM',
    'EqPropAttentionOnlyLM',
    'RecurrentEqPropLM',
    'HybridEqPropLM',
    'LoopedMLPForLM',
    'EQPROP_LM_REGISTRY',
]
