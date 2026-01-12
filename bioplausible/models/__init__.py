"""
EqProp-Torch Models

All neural network architectures supporting Equilibrium Propagation training.
Models use spectral normalization to guarantee Lipschitz constant L < 1 for stable dynamics.
"""

# Re-export utility functions
from .utils import (
    spectral_linear,
    spectral_conv2d,
    estimate_lipschitz,
)

# Re-export models
from .looped_mlp import LoopedMLP, BackpropMLP
from .conv_eqprop import ConvEqProp
from .transformer_eqprop import TransformerEqProp, EqPropAttention
from .lazy_eqprop import LazyEqProp
from .ternary import TernaryEqProp
from .modern_conv_eqprop import ModernConvEqProp
from .causal_transformer_eqprop import CausalTransformerEqProp
from .eqprop_diffusion import EqPropDiffusion
from .neural_cube import NeuralCube
from .backprop_transformer_lm import BackpropTransformerLM
from .feedback_alignment import FeedbackAlignmentEqProp, FeedbackAlignmentLayer
from .dfa_eqprop import DirectFeedbackAlignmentEqProp, DeepDFAEqProp
from .temporal_resonance import TemporalResonanceEqProp
from .chl import ContrastiveHebbianLearning
from .nebc_base import NEBCBase, NEBCRegistry, train_nebc_model, evaluate_nebc_model
from .homeostatic import HomeostaticEqProp
from .hebbian_chain import DeepHebbianChain
from .eqprop_lm_variants import get_eqprop_lm, create_eqprop_lm, list_eqprop_lm_variants

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Utility functions
    'spectral_linear',
    'spectral_conv2d',
    'estimate_lipschitz',
    # Models
    'LoopedMLP',
    'BackpropMLP',
    'ConvEqProp',
    'TransformerEqProp',
    'EqPropAttention',
    # Extra models
    'LazyEqProp',
    'TernaryEqProp',
    'ModernConvEqProp',
    'CausalTransformerEqProp',
    'EqPropDiffusion',
    'NeuralCube',
    'BackpropTransformerLM',
    'FeedbackAlignmentEqProp',
    'FeedbackAlignmentLayer',
    'DirectFeedbackAlignmentEqProp',
    'DeepDFAEqProp',
    'TemporalResonanceEqProp',
    'ContrastiveHebbianLearning',
    'NEBCBase',
    'NEBCRegistry',
    'train_nebc_model',
    'evaluate_nebc_model',
    'HomeostaticEqProp',
    'DeepHebbianChain',
    'get_eqprop_lm',
    'create_eqprop_lm',
    'list_eqprop_lm_variants',
]
