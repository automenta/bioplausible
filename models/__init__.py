"""TorEqProp Release Models Package"""

from .looped_mlp import LoopedMLP, BackpropMLP
from .ternary import TernaryEqProp
from .neural_cube import NeuralCube
from .lazy_eqprop import LazyEqProp, LazyStats
from .feedback_alignment import FeedbackAlignmentEqProp, FeedbackAlignmentLayer
from .kernel import EqPropKernel, compare_memory_autograd_vs_kernel

# Newly ported models
from .temporal_resonance import TemporalResonanceEqProp
from .homeostatic import HomeostaticEqProp
from .conv_eqprop import ConvEqProp
from .modern_conv_eqprop import ModernConvEqProp, SimpleConvEqProp
from .transformer import TransformerEqProp, EqPropAttention
from .causal_transformer_eqprop import CausalTransformerEqProp
from .eqprop_diffusion import EqPropDiffusion

# Language modeling comparison models
from .backprop_transformer_lm import BackpropTransformerLM, create_scaled_model
from .eqprop_lm_variants import (
    get_eqprop_lm, list_eqprop_lm_variants, create_eqprop_lm,
    FullEqPropLM, EqPropAttentionOnlyLM, RecurrentEqPropLM,
    HybridEqPropLM, LoopedMLPForLM
)

# NEBC (Nobody Ever Bothered Club) - Bio-plausible algorithms with SN
from .nebc_base import NEBCBase, NEBCRegistry, register_nebc, train_nebc_model, evaluate_nebc_model, run_nebc_ablation
from .dfa_eqprop import DirectFeedbackAlignmentEqProp, DeepDFAEqProp
from .chl import ContrastiveHebbianLearning, CHLAutoencoder
from .hebbian_chain import DeepHebbianChain, HebbianCube, HebbianLayer

__all__ = [
    'LoopedMLP', 'BackpropMLP',
    'TernaryEqProp',
    'NeuralCube',
    'LazyEqProp', 'LazyStats',
    'FeedbackAlignmentEqProp', 'FeedbackAlignmentLayer',
    'EqPropKernel', 'compare_memory_autograd_vs_kernel',
    'TemporalResonanceEqProp',
    'HomeostaticEqProp',
    'ConvEqProp',
    'ModernConvEqProp', 'SimpleConvEqProp',
    'TransformerEqProp', 'EqPropAttention',
    'CausalTransformerEqProp',
    'EqPropDiffusion',
    # LM comparison models
    'BackpropTransformerLM', 'create_scaled_model',
    'get_eqprop_lm', 'list_eqprop_lm_variants', 'create_eqprop_lm',
    'FullEqPropLM', 'EqPropAttentionOnlyLM', 'RecurrentEqPropLM',
    'HybridEqPropLM', 'LoopedMLPForLM',
    # NEBC models
    'NEBCBase', 'NEBCRegistry', 'register_nebc', 'train_nebc_model', 
    'evaluate_nebc_model', 'run_nebc_ablation',
    'DirectFeedbackAlignmentEqProp', 'DeepDFAEqProp',
    'ContrastiveHebbianLearning', 'CHLAutoencoder',
    'DeepHebbianChain', 'HebbianCube', 'HebbianLayer',
]



