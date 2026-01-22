"""TorEqProp Release Models Package"""

# Language modeling comparison models
from .backprop_transformer_lm import BackpropTransformerLM, create_scaled_model
from .causal_transformer_eqprop import CausalTransformerEqProp
from .chl import CHLAutoencoder, ContrastiveHebbianLearning
from .conv_eqprop import ConvEqProp
from .dfa_eqprop import DeepDFAEqProp, DirectFeedbackAlignmentEqProp
from .eqprop_diffusion import EqPropDiffusion
from .eqprop_lm_variants import (EqPropAttentionOnlyLM, FullEqPropLM,
                                 HybridEqPropLM, LoopedMLPForLM,
                                 RecurrentEqPropLM, create_eqprop_lm,
                                 get_eqprop_lm, list_eqprop_lm_variants)
from .feedback_alignment import FeedbackAlignmentEqProp, FeedbackAlignmentLayer
from .hebbian_chain import DeepHebbianChain, HebbianCube, HebbianLayer
from .homeostatic import HomeostaticEqProp
from .kernel import EqPropKernel, compare_memory_autograd_vs_kernel
from .lazy_eqprop import LazyEqProp, LazyStats
from .looped_mlp import BackpropMLP, LoopedMLP
from .modern_conv_eqprop import ModernConvEqProp, SimpleConvEqProp
# NEBC (Nobody Ever Bothered Club) - Bio-plausible algorithms with SN
from .nebc_base import (NEBCBase, NEBCRegistry, evaluate_nebc_model,
                        register_nebc, run_nebc_ablation, train_nebc_model)
from .neural_cube import NeuralCube
# Newly ported models
from .temporal_resonance import TemporalResonanceEqProp
from .ternary import TernaryEqProp
from .transformer import EqPropAttention, TransformerEqProp

__all__ = [
    "LoopedMLP",
    "BackpropMLP",
    "TernaryEqProp",
    "NeuralCube",
    "LazyEqProp",
    "LazyStats",
    "FeedbackAlignmentEqProp",
    "FeedbackAlignmentLayer",
    "EqPropKernel",
    "compare_memory_autograd_vs_kernel",
    "TemporalResonanceEqProp",
    "HomeostaticEqProp",
    "ConvEqProp",
    "ModernConvEqProp",
    "SimpleConvEqProp",
    "TransformerEqProp",
    "EqPropAttention",
    "CausalTransformerEqProp",
    "EqPropDiffusion",
    # LM comparison models
    "BackpropTransformerLM",
    "create_scaled_model",
    "get_eqprop_lm",
    "list_eqprop_lm_variants",
    "create_eqprop_lm",
    "FullEqPropLM",
    "EqPropAttentionOnlyLM",
    "RecurrentEqPropLM",
    "HybridEqPropLM",
    "LoopedMLPForLM",
    # NEBC models
    "NEBCBase",
    "NEBCRegistry",
    "register_nebc",
    "train_nebc_model",
    "evaluate_nebc_model",
    "run_nebc_ablation",
    "DirectFeedbackAlignmentEqProp",
    "DeepDFAEqProp",
    "ContrastiveHebbianLearning",
    "CHLAutoencoder",
    "DeepHebbianChain",
    "HebbianCube",
    "HebbianLayer",
]
