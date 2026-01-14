from .nebc_base import (
    NEBCBase,
    NEBCRegistry,
    register_nebc,
    train_nebc_model,
    evaluate_nebc_model,
)
from .eqprop_base import EqPropModel, EquilibriumFunction
from .base import BioModel, ModelConfig, register_model, ModelRegistry

# Core Models
from .looped_mlp import LoopedMLP, BackpropMLP
from .conv_eqprop import ConvEqProp
from .modern_conv_eqprop import ModernConvEqProp, SimpleConvEqProp
from .transformer_eqprop import TransformerEqProp
from .causal_transformer_eqprop import CausalTransformerEqProp
from .eqprop_diffusion import EqPropDiffusion
from .lazy_eqprop import LazyEqProp
from .homeostatic import HomeostaticEqProp
from .dfa_eqprop import DirectFeedbackAlignmentEqProp
from .neural_cube import NeuralCube
from .ternary import TernaryEqProp, TernaryEqProp as TernaryWeightMLP
from .chl import ContrastiveHebbianLearning, CHLAutoencoder
from .hebbian_chain import DeepHebbianChain
from .temporal_resonance import (
    TemporalResonanceEqProp,
    TemporalResonanceEqProp as TemporalResonanceNetwork,
)
from .backprop_transformer_lm import BackpropTransformerLM
from .holomorphic_ep import HolomorphicEP
from .deep_ep import DirectedEP
from .finite_nudge_ep import FiniteNudgeEP

# Algorithm-Models (Migrated from algorithms/)
from .standard_eqprop import StandardEqProp
from .simple_fa import StandardFA
from .ada_fa import AdaptiveFeedbackAlignment
from .eq_align import EquilibriumAlignment
from .cf_align import ContrastiveFeedbackAlignment
from .leq_fa import LayerwiseEquilibriumFA
from .eg_fa import EnergyGuidedFA
from .pc_hybrid import PredictiveCodingHybrid
from .sparse_eq import SparseEquilibrium
from .mom_eq import MomentumEquilibrium
from .sto_fa import StochasticFA
from .em_fa import EnergyMinimizingFA

# Feedback Alignment Variants
from .feedback_alignment import FeedbackAlignmentEqProp
from .adaptive_fa import AdaptiveFA

# Language Models
from .eqprop_lm_variants import (
    FullEqPropLM,
    EqPropAttentionOnlyLM,
    RecurrentEqPropLM,
    HybridEqPropLM,
    LoopedMLPForLM,
    get_eqprop_lm,
    create_eqprop_lm,
)

# Export registry
__all__ = [
    # Base
    "BioModel",
    "ModelConfig",
    "register_model",
    "ModelRegistry",
    "NEBCBase",
    "NEBCRegistry",
    "register_nebc",
    "EqPropModel",
    "EquilibriumFunction",
    # Core
    "LoopedMLP",
    "BackpropMLP",
    "ConvEqProp",
    "ModernConvEqProp",
    "SimpleConvEqProp",
    "TransformerEqProp",
    "CausalTransformerEqProp",
    "EqPropDiffusion",
    "LazyEqProp",
    "HomeostaticEqProp",
    "DirectFeedbackAlignmentEqProp",
    "NeuralCube",
    "TernaryWeightMLP",
    "TernaryEqProp",
    "ContrastiveHebbianLearning",
    "CHLAutoencoder",
    "DeepHebbianChain",
    "TemporalResonanceNetwork",
    "TemporalResonanceEqProp",
    "BackpropTransformerLM",
    "HolomorphicEP",
    "DirectedEP",
    "FiniteNudgeEP",
    # Algorithm-Models
    "StandardEqProp",
    "StandardFA",
    "AdaptiveFeedbackAlignment",
    "EquilibriumAlignment",
    "ContrastiveFeedbackAlignment",
    "LayerwiseEquilibriumFA",
    "EnergyGuidedFA",
    "PredictiveCodingHybrid",
    "SparseEquilibrium",
    "MomentumEquilibrium",
    "StochasticFA",
    "EnergyMinimizingFA",
    # FA Variants
    "FeedbackAlignmentEqProp",
    "AdaptiveFA",
    # LM
    "FullEqPropLM",
    "EqPropAttentionOnlyLM",
    "RecurrentEqPropLM",
    "HybridEqPropLM",
    "LoopedMLPForLM",
    "get_eqprop_lm",
    "create_eqprop_lm",
]
