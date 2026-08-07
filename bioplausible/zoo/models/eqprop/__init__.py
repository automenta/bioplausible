"""Equilibrium Propagation Model Zoo Subpackage."""

from .causal_transformer_eqprop import (
    CausalEqPropAttention,
    CausalTransformerEqProp,
)
from .conv_eqprop import ConvEqProp
from .deep_ep import DirectedEP
from .eqprop_diffusion import EqPropDiffusion
from .eqprop_lm_variants import (
    CausalMask,
    EqPropAttentionLM,
    EqPropAttentionOnlyLM,
    FullEqPropLM,
    HybridEqPropLM,
    LoopedMLPForLM,
    RecurrentEqPropLM,
    compare_variants,
    create_eqprop_lm,
    get_eqprop_lm,
    list_eqprop_lm_variants,
    register_eqprop_lm,
)
from .finite_nudge_ep import FiniteNudgeEP
from .graph_eqprop import GraphEqProp
from .hardware_variants import NoisyLoopedMLP, QuantizedLoopedMLP
from .holomorphic_ep import HolomorphicEP
from .homeostatic import HomeostasisMetrics, HomeostaticEqProp
from .lazy_eqprop import LazyEqProp, LazyStats
from .looped_mlp import BackpropMLP, LoopedMLP
from .memory_efficient import (
    MemoryEfficientEqPropModel,
    MemoryEfficientLoopedMLP,
    create_memory_efficient_model,
)
from .modern_conv_eqprop import ModernConvEqProp, SimpleConvEqProp
from .mom_eq import MomentumEquilibrium
from .neural_cube import NeuralCube
from .sparse_eq import SparseEquilibrium
from .standard_eqprop import StandardEqProp
from .temporal_resonance import TemporalResonanceEqProp
from .ternary import TernaryEqProp, TernaryLinear, TernaryQuantize
from .transformer_eqprop import EqPropAttention, TransformerEqProp

__all__: list[str] = [
    "BackpropMLP",
    "CausalEqPropAttention",
    "CausalMask",
    "CausalTransformerEqProp",
    "ConvEqProp",
    "DirectedEP",
    "EqPropAttention",
    "EqPropAttentionLM",
    "EqPropAttentionOnlyLM",
    "EqPropDiffusion",
    "FiniteNudgeEP",
    "FullEqPropLM",
    "GraphEqProp",
    "HolomorphicEP",
    "HomeostasisMetrics",
    "HomeostaticEqProp",
    "HybridEqPropLM",
    "LazyEqProp",
    "LazyStats",
    "LoopedMLP",
    "LoopedMLPForLM",
    "MemoryEfficientEqPropModel",
    "MemoryEfficientLoopedMLP",
    "ModernConvEqProp",
    "MomentumEquilibrium",
    "NeuralCube",
    "NoisyLoopedMLP",
    "QuantizedLoopedMLP",
    "RecurrentEqPropLM",
    "SimpleConvEqProp",
    "SparseEquilibrium",
    "StandardEqProp",
    "TemporalResonanceEqProp",
    "TernaryEqProp",
    "TernaryLinear",
    "TernaryQuantize",
    "TransformerEqProp",
    "compare_variants",
    "create_eqprop_lm",
    "create_memory_efficient_model",
    "get_eqprop_lm",
    "list_eqprop_lm_variants",
    "register_eqprop_lm",
]
