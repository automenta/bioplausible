"""Equilibrium Propagation Model Zoo Subpackage."""

import types

from ..backprop import BackpropMLP
from .causal_transformer_eqprop import (
    CausalEqPropAttention,
    CausalTransformerEqProp,
)
from .conv_eqprop import ConvEqProp
from .diffusion_native import native_diffusion_eqprop
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
from .graph_eqprop import GraphEqProp
from .hardware_variants import NoisyLoopedMLP, QuantizedLoopedMLP
from .holomorphic_ep import HolomorphicEP
from .homeostatic import HomeostasisMetrics, HomeostaticEqProp
from .looped_mlp import native_eqprop_mlp
from .memory_efficient import (
    MemoryEfficientEqPropModel,
    MemoryEfficientLoopedMLP,
    create_memory_efficient_model,
)
from .modern_conv_eqprop import ModernConvEqProp, SimpleConvEqProp
from .momentum_native import native_momentum_eqprop
from .neural_cube import NeuralCube
from .sparse_native import native_sparse_eqprop
from .temporal_resonance import TemporalResonanceEqProp
from .ternary import TernaryEqProp, TernaryLinear, TernaryQuantize
from .ternary_native import native_ternary_eqprop
from .transformer_eqprop import EqPropAttention, TransformerEqProp

# Learning rules for validation tracks
from computronium.core.local_learning.rules.eqprop import LazyEqProp

# Alias for backward compatibility with validation tracks
LoopedMLP = LoopedMLPForLM

__all__: list[str] = sorted(
    name
    for name, obj in vars().items()
    if not name.startswith("_") and not isinstance(obj, types.ModuleType)
)
