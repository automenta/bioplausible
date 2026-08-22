"""Equilibrium Propagation Model Zoo Subpackage."""

import types

from .causal_transformer_eqprop import (
    CausalEqPropAttention,
    CausalTransformerEqProp,
)
from .conv_eqprop import ConvEqProp
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
from ..backprop import BackpropMLP
from .memory_efficient import (
    MemoryEfficientEqPropModel,
    MemoryEfficientLoopedMLP,
    create_memory_efficient_model,
)
from .modern_conv_eqprop import ModernConvEqProp, SimpleConvEqProp
from .neural_cube import NeuralCube
from .temporal_resonance import TemporalResonanceEqProp
from .ternary import TernaryEqProp, TernaryLinear, TernaryQuantize
from .transformer_eqprop import EqPropAttention, TransformerEqProp

__all__: list[str] = sorted(
    name
    for name, obj in vars().items()
    if not name.startswith("_") and not isinstance(obj, types.ModuleType)
)