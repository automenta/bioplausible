"""
EqProp-Torch: Equilibrium Propagation for PyTorch

A production-grade library for training neural networks with Equilibrium Propagation,
featuring spectral normalization for stability and torch.compile for acceleration.
"""

from .acceleration import (
    check_cupy_available,
    compile_model,
    enable_tf32,
    get_optimal_backend,
)
from .core import EqPropTrainer
from .datasets import CharDataset, create_data_loaders, get_lm_dataset, get_vision_dataset
from .generation import generate_from_dataset, generate_text
from .kernel import HAS_CUPY, EqPropKernel
from .models import (
    BackpropMLP,
    ConvEqProp,
    LoopedMLP,
    TransformerEqProp,
)
from .utils import (
    ModelRegistry,
    count_parameters,
    create_model_preset,
    export_to_onnx,
    verify_spectral_norm,
)

# Language model variants (optional import - fails gracefully if dependencies missing)
try:
    from .lm_models import (
        create_eqprop_lm,
        get_eqprop_lm,
        list_eqprop_lm_variants,
    )
    HAS_LM_VARIANTS = True
except ImportError:
    HAS_LM_VARIANTS = False
    get_eqprop_lm = None
    list_eqprop_lm_variants = None

# Bio-plausible research algorithms (optional - from research codebase)
# These are first-class PyTorch nn.Module models, same status as LoopedMLP
try:
    from .bioplausible import (
        ALGORITHM_REGISTRY,
        # Registry
        HAS_BIOPLAUSIBLE,
        AdaptiveFeedbackAlignment,
        # Core algorithms
        BackpropBaseline,
        # Base class
        BaseAlgorithm,
        ContrastiveFeedbackAlignment,
        EnergyGuidedFA,
        EnergyMinimizingFA,
        # Hybrid algorithms
        EquilibriumAlignment,
        LayerwiseEquilibriumFA,
        MomentumEquilibrium,
        PredictiveCodingHybrid,
        SparseEquilibrium,
        StandardEqProp,
        StandardFA,
        StochasticFA,
    )
except ImportError:
    HAS_BIOPLAUSIBLE = False
    BaseAlgorithm = None
    ALGORITHM_REGISTRY = {}


__version__ = "0.1.0"
__all__ = [
    # Trainer
    "EqPropTrainer",
    # Models
    "LoopedMLP",
    "BackpropMLP",
    "ConvEqProp",
    "TransformerEqProp",
    # Kernel
    "EqPropKernel",
    "HAS_CUPY",
    # Utils
    "compile_model",
    "get_optimal_backend",
    "check_cupy_available",
    "enable_tf32",
    # Datasets
    "get_vision_dataset",
    "get_lm_dataset",
    "CharDataset",
    "generate_text",
    "generate_from_dataset",
    "create_data_loaders",
    # Utils
    "export_to_onnx",
    "count_parameters",
    "verify_spectral_norm",
    "create_model_preset",
    "ModelRegistry",
    # LM variants (if available)
    "get_eqprop_lm",
    "list_eqprop_lm_variants",
    "create_eqprop_lm",
    "HAS_LM_VARIANTS",
    # Bio-plausible research algorithms (if available) - first-class models
    "BaseAlgorithm",
    "BackpropBaseline",
    "StandardEqProp",
    "StandardFA",
    "EquilibriumAlignment",
    "AdaptiveFeedbackAlignment",
    "ContrastiveFeedbackAlignment",
    "LayerwiseEquilibriumFA",
    "PredictiveCodingHybrid",
    "EnergyGuidedFA",
    "SparseEquilibrium",
    "MomentumEquilibrium",
    "StochasticFA",
    "EnergyMinimizingFA",
    "HAS_BIOPLAUSIBLE",
    "ALGORITHM_REGISTRY",
]
