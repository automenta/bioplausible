"""
Bio-Plausible Learning Algorithms

Direct imports of research algorithms as first-class PyTorch models.
All algorithms inherit from BaseAlgorithm (nn.Module) and provide both
standard forward() and custom train_step() methods.
"""

try:
    # Import all algorithm classes directly from the algorithms package
    from ..algorithms import (
        ALGORITHM_REGISTRY,
        AdaptiveFeedbackAlignment,
        BackpropBaseline,
        BaseAlgorithm,
        ContrastiveFeedbackAlignment,
        EnergyGuidedFA,
        EnergyMinimizingFA,
        EquilibriumAlignment,
        LayerwiseEquilibriumFA,
        MomentumEquilibrium,
        PredictiveCodingHybrid,
        SparseEquilibrium,
        StandardEqProp,
        StandardFA,
        StochasticFA,
    )
    HAS_BIOPLAUSIBLE = True
except ImportError:
    HAS_BIOPLAUSIBLE = False
    ALGORITHM_REGISTRY = {}


# Re-export all algorithms at module level
__all__ = [
    'BaseAlgorithm',
    'BackpropBaseline',
    'StandardEqProp',
    'StandardFA',
    'EquilibriumAlignment',
    'AdaptiveFeedbackAlignment',
    'ContrastiveFeedbackAlignment',
    'LayerwiseEquilibriumFA',
    'PredictiveCodingHybrid',
    'EnergyGuidedFA',
    'SparseEquilibrium',
    'MomentumEquilibrium',
    'StochasticFA',
    'EnergyMinimizingFA',
    'HAS_BIOPLAUSIBLE',
    'ALGORITHM_REGISTRY',
]
