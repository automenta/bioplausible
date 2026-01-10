"""
Algorithms Package

Novel hybrid learning algorithms combining EqProp and Feedback Alignment.
"""

from .base import (
    BaseAlgorithm,
    AlgorithmConfig,
    BackpropBaseline,
    create_model,
    count_parameters,
)

from .eqprop import StandardEqProp
from .feedback_align import StandardFA
from .eq_align import EquilibriumAlignment
from .ada_fa import AdaptiveFeedbackAlignment

# Radical Variants
from .cf_align import ContrastiveFeedbackAlignment
from .leq_fa import LayerwiseEquilibriumFA
from .pc_hybrid import PredictiveCodingHybrid
from .eg_fa import EnergyGuidedFA
from .sparse_eq import SparseEquilibrium
from .mom_eq import MomentumEquilibrium
from .sto_fa import StochasticFA
from .em_fa import EnergyMinimizingFA

__all__ = [
    'BaseAlgorithm',
    'AlgorithmConfig',
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
    'create_model',
    'count_parameters',
]

# Registry of all available algorithms
ALGORITHM_REGISTRY = {
    'backprop': 'Standard Backpropagation',
    'eqprop': 'Equilibrium Propagation',
    'feedback_alignment': 'Feedback Alignment',
    'eq_align': 'EquilibriumAlignment (EqProp features + FA training)',
    'ada_fa': 'AdaptiveFeedbackAlignment (Evolving feedback)',
    'cf_align': 'ContrastiveFeedbackAlignment (Contrastive via FA)',
    'leq_fa': 'LayerwiseEquilibriumFA (Local settling)',
    'pc_hybrid': 'PredictiveCodingHybrid (PC + FA)',
    'eg_fa': 'EnergyGuidedFA (Energy-weighted updates)',
    'sparse_eq': 'SparseEquilibrium (Top-K dynamics)',
    'mom_eq': 'MomentumEquilibrium (Accelerated settling)',
    'sto_fa': 'StochasticFA (Feedback dropout)',
    'em_fa': 'EnergyMinimizingFA (Energy loss objective)',
}


def list_algorithms():
    """Print all available algorithms."""
    print("Available Algorithms:")
    print("=" * 60)
    for name, desc in ALGORITHM_REGISTRY.items():
        print(f"  {name:20} - {desc}")
    print("=" * 60)
