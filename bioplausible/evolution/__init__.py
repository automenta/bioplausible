"""
EqProp+SN Evolution System

A genetic algorithm-based system for evolving and evaluating 
Equilibrium Propagation + Spectral Normalization model variants.
"""

from .fitness import FitnessScore, compute_fitness
from .breeder import VariationBreeder, ArchConfig
from .evaluator import VariationEvaluator, EvalTier
from .breakthrough import BreakthroughDetector
from .engine import EvolutionEngine

# New modular components
from .base import (
    ModelBuilder, Evaluator, SelectionStrategy, BreedingStrategy,
    TerminationCriterion, EvaluationResult
)
from .config import (
    TIER_CONFIGS, TASK_CONFIGS, MODEL_CONSTRAINTS,
    DEFAULT_FITNESS_WEIGHTS, BreedingConfig
)
from .utils import setup_logger, set_seed, count_parameters, format_time
from .models import ModelRegistry, DefaultModelBuilder
from .algorithm import (
    AlgorithmConfig, AlgorithmBreeder, ALGORITHM_PRESETS,
    UpdateRule, EquilibriumDynamics, GradientApprox, SNStrategy, ActivationFunction
)
from .algorithm_model import AlgorithmVariantModel, build_algorithm_variant

__all__ = [
    # Core components
    'FitnessScore', 'compute_fitness',
    'VariationBreeder', 'ArchConfig',
    'VariationEvaluator', 'EvalTier',
    'BreakthroughDetector',
    'EvolutionEngine',
    # Abstract interfaces
    'ModelBuilder', 'Evaluator', 'SelectionStrategy', 'BreedingStrategy',
    'TerminationCriterion', 'EvaluationResult',
    # Configuration
    'TIER_CONFIGS', 'TASK_CONFIGS', 'MODEL_CONSTRAINTS',
    'DEFAULT_FITNESS_WEIGHTS', 'BreedingConfig',
    # Utilities
    'setup_logger', 'set_seed', 'count_parameters', 'format_time',
    # Model building
    'ModelRegistry', 'DefaultModelBuilder',
    # Algorithm evolution
    'AlgorithmConfig', 'AlgorithmBreeder', 'ALGORITHM_PRESETS',
    'UpdateRule', 'EquilibriumDynamics', 'GradientApprox', 'SNStrategy', 'ActivationFunction',
    'AlgorithmVariantModel', 'build_algorithm_variant',
]

