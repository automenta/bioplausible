"""
EqProp+SN Evolution System

A genetic algorithm-based system for evolving and evaluating
Equilibrium Propagation + Spectral Normalization model variants.
"""

from .algorithm import (ALGORITHM_PRESETS, ActivationFunction,
                        AlgorithmBreeder, AlgorithmConfig, EquilibriumDynamics,
                        GradientApprox, SNStrategy, UpdateRule)
from .algorithm_model import AlgorithmVariantModel, build_algorithm_variant
# New modular components
from .base import (BreedingStrategy, EvaluationResult, Evaluator, ModelBuilder,
                   SelectionStrategy, TerminationCriterion)
from .breakthrough import BreakthroughDetector
from .breeder import ArchConfig, VariationBreeder
from .config import (DEFAULT_FITNESS_WEIGHTS, MODEL_CONSTRAINTS, TASK_CONFIGS,
                     TIER_CONFIGS, BreedingConfig)
from .engine import EvolutionEngine
from .evaluator import EvalTier, VariationEvaluator
from .fitness import FitnessScore, compute_fitness
from .models import DefaultModelBuilder, ModelRegistry
from .utils import count_parameters, format_time, set_seed, setup_logger

__all__ = [
    # Core components
    "FitnessScore",
    "compute_fitness",
    "VariationBreeder",
    "ArchConfig",
    "VariationEvaluator",
    "EvalTier",
    "BreakthroughDetector",
    "EvolutionEngine",
    # Abstract interfaces
    "ModelBuilder",
    "Evaluator",
    "SelectionStrategy",
    "BreedingStrategy",
    "TerminationCriterion",
    "EvaluationResult",
    # Configuration
    "TIER_CONFIGS",
    "TASK_CONFIGS",
    "MODEL_CONSTRAINTS",
    "DEFAULT_FITNESS_WEIGHTS",
    "BreedingConfig",
    # Utilities
    "setup_logger",
    "set_seed",
    "count_parameters",
    "format_time",
    # Model building
    "ModelRegistry",
    "DefaultModelBuilder",
    # Algorithm evolution
    "AlgorithmConfig",
    "AlgorithmBreeder",
    "ALGORITHM_PRESETS",
    "UpdateRule",
    "EquilibriumDynamics",
    "GradientApprox",
    "SNStrategy",
    "ActivationFunction",
    "AlgorithmVariantModel",
    "build_algorithm_variant",
]
