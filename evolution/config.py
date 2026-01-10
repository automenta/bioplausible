"""
Configuration and constants for the evolution system.
"""

from dataclasses import dataclass
from typing import Dict, Tuple


# ============================================================================
# Evaluation Tiers
# ============================================================================

TIER_CONFIGS = {
    1: {  # Smoke test
        'n_samples': 500,
        'epochs': 3,
        'n_seeds': 1,
        'timeout_sec': 30,
        'name': 'smoke',
    },
    2: {  # Quick validation
        'n_samples': 5000,
        'epochs': 10,
        'n_seeds': 1,
        'timeout_sec': 300,
        'name': 'quick',
    },
    3: {  # Full validation
        'n_samples': 60000,
        'epochs': 30,
        'n_seeds': 3,
        'timeout_sec': 1800,
        'name': 'full',
    },
    4: {  # Breakthrough
        'n_samples': 50000,
        'epochs': 100,
        'n_seeds': 5,
        'timeout_sec': 7200,
        'name': 'breakthrough',
    },
}


# ============================================================================
# Task Definitions
# ============================================================================

TASK_CONFIGS = {
    'mnist': {
        'input_dim': 784,
        'output_dim': 10,
        'type': 'classification',
    },
    'fashion': {
        'input_dim': 784,
        'output_dim': 10,
        'type': 'classification',
    },
    'cifar10': {
        'input_dim': 3072,
        'output_dim': 10,
        'type': 'classification',
    },
    'shakespeare': {
        'input_dim': 65,
        'output_dim': 65,
        'type': 'language_modeling',
    },
}


# ============================================================================
# Model Type Constraints
# ============================================================================

MODEL_CONSTRAINTS = {
    'looped_mlp': {
        'max_depth': 100,
        'min_depth': 2,
        'supports_sn': True,
    },
    'transformer': {
        'max_depth': 12,
        'min_depth': 1,
        'supports_sn': True,
        'requires_vocab': True,
    },
    'conv': {
        'max_depth': 50,
        'min_depth': 2,
        'supports_sn': True,
    },
    'hebbian': {
        'max_depth': 1000,
        'min_depth': 10,
        'supports_sn': True,
        'requires_sn': True,  # Hebbian *requires* SN
    },
    'feedback_alignment': {
        'max_depth': 50,
        'min_depth': 2,
        'supports_sn': True,
    },
}


# ============================================================================
# Fitness Weights (for composite scoring)
# ============================================================================

DEFAULT_FITNESS_WEIGHTS = {
    'accuracy': 2.0,
    'perplexity': -1.0,
    'speed': 0.5,
    'memory': -0.3,
    'lipschitz': -1.5,
    'stability': 0.5,
    'generalization': 1.0,
}


# ============================================================================
# Breeding Parameters
# ============================================================================

@dataclass
class BreedingConfig:
    """Configuration for genetic operations."""
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    mutation_strength: float = 0.1  # For Gaussian mutations
    
    # Mutation probabilities for different types
    structural_mutation_prob: float = 0.3  # Depth, width changes
    activation_mutation_prob: float = 0.15
    hyperparameter_mutation_prob: float = 0.3
    topology_mutation_prob: float = 0.1  # Residual connections, etc.
    
    def validate(self):
        """Validate configuration values."""
        assert 0 <= self.mutation_rate <= 1, "mutation_rate must be in [0, 1]"
        assert 0 <= self.crossover_rate <= 1, "crossover_rate must be in [0, 1]"
        assert self.mutation_strength > 0, "mutation_strength must be positive"


# ============================================================================
# Logging Configuration
# ============================================================================

LOG_FORMAT = '[%(levelname)s] %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

VERBOSE_LEVELS = {
    0: 'ERROR',    # Only errors
    1: 'WARNING',  # Warnings and errors
    2: 'INFO',     # Standard progress
    3: 'DEBUG',    # Detailed debug info
}
