"""
Configuration schemas and defaults for Bioplausible experiments.

OmegaConf-based structured configs with Pydantic validation.
"""

from bioplausible.config.defaults import (
    DEFAULT_CONFIGS,
    get_named_config,
    list_named_configs,
    register_default_config,
)
from bioplausible.config.omegaconf import (
    DatasetConfig,
    DomainConfig,
    LightningConfig,
    OptimizerConfig,
    PropagatorConfig,
    ScientistConfig,
    SparsityConfig,
    TrainingConfig,
    get_default_config,
    validate_config,
)
from bioplausible.config.omegaconf import (
    ExperimentModelConfig as ModelConfig,
)
from bioplausible.config.omegaconf import (
    ExperimentSchemaConfig as ExperimentConfig,
)
from bioplausible.config.unified import (
    BaseConfig,
    BaseStructuredConfig,
    config_to_dict,
)

# ──────────────────────────────────────────────
# Merged from config_loader.py
# ──────────────────────────────────────────────
__all__ = [
    # New schema exports
    "BaseConfig",
    "BaseStructuredConfig",
    "DEFAULT_CONFIGS",
    "DatasetConfig",
    "DomainConfig",
    "ExperimentConfig",
    "LightningConfig",
    "ModelConfig",
    "OptimizerConfig",
    "PropagatorConfig",
    "ScientistConfig",
    "SparsityConfig",
    "TrainingConfig",
    "config_to_dict",
    "get_default_config",
    "get_named_config",
    "list_named_configs",
    "register_default_config",
    "validate_config",
]
