"""
Configuration schemas and defaults for Bioplausible experiments.

Unified ExperimentConfig (Sprint 7) is the single source of truth.
Legacy OmegaConf schemas retained for backward compatibility.
"""

from bioplausible.config.defaults import (
    DEFAULT_CONFIGS,
    get_named_config,
    list_named_configs,
    register_default_config,
)
from bioplausible.config.experiment import (
    DataConfig,
    ExperimentConfig,
    HardwareConfig,
    ModelConfig,
    OntologyConfig,
    TrainingConfig,
    from_omegaconf,
    to_deployment_config,
    to_omegaconf,
    to_system_trainer_config,
    to_tile_algorithm_config,
    to_trainer_config,
)
from bioplausible.config.omegaconf import (
    DatasetConfig as LegacyDatasetConfig,
    DomainConfig as LegacyDomainConfig,
    LightningConfig as LegacyLightningConfig,
    OptimizerConfig as LegacyOptimizerConfig,
    PropagatorConfig as LegacyPropagatorConfig,
    ScientistConfig as LegacyScientistConfig,
    SparsityConfig as LegacySparsityConfig,
    TrainingConfig as LegacyTrainingConfig,
    get_default_config,
    validate_config,
)
from bioplausible.config.omegaconf import (
    ExperimentModelConfig as LegacyModelConfig,
)
from bioplausible.config.omegaconf import (
    ExperimentSchemaConfig as LegacyExperimentConfig,
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
    # New unified exports (Sprint 7)
    "DataConfig",
    "ExperimentConfig",
    "HardwareConfig",
    "ModelConfig",
    "OntologyConfig",
    "TrainingConfig",
    "from_omegaconf",
    "to_deployment_config",
    "to_omegaconf",
    "to_system_trainer_config",
    "to_tile_algorithm_config",
    "to_trainer_config",
    # Legacy exports (backward compatibility)
    "BaseConfig",
    "BaseStructuredConfig",
    "DEFAULT_CONFIGS",
    "LegacyDatasetConfig",
    "LegacyDomainConfig",
    "LegacyExperimentConfig",
    "LegacyLightningConfig",
    "LegacyModelConfig",
    "LegacyOptimizerConfig",
    "LegacyPropagatorConfig",
    "LegacyScientistConfig",
    "LegacySparsityConfig",
    "LegacyTrainingConfig",
    "config_to_dict",
    "get_default_config",
    "get_named_config",
    "list_named_configs",
    "register_default_config",
    "validate_config",
]
