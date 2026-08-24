"""
Configuration schemas and defaults for Bioplausible experiments.

Unified ExperimentConfig (Sprint 7) is the single source of truth.
Legacy OmegaConf schemas retained for backward compatibility.
"""

from computronium.config.defaults import (
    DEFAULT_CONFIGS,
    get_named_config,
    list_named_configs,
    register_default_config,
)
from computronium.config.experiment import (
    DataConfig,
    ExperimentConfig,
    HardwareConfig,
    ModelConfig,
    SystemConfig,
    TrainingConfig,
    from_omegaconf,
    to_deployment_config,
    to_omegaconf,
    to_system_trainer_config,
    to_tile_algorithm_config,
    to_trainer_config,
)
from computronium.config.omegaconf import (
    DatasetConfig as LegacyDatasetConfig,
)
from computronium.config.omegaconf import (
    DomainConfig as LegacyDomainConfig,
)
from computronium.config.omegaconf import (
    ExperimentModelConfig as LegacyModelConfig,
)
from computronium.config.omegaconf import (
    ExperimentSchemaConfig as LegacyExperimentConfig,
)
from computronium.config.omegaconf import (
    LightningConfig as LegacyLightningConfig,
)
from computronium.config.omegaconf import (
    OptimizerConfig as LegacyOptimizerConfig,
)
from computronium.config.omegaconf import (
    PropagatorConfig as LegacyPropagatorConfig,
)
from computronium.config.omegaconf import (
    ScientistConfig as LegacyScientistConfig,
)
from computronium.config.omegaconf import (
    SparsityConfig as LegacySparsityConfig,
)
from computronium.config.omegaconf import (
    TrainingConfig as LegacyTrainingConfig,
)
from computronium.config.omegaconf import (
    get_default_config,
    validate_config,
)
from computronium.config.unified import (
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
    "SystemConfig",
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
