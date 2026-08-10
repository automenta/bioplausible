"""
Configuration schemas and defaults for Bioplausible experiments.

OmegaConf-based structured configs with Pydantic validation.
"""

import pathlib
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

from bioplausible.config.defaults import (
    DEFAULT_CONFIGS,
    get_named_config,
    list_named_configs,
    register_default_config,
)
from bioplausible.config.schema import (
    DatasetConfig,
    DomainConfig,
    ExperimentConfig,
    LightningConfig,
    ModelConfig,
    OptimizerConfig,
    PropagatorConfig,
    ScientistConfig,
    SparsityConfig,
    TrainingConfig,
    get_default_config,
    validate_config,
)
from bioplausible.config.unified import (
    BaseConfig,
    BaseStructuredConfig,
    config_to_dict,
)

# ──────────────────────────────────────────────
# Merged from config_loader.py
# ──────────────────────────────────────────────


class ExperimentSchema(BaseModel):
    """Schema for validating experiment configurations."""

    model: str = Field(..., description="Name of the model (e.g., LoopedMLP)")
    task: str = Field(default="mnist", description="Task name")
    hyperparams: dict[str, Any] = Field(
        default_factory=dict, description="Model hyperparameters"
    )
    training: dict[str, Any] = Field(
        default_factory=dict, description="Training settings (lr, epochs)"
    )
    description: str | None = None


class TrainerConfigSchema(BaseModel):
    """Pydantic schema for validating raw ``TrainerConfig`` inputs at the I/O boundary.

    Use ``validate_trainer_config(yaml_dict)`` to fail-fast on invalid
    configs before they reach the OmegaConf/dataclass layer.
    """

    model: str = Field(..., min_length=1)
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    propagator: str | None = Field(default=None, min_length=1)
    propagator_kwargs: dict[str, Any] = Field(default_factory=dict)
    optimizer: str = Field(default="adam", min_length=1)
    optimizer_kwargs: dict[str, Any] = Field(default_factory=dict)
    task: str = Field(default="mnist", min_length=1)
    data_kwargs: dict[str, Any] = Field(default_factory=dict)
    batch_size: int = Field(default=64, ge=1)
    val_batch_size: int | None = Field(default=None, ge=1)
    num_workers: int = Field(default=4, ge=0)
    epochs: int = Field(default=10, ge=1)
    batches_per_epoch: int | None = Field(default=None, ge=1)
    val_batches: int | None = Field(default=None, ge=1)
    grad_clip: float | None = Field(default=1.0, ge=0.0)
    use_compile: bool = False
    compile_mode: str = "reduce-overhead"
    use_lightning: bool = False
    precision: str = "32-true"
    track_energy: bool = True
    track_flops: bool = True
    track_memory: bool = True
    save_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = Field(default=1, ge=1)
    save_best_only: bool = True
    early_stopping_patience: int | None = Field(default=None, ge=1)
    early_stopping_metric: str = "val_loss"
    early_stopping_mode: str = "min"
    log_every_n_steps: int = Field(default=10, ge=1)
    log_dir: str = "logs"
    use_wandb: bool = False
    wandb_project: str | None = None
    seed: int = 42
    deterministic: bool = False
    device: str = "auto"
    tags: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)


def validate_trainer_config(data: dict[str, Any]) -> dict[str, Any]:
    """Validate raw trainer-config dict via ``TrainerConfigSchema``.

    Parameters
    ----------
    data : dict[str, Any]
        Raw config dict (e.g. from YAML or CLI).

    Returns
    -------
    dict[str, Any]
        Validated config dict (same structure, but with defaults filled).

    Raises
    ------
    ValidationError
        If any field fails validation (type, range, or constraint).
    """
    validated = TrainerConfigSchema(**data)
    return validated.model_dump(exclude_unset=False)


def load_config(path: str) -> dict[str, Any]:
    """Load and validate experiment configuration from a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        Dictionary containing the validated configuration.
    """
    if not pathlib.Path(path).exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with pathlib.Path(path).open() as f:
        try:
            raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML config: {e}")
    try:
        validated_config = ExperimentSchema(**raw_config)
        return validated_config.model_dump()
    except ValidationError as e:
        raise ValueError(f"Invalid configuration format: {e}")


__all__ = [
    # New schema exports
    "ExperimentConfig",
    "ExperimentSchema",
    "ModelConfig",
    "OptimizerConfig",
    "PropagatorConfig",
    "SparsityConfig",
    "TrainerConfigSchema",
    "TrainingConfig",
    "DatasetConfig",
    "LightningConfig",
    "DomainConfig",
    "ScientistConfig",
    "get_default_config",
    "validate_config",
    "validate_trainer_config",
    "DEFAULT_CONFIGS",
    "register_default_config",
    "get_named_config",
    "list_named_configs",
    # Unified config hierarchy (REFACTOR.md §1.1)
    "BaseConfig",
    "BaseStructuredConfig",
    "config_to_dict",
    # Merged from config_loader.py
    "load_config",
]
