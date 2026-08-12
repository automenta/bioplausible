"""
Configuration schemas and defaults for Bioplausible experiments.

OmegaConf-based structured configs with Pydantic validation.
"""

from typing import Any

from pydantic import BaseModel, Field

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


__all__ = [
    # New schema exports
    "ExperimentConfig",
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
]
