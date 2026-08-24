"""
OmegaConf-based configuration facades for the Bioplausible platform.

These are the mutable, OmegaConf-structured *document formats* consumed at
the I/O boundary (YAML presets, ``OmegaConf.merge``/``OmegaConf.structured``,
CLI overrides). They are deliberately NOT the canonical internal runtime
configs in :mod:`computronium.config.unified` — each facade's ``to_internal()``
(where one exists) is the single seam into the frozen unified hierarchy.

Renamed from the historical ``config.schema`` module so its classes cannot
collide with the same-named frozen canonical classes in ``unified``.
"""

import time
from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING, OmegaConf

from computronium.config.experiment import ModelConfig as InternalModelConfig
from computronium.core.logging import get_logger

__all__ = [
    "DatasetConfig",
    "DomainConfig",
    "ExperimentModelConfig",
    "ExperimentSchemaConfig",
    "LightningConfig",
    "OptimizerConfig",
    "PropagatorConfig",
    "RunConfig",
    "RunConfigData",
    "RunConfigModel",
    "RunConfigOptimizer",
    "RunConfigTrainer",
    "ScientistConfig",
    "SparsityConfig",
    "TrainingConfig",
    "get_default_config",
    "logger",
    "validate_config",
]
logger = get_logger()


def _register_resolvers() -> None:
    """Register OmegaConf custom resolvers. Safe to call multiple times.

    Only swallows ``ValueError`` raised when a resolver of the same name is
    already registered; any other exception surfaces for debugging.
    """
    try:
        OmegaConf.register_new_resolver("now", time.strftime)
    except ValueError:
        pass
    except Exception:  # broad: best-effort
        logger.exception("Failed to register OmegaConf resolver 'now'")


_register_resolvers()


@dataclass(slots=True)
class ExperimentModelConfig:
    """Configuration for a model component."""

    name: str = "MLP"
    kwargs: dict[str, Any] = field(default_factory=dict)
    compile: bool = False
    compile_mode: str = "reduce-overhead"

    def to_internal(
        self, input_dim: int = 0, output_dim: int = 0
    ) -> InternalModelConfig:
        """Convert to the internal frozen :class:`ModelConfig` used by models.

        Parameters from *kwargs* that match :class:`InternalModelConfig` fields
        are forwarded; the rest land in *extra*.

        Args:
            input_dim: Number of input features (known at task-setup time).
            output_dim: Number of output classes / prediction targets.
        """
        return InternalModelConfig(
            name=self.name,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[],
            extra=self.kwargs,
        )


@dataclass(slots=True)
class PropagatorConfig:
    """Configuration for a propagator/learning rule component."""

    name: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OptimizerConfig:
    """Configuration for an optimizer component."""

    name: str = "adam"
    lr: float = 0.001
    weight_decay: float = 0.0
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SparsityConfig:
    """Configuration for a sparsity component."""

    name: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DatasetConfig:
    """Configuration for datasets."""

    name: str = "mnist"
    batch_size: int = 64
    val_batch_size: int | None = None
    num_workers: int = 4
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrainingConfig:
    """Configuration for training."""

    epochs: int = 10
    batches_per_epoch: int | None = None
    val_batches: int | None = None
    grad_clip: float | None = 1.0
    precision: str = "32-true"
    log_every_n_steps: int = 10
    log_dir: str = "logs"
    save_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 1
    save_best_only: bool = False
    early_stopping_patience: int | None = None
    early_stopping_metric: str = "val_loss"
    early_stopping_mode: str = "min"


@dataclass(slots=True)
class LightningConfig:
    """Configuration for PyTorch Lightning integration."""

    use_lightning: bool = False
    precision: str = "32-true"
    accelerator: str = "auto"
    devices: int = 1
    num_nodes: int = 1
    strategy: str = "auto"


@dataclass(slots=True)
class DomainConfig:
    """Configuration for domain-specific settings."""

    domain: str = "vision"
    task_specific: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ScientistConfig:
    """Configuration for the Scientist / AutoScientist."""

    mode: str = "autonomous"
    max_trials: int = 100
    task_filter: str | None = None
    tier_limit: str | None = None
    num_workers: int = 1
    report_interval: int = 50
    human_approval_gate: bool = False
    llm_backend: str | None = None


@dataclass(slots=True)
class ExperimentSchemaConfig:
    """
    Top-level experiment configuration.

    Usage:
        config = ExperimentSchemaConfig(
            model=ExperimentModelConfig(name="tile_pc"),
            optimizer=OptimizerConfig(name="smep", lr=0.01),
            dataset=DatasetConfig(name="mnist", batch_size=128),
            trainer=TrainingConfig(epochs=20),
        )
        cfg = OmegaConf.structured(config)
        OmegaConf.save(cfg, "config.yaml")
    """

    model: ExperimentModelConfig = field(default_factory=ExperimentModelConfig)
    propagator: PropagatorConfig = field(default_factory=PropagatorConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    sparsity: SparsityConfig = field(default_factory=SparsityConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    trainer: TrainingConfig = field(default_factory=TrainingConfig)
    lightning: LightningConfig = field(default_factory=LightningConfig)
    domain: DomainConfig = field(default_factory=DomainConfig)
    scientist: ScientistConfig = field(default_factory=ScientistConfig)
    seed: int = 42
    device: str = "auto"
    output_dir: str = "results/${now:%Y%m%d_%H%M%S}"
    tags: dict[str, Any] = field(default_factory=dict)
    track_energy: bool = True
    track_flops: bool = True
    track_memory: bool = True
    use_wandb: bool = False
    wandb_project: str | None = None
    deterministic: bool = False


def get_default_config() -> ExperimentSchemaConfig:
    """Get the default experiment configuration."""
    return ExperimentSchemaConfig()


def validate_config(cfg: Any) -> ExperimentSchemaConfig:
    """
    Validate and convert a configuration to ExperimentSchemaConfig.

    Args:
        cfg: Dict, OmegaConf DictConfig, or ExperimentSchemaConfig.

    Returns:
        Validated ExperimentSchemaConfig.
    """
    if isinstance(cfg, ExperimentSchemaConfig):
        return cfg
    if isinstance(cfg, dict):
        return OmegaConf.to_object(
            OmegaConf.merge(
                OmegaConf.structured(ExperimentSchemaConfig),
                OmegaConf.create(cfg),
            )
        )
    return OmegaConf.to_object(
        OmegaConf.merge(
            OmegaConf.structured(ExperimentSchemaConfig),
            cfg,
        )
    )


# ──────────────────────────────────────────────
# Run configuration types for YAML-driven experiments
# ──────────────────────────────────────────────


@dataclass(slots=True)
class RunConfigData:
    """Data section of a :class:`RunConfig` YAML experiment."""

    task: str = MISSING
    batch_size: int = 64
    seq_len: int = 64
    augment: bool = False
    data_fraction: float = 1.0


@dataclass(slots=True)
class RunConfigModel:
    """Model section of a :class:`RunConfig` YAML experiment."""

    name: str = MISSING
    hidden_dim: int = 256
    num_layers: int = 3
    extra: dict[str, Any] = field(default_factory=dict)

    def to_internal(
        self, input_dim: int = 0, output_dim: int = 0
    ) -> InternalModelConfig:
        """Convert to the internal frozen :class:`ModelConfig` used by models."""
        return InternalModelConfig(
            name=self.name,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[self.hidden_dim] * max(self.num_layers, 1),
            extra=self.extra,
        )


@dataclass(slots=True)
class RunConfigOptimizer:
    """Optimizer section of a :class:`RunConfig` YAML experiment."""

    name: str = "adam"
    lr: float = 0.001
    weight_decay: float = 0.0
    beta: float = 0.5
    settle_steps: int = 30
    mode: str = "ep"


@dataclass(slots=True)
class RunConfigTrainer:
    """Trainer section of a :class:`RunConfig` YAML experiment."""

    epochs: int = 10
    batches_per_epoch: int = 100
    grad_clip: float | None = None
    scheduler: str | None = None
    use_compile: bool = True
    track_energy: bool = True


@dataclass(slots=True)
class RunConfig:
    """Top-level YAML-driven configuration consumed by ``run_from_runconfig``."""

    seed: int = 42
    device: str = "auto"
    output_dir: str = "results/${now:%Y%m%d_%H%M%S}"
    data: RunConfigData = field(default_factory=RunConfigData)
    model: RunConfigModel = field(default_factory=RunConfigModel)
    optimizer: RunConfigOptimizer = field(default_factory=RunConfigOptimizer)
    trainer: RunConfigTrainer = field(default_factory=RunConfigTrainer)
    ablation_tags: dict[str, Any] = field(default_factory=dict)
