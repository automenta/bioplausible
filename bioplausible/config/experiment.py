"""Unified Experiment Configuration (Sprint 7).

Single source of truth for all hyperparameters across domains.
Replaces fragmented configs in:
- config/unified.py (ModelConfig, DataConfig, ExperimentConfig, etc.)
- config/omegaconf.py (ExperimentSchemaConfig, RunConfig, etc.)
- core/trainer.py (TrainerConfig, LMTrainingConfig)
- zoo/models/deployments/base.py (DeploymentConfig, ConvDeploymentConfig, etc.)
- core/local_learning/algorithm.py (TileAlgorithmConfig)

ALL FIELDS ARE REQUIRED - NO DEFAULTS.
Experiments must specify every parameter explicitly via YAML or programmatic construction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from bioplausible.core.ontology import (
        CreditAssignmentConfig,
        GeometryConfig,
        ParameterUpdateConfig,
        StateDynamicsConfig,
        SubstrateConfig,
        SystemConfig,
    )

__all__ = [
    "ExperimentConfig",
    "HardwareConfig",
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "SystemConfig",
    "to_omegaconf",
    "from_omegaconf",
    "to_trainer_config",
    "to_system_trainer_config",
    "to_deployment_config",
    "to_tile_algorithm_config",
    "make_vision_preset",
    "make_lm_preset",
    "make_graph_preset",
    "make_rl_preset",
    "make_timeseries_preset",
]


# ──────────────────────────────────────────────
# Core Configuration Primitives (ALL FIELDS REQUIRED)
# ──────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class HardwareConfig:
    """Hardware and execution configuration. All fields required."""

    device: str
    precision: str
    use_compile: bool
    compile_mode: Literal["reduce-overhead", "max-autotune", "default"]
    distributed: bool
    world_size: int
    rank: int
    master_addr: str
    master_port: int
    substrate: Literal[
        "digital", "analog", "memristive", "neuromorphic", "optical", "quantum"
    ]
    substrate_kwargs: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Model architecture configuration. All fields required."""

    # Architecture identification
    name: str
    model_type: Literal["mlp", "conv", "rnn", "transformer", "tile", "equilibrium"]

    # Dimensions
    input_dim: int
    output_dim: int
    hidden_dims: tuple[int, ...]
    num_layers: int

    # Equilibrium/biological parameters
    learning_rate: float
    beta: float
    max_steps: int
    convergence_threshold: float
    convergence_start: int

    # Architecture features
    use_spectral_norm: bool
    spectral_norm_power_iterations: int
    activation: Literal["silu", "relu", "gelu", "tanh", "sigmoid"]
    lipschitz_mode: Literal["power_iteration", "svd"]
    output_scaling_mode: Literal["uniform", "mupc"]
    dropout: float

    # Tile-specific
    neurons_per_tile: int
    tiles_per_layer: int
    algorithm: Literal["ep", "fa", "tp", "pc", "hebbian", "snn", "gnn"]
    mode: Literal["ep", "fa", "backprop", "pc"]
    inference_steps: int
    step_size: float

    # Conv-specific
    input_channels: int
    input_size: int
    conv_channels: tuple[int, ...]
    kernel_sizes: tuple[int, ...]
    use_pooling: bool
    pooling_size: int

    # Transformer-specific
    attention_heads: int
    use_positional_encoding: bool
    use_temporal_attention: bool

    # RNN-specific
    seq_len: int
    hidden_dim: int

    # RL-specific
    obs_dim: int
    action_dim: int
    action_type: Literal["discrete", "continuous"]
    log_std_init: float
    log_std_min: float
    log_std_max: float
    entropy_coef: float
    value_coef: float
    max_grad_norm: float

    # Graph-specific
    node_features: int
    aggregation: Literal["mean", "sum", "max", "attention"]
    readout: Literal["mean", "sum", "max", "attention"]

    # Additional kwargs
    extra: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Training hyperparameters. All fields required."""

    # Optimizer
    optimizer: Literal["adam", "adamw", "sgd", "smep", "muon", "lamb"]
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    eps: float

    # Scheduler
    scheduler: Literal[
        "none", "cosine", "linear", "constant", "cosine_warmup", "step", "exponential"
    ]
    warmup_steps: int
    min_lr_ratio: float
    scheduler_kwargs: dict[str, Any]

    # Training loop
    epochs: int
    batches_per_epoch: int | None
    val_batches: int | None
    max_epoch_time: float
    grad_clip: float | None

    # Batch sizes
    batch_size: int
    val_batch_size: int | None
    num_workers: int

    # Gradient accumulation (LM-style)
    gradient_accumulation_steps: int

    # Mixed precision
    use_amp: bool

    # Logging / monitoring
    log_every_n_steps: int
    track_energy: bool
    track_flops: bool
    track_memory: bool

    # Checkpointing
    save_checkpoints: bool
    checkpoint_dir: str
    save_every_n_epochs: int
    save_best_only: bool

    # Early stopping
    early_stopping_patience: int | None
    early_stopping_metric: str
    early_stopping_mode: Literal["min", "max"]

    # Bio-rule honesty
    allow_bptt_fallback: bool

    # Validation / profiling
    run_validation: bool
    profile_epochs: bool

    # Kernel acceleration
    use_kernel: bool
    kernel_backend: Literal["triton", "cupy", "pytorch", "contrastive"]
    kernel_dtype: Literal["float32", "float16", "bfloat16"]

    # Extra
    extra: dict[str, Any]


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Data and task configuration. All fields required."""

    task: str
    domain: Literal["vision", "lm", "graph", "rl", "timeseries", "tabular"]

    # Data loading
    batch_size: int
    val_batch_size: int | None
    num_workers: int
    seq_len: int
    augment: bool
    data_fraction: float

    # Splits
    train_split: float
    val_split: float
    test_split: float

    # Domain-specific
    data_kwargs: dict[str, Any]


# Re-export SystemConfig from ontology as the unified 5-D config
from bioplausible.core.ontology import SystemConfig


# ──────────────────────────────────────────────
# Top-Level Experiment Configuration (ALL FIELDS REQUIRED)
# ──────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Unified experiment configuration. ALL FIELDS REQUIRED.

    Single source of truth for all hyperparameters. Composes the five
    configuration primitives into a coherent whole.

    Usage:
        config = ExperimentConfig(
            model=ModelConfig(...),
            training=TrainingConfig(...),
            data=DataConfig(...),
            hardware=HardwareConfig(...),
            system=SystemConfig(...),
            name="my_experiment",
            seed=42,
            description="",
            tags=(),
            output_dir="results",
            use_wandb=False,
            wandb_project=None,
            wandb_entity=None,
            deterministic=False,
        )
    """

    # Core components (all required)
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    hardware: HardwareConfig
    system: SystemConfig

    # Metadata
    name: str
    seed: int
    description: str
    tags: tuple[str, ...]

    # Output
    output_dir: str
    use_wandb: bool
    wandb_project: str | None
    wandb_entity: str | None

    # Determinism
    deterministic: bool


# ──────────────────────────────────────────────
# Conversion Helpers (to legacy configs)
# ──────────────────────────────────────────────


def to_omegaconf(config: ExperimentConfig) -> Any:
    """Convert ExperimentConfig to OmegaConf structured config (for YAML I/O)."""
    from omegaconf import OmegaConf

    return OmegaConf.structured(config)


def from_omegaconf(cfg: Any) -> ExperimentConfig:
    """Create ExperimentConfig from OmegaConf config (loaded from YAML)."""
    from omegaconf import OmegaConf

    merged = OmegaConf.merge(ExperimentConfig, cfg)
    return OmegaConf.to_object(merged)  # type: ignore[return-value]


def to_trainer_config(config: ExperimentConfig):
    """Convert to core.trainer.TrainerConfig for CoreTrainer."""
    from bioplausible.core.trainer import TrainerConfig

    return TrainerConfig(
        model=config.model.name,
        model_kwargs={
            **config.model.extra,
            "hidden_dims": list(config.model.hidden_dims),
            "num_layers": config.model.num_layers,
            "learning_rate": config.model.learning_rate,
            "beta": config.model.beta,
            "max_steps": config.model.max_steps,
            "convergence_threshold": config.model.convergence_threshold,
            "convergence_start": config.model.convergence_start,
            "use_spectral_norm": config.model.use_spectral_norm,
            "activation": config.model.activation,
        },
        propagator=config.model.extra.get("propagator"),
        propagator_kwargs=config.model.extra.get("propagator_kwargs", {}),
        optimizer=config.training.optimizer,
        optimizer_kwargs={
            "lr": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
            **config.training.scheduler_kwargs,
        },
        task=config.data.task,
        data_kwargs=config.data.data_kwargs,
        batch_size=config.training.batch_size,
        val_batch_size=config.training.val_batch_size or config.data.val_batch_size,
        num_workers=config.data.num_workers,
        epochs=config.training.epochs,
        batches_per_epoch=config.training.batches_per_epoch,
        val_batches=config.training.val_batches,
        max_epoch_time=config.training.max_epoch_time,
        grad_clip=config.training.grad_clip,
        use_compile=config.hardware.use_compile,
        compile_mode=config.hardware.compile_mode,
        precision=config.hardware.precision,
        track_energy=config.training.track_energy,
        track_flops=config.training.track_flops,
        track_memory=config.training.track_memory,
        allow_bptt_fallback=config.training.allow_bptt_fallback,
        run_validation=config.training.run_validation,
        profile_epochs=config.training.profile_epochs,
        target_hardware=config.hardware.substrate
        if config.hardware.substrate != "digital"
        else None,
        use_kernel=config.training.use_kernel,
        kernel_backend=config.training.kernel_backend,
        kernel_dtype=config.training.kernel_dtype,
        save_checkpoints=config.training.save_checkpoints,
        checkpoint_dir=config.training.checkpoint_dir,
        save_every_n_epochs=config.training.save_every_n_epochs,
        save_best_only=config.training.save_best_only,
        early_stopping_patience=config.training.early_stopping_patience,
        early_stopping_metric=config.training.early_stopping_metric,
        early_stopping_mode=config.training.early_stopping_mode,
        log_every_n_steps=config.training.log_every_n_steps,
        log_dir=config.output_dir,
        use_wandb=config.use_wandb,
        wandb_project=config.wandb_project,
        seed=config.seed,
        deterministic=config.deterministic,
        device=config.hardware.device,
        tags={tag: True for tag in config.tags} if config.tags else {},
        extra={
            **config.training.extra,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "use_amp": config.training.use_amp,
            "min_lr_ratio": config.training.min_lr_ratio,
            "scheduler": config.training.scheduler,
            "warmup_steps": config.training.warmup_steps,
        },
    )


def to_system_trainer_config(config: ExperimentConfig):
    """Convert to core.system_trainer.SystemTrainerConfig."""
    from bioplausible.core.system_trainer import SystemTrainerConfig

    return SystemTrainerConfig(
        max_epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        val_batch_size=config.training.val_batch_size or config.data.val_batch_size,
        device=config.hardware.device,
        grad_clip=config.training.grad_clip,
        track_energy=config.training.track_energy,
        track_flops=config.training.track_flops,
        track_memory=config.training.track_memory,
        log_every_n_steps=config.training.log_every_n_steps,
        seed=config.seed,
        deterministic=config.deterministic,
    )


def to_deployment_config(config: ExperimentConfig):
    """Convert to zoo.models.deployments.base.DeploymentConfig."""
    from bioplausible.zoo.models.deployments.base import DeploymentConfig

    # Narrow algorithm to DeploymentConfig's Literal
    algo = config.model.algorithm
    if algo == "gnn":
        algo = "ep"  # fallback
    # Narrow mode to DeploymentConfig's Literal
    mode = config.model.mode
    if mode == "fa":
        mode = "ep"  # fallback
    # Narrow activation to DeploymentConfig's Literal
    activation = config.model.activation
    if activation == "sigmoid":
        activation = "tanh"  # fallback

    return DeploymentConfig(
        neurons_per_tile=config.model.neurons_per_tile,
        tiles_per_layer=config.model.tiles_per_layer,
        num_fc_layers=config.model.num_layers,
        learning_rate=config.model.learning_rate,
        dropout=config.model.dropout,
        weight_decay=config.training.weight_decay,
        algorithm=algo,
        mode=mode,
        inference_steps=config.model.inference_steps,
        step_size=config.model.step_size,
        beta=config.model.beta,
        activation=activation,
        task_type=config.model.extra.get("task_type", "classification"),
        equitile_kwargs=config.model.extra,
    )


def to_tile_algorithm_config(config: ExperimentConfig):
    """Convert to core.local_learning.algorithm.TileAlgorithmConfig."""
    from bioplausible.core.local_learning.algorithm import TileAlgorithmConfig

    return TileAlgorithmConfig(
        input_dim=config.model.input_dim,
        output_dim=config.model.output_dim,
        neurons_per_tile=config.model.neurons_per_tile,
        tiles_per_layer=config.model.tiles_per_layer,
        num_hidden_layers=config.model.num_layers,
        use_skip_connections=config.model.extra.get("use_skip_connections", False),
        algorithm=config.model.algorithm,
        beta=config.model.beta,
        step_size=config.model.step_size,
        lambda_error=config.model.extra.get("lambda_error", 1.0),
        clamp_min=config.model.extra.get("clamp_min", -10.0),
        clamp_max=config.model.extra.get("clamp_max", 10.0),
        clamp=config.model.extra.get("clamp", True),
        free_steps=config.model.max_steps,
        nudged_steps=config.model.max_steps,
        learning_rate=config.model.learning_rate,
        importance_lr=config.training.scheduler_kwargs.get("importance_lr", 0.01),
        mode=config.model.mode,
        extra=config.model.extra,
    )


# ──────────────────────────────────────────────
# Preset Factories (for common experiment templates)
# ──────────────────────────────────────────────


def _base_hardware() -> HardwareConfig:
    return HardwareConfig(
        device="cuda",
        precision="32-true",
        use_compile=False,
        compile_mode="reduce-overhead",
        distributed=False,
        world_size=1,
        rank=0,
        master_addr="localhost",
        master_port=29500,
        substrate="digital",
        substrate_kwargs={},
    )


def _base_training() -> TrainingConfig:
    return TrainingConfig(
        optimizer="adam",
        learning_rate=0.001,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        scheduler="cosine_warmup",
        warmup_steps=100,
        min_lr_ratio=0.1,
        scheduler_kwargs={},
        epochs=10,
        batches_per_epoch=None,
        val_batches=None,
        max_epoch_time=0.0,
        grad_clip=1.0,
        batch_size=64,
        val_batch_size=None,
        num_workers=4,
        gradient_accumulation_steps=1,
        use_amp=True,
        log_every_n_steps=10,
        track_energy=True,
        track_flops=True,
        track_memory=True,
        save_checkpoints=True,
        checkpoint_dir="checkpoints",
        save_every_n_epochs=1,
        save_best_only=True,
        early_stopping_patience=None,
        early_stopping_metric="val_loss",
        early_stopping_mode="min",
        allow_bptt_fallback=True,
        run_validation=True,
        profile_epochs=False,
        use_kernel=False,
        kernel_backend="triton",
        kernel_dtype="float32",
        extra={},
    )


def _base_data(domain: str, task: str) -> DataConfig:
    return DataConfig(
        task=task,
        domain=domain,  # type: ignore[arg-type]
        batch_size=64,
        val_batch_size=None,
        num_workers=4,
        seq_len=64,
        augment=False,
        data_fraction=1.0,
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        data_kwargs={},
    )


def _base_system() -> SystemConfig:
    from bioplausible.core.ontology import (
        SubstrateConfig,
        GeometryConfig,
        StateDynamicsConfig,
        CreditAssignmentConfig,
        ParameterUpdateConfig,
        SystemConfig,
    )

    return SystemConfig(
        substrate=SubstrateConfig.digital(),
        geometry=GeometryConfig.feedforward(input_dim=0, output_dim=0, hidden_dims=()),
        dynamics=StateDynamicsConfig.instantaneous(),
        credit=CreditAssignmentConfig.gradient(),
        update=ParameterUpdateConfig.euclidean(),
    )


def _override_config(base, **overrides):
    """Create a new frozen dataclass instance with overridden fields."""
    return type(base)(**{**asdict(base), **overrides})


def make_vision_preset(
    *,
    name: str = "vision_mlp",
    seed: int = 42,
    hidden_dims: tuple[int, ...] = (256, 256, 256),
    learning_rate: float = 0.001,
    epochs: int = 10,
    task: str = "mnist",
    batch_size: int = 64,
) -> ExperimentConfig:
    """Create a vision experiment preset (MLP on MNIST/CIFAR)."""
    return ExperimentConfig(
        model=ModelConfig(
            name="MLP",
            model_type="mlp",
            input_dim=784,
            output_dim=10,
            hidden_dims=hidden_dims,
            num_layers=len(hidden_dims),
            learning_rate=learning_rate,
            beta=0.2,
            max_steps=30,
            convergence_threshold=1e-3,
            convergence_start=5,
            use_spectral_norm=True,
            spectral_norm_power_iterations=5,
            activation="silu",
            lipschitz_mode="power_iteration",
            output_scaling_mode="mupc",
            dropout=0.0,
            neurons_per_tile=48,
            tiles_per_layer=4,
            algorithm="ep",
            mode="ep",
            inference_steps=10,
            step_size=0.1,
            input_channels=3,
            input_size=32,
            conv_channels=(32, 64, 128),
            kernel_sizes=(3, 3, 3),
            use_pooling=True,
            pooling_size=2,
            attention_heads=4,
            use_positional_encoding=True,
            use_temporal_attention=True,
            seq_len=64,
            hidden_dim=64,
            obs_dim=8,
            action_dim=4,
            action_type="discrete",
            log_std_init=0.0,
            log_std_min=-20.0,
            log_std_max=2.0,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=0.5,
            node_features=10,
            aggregation="mean",
            readout="mean",
            extra={},
        ),
        training=_override_config(
            _base_training(),
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
        ),
        data=_base_data("vision", task),
        hardware=_base_hardware(),
        system=_base_system(),
        name=name,
        seed=seed,
        description=f"Vision {task} with MLP",
        tags=("vision", task),
        output_dir="results",
        use_wandb=False,
        wandb_project=None,
        wandb_entity=None,
        deterministic=False,
    )


def make_lm_preset(
    *,
    name: str = "lm_mlp",
    seed: int = 42,
    hidden_dims: tuple[int, ...] = (512, 512, 512, 512),
    learning_rate: float = 3e-4,
    epochs: int = 20,
    task: str = "tiny_shakespeare",
    batch_size: int = 32,
    seq_len: int = 256,
    vocab_size: int = 65,
) -> ExperimentConfig:
    """Create a language modeling experiment preset."""
    return ExperimentConfig(
        model=ModelConfig(
            name="MLP",
            model_type="mlp",
            input_dim=seq_len,
            output_dim=vocab_size,
            hidden_dims=hidden_dims,
            num_layers=len(hidden_dims),
            learning_rate=learning_rate,
            beta=0.2,
            max_steps=30,
            convergence_threshold=1e-3,
            convergence_start=5,
            use_spectral_norm=True,
            spectral_norm_power_iterations=5,
            activation="gelu",
            lipschitz_mode="power_iteration",
            output_scaling_mode="mupc",
            dropout=0.1,
            neurons_per_tile=48,
            tiles_per_layer=4,
            algorithm="ep",
            mode="ep",
            inference_steps=10,
            step_size=0.1,
            input_channels=3,
            input_size=32,
            conv_channels=(32, 64, 128),
            kernel_sizes=(3, 3, 3),
            use_pooling=True,
            pooling_size=2,
            attention_heads=4,
            use_positional_encoding=True,
            use_temporal_attention=True,
            seq_len=seq_len,
            hidden_dim=512,
            obs_dim=8,
            action_dim=4,
            action_type="discrete",
            log_std_init=0.0,
            log_std_min=-20.0,
            log_std_max=2.0,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=0.5,
            node_features=10,
            aggregation="mean",
            readout="mean",
            extra={"vocab_size": vocab_size},
        ),
        training=TrainingConfig(
            **asdict(_base_training()),
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            scheduler="cosine_warmup",
            warmup_steps=100,
            use_amp=True,
            gradient_accumulation_steps=4,
        ),
        data=DataConfig(
            task=task,
            domain="lm",
            batch_size=batch_size,
            val_batch_size=None,
            num_workers=4,
            seq_len=seq_len,
            augment=False,
            data_fraction=1.0,
            train_split=0.9,
            val_split=0.05,
            test_split=0.05,
            data_kwargs={"vocab_size": vocab_size},
        ),
        hardware=_base_hardware(),
        system=_base_system(),
        name=name,
        seed=seed,
        description=f"LM {task} with MLP",
        tags=("lm", task),
        output_dir="results",
        use_wandb=False,
        wandb_project=None,
        wandb_entity=None,
        deterministic=False,
    )


def make_graph_preset(
    *,
    name: str = "graph_gnn",
    seed: int = 42,
    hidden_dims: tuple[int, ...] = (64, 64, 64),
    learning_rate: float = 0.001,
    epochs: int = 50,
    task: str = "cora",
    batch_size: int = 1,
    node_features: int = 1433,
    num_classes: int = 7,
) -> ExperimentConfig:
    """Create a graph experiment preset."""
    return ExperimentConfig(
        model=ModelConfig(
            name="GNN",
            model_type="transformer",
            input_dim=node_features,
            output_dim=num_classes,
            hidden_dims=hidden_dims,
            num_layers=len(hidden_dims),
            learning_rate=learning_rate,
            beta=0.2,
            max_steps=30,
            convergence_threshold=1e-3,
            convergence_start=5,
            use_spectral_norm=True,
            spectral_norm_power_iterations=5,
            activation="relu",
            lipschitz_mode="power_iteration",
            output_scaling_mode="mupc",
            dropout=0.5,
            neurons_per_tile=48,
            tiles_per_layer=4,
            algorithm="gnn",
            mode="ep",
            inference_steps=10,
            step_size=0.1,
            input_channels=3,
            input_size=32,
            conv_channels=(32, 64, 128),
            kernel_sizes=(3, 3, 3),
            use_pooling=True,
            pooling_size=2,
            attention_heads=4,
            use_positional_encoding=True,
            use_temporal_attention=True,
            seq_len=64,
            hidden_dim=64,
            obs_dim=8,
            action_dim=4,
            action_type="discrete",
            log_std_init=0.0,
            log_std_min=-20.0,
            log_std_max=2.0,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=0.5,
            node_features=node_features,
            aggregation="mean",
            readout="mean",
            extra={},
        ),
        training=_override_config(
            _base_training(),
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
        ),
        data=_base_data("graph", task),
        hardware=_base_hardware(),
        system=_base_system(),
        name=name,
        seed=seed,
        description=f"Graph {task} with GNN",
        tags=("graph", task),
        output_dir="results",
        use_wandb=False,
        wandb_project=None,
        wandb_entity=None,
        deterministic=False,
    )


def make_rl_preset(
    *,
    name: str = "rl_ppo",
    seed: int = 42,
    hidden_dims: tuple[int, ...] = (128, 128),
    learning_rate: float = 3e-4,
    epochs: int = 100,
    task: str = "cartpole",
    obs_dim: int = 4,
    action_dim: int = 2,
    action_type: Literal["discrete", "continuous"] = "discrete",
) -> ExperimentConfig:
    """Create an RL experiment preset."""
    return ExperimentConfig(
        model=ModelConfig(
            name="RL",
            model_type="mlp",
            input_dim=obs_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
            num_layers=len(hidden_dims),
            learning_rate=learning_rate,
            beta=0.2,
            max_steps=30,
            convergence_threshold=1e-3,
            convergence_start=5,
            use_spectral_norm=False,
            spectral_norm_power_iterations=5,
            activation="tanh",
            lipschitz_mode="power_iteration",
            output_scaling_mode="mupc",
            dropout=0.0,
            neurons_per_tile=48,
            tiles_per_layer=4,
            algorithm="ep",
            mode="ep",
            inference_steps=10,
            step_size=0.1,
            input_channels=3,
            input_size=32,
            conv_channels=(32, 64, 128),
            kernel_sizes=(3, 3, 3),
            use_pooling=True,
            pooling_size=2,
            attention_heads=4,
            use_positional_encoding=True,
            use_temporal_attention=True,
            seq_len=64,
            hidden_dim=128,
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_type=action_type,
            log_std_init=0.0,
            log_std_min=-20.0,
            log_std_max=2.0,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=0.5,
            node_features=10,
            aggregation="mean",
            readout="mean",
            extra={},
        ),
        training=TrainingConfig(
            **asdict(_base_training()),
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=64,
            optimizer="adam",
            scheduler="linear",
        ),
        data=_base_data("rl", task),
        hardware=_base_hardware(),
        system=_base_system(),
        name=name,
        seed=seed,
        description=f"RL {task} with PPO-style MLP",
        tags=("rl", task),
        output_dir="results",
        use_wandb=False,
        wandb_project=None,
        wandb_entity=None,
        deterministic=False,
    )


def make_timeseries_preset(
    *,
    name: str = "ts_forecast",
    seed: int = 42,
    hidden_dims: tuple[int, ...] = (64, 64, 64),
    learning_rate: float = 0.001,
    epochs: int = 50,
    task: str = "etth1",
    input_dim: int = 7,
    output_dim: int = 1,
    seq_len: int = 96,
    pred_len: int = 24,
) -> ExperimentConfig:
    """Create a time-series forecasting experiment preset."""
    return ExperimentConfig(
        model=ModelConfig(
            name="Timeseries",
            model_type="rnn",
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            num_layers=len(hidden_dims),
            learning_rate=learning_rate,
            beta=0.2,
            max_steps=30,
            convergence_threshold=1e-3,
            convergence_start=5,
            use_spectral_norm=False,
            spectral_norm_power_iterations=5,
            activation="tanh",
            lipschitz_mode="power_iteration",
            output_scaling_mode="mupc",
            dropout=0.1,
            neurons_per_tile=48,
            tiles_per_layer=4,
            algorithm="ep",
            mode="ep",
            inference_steps=10,
            step_size=0.1,
            input_channels=3,
            input_size=32,
            conv_channels=(32, 64, 128),
            kernel_sizes=(3, 3, 3),
            use_pooling=True,
            pooling_size=2,
            attention_heads=4,
            use_positional_encoding=True,
            use_temporal_attention=True,
            seq_len=seq_len,
            hidden_dim=64,
            obs_dim=8,
            action_dim=4,
            action_type="discrete",
            log_std_init=0.0,
            log_std_min=-20.0,
            log_std_max=2.0,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=0.5,
            node_features=10,
            aggregation="mean",
            readout="mean",
            extra={"pred_len": pred_len},
        ),
        training=TrainingConfig(
            **asdict(_base_training()),
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=32,
        ),
        data=DataConfig(
            task=task,
            domain="timeseries",
            batch_size=32,
            val_batch_size=None,
            num_workers=4,
            seq_len=seq_len,
            augment=False,
            data_fraction=1.0,
            train_split=0.7,
            val_split=0.15,
            test_split=0.15,
            data_kwargs={"pred_len": pred_len},
        ),
        hardware=_base_hardware(),
        system=_base_system(),
        name=name,
        seed=seed,
        description=f"Timeseries {task} forecasting",
        tags=("timeseries", task),
        output_dir="results",
        use_wandb=False,
        wandb_project=None,
        wandb_entity=None,
        deterministic=False,
    )
