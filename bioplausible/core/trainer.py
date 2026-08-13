"""
CoreTrainer: Unified Training Class

Replaces multiple runners (runner.py, SupervisedTrainer, etc.).
Accepts a config dict/YAML/OmegaConf specifying model, propagator, optimizer, data,
and trainer_args. Uses Lightning for distributed but provides a clean local-first API.
"""

import json
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeIs, cast

import optuna
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from bioplausible.core.checkpoint import (
    load_checkpoint as load_checkpoint_file,
)
from bioplausible.core.checkpoint import (
    save_checkpoint as save_checkpoint_file,
)
from bioplausible.core.ebm import EBMTrainer, EnergyModel
from bioplausible.core.logging import get_logger
from bioplausible.core.losses import compute_accuracy, compute_loss
from bioplausible.core.metrics import BaseMetrics
from bioplausible.core.profiling import EnergyTracker
from bioplausible.core.registry import (
    ComponentCategory,
    IncompatibilityError,
    Registry,
)
from bioplausible.core.utils.device import get_device
from bioplausible.domains.base import DomainType
from bioplausible.execution.callbacks import ExecutionCallback
from bioplausible.utils import count_parameters

if TYPE_CHECKING:
    from bioplausible.domains import TaskProtocol

logger = get_logger()


class _LearningRuleOptimizer(Protocol):
    """Duck-typed view of a zoo ``LearningRuleOptimizer`` callable surface.

    Keeps ``core.trainer`` decoupled from the ``bioplausible.zoo`` package
    (Sprint 0.5): the trainer narrows via a marker attribute instead of
    importing the concrete class, so light consumers of ``core.trainer`` never
    pull the whole zoo (→ torchvision/lightning/optuna).
    """

    def step(self, x: torch.Tensor, target: torch.Tensor | None = None) -> None: ...


def _is_learning_rule_optimizer(o: object) -> TypeIs[_LearningRuleOptimizer]:
    """Type-narrowing guard for the learning-rule-optimizer calling convention."""
    return bool(getattr(type(o), "_is_learning_rule", False))


def _default_output_base() -> Path:
    """Get base output directory. Uses unique tempdir during pytest runs."""
    if env_dir := os.environ.get("BIOPL_OUTPUT_DIR"):
        return Path(env_dir)
    if "PYTEST_CURRENT_TEST" in os.environ or "pytest" in os.environ.get("_", ""):
        return Path(tempfile.mkdtemp(prefix="biopl-runs-"))
    return Path("logs")


class TrainerProtocol(Protocol):
    """Protocol for training interfaces.

    Both ``CoreTrainer`` and the legacy ``_TaskTrainer`` satisfy this
    protocol, allowing callers to train via a uniform ``train_epoch()``
    interface regardless of how the trainer was constructed.
    """

    def train_epoch(self) -> dict[str, float]: ...


def _make_ebm_trainer(config: TrainerConfig, model: nn.Module) -> EBMTrainer:
    """Create an EBMTrainer from trainer config."""
    return EBMTrainer(
        model,
        lr=config.optimizer_kwargs.get("lr", 0.01),
        free_steps=config.extra.get("free_steps", 30),
        nudged_steps=config.extra.get("nudged_steps"),
        beta=config.extra.get("beta", 0.1),
        clip_grad_norm=config.grad_clip,
    )


@dataclass
class TrainerConfig:
    """Configuration for CoreTrainer."""

    # Model
    model: str  # Registry name
    model_kwargs: dict[str, Any] = field(default_factory=dict)

    # Propagator / Learning Rule (optional, can be part of model)
    propagator: str | None = None
    propagator_kwargs: dict[str, Any] = field(default_factory=dict)

    # Optimizer
    optimizer: str = "adam"
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)

    # Data
    task: str = "mnist"  # Task name (mnist, cifar10, shakespeare, etc.)
    data_kwargs: dict[str, Any] = field(default_factory=dict)
    batch_size: int = 64
    val_batch_size: int | None = None
    num_workers: int = 4

    # Training
    epochs: int = 10
    batches_per_epoch: int | None = None
    val_batches: int | None = None
    # Hard per-epoch wall-clock budget in seconds. When >0, a training epoch
    # stops consuming batches once the budget is exhausted (measured from the
    # epoch's first batch). Bounds slow-settling rule models (eqprop) so a
    # shallow breadth probe cannot burn the whole sweep on one epoch.
    max_epoch_time: float = 0.0
    grad_clip: float | None = 1.0
    use_compile: bool = False
    compile_mode: str = "reduce-overhead"
    use_lightning: bool = False
    precision: str = "32-true"  # "16-mixed", "bf16-mixed", "32-true"

    # Energy/Monitoring
    track_energy: bool = True
    track_flops: bool = True
    track_memory: bool = True

    # Bio-rule honesty: when False, a model that silently degrades to plain
    # BPTT (no train_step, no learning-rule propagator) is not silently
    # accepted — a loud warning is raised and `training_path="bptt"` is
    # recorded so the sweep can flag it as a DEFECT. Bio families set this
    # False in the sweep; backprop/forward models keep it True.
    allow_bptt_fallback: bool = True

    # Profiling / eval toggles (plan §10): disabling these isolates the raw
    # train-loop floor from validation and per-batch instrumentation overhead.
    run_validation: bool = True
    profile_epochs: bool = False

    # Hardware target (plan §17): swaps in a substrate-faithful model facade so
    # `cost_of_plausibility` is hardware-aware rather than idealized-digital-GPU.
    # ``None``/``"gpu"`` = digital reference; ``"fpga"`` → ``QuantizedLoopedMLP``;
    # ``"analog"`` → ``NoisyLoopedMLP``. Only affects the LoopedMLP equilibrium
    # family; inert for all other models. Stored as ``str`` (not ``Literal``)
    # because ``TrainerConfig`` round-trips through OmegaConf, which does not
    # yet serialize ``typing.Literal`` fields.
    target_hardware: str | None = None

    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 1
    save_best_only: bool = True

    # Early stopping
    early_stopping_patience: int | None = None
    early_stopping_metric: str = "val_loss"
    early_stopping_mode: str = "min"

    # Logging
    log_every_n_steps: int = 10
    log_dir: str = "logs"
    use_wandb: bool = False
    wandb_project: str | None = None

    # Reproducibility
    seed: int = 42
    deterministic: bool = False

    # Device
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"

    # Ablation/experiment tags
    tags: dict[str, Any] = field(default_factory=dict)

    # Extra
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> TrainerConfig:
        """Load config from YAML file."""
        with Path(path).open() as f:
            cfg = OmegaConf.load(f)
        return cls.from_dictconfig(cfg)

    @classmethod
    def from_dictconfig(cls, cfg: DictConfig) -> TrainerConfig:
        """Create from OmegaConf DictConfig."""
        # Merge with defaults
        default = OmegaConf.structured(cls)
        merged = OmegaConf.merge(default, cfg)
        return OmegaConf.to_object(merged)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainerConfig:
        """Create from plain dict."""
        return cls.from_dictconfig(OmegaConf.create(d))

    def to_dict(self) -> dict[str, Any]:
        """Convert to plain dict."""
        return OmegaConf.to_container(OmegaConf.structured(self), resolve=True)


@dataclass
class LMTrainingConfig(TrainerConfig):
    """LM trainer config — step-based knobs on the unified trainer config.

    Reconciles the former ``equitile/lm/training.py:TrainingConfig`` onto the
    shared :class:`TrainerConfig` hierarchy (REFACTOR.md §1): epoch-wise knobs
    (``epochs``, ``device``, ``num_workers``, ``checkpoint_dir``) are inherited
    unchanged; only step-based LM cadence and scheduler knobs remain local.

    Attributes:
        model: Optional registry name — the LM trainer binds an explicit model
            instance, so this is normally ``None``.
        learning_rate: Peak learning rate.
        warmup_steps: Warmup steps for the LR schedule.
        weight_decay: AdamW weight decay.
        use_amp: Enable automatic mixed precision.
        gradient_accumulation_steps: Steps over which gradients accumulate.
        gradient_clip: Gradient clipping norm.
        lr_schedule: LR schedule type ("cosine", "linear", "constant").
        min_lr_ratio: Minimum LR ratio for the cosine schedule.
        save_every: Save a checkpoint every N steps.
        eval_every: Run validation every N steps.
        log_every: Log metrics every N steps.
        generate_every: Generate samples every N steps.
    """

    model: str | None = None

    learning_rate: float = 3e-4
    warmup_steps: int = 100
    weight_decay: float = 0.1

    use_amp: bool = True
    gradient_accumulation_steps: int = 1
    gradient_clip: float = 1.0

    lr_schedule: str = "cosine"
    min_lr_ratio: float = 0.1

    save_every: int = 500
    eval_every: int = 100
    log_every: int = 10
    generate_every: int = 200

    def __post_init__(self) -> None:
        """Resolve ``device="auto"`` to a concrete backend."""
        if self.device == "auto":
            self.device = str(get_device())


@dataclass(frozen=True, slots=True)
class TrainingMetrics(BaseMetrics):
    """Metrics from a training step/epoch.

    Inherits the canonical ``epoch``/``step``/``extra`` shape from
    :class:`~bioplausible.core.metrics.BaseMetrics`; ``step`` is set to
    ``global_step`` at construction time so the metrics carry the update
    index within the epoch.
    """

    train_loss: float = 0.0
    train_acc: float = 0.0
    val_loss: float | None = None
    val_acc: float | None = None
    val_perplexity: float | None = None
    learning_rate: float | None = None
    epoch_time: float = 0.0
    samples_seen: int = 0

    # Energy metrics
    energy_proxy: float | None = None
    forward_flops: int | None = None
    backward_flops: int | None = None
    wall_time_ms: float | None = None
    peak_memory_mb: float | None = None
    requires_backward: bool | None = None


class CoreTrainer:
    """
    Unified training interface for all bioplausible models.

    Usage:
        config = TrainerConfig(
            model="tile_pc",
            model_kwargs={"input_dim": 784, "hidden_dim": 256, "output_dim": 10},
            optimizer="smep",
            optimizer_kwargs={"lr": 0.01, "beta": 0.5},
            task="mnist",
            epochs=10,
            track_energy=True
        )
        trainer = CoreTrainer(config)
        history = trainer.fit()

    Or from YAML:
        trainer = CoreTrainer.from_yaml("config.yaml")
        history = trainer.fit()
    """

    def __init__(self, config: TrainerConfig | dict[str, Any] | str):
        """
        Initialize trainer.

        Args:
            config: TrainerConfig, dict, or path to YAML config file
        """
        if isinstance(config, str):
            self.config = TrainerConfig.from_yaml(config)
        elif isinstance(config, dict):
            self.config = TrainerConfig.from_dict(config)
        elif isinstance(config, TrainerConfig):
            self.config = config
        else:
            raise TypeError(f"Expected TrainerConfig, dict, or str, got {type(config)}")

        # Set seed
        self._set_seed(self.config.seed)

        # Determine device
        self.device = self._resolve_device(self.config.device)

        # Initialize components
        self.model: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.propagator = None
        self.train_loader = None
        self.val_loader = None
        self.task_obj = None
        self.loss_fn: nn.Module | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = (
            float("inf") if self.config.early_stopping_mode == "min" else -float("inf")
        )
        self.patience_counter = 0
        self.history: list[TrainingMetrics] = []
        self._hardware_meta: dict[str, object] = {}

        # Self-diagnosis telemetry (EXPERIMENT_PLAN5 §1): which credit-assignment
        # path each train step actually took. Surfaced per-epoch via
        # ``TrainingMetrics.extra["training_paths"]`` and consumed by the probe
        # driver so the sweep can flag silent BPTT fallbacks as defects.
        self._training_path_counts: dict[str, int] = {}

        # BPTT-fallback warning dedup: the "silent degradation to backprop"
        # warning is emitted once per run, not once per batch, so a probe that
        # degrades does not flood the log (one warning per probe, then the
        # per-epoch training_path='bptt' telemetry records the defect).
        self._bptt_fallback_warned = False

        # Output directory (session-scoped tempdir when pytest is detected)
        self.output_dir = (
            _default_output_base() / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        self._save_config()

        # Callbacks
        self._callbacks: list[Callable] = []
        self._execution_callbacks: list[ExecutionCallback] = []

        logger.info("CoreTrainer initialized on %s", self.device)
        logger.info("Output dir: %s", self.output_dir)

    @classmethod
    def from_yaml(cls, path: str) -> CoreTrainer:
        """Create trainer from YAML config file."""
        return cls(TrainerConfig.from_yaml(path))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CoreTrainer:
        """Create trainer from dict."""
        return cls(TrainerConfig.from_dict(d))

    @classmethod
    def from_task(
        cls,
        model: nn.Module,
        task: TaskProtocol,
        device: str = "cpu",
        optimizer: torch.optim.Optimizer | None = None,
        epochs: int = 1,
        **kwargs: object,
    ) -> CoreTrainer:
        """Create a CoreTrainer from a pre-built model and task.

        Bypasses the config-driven data/setup path. The returned trainer
        uses ``task.get_batch()`` instead of data loaders, matching the
        behaviour of the legacy ``_TaskTrainer``.

        Args:
            model: Already-initialized model.
            task: ``BaseTask`` instance providing ``get_batch`` /
                ``compute_metrics``.
            device: Target device.
            optimizer: Pre-built optimizer (or ``None``).
            epochs: Training epochs (primarily for logging).
            **kwargs: Additional fields for the internal ``TrainerConfig``
                (e.g. ``grad_clip``, ``track_energy``, ``batches_per_epoch``,
                ``output_dir``, ``batch_size``, ``use_compile``, ``ablation_tags``).

        Returns:
            Configured ``CoreTrainer`` ready to call ``train_epoch()``.
        """
        # Resolve loss function matching the task's output geometry.
        task_type = getattr(task, "task_type", "vision")
        output_dim = getattr(task, "output_dim", 0)
        if task_type == DomainType.TABULAR and output_dim == 1:
            loss_fn = nn.MSELoss()
        else:
            loss_fn = nn.CrossEntropyLoss()

        cfg_kw = {
            "model": getattr(model, "algorithm_name", model.__class__.__name__),
            "task": getattr(task, "name", "custom"),
            "device": device,
            "epochs": epochs,
            "batch_size": kwargs.pop("batch_size", 64),
            "batches_per_epoch": kwargs.pop("batches_per_epoch", 100),
            "grad_clip": kwargs.pop("grad_clip", 0.0),
            "track_energy": kwargs.pop("track_energy", False),
            "log_dir": kwargs.pop(
                "output_dir", os.path.join(tempfile.gettempdir(), "bioplausible")
            ),
            "use_compile": kwargs.pop("use_compile", False),
            "tags": kwargs.pop("ablation_tags", {}),
        }
        cfg_kw.update(kwargs)
        config = TrainerConfig.from_dict(cfg_kw)
        trainer = cls(config)
        trainer.model = model.to(device)
        trainer.optimizer = optimizer
        trainer.task_obj = task
        trainer.loss_fn = loss_fn
        trainer.train_loader = None
        trainer.val_loader = None
        return trainer

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        from bioplausible.core.utils.seeds import set_all_seeds

        set_all_seeds(seed, deterministic=self.config.deterministic)

    def _resolve_device(self, device: str) -> torch.device:
        """Resolve device string to torch.device."""
        from bioplausible.core.utils.device import get_device

        return get_device(device)

    def _save_config(self) -> None:
        """Save config to output directory."""
        config_path = self.output_dir / "config.yaml"
        with Path(config_path).open("w") as f:
            OmegaConf.save(OmegaConf.structured(self.config), f)

    def setup(self) -> None:
        """Setup model, optimizer, data loaders, and propagator."""
        logger.info("Setting up trainer components...")

        # 1. Setup data
        self._setup_data()

        # 2. Create model
        self._create_model()

        # 3. Create propagator if specified
        self._create_propagator()

        # 4. Create optimizer (learning-rule optimizers eagerly; standard
        # torch optimizers lazily on first BPTT step — see ``_ensure_optimizer``).
        if self._needs_eager_optimizer():
            self._create_optimizer()

        # 5. Compile model if requested
        if self.config.use_compile and not self._is_kernal_model():
            try:
                self.model = torch.compile(self.model, mode=self.config.compile_mode)
                logger.info("Model compiled with mode=%s", self.config.compile_mode)
            except (RuntimeError, TypeError, ValueError) as e:
                logger.warning("Compilation failed: %s", e)

        # 6. Move to device
        self.model = self.model.to(self.device)

        logger.info("Setup complete")

    def _setup_data(self) -> None:
        """Setup data loaders via unified task resolution."""
        logger.info("Setting up data for task: %s", self.config.task)

        from bioplausible.config.unified import DataConfig
        from bioplausible.domains.base import DomainType, TaskSplit
        from bioplausible.domains.registry import resolve_task_from_data_config

        data_config = DataConfig(
            name=self.config.task,
            task=self.config.task,
            batch_size=self.config.batch_size,
            val_batch_size=self.config.val_batch_size,
            num_workers=self.config.num_workers,
            seq_len=self.config.data_kwargs.get("seq_len", 64),
            augment=self.config.data_kwargs.get("augment", False),
            data_fraction=self.config.data_kwargs.get("data_fraction", 1.0),
            data_kwargs=self.config.data_kwargs,
        )

        task_obj = resolve_task_from_data_config(data_config, device=str(self.device))

        # For LM tasks (and other non-DataLoader tasks), store task_obj and use get_batch
        # For vision/tabular tasks, use DataLoaders directly so test overrides work
        if task_obj.spec.domain_type in (
            DomainType.LM,
            DomainType.RL,
            DomainType.GRAPH,
            DomainType.TIMESERIES,
        ):
            self.task_obj = task_obj
            self.train_loader = None
            self.val_loader = None
        else:
            self.task_obj = None
            self.train_loader = task_obj.get_dataloader(TaskSplit.TRAIN)
            self.val_loader = task_obj.get_dataloader(TaskSplit.VAL)

        # Store vocab_size for model creation (LM tasks)
        if hasattr(task_obj, "vocab_size"):
            self.config.model_kwargs.setdefault("vocab_size", task_obj.vocab_size)

        train_len = len(self.train_loader) if self.train_loader else 0
        val_len = len(self.val_loader) if self.val_loader else 0
        logger.info("Data loaders created: train=%d, val=%d", train_len, val_len)

    def _create_model(self) -> None:
        """Create model from registry via the single construction layer."""
        # Check if model is registered in new registry
        try:
            model_cls = Registry.get(ComponentCategory.MODEL, self.config.model)
        except ValueError:
            available = Registry.list(ComponentCategory.MODEL).get("model", [])
            raise ValueError(
                f"Model '{self.config.model}' not registered. Available: {available}"
            )
        from bioplausible.core.construction import construct_model

        # model_kwargs carries the resolved scalar input/output dims (build_model
        # path); fall back to 0 only if an unusual caller omitted them.
        input_dim = int(self.config.model_kwargs.get("input_dim") or 0)
        output_dim = int(self.config.model_kwargs.get("output_dim") or 0)
        self.model = construct_model(
            model_cls,
            self.config.model_kwargs,
            input_dim=input_dim,
            output_dim=output_dim,
            model_name=self.config.model,
        )
        self.model = self._apply_hardware(cast("nn.Module", self.model))

        logger.info(
            "Model created: %s (%d params)",
            self.model.__class__.__name__,
            count_parameters(self.model, trainable_only=False),
        )

    def _apply_hardware(self, model: nn.Module) -> nn.Module:
        """Swap in a substrate-faithful facade when ``target_hardware`` is set.

        Only the LoopedMLP equilibrium family has a substrate approximation
        (the ``quantized_looped_mlp`` / ``noisy_looped_mlp`` facades inherit
        from it), so the knob is a no-op for every other model. The facade is
        rebuilt from the same ``model_kwargs`` plus the hardware depth default
        (``bits`` for FPGA, ``noise_level`` for analog).

        The chosen substrate is recorded on ``self._hardware_meta`` so each
        epoch's :class:`TrainingMetrics.extra` exposes it to the probe driver.
        """
        hardware = self.config.target_hardware
        self._hardware_meta = {}
        if not hardware or hardware == "gpu":
            return model

        from bioplausible.zoo.models.eqprop import LoopedMLP
        from bioplausible.zoo.models.eqprop.hardware_variants import (
            NoisyLoopedMLP,
            QuantizedLoopedMLP,
        )

        if not isinstance(model, LoopedMLP):
            return model

        if hardware == "fpga":
            kwargs = dict(self.config.model_kwargs)
            kwargs.setdefault("bits", 8)
            self._hardware_meta = {
                "target_hardware": "fpga",
                "bits": int(kwargs["bits"]),
            }
            swapped: nn.Module = QuantizedLoopedMLP(**kwargs)
        elif hardware == "analog":
            kwargs = dict(self.config.model_kwargs)
            kwargs.setdefault("noise_level", 0.05)
            self._hardware_meta = {
                "target_hardware": "analog",
                "noise_level": float(kwargs["noise_level"]),
            }
            swapped = NoisyLoopedMLP(**kwargs)
        else:  # pragma: no cover  # Literal type narrows the possible values
            return model

        logger.info("Hardware target '%s': %s", hardware, swapped.__class__.__name__)
        return swapped

    def _is_kernal_model(self) -> bool:
        """Check if model uses kernel backend (not compatible with torch.compile)."""
        return getattr(self.model, "backend", "pytorch") == "kernel"

    def _create_propagator(self) -> None:
        """Create propagator/learning rule if specified.

        If the configured propagator name is a *model-side* learning rule
        (registered in ``Registry._ALIASES``, e.g. ``"ff"`` → model
        ``"forward_forward"``), the model already owns the training step via
        its ``train_step`` method — no propagator object is needed and this
        is a no-op. Genuine propagators (eqprop, hebbian, etc.) are
        instantiated as ``LearningRuleOptimizer`` instances.
        """
        if not self.config.propagator:
            return

        prop_name = self.config.propagator

        # Model-side learners: the model's train_step handles training.
        if prop_name in Registry.aliases():
            logger.info(
                "Propagator %r is a model-side learner — "
                "the model's train_step handles training; skipping propagator.",
                prop_name,
            )
            return

        logger.info("Creating propagator: %s", prop_name)

        # Check capability compatibility (REFACTOR3 §4-5).
        model_name = self.config.model

        try:
            prop_meta = Registry.get_metadata(ComponentCategory.PROPAGATOR, prop_name)
            model_meta = Registry.get_metadata(ComponentCategory.MODEL, model_name)
            required = set(prop_meta.requires)
            provided = set(model_meta.provides)
            missing = required - provided
            if missing:
                raise IncompatibilityError(
                    f"Propagator '{prop_name}' requires capabilities {missing}, "
                    f"but model '{model_name}' only provides {provided}. "
                    f"Fix: implement transition_modules() on your model, "
                    f"or use a compatible propagator (e.g., 'backprop', 'fa')."
                )
        except ValueError:
            # If metadata not found, fall through to try creating anyway.
            pass

        try:
            prop_cls = Registry.get(ComponentCategory.PROPAGATOR, prop_name)
        except ValueError:
            logger.warning("Propagator %s not in registry, skipping", prop_name)
            return
        # Every registered propagator is a LearningRuleOptimizer with the
        # signature `(params, model, **kwargs)` (params first, then the model).
        # Constructing with the raw model as the first positional arg (the old
        # `prop_cls(self.model, ...)`) bound the *model* to `params` and left
        # `model` as a required kwarg — a hard TypeError for the whole family.
        self.propagator = prop_cls(
            list(self.model.parameters()), self.model, **self.config.propagator_kwargs
        )

    def _needs_eager_optimizer(self) -> bool:
        """Return True if the configured optimizer is a learning-rule optimizer.

        Learning-rule optimizers drive the Phase-3 update path in
        ``_train_step`` and therefore must exist at setup time. Standard torch
        optimizers (adam/sgd/… via the ``torch.optim`` fallback) are only used
        by the BPTT fallback, so they are created lazily on the first BPTT step
        — a self-updating rule (equilibrium, model-side ``train_step``) never
        allocates a phantom optimizer it does not use.
        """
        if not self.config.optimizer:
            return False
        try:
            meta = Registry.get_metadata(
                ComponentCategory.OPTIMIZER, self.config.optimizer
            )
        except ValueError:
            return False
        return meta.credit_assignment_type in {
            "equilibrium",
            "hebbian",
            "target",
            "forward-only",
            "spiking",
        }

    def _ensure_optimizer(self) -> None:
        """Create the standard optimizer lazily on first BPTT need."""
        if self.optimizer is None:
            self._create_optimizer()

    def _create_optimizer(self) -> None:
        """Create optimizer."""
        # Check if optimizer is in new registry
        try:
            opt_cls = Registry.get(ComponentCategory.OPTIMIZER, self.config.optimizer)

            # Check if it's a learning rule optimizer (needs model)
            meta = Registry.get_metadata(
                ComponentCategory.OPTIMIZER, self.config.optimizer
            )
            if meta.credit_assignment_type in [
                "equilibrium",
                "hebbian",
                "target",
                "forward-only",
                "spiking",
            ]:
                self.optimizer = opt_cls(
                    self.model.parameters(),
                    model=self.model,
                    **self.config.optimizer_kwargs,
                )
            else:
                self.optimizer = opt_cls(
                    self.model.parameters(), **self.config.optimizer_kwargs
                )
        except ValueError:
            # Fall back to torch.optim
            opt_cls = getattr(torch.optim, self.config.optimizer, None)
            if opt_cls is None:
                logger.warning(
                    "Optimizer %s not found in registry or torch.optim, using Adam",
                    self.config.optimizer,
                )
                opt_cls = torch.optim.Adam
            self.optimizer = opt_cls(
                self.model.parameters(), **self.config.optimizer_kwargs
            )

    def fit(
        self, scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    ) -> list[TrainingMetrics]:
        """
        Run training loop.

        Args:
            scheduler: Optional LR scheduler that is stepped at the end of
                each epoch (epoch-wise scheduling only; per-step schedulers
                should be invoked via a custom training loop instead).

        Returns:
            List of TrainingMetrics for each epoch
        """
        if scheduler is not None:
            self.scheduler = scheduler
        if self.model is None:
            self.setup()

        logger.info("Starting training for %d epochs", self.config.epochs)

        batches_per_epoch = self._resolve_batches_per_epoch()
        val_batches = self.config.val_batches or 20

        self._train_epochs_loop(batches_per_epoch, val_batches)

        logger.info("Training complete")
        return self.history

    def _train_epochs_loop(self, batches_per_epoch: int, val_batches: int) -> None:
        """Run the epoch loop, handling checkpoints, callbacks, and early stop."""
        try:
            if self.config.profile_epochs:
                self._profile_loop(batches_per_epoch, val_batches)
                return
            for epoch in range(self.config.epochs):
                self.current_epoch = epoch
                epoch_start = time.time()
                epoch_metrics = self._run_epoch(
                    epoch, epoch_start, batches_per_epoch, val_batches
                )
                if epoch_metrics is None:
                    break
                # A shallow-probe ``max_epoch_time`` budget that exhausted the
                # epoch means the run is truncated: remaining epochs carry no
                # signal (they'd hit the same budget immediately) and the probe
                # already routes this run to ``epoch_time_truncated``. Stop here
                # so the remaining epochs + validation passes don't waste the
                # GPU budget.
                if epoch_metrics.extra.get("epoch_time_budget_stopped"):
                    logger.info(
                        "epoch=%-2d hit max_epoch_time — stopping run early "
                        "(epoch_time_truncated)",
                        epoch,
                    )
                    break
                if self._handle_epoch_end(epoch_metrics):
                    break
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except (RuntimeError, OSError, ValueError) as e:
            logger.error("Training failed: %s", e, exc_info=True)
            raise
        finally:
            self._save_history()

    def _profile_loop(self, batches_per_epoch: int, val_batches: int) -> None:
        """Run one epoch under ``torch.profiler`` and dump a Chrome trace.

        Exports the trace to the run's output directory (plan §3 protocol):
        ``profile_e0.trace``. Useful for identifying the top time consumers in
        the CoreTrainer epoch loop (data / forward / backward / validation).
        """
        from torch import profiler

        trace_path = self.output_dir / "profile_e0.trace"
        logger.info("Profiling one epoch → %s", trace_path)

        # Run the profiled epoch directly so the epoch-loop metrics path is
        # exercised, mirroring a normal fit().
        self.current_epoch = 0
        epoch_start = time.time()
        with profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            self._run_epoch(0, epoch_start, batches_per_epoch, val_batches)
        prof.export_chrome_trace(str(trace_path))

        top = [
            (event.self_cpu_time_total, event.key)
            for event in prof.key_averages()
            if event.self_cpu_time_total > 0
        ]
        top.sort(reverse=True)
        logger.info("Top profiler consumers:")
        for us, key in top[:10]:
            logger.info("  %8.1f ms  %s", us / 1000.0, key)
        logger.info("Profile complete")

    def _handle_epoch_end(self, epoch_metrics: TrainingMetrics) -> bool:
        """Post-step bookkeeping for a finished epoch.

        Returns True if early stopping triggered (caller should break).
        """
        epoch = self.current_epoch

        self.history.append(epoch_metrics)
        self._log_epoch(epoch_metrics)
        self._run_callbacks(epoch_metrics)
        self._fire_execution_hook("on_epoch_end", epoch, epoch_metrics)

        if self.config.save_checkpoints and self._should_save_checkpoint(epoch_metrics):
            self._save_checkpoint(epoch_metrics)

        if self._check_early_stopping(epoch_metrics):
            logger.info("Early stopping triggered at epoch %d", epoch)
            return True

        if self.scheduler is not None:
            if self._is_kernal_model():
                logger.warning(
                    "LR scheduler has no effect in kernel mode "
                    "(kernel manages its own updates)"
                )
            self.scheduler.step()
        return False

    def _resolve_batches_per_epoch(self) -> int:
        """Determine the number of batches per epoch for training."""
        if self.config.batches_per_epoch:
            return self.config.batches_per_epoch
        if self.train_loader:
            return len(self.train_loader)
        return 100  # Default for task-based

    def _run_epoch(
        self,
        epoch: int,
        epoch_start: float,
        batches_per_epoch: int,
        val_batches: int,
    ) -> TrainingMetrics | None:
        """Run a single training epoch and return its metrics (None if aborted)."""
        train_metrics = self._train_epoch(batches_per_epoch)
        val_metrics = self.validate(val_batches) if self.config.run_validation else {}

        extra_keys = [
            "loss",
            "accuracy",
            "samples_seen",
            "energy_proxy",
            "forward_flops",
            "backward_flops",
            "wall_time_ms",
            "peak_memory_mb",
            "requires_backward",
        ]
        return TrainingMetrics(
            epoch=epoch,
            step=self.global_step,
            train_loss=train_metrics.get("loss", 0.0),
            train_acc=train_metrics.get("accuracy", 0.0),
            val_loss=val_metrics.get("val_loss"),
            val_acc=val_metrics.get("val_accuracy"),
            val_perplexity=val_metrics.get("val_perplexity"),
            learning_rate=self._get_lr(),
            epoch_time=time.time() - epoch_start,
            samples_seen=train_metrics.get("samples_seen", 0),
            energy_proxy=train_metrics.get("energy_proxy"),
            forward_flops=train_metrics.get("forward_flops"),
            backward_flops=train_metrics.get("backward_flops"),
            wall_time_ms=train_metrics.get("wall_time_ms"),
            peak_memory_mb=train_metrics.get("peak_memory_mb"),
            requires_backward=train_metrics.get("requires_backward"),
            extra={
                "training_paths": dict(self._training_path_counts),
                **self._hardware_meta,
                **{k: v for k, v in train_metrics.items() if k not in extra_keys},
            },
        )

    def train_epoch(self) -> dict[str, float]:
        """Public single-epoch runner matching ``TrainerProtocol``.

        Delegates to the internal ``_train_epoch``, using the configured
        ``batches_per_epoch`` from ``self.config`` (fallback: 100).
        """
        batches = (
            self.config.batches_per_epoch
            if self.config.batches_per_epoch
            else (len(self.train_loader) if self.train_loader else 100)
        )
        return self._train_epoch(batches)

    def _get_batch_data(
        self, loader: str, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch a training/validation batch from ``task_obj`` or DataLoader iterator."""
        if self.task_obj is not None:
            return self.task_obj.get_batch(loader, batch_size)
        iter_name = f"_{loader}_iter"
        loader_name = f"{loader}_loader"
        try:
            return next(getattr(self, iter_name))
        except AttributeError:
            setattr(self, iter_name, iter(getattr(self, loader_name)))
            return next(getattr(self, iter_name))
        except StopIteration:
            setattr(self, iter_name, iter(getattr(self, loader_name)))
            return next(getattr(self, iter_name))

    def _train_epoch(self, batches_per_epoch: int) -> dict[str, object]:
        """Run one training epoch."""
        self.model.train()

        from collections import defaultdict

        import numpy as np

        metrics_agg: defaultdict[str, list[float]] = defaultdict(list)
        samples_seen = 0
        batch_size = self.config.batch_size
        use_task = self.task_obj is not None

        epoch_start = time.time()
        max_epoch_time = float(self.config.max_epoch_time or 0.0)
        budget_stopped = False

        for batch_idx in range(batches_per_epoch):
            if max_epoch_time > 0.0 and time.time() - epoch_start >= max_epoch_time:
                logger.info(
                    "epoch time budget reached at batch=%d (%.0fs), stopping epoch",
                    batch_idx,
                    time.time() - epoch_start,
                )
                budget_stopped = True
                break
            x, y = (
                self._get_batch_data("train", batch_size)
                if use_task
                else self._get_batch_data("train", batch_size)
            )
            x, y = x.to(self.device), y.to(self.device)
            samples_seen += x.shape[0]

            # Energy tracking
            if self.config.track_energy:
                requires_backward = self._model_requires_backward()

                with EnergyTracker(
                    self.model,
                    requires_backward=requires_backward,
                    global_step=self.global_step,
                ) as et:
                    step_metrics = self._train_step(x, y)

                if et.profile:
                    step_metrics["energy_proxy"] = et.profile.energy_proxy
                    step_metrics["forward_flops"] = et.profile.forward_flops
                    step_metrics["backward_flops"] = et.profile.backward_flops
                    step_metrics["wall_time_ms"] = et.profile.wall_time_ms
                    step_metrics["peak_memory_mb"] = et.profile.peak_memory_mb
                    step_metrics["requires_backward"] = int(
                        et.profile.requires_backward
                    )
            else:
                step_metrics = self._train_step(x, y)

            # Run-wide numerical-health guard across EVERY training path
            # (energy, propagator, model train_step, BPTT). A non-finite loss
            # aborts the probe immediately — no point burning the remaining
            # epoch budget on a run that has already diverged.
            _step_loss = step_metrics.get("loss")
            if _step_loss is not None and not isfinite(float(_step_loss)):
                from bioplausible.core.exceptions import NumericalInstabilityError

                raise NumericalInstabilityError(
                    f"non-finite loss ({_step_loss!r}) for model="
                    f"{self.config.model!r} at global_step={self.global_step}"
                )

            # Some training paths (e.g. a learning-rule *propagator* whose
            # ``step()`` only applies updates and returns ``None``) do not report
            # a per-step loss/accuracy. Without them the epoch aggregates to
            # zero, which silently breaks the sweep's liveness gate (every such
            # rule looks "dead"). Backfill train loss/acc from a lightweight
            # no-grad forward so the epoch metrics — and the gate — are real.
            if "loss" not in step_metrics or "accuracy" not in step_metrics:
                with torch.no_grad():
                    _logits = self.model(self._adapt_input(x))
                if "loss" not in step_metrics:
                    step_metrics["loss"] = float(compute_loss(self.loss_fn, _logits, y))
                if "accuracy" not in step_metrics:
                    step_metrics["accuracy"] = float(compute_accuracy(_logits, y))

            for k, v in step_metrics.items():
                if isinstance(v, (int, float)):
                    metrics_agg[k].append(v)

            self.global_step += 1

            # Log step
            if self.global_step % self.config.log_every_n_steps == 0:
                self._log_step(step_metrics, batch_idx, batches_per_epoch)

            # Execution callbacks (Sprint 3.4) — scalar telemetry only
            self._fire_execution_hook(
                "on_step_end",
                self.global_step,
                step_metrics.get("loss", float("nan")),
                self._compute_grad_norms(self.model),
            )
            energy = step_metrics.get("energy_proxy") or step_metrics.get("energy")
            if energy is not None and isinstance(energy, (int, float)):
                self._fire_execution_hook(
                    "on_settling_step", self.global_step, float(energy)
                )

        # Average metrics
        avg_metrics: dict[str, object] = {
            k: np.mean(v) for k, v in metrics_agg.items() if v
        }
        avg_metrics["samples_seen"] = samples_seen
        avg_metrics["epoch_time_budget_stopped"] = budget_stopped
        return avg_metrics

    def _adapt_input(self, x: torch.Tensor) -> torch.Tensor:
        """Adapt input tensor to the model's expected spatial format.

        Models declare their input representation via ``model.input_format``:
        - ``"spatial"`` (conv models) expect a 4D ``(B, C, H, W)`` tensor.
        - ``"flat"`` (default, MLP / equilibrium models) expect a 2D
          ``(B, input_dim)`` tensor.

        For flat models we reshape the spatial batch to 2D, preserving the
        per-sample content (no information is discarded by the architecture it
        was designed for). Spatial models receive the tensor untouched so conv
        feature extraction can leverage the retained spatial structure.
        """
        if x.dim() <= 2:
            return x
        if getattr(self.model, "input_format", "flat") == "spatial":
            return x
        return x.view(x.size(0), -1)

    def _train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        """Single training step.

        Dispatches via a structural match to the active learning rule:
        1. EnergyModel protocol (match/case) — energy-based models.
        2. Explicit propagator= — a registered LearningRuleOptimizer
           (e.g. eq_prop, feedback_alignment). Model-side rules
           (``"ff"``, ``"pepita"``, …) are aliases that resolve to the
           model's own train_step (Phase 3), so no propagator object is
           created for them.
        3. Model-side train_step — bio-plausible models that own their
           learning rule (``ForwardForwardNet``, ``DifferenceTargetProp``, …).
        4. Learning-rule optimizer= — an optimizer that *is* a learning
           rule (collapsed toward Phase 3 as propagator logic migrates).
        5. Standard BPTT fallback — plain ``forward`` + ``loss.backward()``.
        """
        x = self._adapt_input(x)

        # Phase 1: EnergyModel — clean structural dispatch
        match self.model:
            case EnergyModel():
                self._record_path("energy")
                return _make_ebm_trainer(self.config, self.model).train_step(x, y)

        # Phase 2: Explicit propagator= — a registered LearningRuleOptimizer
        # (eq_prop, hebbian, etc.). Model-side rules ("ff", "pepita", …) are
        # aliases resolved in _create_propagator, so self.propagator is None
        # for them and we fall through to Phase 3.
        rule = self.propagator
        if rule is not None and _is_learning_rule_optimizer(rule):
            self._record_path("propagator")
            return rule.step(x=x, target=y) or {}

        # Phase 3: Model-side custom train_step (bio-plausible models)
        # Probe with real data — the only reliable way to verify
        # that train_step returns meaningful metrics vs NotImplementedError/None.
        # A base-class train_step raises NotImplementedError to signal "use BPTT";
        # catch it and fall through to Phase 4/5 rather than aborting the epoch.
        if hasattr(self.model, "train_step"):
            try:
                metrics = self.model.train_step(x, y)
            except NotImplementedError:
                metrics = None
            if metrics is not None:
                self._record_path("model_train_step")
                return metrics

        # Phase 4: Learning-rule optimizer (optimizer= that is a LearningRuleOptimizer)
        rule = self.optimizer
        if rule is not None and _is_learning_rule_optimizer(rule):
            self._record_path("propagator")
            return rule.step(x=x, target=y) or {}

        # Phase 5: Standard forward/backward
        if getattr(self.model, "gradient_method", None) == "equilibrium":
            self._record_path("implicit_equilibrium")
        else:
            self._record_path("bptt")
        return self._bptt_step(x, y)

    def _record_path(self, path: str) -> None:
        """Increment the credit-assignment path counter for self-diagnosis."""
        self._training_path_counts[path] = self._training_path_counts.get(path, 0) + 1

    def _check_numerical_health(
        self, logits: torch.Tensor, loss: torch.Tensor, y: torch.Tensor
    ) -> None:
        """Raise optuna.TrialPruned if numerical pathologies detected.

        Catches: NaN/Inf, collapsed outputs, exploded gradients, constant predictions.
        Does NOT prune on "low loss" — that's a success signal, not a pathology.
        """
        if not torch.isfinite(loss).all():
            raise optuna.TrialPruned("Non-finite loss")

        # Collapsed outputs: all logits nearly identical
        if logits.std() < 1e-6:
            raise optuna.TrialPruned(f"Collapsed logits (std={logits.std():.2e})")

        # Constant predictions: all samples -> same class with high confidence
        preds = logits.argmax(dim=1)
        if preds.unique().numel() == 1 and logits.max() > 10:
            raise optuna.TrialPruned("Constant high-confidence predictions")

        # Weight explosion check (sample a few params)
        for p in self.model.parameters():
            if p.requires_grad and p.abs().max() > 1e6:
                raise optuna.TrialPruned(f"Weight explosion (max={p.abs().max():.2e})")

        # Gradient explosion (after backward, before clip)
        total_grad_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm**0.5
        if total_grad_norm > 100:
            raise optuna.TrialPruned(f"Gradient explosion (norm={total_grad_norm:.1f})")

    def _bptt_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        """Standard backpropagation training step."""
        x = self._adapt_input(x)

        # Biological-rule honesty: if the caller disallowed the BPTT fallback
        # and this is a *plain* unrolled backward (not the O(1) implicit
        # equilibrium path, which legitimately routes through forward/backward),
        # raise a loud warning so the silent degradation is never masked. Warn
        # once per run, not once per batch — a probe that degrades to BPTT
        # floods the log if we log on every step.
        if (
            not self.config.allow_bptt_fallback
            and getattr(self.model, "gradient_method", None) != "equilibrium"
            and not self._bptt_fallback_warned
        ):
            self._bptt_fallback_warned = True
            logger.warning(
                "BPTT fallback used on model=%s, but allow_bptt_fallback=False — "
                "recording training_path='bptt' (flagged as DEFECT by sweep)",
                self.config.model,
            )

        self._ensure_optimizer()
        if self.optimizer:
            self.optimizer.zero_grad()

        logits = self.model(x)
        loss = compute_loss(self.loss_fn, logits, y)
        loss.backward()

        # Numerical health check — prune trial on actual pathologies
        self._check_numerical_health(logits, loss, y)

        # Gradient clipping
        if self.config.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )

        if self.optimizer:
            self.optimizer.step()

        # Compute metrics
        metrics: dict[str, float] = {"loss": loss.item()}
        if self.task_obj is not None and hasattr(self.task_obj, "compute_metrics"):
            with torch.no_grad():
                task_metrics = self.task_obj.compute_metrics(
                    logits.detach(), y, loss.item()
                )
                metrics.update(task_metrics)
        else:
            metrics["accuracy"] = compute_accuracy(logits, y)

        return metrics

    def validate(self, val_batches: int) -> dict[str, object]:
        """Run ``val_batches`` batches of validation and return averaged metrics."""
        if self.val_loader is None and self.task_obj is None:
            return {"val_loss": float("nan"), "val_accuracy": float("nan")}

        self.model.eval()

        import numpy as np

        val_losses: list[float] = []
        val_accs: list[float] = []
        val_perplexities: list[float] = []
        batch_size = self.config.val_batch_size or self.config.batch_size

        with torch.no_grad():
            for _ in range(val_batches):
                x, y = self._get_batch_data("val", batch_size)
                x, y = x.to(self.device), y.to(self.device)

                x = self._adapt_input(x)
                # Models with a custom val_step (e.g. diffusion denoisers that
                # need a timestep `t`) must be evaluated through it rather than
                # a bare forward() call.
                if hasattr(self.model, "val_step"):
                    val_metrics = self.model.val_step(x, y)
                    val_losses.append(val_metrics.get("loss", float("nan")) * 1.0)
                    val_accs.append(val_metrics.get("accuracy", 0.0))
                    continue
                logits = self.model(x)
                loss = compute_loss(self.loss_fn, logits, y)
                val_losses.append(loss.item())

                # Accuracy from task or fallback
                if self.task_obj is not None and hasattr(
                    self.task_obj, "compute_metrics"
                ):
                    step_metrics = self.task_obj.compute_metrics(logits, y, loss.item())
                    val_accs.append(step_metrics.get("accuracy", 0.0))
                else:
                    val_accs.append(compute_accuracy(logits, y))

                # Perplexity for LM
                if self.task_obj and self.task_obj.task_type == DomainType.LM:
                    val_perplexities.append(np.exp(min(loss.item(), 10)))

        result: dict[str, object] = {
            "val_loss": float(np.mean(val_losses)) if val_losses else float("nan"),
            "val_accuracy": float(np.mean(val_accs)) if val_accs else float("nan"),
        }

        if val_perplexities:
            result["val_perplexity"] = np.mean(val_perplexities)

        return result

    def _get_lr(self) -> float | None:
        """Get current learning rate."""
        if self.optimizer and hasattr(self.optimizer, "param_groups"):
            return self.optimizer.param_groups[0].get("lr")
        return None

    def _log_epoch(self, metrics: TrainingMetrics) -> None:
        """Log epoch metrics."""
        msg = ("Epoch %d: Train Loss=%.4f, Train Acc=%.4f") % (
            metrics.epoch,
            metrics.train_loss,
            metrics.train_acc,
        )
        if metrics.val_loss is not None:
            msg += ", Val Loss=%.4f, Val Acc=%.4f" % (
                metrics.val_loss,
                metrics.val_acc,
            )
        if metrics.val_perplexity is not None:
            msg += ", Val PPL=%.2f" % metrics.val_perplexity
        if metrics.learning_rate is not None:
            msg += ", LR=%.2e" % metrics.learning_rate
        msg += ", Time=%.1fs" % metrics.epoch_time

        logger.info(msg)

    def _log_step(self, metrics: dict[str, float], step: int, total: int) -> None:
        """Log step metrics."""
        loss = metrics.get("loss", 0)
        acc = metrics.get("accuracy", 0)
        logger.debug("Step %d/%d: Loss=%.4f, Acc=%.4f", step, total, loss, acc)

    def _run_callbacks(self, metrics: TrainingMetrics) -> None:
        """Run registered callbacks."""
        for cb in self._callbacks:
            try:
                cb(self, metrics)
            except Exception as e:  # broad: user callbacks may raise anything
                logger.warning("Callback failed: %s", e)

    def add_callback(self, callback: Callable) -> None:
        """Add a callback function."""
        self._callbacks.append(callback)

    def add_execution_callback(self, callback: ExecutionCallback) -> None:
        """Register an ``ExecutionCallback`` for live telemetry.

        Hooks fire best-effort (a raising callback is logged and
        swallowed) and receive scalar metrics only, so UI listeners
        (e.g. the NiceGUI demo) can never corrupt training state.

        Args:
            callback: Object implementing the ``ExecutionCallback`` protocol.
        """
        self._execution_callbacks.append(callback)

    def _fire_execution_hook(self, name: str, *args: object) -> None:
        """Fire an execution callback hook on all registered callbacks."""
        for cb in self._execution_callbacks:
            try:
                getattr(cb, name)(*args)
            except Exception as e:  # broad: external listener may raise anything
                logger.warning("Execution callback %r.%s failed: %s", cb, name, e)

    def _model_requires_backward(self) -> bool:
        """Whether the current model needs backward for energy tracking.

        Looks the model up by its **registered** name (``config.model``), not
        the display ``algorithm_name`` — the latter is a presentation label
        (e.g. ``"HebbianChain"``) that has no registry entry, which previously
        made every such lookup miss and warn per batch. Unregistered models
        fall back to ``requires_backward = True`` and warn once.
        """
        name = self.config.model
        warned = getattr(CoreTrainer, "_missing_metadata_warned", None)
        if warned is None:
            warned = CoreTrainer._missing_metadata_warned = set()
        value = True
        try:
            meta = Registry.get_metadata(ComponentCategory.MODEL, name)
            value = meta.requires_backward
        except ValueError, KeyError:
            if name not in warned:
                warned.add(name)
                logger.warning("Could not fetch metadata for %s", name)
        return value

    @staticmethod
    def _compute_grad_norms(model: nn.Module) -> dict[str, float]:
        """Compute per-parameter L2 gradient norms for the model."""
        norms: dict[str, float] = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                norms[name] = param.grad.norm().item()
        return norms

    def _should_save_checkpoint(self, metrics: TrainingMetrics) -> bool:
        """Determine if checkpoint should be saved."""
        if not self.config.save_checkpoints:
            return False

        if self.current_epoch % self.config.save_every_n_epochs != 0:
            return False

        if self.config.save_best_only and metrics.val_loss is not None:
            if self.config.early_stopping_mode == "min":
                is_best = metrics.val_loss < self.best_val_metric
            else:
                is_best = metrics.val_loss > self.best_val_metric

            if is_best:
                self.best_val_metric = metrics.val_loss
                return True
            return False

        return True

    def _save_checkpoint(self, metrics: TrainingMetrics) -> None:
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        path = (
            checkpoint_dir / f"epoch_{self.current_epoch}_val_{metrics.val_loss:.4f}.pt"
        )

        save_checkpoint_file(
            path,
            {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": (
                    self.optimizer.state_dict() if self.optimizer else None
                ),
                "metrics": metrics.to_dict(),
                "config": self.config.to_dict(),
                "global_step": self.global_step,
            },
        )

        logger.info("Checkpoint saved: %s", path)

    def _check_early_stopping(self, metrics: TrainingMetrics) -> bool:
        """Check early stopping condition."""
        if self.config.early_stopping_patience is None:
            return False

        if metrics.val_loss is None:
            return False

        if self.config.early_stopping_mode == "min":
            improved = metrics.val_loss < self.best_val_metric
        else:
            improved = metrics.val_loss > self.best_val_metric

        if improved:
            self.best_val_metric = metrics.val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        return self.patience_counter >= self.config.early_stopping_patience

    def _save_history(self) -> None:
        """Save training history to JSON."""
        history_path = self.output_dir / "history.json"
        with Path(history_path).open("w") as f:
            json.dump([m.to_dict() for m in self.history], f, indent=2)

        # Also save as JSONL for streaming
        jsonl_path = self.output_dir / "history.jsonl"
        with Path(jsonl_path).open("w") as f:
            for m in self.history:
                f.write(json.dumps(m.to_dict()) + "\n")

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = load_checkpoint_file(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        opt_state = checkpoint.get("optimizer_state_dict")
        if self.optimizer and opt_state is not None:
            self.optimizer.load_state_dict(opt_state)
        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.history = [TrainingMetrics(**m) for m in checkpoint.get("metrics", [])]
        logger.info("Loaded checkpoint from epoch %d", self.current_epoch)

    def search(
        self, param_space: dict[str, object], n_trials: int = 20
    ) -> dict[str, object]:
        """
        Run hyperparameter search using Optuna.

        Args:
            param_space: Dict of parameter names to Optuna distributions
            n_trials: Number of trials

        Returns:
            Best parameters and metrics
        """
        import optuna

        def objective(trial: optuna.Trial) -> float:
            # Sample parameters
            for name, dist in param_space.items():
                if hasattr(dist, "__call__"):
                    _ = dist(trial)
                else:
                    _ = (
                        trial.suggest_categorical(name, dist)
                        if isinstance(dist, list)
                        else dist
                    )
                # Update config
                # This is simplified - would need proper config merging

            # Create new trainer with sampled config
            # Run training and return validation metric

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "trials": len(study.trials),
        }

    def export_onnx(self, path: str, input_shape: tuple[int, ...] = (1, 784)) -> None:
        """Export model to ONNX."""
        from bioplausible.utils import export_to_onnx

        export_to_onnx(self.model, path, input_shape, device=self.device)

    def get_history_dataframe(self):
        """Get history as pandas DataFrame."""
        try:
            import pandas as pd

            return pd.DataFrame([m.to_dict() for m in self.history])
        except ImportError:
            logger.warning("pandas not available")
            return None


def _convert_dictconfig(obj):
    """Deeply convert OmegaConf DictConfig to native dicts."""
    if hasattr(obj, "_is_dict"):
        return OmegaConf.to_container(obj, resolve=True)
    elif isinstance(obj, list):
        return [_convert_dictconfig(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: _convert_dictconfig(v) for k, v in obj.items()}
    return obj


def run_from_runconfig(cfg: object) -> dict[str, object]:
    """Run an experiment from an OmegaConf-based ``RunConfig``.

    Accepts a ``RunConfig`` (defined in :mod:`bioplausible.config.omegaconf`)
    produced by loading the YAML experiment configs in ``configs/``.

    Args:
        cfg: ``RunConfig`` instance with ``data``/``model``/``optimizer``/
            ``trainer`` sections.

    Returns:
        Dict with ``history`` (list of per-epoch metric dicts) and
        ``final_val_accuracy``.
    """
    import json

    from bioplausible.config.unified import DataConfig
    from bioplausible.domains.registry import resolve_task_from_data_config

    torch.manual_seed(cfg.seed)

    device = _resolve_runconfig_device(cfg)

    data_config = DataConfig(
        name=cfg.data.task,
        task=cfg.data.task,
        batch_size=cfg.data.batch_size,
        val_batch_size=None,
        num_workers=4,
        seq_len=cfg.data.seq_len,
        augment=cfg.data.augment,
        data_fraction=cfg.data.data_fraction,
        data_kwargs={},
    )

    task = resolve_task_from_data_config(data_config, device=device)

    model = _build_runconfig_model(cfg, task, device)
    optimizer = _build_runconfig_optimizer(cfg, model)

    ablation_tags = _convert_dictconfig(cfg.ablation_tags)

    trainer = task.create_trainer(
        model=model,
        optimizer=optimizer,
        epochs=cfg.trainer.epochs,
        batches_per_epoch=cfg.trainer.batches_per_epoch,
        grad_clip=cfg.trainer.grad_clip,
        use_compile=cfg.trainer.use_compile,
        track_energy=cfg.trainer.track_energy,
        ablation_tags=ablation_tags,
        output_dir=cfg.output_dir,
        device=device,
    )

    try:
        results = _run_runconfig_epochs(trainer, cfg)
    except optuna.exceptions.TrialPruned as exc:
        # No study to consume the prune in a standalone run; record the revert
        # instead of crashing (PLAN4 S0d).
        _record_pruned_failure(cfg, device, exc)
        return {"history": [], "final_val_accuracy": 0.0, "status": "error"}

    Path(cfg.output_dir).mkdir(exist_ok=True, parents=True)
    clean_results = _convert_dictconfig(results)
    with (Path(cfg.output_dir) / "results.json").open("w") as f:
        json.dump(clean_results, f, indent=4)

    return {
        "history": clean_results,
        "final_val_accuracy": (
            clean_results[-1].get("val_accuracy", 0.0) if clean_results else 0.0
        ),
        "status": "completed",
    }


def _record_pruned_failure(cfg: object, device: str, exc: Exception) -> None:
    """Persist an optuna-prune that escaped the standalone runner as a revert.

    Inside a live HPO study the pruner consumes ``TrialPruned``; in a standalone
    ``run_from_runconfig`` call there is no study to sink it, so it surfaced as a
    crash. Record it on the failure side of the sink (honors "record reverts"):
    the AutoScientist then knows not to re-burn the same config (PLAN4 S0d).
    """
    from bioplausible.experiment.result_sink import record_experiment_result

    record_experiment_result(
        model=cfg.model.name,
        task=cfg.data.task,
        status="error",
        seed=cfg.seed,
        epochs=cfg.trainer.epochs,
        device=device,
        extra={"error": str(exc)},
    )


def _resolve_runconfig_device(cfg: object) -> str:
    """Resolve the target device from a RunConfig's ``device`` field."""
    from bioplausible.core.utils.device import get_device

    if cfg.device == "auto":
        return str(get_device())
    return cfg.device


def _build_runconfig_model(cfg: object, task: object, device: str) -> nn.Module:
    """Instantiate and move a model to ``device`` from a RunConfig."""
    from bioplausible.core.construction import construct_model

    extra_kwargs = _convert_dictconfig(cfg.model.extra)
    config: dict[str, object] = {"hidden_dim": cfg.model.hidden_dim}
    if hasattr(cfg.model, "num_layers"):
        config["num_layers"] = cfg.model.num_layers
    config.update(extra_kwargs)

    model_cls = Registry.get(ComponentCategory.MODEL, cfg.model.name)
    model = construct_model(
        model_cls,
        config,
        input_dim=task.input_dim,
        output_dim=task.output_dim,
        model_name=cfg.model.name,
    )
    return model.to(device)


def _build_runconfig_optimizer(cfg: object, model: nn.Module) -> object:
    """Build the optimizer from a RunConfig, trying both call signatures."""
    opt_kwargs: dict[str, object] = {
        "lr": cfg.optimizer.lr,
        "weight_decay": cfg.optimizer.weight_decay,
    }
    if cfg.optimizer.name.startswith("mep") or cfg.optimizer.name in [
        "smep",
        "sdmep",
        "local_ep",
        "natural_ep",
        "muon_backprop",
    ]:
        if hasattr(cfg.optimizer, "beta"):
            opt_kwargs["beta"] = cfg.optimizer.beta
        if hasattr(cfg.optimizer, "settle_steps"):
            opt_kwargs["settle_steps"] = cfg.optimizer.settle_steps
        if hasattr(cfg.optimizer, "mode"):
            opt_kwargs["mode"] = cfg.optimizer.mode

    opt_cls = Registry.get(ComponentCategory.OPTIMIZER, cfg.optimizer.name)

    # Some optimizers (learning-rule propagators) require the model, while
    # plain torch.optim optimizers do not. Attempt both call signatures.
    try:
        return opt_cls(model.parameters(), model=model, **opt_kwargs)
    except TypeError:
        return opt_cls(model.parameters(), **opt_kwargs)


def _run_runconfig_epochs(trainer: object, cfg: object) -> list[dict[str, object]]:
    """Run training epochs, adopting the trainer's scheduling interface."""
    results: list[dict[str, object]] = []

    if hasattr(trainer, "train_epoch"):
        for _ in range(cfg.trainer.epochs):
            results.append(trainer.train_epoch())
    elif hasattr(trainer, "run"):
        history = trainer.run()
        if isinstance(history, dict) and "rewards" in history:
            for i, r in enumerate(history["rewards"]):
                results.append({"epoch": i, "reward": r, "val_accuracy": r})
    else:
        trainer.fit(train_loader=None, epochs=cfg.trainer.epochs)

    return results


__all__ = [
    "CoreTrainer",
    "LMTrainingConfig",
    "TrainerConfig",
    "TrainerProtocol",
    "TrainingMetrics",
    "run_from_runconfig",
]
