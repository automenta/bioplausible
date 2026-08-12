"""Unified configuration hierarchy for Bioplausible.

REFACTOR.md §1.1 (revised): Single source of truth for configuration
dataclasses that work equally well as frozen runtime objects and OmegaConf
structured configs.

The original REFACTOR.md proposed frozen dataclasses that "would break
OmegaConf compatibility". Testing confirms OmegaConf 2.3+ handles
``@dataclass(frozen=True, slots=True)`` correctly for ``OmegaConf.structured()``,
``OmegaConf.merge()``, and ``OmegaConf.to_object()`` — the YAML save/load
round-trip requires a single ``OmegaConf.merge(cls, cfg)`` step after
``OmegaConf.load()`` (one line).

Therefore this module defines frozen runtime configs directly — no separate
"schema" mirror is needed. Migrate existing configs to this pattern by:
1. Making the dataclass ``frozen=True, slots=True``
2. Using :func:`load_config` / :func:`save_config` instead of raw OmegaConf calls
3. Adding ``to_internal()`` only if the config must interoperate with an
   existing non-frozen structured config (gradual migration path)

Example:
    >>> from bioplausible.config.unified import BaseConfig, load_config, save_config
    >>> @dataclass(frozen=True, slots=True)
    ... class MyConfig(BaseConfig):
    ...     lr: float = 0.01
    >>> cfg = MyConfig(name="test", lr=0.05)
    >>> save_config(cfg, "my_config.yaml")   # writes YAML
    >>> loaded = load_config(MyConfig, "my_config.yaml")  # reads YAML
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

__all__ = [
    "BaseConfig",
    "BaseStructuredConfig",
    "DeviceStr",
    "ExperimentConfig",
    "ExperimentRunnerConfig",
    "LayerRole",
    "ModelConfig",
    "ReproducibilityConfig",
    "compute_hidden_dims",
    "config_to_dict",
    "load_config",
    "resolve_hidden_dims",
    "save_config",
]


DeviceStr = str


@dataclass(frozen=True, slots=True)
class BaseConfig:
    """Frozen runtime base with fields common to all configs.

    Attributes:
        name: Human-readable identifier for the config instance.
        seed: Random seed for reproducibility.
        device: Target device ("auto", "cpu", "cuda", "mps").
    """

    name: str = "default"
    seed: int = 42
    device: DeviceStr = "auto"


@dataclass
class BaseStructuredConfig:
    """OmegaConf-compatible mirror of :class:`BaseConfig`.

    Non-frozen so ``OmegaConf.merge`` can create new instances during YAML
    deserialization. Use :meth:`to_internal` to obtain the frozen runtime
    equivalent, or prefer :func:`load_config` which handles the conversion.
    """

    name: str = "default"
    seed: int = 42
    device: DeviceStr = "auto"

    def to_internal(self) -> BaseConfig:
        """Convert to the frozen runtime :class:`BaseConfig`."""
        return BaseConfig(
            name=self.name,
            seed=self.seed,
            device=self.device,
        )


@dataclass(frozen=True, slots=True)
class BaseStructuredDefaults:
    """Frozen mirror of :class:`BaseConfig` for direct OmegaConf use.

    This exists for the common case where you want a frozen config that also
    round-trips through YAML without a separate ``to_internal`` step. See
    :func:`load_config` and :func:`save_config` for the helpers.
    """

    name: str = "default"
    seed: int = 42
    device: DeviceStr = "auto"


# ──────────────────────────────────────────────
# Model configuration (migrated from core/config.py)
# ──────────────────────────────────────────────


#: Role of a layer within a model: "hidden" or "output".
LayerRole = Literal["hidden", "output"]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Configuration for a bio-plausible model.

    Migrated from :mod:`bioplausible.core.config` (REFACTOR.md §1.1). Kept as
    a standalone frozen dataclass (NOT extending :class:`BaseConfig`) because
    the required ``name``/``input_dim``/``output_dim`` fields cannot follow
    ``BaseConfig``'s defaulted ``seed``/``device`` in dataclass MRO.
    """

    name: str
    input_dim: int
    output_dim: int
    hidden_dims: list[int] = field(default_factory=list)

    # Training hyperparameters
    learning_rate: float = 0.001
    beta: float = 0.2  # For EqProp
    # Maximum number of equilibrium steps
    max_steps: int = 30
    # Equilibrium settling early-stop parameters
    convergence_threshold: float = 1e-3
    convergence_start: int = 5

    # Architecture
    use_spectral_norm: bool = True
    # Power iterations for the spectral-norm parametrization. Lower = cheaper
    # equilibrium settles (each power iteration is a forward+tranpose multiply);
    # the coarse sweep can set 0 to drop spectral-norm cost from the map.
    spectral_norm_power_iterations: int = 5
    activation: str = "silu"
    lipschitz_mode: str = "power_iteration"  # "power_iteration" or "svd"

    # μPC (Maximal Update Parameterization) output-node scaling
    # "mupc": output layer skips the √L scaling factor applied to hidden layers
    # "uniform": all layers get the same scaling (backward compat / ablation)
    output_scaling_mode: Literal["uniform", "mupc"] = "mupc"

    # Additional kwargs
    extra: dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration."""
        # input_dim can be 0 for Conv models (placeholder)
        val = self.input_dim
        if isinstance(val, tuple):
            import math

            val = math.prod(val)
        if val < 0:
            raise ValueError(f"input_dim must be >= 0, got {val}")
        # Use object.__setattr__ because frozen=True
        if isinstance(self.input_dim, tuple):
            object.__setattr__(self, "input_dim", val)
        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be > 0, got {self.output_dim}")


def resolve_hidden_dims(
    config: ModelConfig | None, hidden_dim: int | None
) -> list[int]:
    """Resolve the ``hidden_dims`` list from a ``ModelConfig`` or fallback.

    Returns ``config.hidden_dims`` if non-empty; otherwise falls back to
    ``[hidden_dim]`` if set; otherwise ``[]``.
    """
    if config is not None and config.hidden_dims:
        return config.hidden_dims
    if hidden_dim is not None:
        return [hidden_dim]
    return []


def compute_hidden_dims(
    hidden_dim: int | None, num_layers: int, max_layers: int = 5
) -> list[int]:
    """Compute a ``hidden_dims`` list for a ``build`` classmethod.

    Returns ``[hidden_dim] * min(num_layers, max_layers)`` when
    ``hidden_dim`` is set, else ``[]``.
    """
    if hidden_dim is None:
        return []
    return [hidden_dim] * min(num_layers, max_layers)


def _build_model_config(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]  # contract for zoo models' build classmethods
    spec,
    input_dim: int,
    output_dim: int,
    hidden_dim: int | None,
    num_layers: int,
    kwargs: dict[str, object],
    *,
    learning_rate: float | None = None,
    beta: float | None = None,
    max_steps: int | None = None,
    use_spectral_norm: bool | None = None,
    convergence_threshold: float | None = None,
    convergence_start: int | None = None,
) -> ModelConfig:
    """Construct a ``ModelConfig`` from the standard ``build`` classmethod parameters.

    Handles the common ``spec.name``, ``compute_hidden_dims``, and
    ``kwargs`` wiring. Optional overrides are passed through to the
    ``ModelConfig`` constructor; if *not* provided, the corresponding
    ``ModelConfig`` defaults apply.
    """
    # Collect overrides that match ModelConfig fields so they can be applied to
    # the (frozen) config after construction. ``None`` entries are filtered by
    # the apply loop below, so named ``build`` params that weren't provided are
    # harmless; explicit kwargs take precedence over them.
    overrides: dict[str, object] = {
        "learning_rate": learning_rate,
        "beta": beta,
        "max_steps": max_steps,
        "convergence_threshold": convergence_threshold,
        "convergence_start": convergence_start,
        "use_spectral_norm": use_spectral_norm,
    }

    kw_beta = kwargs.get("beta")
    if isinstance(kw_beta, float | int):
        overrides["beta"] = kw_beta

    kw_max_steps = kwargs.get("max_steps")
    if isinstance(kw_max_steps, int):
        overrides["max_steps"] = kw_max_steps

    kw_threshold = kwargs.get("convergence_threshold")
    if isinstance(kw_threshold, float | int):
        overrides["convergence_threshold"] = float(kw_threshold)

    kw_start = kwargs.get("convergence_start")
    if isinstance(kw_start, int):
        overrides["convergence_start"] = kw_start

    config = ModelConfig(
        name=spec.name,
        input_dim=input_dim if input_dim is not None else 0,
        output_dim=output_dim,
        hidden_dims=compute_hidden_dims(hidden_dim, num_layers),
        extra=kwargs,
    )
    # Apply overrides after construction (frozen — use object.__setattr__).
    for field_name, value in overrides.items():
        if value is not None:
            object.__setattr__(config, field_name, value)

    return config


def load_config(cls: type, path: str | Path) -> Any:
    """Load a frozen dataclass config from a YAML file.

    Merges the YAML with ``cls`` (the frozen dataclass) so that defaults
    from the dataclass definition fill in missing keys, then converts
    to a runtime instance via :func:`OmegaConf.to_object`.

    Args:
        cls: The frozen dataclass config class (e.g. ``MyConfig``).
        path: Path to the YAML file.

    Returns:
        Instance of *cls* with values from the YAML merged over dataclass
        defaults.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    yaml_cfg = OmegaConf.load(p)
    merged = OmegaConf.merge(cls, yaml_cfg)
    obj = OmegaConf.to_object(merged)
    if not isinstance(obj, cls):
        return obj
    return obj


def save_config(obj: Any, path: str | Path) -> None:
    """Save a dataclass config instance to a YAML file.

    Args:
        obj: A dataclass instance (frozen or mutable).
        path: Destination YAML file path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.structured(obj), str(p))


def config_to_dict(obj: Any) -> dict[str, Any]:
    """Convert a dataclass config to a plain dict, omitting ``None`` values.

    Like :meth:`~bioplausible.core.metrics.BaseMetrics.to_dict`, this
    strips ``None`` entries so the result is JSON-serialisable and
    losslessly reconstructable.
    """
    return {k: v for k, v in asdict(obj).items() if v is not None}


# ──────────────────────────────────────────────
# Experiment configuration (standardized bases)
# ──────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ExperimentConfig(BaseConfig):
    """Base configuration for reproducible experiment tracking.

    Standardized on :class:`BaseConfig` pattern (REFACTOR.md §1).
    Fields common to all experiment configs: name, seed, device.
    Domain-specific fields should be added in subclasses.
    """

    description: str = ""
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ReproducibilityConfig(BaseConfig):
    """Configuration for reproducibility tracking (core.utils.reproducibility).

    Extends :class:`BaseConfig` with experiment-specific config dicts.
    """

    model_config: dict[str, object] = field(default_factory=dict)
    training_config: dict[str, object] = field(default_factory=dict)
    data_config: dict[str, object] = field(default_factory=dict)
    hardware_config: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExperimentRunnerConfig(BaseConfig):
    """Configuration for experiment runner (experiments.utils).

    Extends :class:`BaseConfig` with model/optimizer/training parameters.
    """

    model_name: str = ""
    optimizer_name: str = ""
    model_params: dict[str, object] = field(default_factory=dict)
    optimizer_params: dict[str, object] = field(default_factory=dict)
    epochs: int = 10
    batches_per_epoch: int = 100
    eval_batches: int = 20
    track_metrics: bool = True
    verbose: bool = True
