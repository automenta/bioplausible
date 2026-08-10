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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

__all__ = [
    "BaseConfig",
    "BaseStructuredConfig",
    "DeviceStr",
    "config_to_dict",
    "load_config",
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
