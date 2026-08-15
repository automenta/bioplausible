"""
Default experiment configurations for common scenarios.

Named presets registered here extend the schema-level defaults returned by
:func:`bioplausible.config.omegaconf.get_default_config`. External code can
register additional presets by calling :func:`register_default_config` at
import time (e.g., in a plugin or site-customization module) and then look
them up by name via :func:`get_named_config`.

The :data:`DEFAULT_CONFIGS` dict remains exported for back-compat inspection
but callers should prefer the accessors below so the registry can evolve
without breaking direct-dict manipulation.
"""

from omegaconf import OmegaConf

from bioplausible.config.omegaconf import ExperimentSchemaConfig
from bioplausible.core.logging import get_logger

__all__ = [
    "DEFAULT_CONFIGS",
    "get_named_config",
    "list_named_configs",
    "register_default_config",
]
DEFAULT_CONFIGS: dict[str, ExperimentSchemaConfig] = {}

_logger = get_logger()


def _make_writable(node) -> None:
    """Recursively make an OmegaConf config writable."""
    if not OmegaConf.is_config(node):
        return
    node._set_flag("readonly", False)
    node._set_flag("struct", False)
    for k in node:
        _make_writable(node[k])


def register_default_config(name: str, overrides: dict) -> None:
    """Register a named experiment preset by merging overrides into the base.

    Re-registering an existing ``name`` overwrites the previous entry and
    emits a warning, so plugins can override built-ins.

    Raises:
        ValueError: If ``overrides`` is not a mapping or produces an
            invalid ``ExperimentConfig`` (e.g., unknown field).
    """
    if not isinstance(overrides, dict):
        raise ValueError(f"overrides must be a dict, got {type(overrides).__name__}")
    if name in DEFAULT_CONFIGS:
        _logger.warning("Overwriting default config preset %r", name)
    base = OmegaConf.structured(ExperimentSchemaConfig)
    _make_writable(base)
    merged = OmegaConf.merge(base, OmegaConf.create(overrides))
    DEFAULT_CONFIGS[name] = OmegaConf.to_object(merged)


def _register_default(name: str, overrides: dict) -> None:
    """Back-compat alias for :func:`register_default_config`."""
    register_default_config(name, overrides)


def get_named_config(name: str) -> ExperimentSchemaConfig:
    """Look up a registered named preset.

    Returns a deep copy: mutating the returned object does not affect the
    registry entry. Raises ``KeyError`` with the list of available presets
    if ``name`` is unknown (so callers see discoverable options in the
    traceback).
    """
    try:
        cfg = DEFAULT_CONFIGS[name]
    except KeyError as e:
        available = ", ".join(sorted(DEFAULT_CONFIGS)) or "<none>"
        raise KeyError(
            f"No default config preset named {name!r}. Available: {available}"
        ) from e
    return OmegaConf.to_object(OmegaConf.structured(cfg))


def list_named_configs() -> list[str]:
    """Return the preset names registered so far, sorted alphabetically."""
    return sorted(DEFAULT_CONFIGS)


# ---- Vision benchmarks ----

_register_default(
    "vision_mlp",
    {
        "model": {"name": "MLP", "kwargs": {"hidden_dim": 256, "num_layers": 3}},
        "optimizer": {"name": "adam", "lr": 0.001},
        "dataset": {"name": "mnist", "batch_size": 64},
        "trainer": {"epochs": 10},
    },
)

_register_default(
    "vision_eqprop",
    {
        "model": {"name": "EqPropMLP", "kwargs": {"hidden_dim": 256, "num_layers": 3}},
        "optimizer": {"name": "adam", "lr": 0.01},
        "dataset": {"name": "mnist", "batch_size": 64},
        "trainer": {"epochs": 10},
    },
)

_register_default(
    "vision_ff",
    {
        "model": {
            "name": "ForwardForwardNet",
            "kwargs": {"hidden_dim": 256, "num_layers": 3},
        },
        "optimizer": {"name": "adam", "lr": 0.01},
        "dataset": {"name": "mnist", "batch_size": 64},
        "trainer": {"epochs": 10},
    },
)

_register_default(
    "vision_equitile",
    {
        "model": {
            "name": "tile_pc",
            "kwargs": {"neurons_per_tile": 48, "tiles_per_layer": 4},
        },
        "optimizer": {"name": "adam", "lr": 0.01},
        "dataset": {"name": "mnist", "batch_size": 64},
        "trainer": {"epochs": 10},
    },
)

# ---- MEP benchmarks ----

_register_default(
    "vision_mep_smep",
    {
        "model": {"name": "MLP", "kwargs": {"hidden_dim": 256, "num_layers": 3}},
        "propagator": {"name": "smep", "kwargs": {"beta": 0.5}},
        "optimizer": {"name": "adam", "lr": 0.01},
        "dataset": {"name": "mnist", "batch_size": 64},
        "trainer": {"epochs": 10},
    },
)

# ---- LM benchmarks ----

_register_default(
    "lm_mlp",
    {
        "model": {"name": "MLP", "kwargs": {"hidden_dim": 512, "num_layers": 4}},
        "optimizer": {"name": "adamw", "lr": 0.0003},
        "dataset": {"name": "tiny_shakespeare", "batch_size": 32},
        "domain": {"domain": "lm"},
        "trainer": {"epochs": 20},
    },
)

# ---- Ablation configs ----

_register_default(
    "ablation_quick",
    {
        "model": {"name": "MLP", "kwargs": {"hidden_dim": 128, "num_layers": 2}},
        "optimizer": {"name": "sgd", "lr": 0.01},
        "dataset": {"name": "digits", "batch_size": 32},
        "trainer": {"epochs": 3, "batches_per_epoch": 50},
        "track_energy": True,
    },
)
