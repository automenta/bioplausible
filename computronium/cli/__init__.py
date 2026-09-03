"""Bioplausible CLI Tools.

Lazy (Sprint 0.5): this package previously eagerly imported ``cli.__main__``
(→ rank → analysis → hyperopt → execution), a chain with a latent circular
import (``execution.engine`` imports ``from computronium.hyperopt import
PatientLevel``). That chain only survived because the old eager top-level
``computronium/__init__.py`` happened to pre-import ``hyperopt``/``execution``
first. With lazy top-level imports that pre-warming is gone, so the console
scripts now expose their entry points on demand instead.
"""

_LAZY: dict[str, tuple[str, str | None]] = {  # ruff: ignore[non-empty-init-module]
    "main": ("computronium.cli.__main__", "main"),
    "run_benchmark": ("computronium.cli.run", "run_benchmark"),
    "run_core_train": ("computronium.cli.run", "run_core_train"),
    "run_from_yaml": ("computronium.cli.run", "run_from_yaml"),
    "run_search": ("computronium.cli.run", "run_search"),
    "run_training": ("computronium.cli.run", "run_training"),
}

__all__ = sorted(_LAZY)  # ruff: ignore[invalid-all-format]


def __getattr__(name: str) -> object:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = _LAZY[name]
    module = __import__(module_name, fromlist=[attr] if attr else ["*"])
    value: object = module if attr is None else getattr(module, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
