"""Bioplausible CLI Tools.

Lazy (Sprint 0.5): this package previously eagerly imported ``cli.__main__``
(→ rank → analysis → hyperopt → execution), a chain with a latent circular
import (``execution.engine`` imports ``from bioplausible.hyperopt import
PatientLevel``). That chain only survived because the old eager top-level
``bioplausible/__init__.py`` happened to pre-import ``hyperopt``/``execution``
first. With lazy top-level imports that pre-warming is gone, so the console
scripts now expose their entry points on demand instead.
"""

_LAZY: dict[str, tuple[str, str | None]] = {
    "main": ("bioplausible.cli.__main__", "main"),
    "inspect_model": ("bioplausible.cli.lab", "inspect_model"),
    "list_models": ("bioplausible.cli.run", "list_models"),
    "run_benchmark": ("bioplausible.cli.run", "run_benchmark"),
    "run_core_train": ("bioplausible.cli.run", "run_core_train"),
    "run_from_yaml": ("bioplausible.cli.run", "run_from_yaml"),
    "run_search": ("bioplausible.cli.run", "run_search"),
    "run_training": ("bioplausible.cli.run", "run_training"),
}

__all__ = sorted(_LAZY)


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
