"""Bioplausible CLI Tools."""

# Re-exported for the ``eqprop-verify`` console-script entry point
# (pyproject.toml: bioplausible.cli:main).
from bioplausible.cli.__main__ import main as main
from bioplausible.cli.lab import inspect_model
from bioplausible.cli.run import (
    list_models,
    run_benchmark,
    run_core_train,
    run_from_yaml,
    run_search,
    run_training,
)

__all__ = [
    "inspect_model",
    "list_models",
    "main",
    "run_benchmark",
    "run_core_train",
    "run_from_yaml",
    "run_search",
    "run_training",
]
