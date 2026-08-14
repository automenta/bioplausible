"""
Execution Engine: Autonomous Discovery Execution Engine.

This module manages continuous experiment execution for research automation.

Key Components:
    - ExecutionEngine: Main agent class running the discovery loop
    - ExecutionStrategy: Decides what experiments to run next
    - ExperimentState: Tracks progress and historical results
    - ExperimentTask: Individual experiment specification
    - DecisionLogger: Records scientific decisions

Usage:
    from bioplausible.execution import ExecutionEngine

    engine = ExecutionEngine(db_path="artifacts/bioplausible.db")
    engine.run()  # Start continuous discovery
"""

# Lazy package init (Sprint 0.5 module boundary). The old eager ``engine`` /
# ``strategy`` imports created a genuine circular import:
#   bioplausible.hyperopt → execution._guards → execution/__init__ → engine →
#   strategy → `from bioplausible.hyperopt import PatientLevel`  (hyperopt only
#   partially initialized → ImportError).
# It was masked only when the old eager top-level `bioplausible/__init__.py`
# happened to import `hyperopt`/`execution` in a working order first. Lazily
# exposing the symbols means importing (say) `execution.callbacks` — all the
# `core.trainer` needs — no longer drags in `execution.engine`/`strategy` →
# `hyperopt` → `zoo`, which also slims the `core` import graph.

_LAZY: dict[str, tuple[str, str | None]] = {
    "BaseExecutionCallback": (
        "bioplausible.execution.callbacks",
        "BaseExecutionCallback",
    ),
    "DecisionLogger": ("bioplausible.execution._state", "DecisionLogger"),
    "ExecutionCallback": ("bioplausible.execution.callbacks", "ExecutionCallback"),
    "ExecutionEngine": ("bioplausible.execution.engine", "ExecutionEngine"),
    "ExecutionStrategy": ("bioplausible.execution.strategy", "ExecutionStrategy"),
    "ExperimentState": ("bioplausible.execution._state", "ExperimentState"),
    "ExperimentTask": ("bioplausible.execution.task", "ExperimentTask"),
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
