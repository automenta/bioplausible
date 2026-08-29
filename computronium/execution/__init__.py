"""
Execution Engine: Autonomous Discovery Execution Engine.

This module manages continuous experiment execution for research automation.

Key Components:
    - ExecutionEngine: Main agent class running the discovery loop
    - ExecutionStrategy: Decides what experiments to run next
    - ExperimentState: Tracks progress and historical results
    - ExperimentTask: Individual experiment specification
    - DecisionLogger: Records scientific decisions
    - CandidateGenerator: Generates experiment candidates
    - Task weights, criteria, and lifecycle management

Usage:
    from computronium.execution import ExecutionEngine

    engine = ExecutionEngine(db_path="artifacts/computronium.db")
    engine.run()  # Start continuous discovery
"""

# Lazy package init (Sprint 0.5 module boundary). The old eager ``engine`` /
# ``strategy`` imports created a genuine circular import:
#   computronium.hyperopt → execution._guards → execution/__init__ → engine →
#   strategy → `from computronium.hyperopt import PatientLevel`  (hyperopt only
#   partially initialized → ImportError).
# It was masked only when the old eager top-level `computronium/__init__.py`
# happened to import `hyperopt`/`execution` in a working order first. Lazily
# exposing the symbols means importing (say) `execution.callbacks` — all the
# `core.trainer` needs — no longer drags in `execution.engine`/`strategy` →
# `hyperopt` → `zoo`, which also slims the `core` import graph.

_LAZY: dict[str, tuple[str, str | None]] = {
    "BaseExecutionCallback": (
        "computronium.execution.callbacks",
        "BaseExecutionCallback",
    ),
    "CandidateGenerator": (
        "computronium.execution.candidate_gen",
        "CandidateGenerator",
    ),
    "DecisionLogger": ("computronium.execution._state", "DecisionLogger"),
    "ExecutionCallback": ("computronium.execution.callbacks", "ExecutionCallback"),
    "ExecutionEngine": ("computronium.execution.engine", "ExecutionEngine"),
    "ExecutionStrategy": ("computronium.execution.lifecycle", "ExecutionStrategy"),
    "ExperimentState": ("computronium.execution._state", "ExperimentState"),
    "ExperimentTask": ("computronium.execution.task", "ExperimentTask"),
    "ExecutionStrategyConfig": (
        "computronium.execution.candidate_gen",
        "ExecutionStrategyConfig",
    ),
    "check_criterion": ("computronium.execution.criteria", "check_criterion"),
    "CRITERIA": ("computronium.execution.criteria", "CRITERIA"),
    "TASK_WEIGHTS": ("computronium.execution.task_weights", "TASK_WEIGHTS"),
    "TASK_GROUPS": ("computronium.execution.task_weights", "TASK_GROUPS"),
    "TIER_ORDER": ("computronium.execution.task_weights", "TIER_ORDER"),
    "calculate_complexity_penalty": (
        "computronium.execution.task_weights",
        "calculate_complexity_penalty",
    ),
    "calculate_future_boost": (
        "computronium.execution.task_weights",
        "calculate_future_boost",
    ),
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
