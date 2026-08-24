"""
Domains Package

Domain abstraction layer with standard interfaces for vision, LM, RL, graph,
tabular, time series, and scientific simulation.

Also re-exports the merged task factory (``create_task``) and training
utilities (``_TaskTrainer``, ``TaskProtocol``) that originated in
``hyperopt/tasks.py``.
"""

from computronium.domains.base import (
    Batch,
    DomainSpec,
    DomainTask,
    DomainType,
    Metrics,
    TaskSplit,
)
from computronium.domains.factory import CharNGramTask, create_task
from computronium.domains.graph import GraphTask
from computronium.domains.lm import LMTask
from computronium.domains.registry import SUPPORTED_TASKS, TaskSpec, resolve_task
from computronium.domains.rl import RLTask
from computronium.domains.scientific import ScientificTask
from computronium.domains.tabular import TabularTask
from computronium.domains.timeseries import TimeSeriesTask
from computronium.domains.trainer import TaskProtocol, _resolve_task_loss, _TaskTrainer
from computronium.domains.vision import VisionTask

# Registry for domain tasks
_DOMAIN_REGISTRY = {
    "vision": VisionTask,
    "lm": LMTask,
    "rl": RLTask,
    "graph": GraphTask,
    "tabular": TabularTask,
    "timeseries": TimeSeriesTask,
    "scientific": ScientificTask,
}


def create_domain_task(domain: str, name: str, **kwargs) -> DomainTask:
    """Create a domain task by name."""
    if domain not in _DOMAIN_REGISTRY:
        raise ValueError(
            f"Unknown domain: {domain}. Available: {list(_DOMAIN_REGISTRY.keys())}"
        )

    task_class = _DOMAIN_REGISTRY[domain]
    return task_class(name=name, **kwargs)


def register_domain_task(domain: str, task_class: type) -> None:
    """Register a new domain task class."""
    _DOMAIN_REGISTRY[domain] = task_class


def list_domains() -> list:
    """List available domains."""
    return list(_DOMAIN_REGISTRY.keys())


__all__ = [
    # Base classes
    "DomainTask",
    "DomainType",
    "DomainSpec",
    "TaskSplit",
    "Batch",
    "Metrics",
    # Concrete tasks
    "VisionTask",
    "LMTask",
    "RLTask",
    "GraphTask",
    "TabularTask",
    "TimeSeriesTask",
    "ScientificTask",
    "CharNGramTask",
    # Task registry
    "SUPPORTED_TASKS",
    "TaskSpec",
    "resolve_task",
    # Factory
    "create_domain_task",
    "register_domain_task",
    "list_domains",
    "create_task",
    # Training utilities
    "TaskProtocol",
    "_TaskTrainer",
    "_resolve_task_loss",
]
