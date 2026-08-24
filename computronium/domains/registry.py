"""Cross-domain task registry (architecture §5, §8).

Single source of task *names* for scheduling; geometry is derived from the
concrete :class:`DomainTask` each name maps to (via the domain factory), never
hardcoded here. This is the right home for the registry because it spans every
domain — vision, language, RL, tabular — not just vision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from computronium.config.unified import DataConfig
    from computronium.domains.base import DomainTask

__all__ = [
    "SUPPORTED_TASKS",
    "TaskSpec",
    "resolve_task",
    "resolve_task_from_data_config",
]

# Network-fetching tasks (cifar100/svhn, the graph datasets) are excluded:
# geometry resolution is offline, and architecture §11 defers the full Phase-0
# breadth. The MLP-parity tier only schedules the offline-resolvable set.
SUPPORTED_TASKS: frozenset[str] = frozenset({
    # vision (incl. toy boolean/toy-classification datasets)
    "xor",
    "spiral",
    "circles",
    "digits",
    "mnist",
    "fashion_mnist",
    "kmnist",
    "usps",
    "cifar10",
    "cifar100",
    "svhn",
    # language
    "tiny_shakespeare",
    "char_ngram",
    "wikitext2",
    "penn_treebank",
    # rl
    "pendulum",
    "acrobot",
    "cartpole",
    "mountain_car",
    "lunar_lander",
    # graph
    "cora",
    "citeseer",
    "pubmed",
    # tabular
    "breast_cancer",
    "iris",
    "wine",
    "diabetes",
    "california_housing",
})


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Resolved geometry facts for a task (single source of truth).

    ``input_dim`` is the *flattened* MLP input size; ``output_dim`` the class
    count. Values are derived from the concrete ``DomainTask`` the name maps to
    (via the domain factory), never hardcoded here.
    """

    name: str
    input_dim: int
    output_dim: int


def resolve_task(name: str) -> TaskSpec:
    """Resolve a task name to its concrete geometry.

    Builds the actual ``DomainTask`` via the domain factory and reads its own
    ``input_dim``/``output_dim`` (flattening spatial input shapes), so geometry
    always matches the task's real data — covering every domain, not just
    vision.

    Args:
        name: Task name (e.g. ``"mnist"``, ``"cifar10"``, ``"usps"``).

    Returns:
        The :class:`TaskSpec` derived from the concrete task.

    Raises:
        ValueError: If ``name`` is not a known task.
    """
    if name not in SUPPORTED_TASKS:
        raise ValueError(  # ruff: ignore[raise-vanilla-args]  # descriptive message is the public API
            f"unknown task {name!r}; available: {sorted(SUPPORTED_TASKS)}"
        )
    from computronium.domains.factory import create_task

    task = create_task(name, device="cpu", quick_mode=True)
    task.setup()
    input_dim = task.input_dim
    if isinstance(input_dim, (tuple, list)):
        input_dim = math.prod(int(d) for d in input_dim)
    return TaskSpec(
        name=name, input_dim=int(input_dim), output_dim=int(task.output_dim)
    )


def resolve_task_from_data_config(
    config: DataConfig, device: str = "cpu"
) -> DomainTask:
    """Resolve a :class:`DataConfig` to a concrete :class:`DomainTask`.

    This is the single canonical resolution path for all task/geometry
    derivation. It replaces the scattered ``create_task``/``resolve_task``/
    ``_setup_data``/``_get_train_loader`` calls.

    Args:
        config: Data configuration specifying the task and loading parameters.
        device: Target device for the task.

    Returns:
        A fully set up :class:`DomainTask` with data loaders ready.
    """
    from computronium.domains.factory import create_task

    if config.task not in SUPPORTED_TASKS:
        raise ValueError(
            f"unknown task {config.task!r}; available: {sorted(SUPPORTED_TASKS)}"
        )

    task = create_task(
        config.task, device=device, quick_mode=False, **config.data_kwargs
    )
    task.batch_size = config.batch_size
    task.setup()
    return task
