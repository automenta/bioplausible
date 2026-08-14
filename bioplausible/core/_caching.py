"""Reusable experiment caches (REFACTOR5 EXPERIMENT CACHING).

These live in ``core`` (L1) so both ``CoreTrainer`` and the L5/L6 orchestration
layers can consume them without an upward import. They are strictly optional:
``CoreTrainer`` consults them only when provided (default ``None`` preserves the
current per-probe construction behavior). Caches are opt-in and thread-safe
(no GIL reliance per PEP 703).
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable


def _stable_hash(obj: object) -> str:
    """Order-independent SHA256 of a JSON-serializable object."""
    canonical = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class DatasetCache:
    """Cache resolved domain task objects to skip dataset (re)construction.

    Keys on the data-relevant config; values are resolved task objects whose
    dataset is already materialized, so re-deriving fresh ``DataLoader``
    instances on a hit is cheap. LRU-bounded to cap dataset memory.
    """

    def __init__(self, max_entries: int = 16) -> None:
        self._max = max_entries
        self._items: OrderedDict[tuple[object, ...], object] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def key(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]  # the key must capture every dataset-affecting dim
        task: str,
        data_kwargs: dict[str, object],
        batch_size: int,
        num_workers: int,
        seed: int,
        device: str,
    ) -> tuple[object, ...]:
        """Build the cache key from dataset-affecting inputs."""
        return (
            task,
            _stable_hash(data_kwargs),
            batch_size,
            num_workers,
            seed,
            device,
        )

    def get(self, key: tuple[object, ...]) -> object | None:
        with self._lock:
            return self._items.get(key)

    def put(self, key: tuple[object, ...], task_obj: object) -> None:
        with self._lock:
            self._items.pop(key, None)
            self._items[key] = task_obj
            while len(self._items) > self._max:
                self._items.popitem(last=False)


class ModelCache:
    """Cache constructed model templates to skip per-probe re-instantiation.

    Keys on the model name and a config hash; values are CPU templates. Hits
    return a fresh ``deepcopy`` so every probe gets independent parameters
    (state dicts are never shared). LRU-bounded to cap resident model memory.
    """

    def __init__(
        self,
        max_entries: int = 32,
        copy_fn: Callable[[nn.Module], nn.Module] = copy.deepcopy,
    ) -> None:
        self._max = max_entries
        self._copy = copy_fn
        self._items: OrderedDict[tuple[object, ...], nn.Module] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def key(  # the key must capture every model-affecting dim
        model_name: str,
        config: dict[str, object],
        input_dim: int | tuple[int, ...],
        output_dim: int,
        device: str,
    ) -> tuple[object, ...]:
        """Build the cache key from model-affecting inputs."""
        return (
            model_name,
            _stable_hash(config),
            repr(input_dim),
            output_dim,
            device,
        )

    def get(self, key: tuple[object, ...]) -> nn.Module | None:
        with self._lock:
            template = self._items.get(key)
            if template is None:
                return None
            self._items.move_to_end(key)
            return self._copy(template)

    def put(self, key: tuple[object, ...], model: nn.Module) -> None:
        # Store a detached CPU template so hits are device-agnostic.
        template = model.to(torch.device("cpu"))
        with self._lock:
            self._items.pop(key, None)
            self._items[key] = template
            while len(self._items) > self._max:
                self._items.popitem(last=False)
