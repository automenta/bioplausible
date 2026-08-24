"""Tests for the REFACTOR5 experiment cache layer."""

import numpy as np
import torch
from torch import nn

from computronium.core._caching import DatasetCache, ModelCache, _stable_hash


def test_dataset_cache_roundtrip_and_eviction():
    cache = DatasetCache(max_entries=2)
    key = cache.key("mnist", {"augment": True}, 64, 0, 1, "cpu")
    obj = object()
    cache.put(key, obj)
    assert cache.get(key) is obj
    # LRU: add two more, evicting the first.
    k2 = cache.key("mnist", {"augment": True}, 128, 0, 1, "cpu")
    k3 = cache.key("fmnist", {"augment": False}, 64, 0, 1, "cpu")
    cache.put(k2, object())
    cache.put(k3, object())
    assert cache.get(key) is None


def test_dataset_cache_key_is_order_independent():
    cache = DatasetCache()
    a = cache.key("mnist", {"augment": True, "fraction": 0.5}, 64, 0, 1, "cpu")
    b = cache.key("mnist", {"fraction": 0.5, "augment": True}, 64, 0, 1, "cpu")
    assert a == b
    c = cache.key("mnist", {"augment": True, "fraction": 0.5}, 64, 0, 2, "cpu")
    assert a != c


def test_model_cache_returns_fresh_params():
    cache = ModelCache()
    key = cache.key("mlp", {"hidden_dim": 8}, 4, 2, "cpu")

    model = nn.Linear(4, 2)
    with torch.no_grad():
        model.weight.fill_(0.5)
    cache.put(key, model)

    first = cache.get(key)
    second = cache.get(key)
    assert first is not None and second is not None
    assert first is not second
    assert first.weight.allclose(torch.full_like(first.weight, 0.5))
    # Mutating one copy must not affect the template or other copies.
    with torch.no_grad():
        first.weight.fill_(1.0)
    assert second.weight.allclose(torch.full_like(second.weight, 0.5))


def test_model_cache_lru_eviction():
    cache = ModelCache(max_entries=2)
    k1 = cache.key("m", {"h": 1}, 1, 1, "cpu")
    k2 = cache.key("m", {"h": 2}, 1, 1, "cpu")
    k3 = cache.key("m", {"h": 3}, 1, 1, "cpu")
    cache.put(k1, nn.Linear(1, 1))
    cache.put(k2, nn.Linear(1, 1))
    cache.get(k1)  # touch k1 -> k2 becomes LRU
    cache.put(k3, nn.Linear(1, 1))
    assert cache.get(k2) is None
    assert cache.get(k1) is not None


def test_stable_hash_handles_non_json_and_nested():
    # Nested dicts + lists are order-independent and hash alike.
    a = _stable_hash({"cfg": {"b": 1, "a": [1, 2, 3]}, "n": 5})
    b = _stable_hash({"n": 5, "cfg": {"a": [1, 2, 3], "b": 1}})
    assert a == b
    # Non-JSON-serializable values degrade to a stable repr-based token.
    c = _stable_hash({"t": torch.zeros(2, 2)})
    d = _stable_hash({"t": torch.zeros(2, 2)})
    assert c == d
    # Distinguishing types must not collide even with equal repr.
    e = _stable_hash({"v": {"a": 1}})
    f = _stable_hash({"v": {"a": 1.0}})
    assert e != f
    # NumPy scalars/arrays are supported.
    g = _stable_hash({"arr": np.array([1, 2])})
    assert g == _stable_hash({"arr": np.array([1, 2])})
