"""Tests for named default-config registry (bioplausible.config.defaults)."""

import pytest

from bioplausible.config import (
    get_named_config,
    list_named_configs,
    register_default_config,
)


def test_list_named_configs_includes_builtins():
    names = list_named_configs()
    assert isinstance(names, list)
    assert len(names) >= 7
    # Built-ins present
    assert "vision_mlp" in names
    assert "vision_eqprop" in names
    assert "lm_mlp" in names
    assert "ablation_quick" in names
    # Sorted
    assert names == sorted(names)


def test_get_named_config_returns_copy():
    cfg1 = get_named_config("vision_mlp")
    cfg2 = get_named_config("vision_mlp")
    assert cfg1 is not cfg2  # deep copy, not same object
    assert cfg1.model.name == "MLP"
    assert cfg1.trainer.epochs == 10


def test_unknown_name_raises_keyerror_with_available_list():
    with pytest.raises(KeyError) as exc:
        get_named_config("nonexistent_xyz")
    msg = str(exc.value)
    assert "nonexistent_xyz" in msg
    assert "Available" in msg
    assert "vision_mlp" in msg  # lists actual presets


def test_register_default_config_creates_new_preset():
    before = set(list_named_configs())
    register_default_config(
        "test_new_preset",
        {
            "model": {"name": "MLP", "kwargs": {"hidden_dim": 32}},
            "trainer": {"epochs": 2},
        },
    )
    after = set(list_named_configs())
    assert after == before | {"test_new_preset"}

    cfg = get_named_config("test_new_preset")
    assert cfg.model.name == "MLP"
    assert cfg.model.kwargs["hidden_dim"] == 32
    assert cfg.trainer.epochs == 2


def test_register_default_config_overwrites_with_warning(caplog):
    caplog.set_level("WARNING")
    register_default_config(
        "test_overwrite",
        {"model": {"name": "MLP", "kwargs": {"hidden_dim": 1}}},
    )
    # Overwrite
    register_default_config(
        "test_overwrite",
        {"model": {"name": "MLP", "kwargs": {"hidden_dim": 999}}},
    )
    assert any(
        "Overwriting default config preset 'test_overwrite'" in r.message
        for r in caplog.records
    )
    cfg = get_named_config("test_overwrite")
    assert cfg.model.kwargs["hidden_dim"] == 999


def test_register_default_config_rejects_non_dict():
    with pytest.raises(ValueError, match="overrides must be a dict"):
        register_default_config("bad", "not a dict")
    with pytest.raises(ValueError, match="overrides must be a dict"):
        register_default_config("bad", 123)


def test_default_configs_dict_is_read_only_by_convention():
    # The module exports the dict but mutation doesn't persist through accessors.
    # This documents the intended usage: callers use get_named_config / register_*.
    cfg = get_named_config("vision_mlp")
    cfg.model.name = "Mutated"
    fresh = get_named_config("vision_mlp")
    assert fresh.model.name == "MLP"  # original unaffected
