"""Tests for the unified config hierarchy (REFACTOR.md §1.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from bioplausible.config.unified import (
    BaseConfig,
    BaseStructuredConfig,
    config_to_dict,
    load_config,
    save_config,
)

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class _SampleConfig(BaseConfig):
    """Minimal config for round-trip tests."""

    lr: float = 0.01
    hidden_dims: list[int] = field(default_factory=lambda: [32, 64])


@dataclass(frozen=True, slots=True)
class _OverrideConfig(BaseConfig):
    """Config that overrides an inherited field default."""

    device: str = "cuda"
    extra: str = "val"


class TestBaseConfig:
    def test_defaults(self) -> None:
        cfg = BaseConfig()
        assert cfg.name == "default"
        assert cfg.seed == 42
        assert cfg.device == "auto"

    def test_frozen(self) -> None:
        cfg = BaseConfig(name="test")
        with pytest.raises(AttributeError):
            cfg.name = "other"

    def test_slots(self) -> None:
        cfg = BaseConfig()
        with pytest.raises(AttributeError):
            cfg.new_attr = "x"


class TestConfigToDict:
    def test_omits_none(self) -> None:
        @dataclass(frozen=True, slots=True)
        class Cfg(BaseConfig):
            val: float | None = None

        d = config_to_dict(Cfg(val=None))
        assert "val" not in d
        assert "name" in d

    def test_includes_nested(self) -> None:
        d = config_to_dict(_SampleConfig(name="n", lr=0.05))
        assert d["hidden_dims"] == [32, 64]


class TestYamlRoundTrip:
    def test_round_trip_full(self, tmp_path: Path) -> None:
        cfg = _SampleConfig(name="test", seed=7, lr=0.05)
        path = tmp_path / "cfg.yaml"
        save_config(cfg, path)
        loaded = load_config(_SampleConfig, path)
        assert isinstance(loaded, _SampleConfig)
        assert loaded.name == "test"
        assert loaded.seed == 7
        assert loaded.lr == 0.05
        assert loaded.hidden_dims == [32, 64]

    def test_round_trip_partial_yaml(self, tmp_path: Path) -> None:
        """YAML with only some keys filled; defaults from dataclass apply."""
        cfg = _SampleConfig(name="partial", lr=0.1)
        path = tmp_path / "partial.yaml"
        save_config(cfg, path)
        loaded = load_config(_SampleConfig, path)
        assert loaded.name == "partial"
        assert loaded.lr == 0.1
        assert loaded.seed == 42  # default from dataclass

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(_SampleConfig, tmp_path / "nope.yaml")


class TestBaseStructuredConfig:
    def test_to_internal(self) -> None:
        s = BaseStructuredConfig(name="x", seed=99, device="cpu")
        rt = s.to_internal()
        assert isinstance(rt, BaseConfig)
        assert rt.name == "x"
        assert rt.seed == 99
        assert rt.device == "cpu"


class TestChildOverride:
    def test_overrides_parent_default(self) -> None:
        """Child can override a parent field default (e.g. device → cuda)."""
        cfg = _OverrideConfig()
        assert cfg.device == "cuda"
        assert cfg.name == "default"
        assert cfg.seed == 42

    def test_isinstance_base(self) -> None:
        cfg = _OverrideConfig()
        assert isinstance(cfg, BaseConfig)
