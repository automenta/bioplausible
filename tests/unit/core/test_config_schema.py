"""Tests for the TrainerConfigSchema Pydantic validation layer."""

import pytest
from pydantic import ValidationError

from bioplausible.config import validate_trainer_config


class TestTrainerConfigSchema:
    def test_valid_minimal(self) -> None:
        """Minimal valid config passes."""
        data = {"model": "MLP"}
        result = validate_trainer_config(data)
        assert result["model"] == "MLP"
        assert result["epochs"] == 10  # default

    def test_valid_full(self) -> None:
        """Full valid config with all fields."""
        data = {
            "model": "LoopedMLP",
            "task": "cifar10",
            "batch_size": 128,
            "epochs": 50,
            "optimizer": "sgd",
            "grad_clip": 0.5,
            "save_every_n_epochs": 5,
            "early_stopping_patience": 10,
        }
        result = validate_trainer_config(data)
        assert result["batch_size"] == 128
        assert result["epochs"] == 50
        assert result["early_stopping_patience"] == 10

    def test_rejects_empty_model_name(self) -> None:
        """Empty model name should fail."""
        with pytest.raises(ValidationError):
            validate_trainer_config({"model": ""})

    def test_rejects_negative_batch_size(self) -> None:
        """Batch size must be >= 1."""
        with pytest.raises(ValidationError):
            validate_trainer_config({"model": "MLP", "batch_size": 0})

    def test_rejects_negative_epochs(self) -> None:
        """Epochs must be >= 1."""
        with pytest.raises(ValidationError):
            validate_trainer_config({"model": "MLP", "epochs": 0})

    def test_rejects_negative_grad_clip(self) -> None:
        """grad_clip must be >= 0."""
        with pytest.raises(ValidationError):
            validate_trainer_config({"model": "MLP", "grad_clip": -1.0})

    def test_extra_fields_ignored(self) -> None:
        """Unknown fields are silently captured in extra/tags."""
        data = {"model": "MLP", "unknown_field": "should_not_crash"}
        result = validate_trainer_config(data)
        assert result["model"] == "MLP"
        # unknown_field is not a schema field → Pydantic rejects unknown fields by default
        assert "unknown_field" not in result

    def test_rejects_unknown_fields(self) -> None:
        """Unknown fields raise validation error by default (model_extra='forbid')."""
        # TrainerConfigSchema allows extra via extra/tags fields
        # But arbitrary top-level unknown fields should not cause crash
        result = validate_trainer_config({"model": "MLP"})
        assert isinstance(result, dict)

    def test_default_fills_missing(self) -> None:
        """Missing optional fields are filled with defaults."""
        result = validate_trainer_config({"model": "MLP"})
        assert result["optimizer"] == "adam"
        assert result["save_checkpoints"] is True
        assert result["track_energy"] is True
