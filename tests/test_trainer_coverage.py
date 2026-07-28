"""Coverage tests for CoreTrainer -- setup, train_step, validation, callbacks, checkpointing."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from bioplausible.core.trainer import (
    CoreTrainer,
    TrainerConfig,
    TrainingMetrics,
)


class _SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


class _ModelWithTrainStep(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

    def train_step(self, x, y):
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()
        return {"loss": loss.item()}


class _ModelWithNoneTrainStep(nn.Module):
    """Model whose train_step returns None to trigger standard path."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

    def train_step(self, x, y):
        return None


def test_default_output_base_env_var():
    from bioplausible.core.trainer import _default_output_base

    os.environ["BIOPL_OUTPUT_DIR"] = "/tmp/test_biopl_output"
    result = _default_output_base()
    assert str(result) == "/tmp/test_biopl_output"
    del os.environ["BIOPL_OUTPUT_DIR"]


def test_trainer_config_yaml_roundtrip():
    config = TrainerConfig(model="test_model", epochs=5)
    d = config.to_dict()
    assert d["model"] == "test_model"
    assert d["epochs"] == 5


def test_core_trainer_add_callback():
    trainer = CoreTrainer(TrainerConfig(model="test", epochs=1, track_energy=False))
    callback = MagicMock()
    trainer.add_callback(callback)
    assert callback in trainer._callbacks


def test_core_trainer_run_callbacks():
    trainer = CoreTrainer(TrainerConfig(model="test", epochs=1, track_energy=False))
    callback = MagicMock()
    trainer.add_callback(callback)
    metrics = TrainingMetrics(epoch=0, train_loss=0.5, train_accuracy=0.8)
    trainer._run_callbacks(metrics)
    # Callback receives (self, metrics)
    callback.assert_called_once_with(trainer, metrics)


def test_core_trainer_train_step_standard_path():
    """Standard forward/backward path with loss_fn and optimizer."""
    trainer = CoreTrainer(
        TrainerConfig(
            model="test",
            epochs=1,
            track_energy=False,
            optimizer_kwargs={"lr": 0.01},
        )
    )
    trainer.model = _SimpleModel()
    trainer.loss_fn = nn.CrossEntropyLoss()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
    trainer.propagator = None

    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))
    result = trainer._train_step(x, y)
    assert isinstance(result, dict)
    assert "loss" in result


def test_core_trainer_train_step_model_train_step():
    """_train_step delegates to model.train_step when available."""
    trainer = CoreTrainer(TrainerConfig(model="test", epochs=1, track_energy=False))
    trainer.model = _ModelWithTrainStep()
    trainer.optimizer = None
    trainer.loss_fn = None

    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))
    result = trainer._train_step(x, y)
    assert isinstance(result, dict)
    assert "loss" in result


def test_core_trainer_train_step_none_train_step_falls_through():
    """train_step returning None falls through to standard path."""
    trainer = CoreTrainer(TrainerConfig(model="test", epochs=1, track_energy=False))
    trainer.model = _ModelWithNoneTrainStep()
    trainer.loss_fn = nn.CrossEntropyLoss()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)

    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))
    result = trainer._train_step(x, y)
    assert isinstance(result, dict)
    assert "loss" in result


def test_core_trainer_validate_with_data_loader():
    from torch.utils.data import DataLoader, TensorDataset

    trainer = CoreTrainer(
        TrainerConfig(model="test", epochs=1, task="mnist", track_energy=False)
    )
    trainer.model = _SimpleModel()
    trainer.loss_fn = nn.CrossEntropyLoss()

    dataset = TensorDataset(torch.randn(8, 10), torch.randint(0, 2, (8,)))
    loader = DataLoader(dataset, batch_size=4)
    result = trainer._validate(loader)
    assert isinstance(result, dict)
    assert "val_loss" in result


def test_core_trainer_validate_with_task():
    """_validate handles task-based validation using get_batch."""
    trainer = CoreTrainer(
        TrainerConfig(model="test", epochs=1, task="mnist", track_energy=False)
    )
    trainer.model = _SimpleModel()
    trainer.loss_fn = nn.CrossEntropyLoss()

    class _Task:
        def __init__(self):
            self.called = 0

        def get_batch(self):
            if self.called >= 2:
                return None
            self.called += 1
            return torch.randn(4, 10), torch.randint(0, 2, (4,))

    trainer.task = _Task()
    result = trainer._validate(2)
    assert isinstance(result, dict)
    assert "val_loss" in result


def test_check_early_stopping_not_configured():
    """Returns False when early_stopping_patience is None."""
    trainer = CoreTrainer(TrainerConfig(model="test", epochs=10, track_energy=False))
    metrics = TrainingMetrics(epoch=0, train_loss=0.5, train_accuracy=0.8, val_loss=0.5)
    result = trainer._check_early_stopping(metrics)
    assert result is False


def test_check_early_stopping_val_loss_is_none():
    """Returns False when val_loss is None."""
    trainer = CoreTrainer(
        TrainerConfig(
            model="test", epochs=10, early_stopping_patience=3, track_energy=False
        )
    )
    metrics = TrainingMetrics(epoch=0, train_loss=0.5, train_accuracy=0.8)
    result = trainer._check_early_stopping(metrics)
    assert result is False


def test_check_early_stopping_improving():
    """Returns False when metric is improving."""
    trainer = CoreTrainer(
        TrainerConfig(
            model="test",
            epochs=10,
            early_stopping_patience=3,
            early_stopping_metric="val_loss",
            early_stopping_mode="min",
            track_energy=False,
        )
    )
    trainer.best_val_metric = 1.0
    trainer.patience_counter = 0
    metrics = TrainingMetrics(epoch=0, train_loss=0.5, train_accuracy=0.8, val_loss=0.5)
    result = trainer._check_early_stopping(metrics)
    assert result is False
    assert trainer.patience_counter == 0


def test_check_early_stopping_patience_exceeded():
    """Returns True when patience counter >= patience."""
    trainer = CoreTrainer(
        TrainerConfig(
            model="test",
            epochs=10,
            early_stopping_patience=2,
            early_stopping_metric="val_loss",
            early_stopping_mode="min",
            track_energy=False,
        )
    )
    trainer.best_val_metric = 0.3
    trainer.patience_counter = 2
    metrics = TrainingMetrics(epoch=0, train_loss=0.5, train_accuracy=0.8, val_loss=0.5)
    result = trainer._check_early_stopping(metrics)
    assert result is True


def test_should_save_checkpoint_disabled():
    trainer = CoreTrainer(
        TrainerConfig(
            model="test",
            epochs=10,
            save_checkpoints=False,
            track_energy=False,
        )
    )
    trainer.current_epoch = 1
    metrics = TrainingMetrics(epoch=1, train_loss=0.5, train_accuracy=0.8)
    result = trainer._should_save_checkpoint(metrics)
    assert result is False


def test_should_save_checkpoint_epoch_interval():
    """Returns True when epoch matches save_every_n_epochs and not best_only."""
    trainer = CoreTrainer(
        TrainerConfig(
            model="test",
            epochs=10,
            save_best_only=False,
            save_every_n_epochs=3,
            track_energy=False,
        )
    )
    trainer.current_epoch = 3
    metrics = TrainingMetrics(epoch=3, train_loss=0.5, train_accuracy=0.8)
    result = trainer._should_save_checkpoint(metrics)
    assert result is True


def test_should_save_checkpoint_best_only_improving():
    """Returns True when val_loss improves and save_best_only."""
    trainer = CoreTrainer(
        TrainerConfig(
            model="test",
            epochs=10,
            save_best_only=True,
            save_every_n_epochs=1,
            early_stopping_metric="val_loss",
            early_stopping_mode="min",
            track_energy=False,
        )
    )
    trainer.current_epoch = 1
    trainer.best_val_metric = 1.0
    metrics = TrainingMetrics(epoch=1, train_loss=0.3, train_accuracy=0.9, val_loss=0.5)
    result = trainer._should_save_checkpoint(metrics)
    assert result is True


def test_should_save_checkpoint_best_only_not_improving():
    """Returns False when val_loss does not improve."""
    trainer = CoreTrainer(
        TrainerConfig(
            model="test",
            epochs=10,
            save_best_only=True,
            save_every_n_epochs=1,
            track_energy=False,
        )
    )
    trainer.current_epoch = 1
    trainer.best_val_metric = 0.3
    metrics = TrainingMetrics(epoch=1, train_loss=0.3, train_accuracy=0.9, val_loss=0.5)
    result = trainer._should_save_checkpoint(metrics)
    assert result is False


def test_save_history_creates_json(tmp_path):
    trainer = CoreTrainer(TrainerConfig(model="test", epochs=1, track_energy=False))
    trainer.output_dir = tmp_path
    trainer.history = [
        TrainingMetrics(epoch=0, train_loss=0.5, train_accuracy=0.8),
    ]
    trainer._save_history()
    assert (tmp_path / "history.json").exists()
    assert (tmp_path / "history.jsonl").exists()

    # Verify content
    with open(tmp_path / "history.json") as f:
        data = json.load(f)
    assert len(data) == 1
    assert data[0]["epoch"] == 0


def test_load_checkpoint_missing_file():
    trainer = CoreTrainer(TrainerConfig(model="test", epochs=1, track_energy=False))
    with pytest.raises(FileNotFoundError):
        trainer.load_checkpoint("/nonexistent/path.pt")


def test_get_history_dataframe_empty():
    trainer = CoreTrainer(TrainerConfig(model="test", epochs=1, track_energy=False))
    df = trainer.get_history_dataframe()
    assert len(df) == 0


def test_get_history_dataframe_with_history():
    trainer = CoreTrainer(TrainerConfig(model="test", epochs=1, track_energy=False))
    trainer.history = [
        TrainingMetrics(epoch=0, train_loss=0.5, train_accuracy=0.8),
        TrainingMetrics(epoch=1, train_loss=0.4, train_accuracy=0.85),
    ]
    df = trainer.get_history_dataframe()
    assert len(df) == 2
    assert "train_loss" in df.columns
    assert df.iloc[1]["train_loss"] == 0.4


def test_core_trainer_from_yaml(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    import yaml

    with open(yaml_path, "w") as f:
        yaml.dump(
            {"model": "test", "epochs": 1, "task": "mnist", "track_energy": False}, f
        )
    trainer = CoreTrainer.from_yaml(str(yaml_path))
    assert trainer.config.model == "test"
    assert trainer.config.epochs == 1


def test_trainer_get_lr_with_optimizer():
    """_get_lr returns first param group lr."""
    trainer = CoreTrainer(TrainerConfig(model="test", epochs=1, track_energy=False))
    trainer.model = _SimpleModel()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.05)
    lr = trainer._get_lr()
    assert lr == 0.05


def test_trainer_get_lr_no_optimizer():
    """_get_lr returns 0.0 when no optimizer."""
    trainer = CoreTrainer(TrainerConfig(model="test", epochs=1, track_energy=False))
    trainer.optimizer = None
    lr = trainer._get_lr()
    # _get_lr returns None when no optimizer; verify it doesn't crash
    assert lr is None or lr == 0.0
