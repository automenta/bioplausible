"""Tests for the CoreTrainer."""

import pytest

from bioplausible.core.trainer import (
    CoreTrainer,
    TrainerConfig,
    TrainingMetrics,
)


def test_trainer_config_defaults():
    """Test TrainerConfig default values."""
    config = TrainerConfig(model="test_model")
    assert config.epochs == 10
    assert config.batch_size == 64
    assert config.optimizer == "adam"
    assert config.task == "mnist"
    assert config.device == "auto"
    assert config.track_energy is True


def test_trainer_config_from_dict():
    """Test creating TrainerConfig from dict."""
    config = TrainerConfig.from_dict({
        "model": "test_model",
        "epochs": 20,
        "batch_size": 128,
        "optimizer": "sgd",
    })
    assert config.model == "test_model"
    assert config.epochs == 20
    assert config.batch_size == 128
    assert config.optimizer == "sgd"


def test_training_metrics():
    """Test TrainingMetrics creation and serialization."""
    metrics = TrainingMetrics(
        epoch=0,
        train_loss=0.5,
        train_accuracy=0.8,
        val_loss=0.4,
        val_accuracy=0.85,
        epoch_time=1.0,
    )
    assert metrics.train_loss == pytest.approx(0.5)
    assert metrics.val_accuracy == pytest.approx(0.85)

    d = metrics.to_dict()
    assert d["epoch"] == 0
    assert d["train_loss"] == pytest.approx(0.5)
    assert d["val_accuracy"] == pytest.approx(0.85)


def test_training_metrics_partial():
    """Test TrainingMetrics with only required fields."""
    metrics = TrainingMetrics(epoch=0, train_loss=0.5, train_accuracy=0.0)
    assert metrics.val_loss is None
    assert metrics.energy_proxy is None


def test_trainer_config_to_dict():
    """Test TrainerConfig serialization."""
    config = TrainerConfig(model="test_model")
    d = config.to_dict()
    assert d["model"] == "test_model"
    assert d["epochs"] == 10


def test_core_trainer_from_dict():
    """Test creating CoreTrainer from dict."""
    trainer = CoreTrainer.from_dict({
        "model": "test",
        "epochs": 1,
        "task": "mnist",
        "track_energy": False,
    })
    assert trainer.config.model == "test"
    assert trainer.config.epochs == 1


def test_core_trainer_initialization():
    """Test CoreTrainer initialization."""
    config = TrainerConfig(
        model="test",
        epochs=1,
        track_energy=False,
        task="mnist",
    )
    trainer = CoreTrainer(config)
    assert trainer is not None
    assert trainer.config.model == "test"
    assert trainer.config.epochs == 1


REGISTERED_PROPAGATORS = [
    "backprop",
    "feedback_alignment",
    "direct_fa",
    "adaptive_fa",
    "stochastic_fa",
    "contrastive_fa",
    "eq_prop",
    "adam_eq_prop",
    "holomorphic_eq_prop",
    "finite_nudge_eq_prop",
    "lazy_eq_prop",
    "contrastive_hebbian_learning",
]


@pytest.mark.parametrize("propagator", REGISTERED_PROPAGATORS)
def test_propagator_constructs_with_correct_signature(propagator):
    """All registered propagators are LearningRuleOptimizer `(params, model, ...)`.

    Regression for the CoreTrainer `_create_propagator` bug that bound the
    *model* to the `params` positional arg (`prop_cls(self.model, ...)`), which
    raised a TypeError for the entire propagator family.
    """
    config = TrainerConfig(
        model="backprop_mlp",
        propagator=propagator,
        task="digits",
        model_kwargs={"input_dim": 64, "hidden_dim": 16, "output_dim": 10},
        epochs=1,
    )
    trainer = CoreTrainer(config)
    trainer.setup()
    assert trainer.propagator is not None
    assert trainer.propagator.model is trainer.model


def test_feedback_alignment_propagator_trains():
    """FA as a propagator (not just the self-training FA *model*) fits end-to-end."""
    config = TrainerConfig(
        model="backprop_mlp",
        propagator="feedback_alignment",
        task="digits",
        model_kwargs={"input_dim": 64, "hidden_dim": 16, "output_dim": 10},
        epochs=1,
    )
    history = CoreTrainer(config).fit()
    assert len(history) == 1
    assert history[-1].val_accuracy > 0.0
