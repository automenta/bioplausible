"""CoreTrainer + deployment-model integration (Pillar A).

Verifies the single training path: a substrate deployment model (``conv_equitile``)
is constructed via the canonical ``construct_model`` funnel, receives spatial input
(``input_format="spatial"``), and trains through ``CoreTrainer``'s Phase-3
model-side ``train_step`` dispatch.
"""

import pytest

import bioplausible.zoo  # ruff: ignore[unused-import]  # registration side effect
from bioplausible.core.trainer import CoreTrainer, TrainerConfig

_TILE_KWARGS = {
    "conv_channels": [4, 8],
    "neurons_per_tile": 16,
    "tiles_per_layer": 1,
    "num_fc_layers": 1,
    "learning_rate": 1e-3,
}


def _trainer_config(**overrides: object) -> TrainerConfig:
    """Build a minimal one-epoch CoreTrainer config for a deployment model."""
    kwargs = {
        "model": "conv_equitile",
        "task": "mnist",
        "model_kwargs": dict(_TILE_KWARGS),
        "epochs": 1,
        "batch_size": 16,
        "batches_per_epoch": 3,
        "track_energy": False,
        "track_flops": False,
        "track_memory": False,
        "save_checkpoints": False,
        "run_validation": False,
    }
    kwargs.update(overrides)
    return TrainerConfig(**kwargs)


def test_deployment_model_constructs_via_coretrainer() -> None:
    """ConvEquiTile is constructed through the single construction funnel."""
    trainer = CoreTrainer(_trainer_config())
    trainer.setup()

    assert trainer.model is not None
    assert trainer.model.__class__.__name__ == "ConvEquiTile"
    assert trainer.model.config.mode == "pc"
    assert hasattr(trainer.model, "train_step")


def test_deployment_model_receives_spatial_input() -> None:
    """Spatial (tuple) task geometry threads to the model, which declares ``input_format``."""
    trainer = CoreTrainer(_trainer_config())
    trainer.setup()

    # input_dim tuple (1, 28, 28) must be preserved, not int()-coerced.
    assert trainer.config.model_kwargs["input_dim"] == (1, 28, 28)
    assert trainer.config.model_kwargs["output_dim"] == 10
    assert getattr(trainer.model, "input_format", "flat") == "spatial"


def test_deployment_model_trains_through_coretrainer() -> None:
    """A full training epoch runs through CoreTrainer's model-side train_step."""
    trainer = CoreTrainer(_trainer_config())
    history = trainer.fit()

    assert len(history) == 1
    assert history[-1].train_loss > 0.0
    # Phase-3 dispatch: the model's own train_step must be the recorded path.
    paths = history[-1].extra.get("training_paths", {})
    assert paths.get("model_train_step", 0) >= 1


def test_deployment_model_build_accepts_tuple_input_dim() -> None:
    """ConvEquiTile.build flattens spatial tuples via math.prod (Pillar C contract)."""
    from bioplausible.core.construction import construct_model
    from bioplausible.core.registry import ComponentCategory, Registry

    model_cls = Registry.get(ComponentCategory.MODEL, "conv_equitile")
    model = construct_model(
        model_cls,
        dict(_TILE_KWARGS),
        input_dim=(1, 28, 28),
        output_dim=10,
        model_name="conv_equitile",
    )

    assert model is not None
    assert model.config.input_channels == 1
    assert model.config.input_size == 28
    assert model.config.num_classes == 10


def test_deployment_model_spatial_forward_via_adapt_input() -> None:
    """CoreTrainer._adapt_input preserves 4D tensors for spatial models."""
    import torch

    trainer = CoreTrainer(_trainer_config())
    trainer.setup()

    x = torch.randn(4, 1, 28, 28)
    adapted = trainer._adapt_input(x)
    assert adapted.shape == (4, 1, 28, 28)

    with pytest.raises(RuntimeError):
        # Flattened input to a spatial model must fail loudly.
        trainer._adapt_input(x.view(4, -1))
        trainer.model(x.view(4, -1))
