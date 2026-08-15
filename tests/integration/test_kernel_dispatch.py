"""Kernel dispatch integration (REFACTOR7 Phase 2-9 opt-in wiring).

Verifies that :meth:`CoreTrainer._wrap_with_kernel` attaches a ``KernelBackend``
instance to the model when ``use_kernel=True`` and the model's family has a
registered backend, and that the plain (kernel-off) path leaves the model
untouched. This exercises the trainer-level dispatch seam that the per-family
parity suites validate at the backend level.
"""

from __future__ import annotations

import pytest

from bioplausible.acceleration import get_algorithm_kernels
from bioplausible.core.trainer import CoreTrainer, TrainerConfig


@pytest.fixture(scope="module", autouse=True)
def _populate_kernel_registry():
    get_algorithm_kernels()
    yield


@pytest.mark.parametrize(
    ("model_name", "family"),
    [
        ("feedback_alignment", "fa"),
        ("backprop_mlp", "backprop"),
    ],
)
def test_kernel_dispatch_attaches_backend(model_name, family):
    """``use_kernel=True`` wraps the model with the family's kernel backend."""
    config = TrainerConfig(
        model=model_name,
        task="digits",
        model_kwargs={
            "input_dim": 64,
            "hidden_dim": 16,
            "output_dim": 10,
            "num_layers": 2,
        },
        epochs=1,
        use_kernel=True,
        kernel_backend="cpu",
        track_energy=False,
    )
    trainer = CoreTrainer(config)
    trainer.setup()
    backend = getattr(trainer.model, "_kernel_backend", None)
    assert backend is not None, (
        f"{model_name} (family {family}) was not wrapped with a kernel backend"
    )
    assert backend._config is not None
    assert backend.name == family


@pytest.mark.parametrize("model_name", ["feedback_alignment", "backprop_mlp"])
def test_kernel_off_leaves_model_untouched(model_name):
    """Default ``use_kernel=False`` must not attach a kernel backend."""
    config = TrainerConfig(
        model=model_name,
        task="digits",
        model_kwargs={
            "input_dim": 64,
            "hidden_dim": 16,
            "output_dim": 10,
            "num_layers": 2,
        },
        epochs=1,
        use_kernel=False,
        track_energy=False,
    )
    trainer = CoreTrainer(config)
    trainer.setup()
    assert getattr(trainer.model, "_kernel_backend", None) is None
