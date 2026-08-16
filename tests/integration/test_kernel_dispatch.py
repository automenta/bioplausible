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


def test_kernel_backend_consumed_in_train_step():
    """A uniform-interface backend attached by ``use_kernel=True`` is actually
    driven during training (forward → backward → update_weights), so the
    ``dispatch_train_step`` kernel seam is not a dead path.

    Uses ``standard_fa`` (plain ``nn.Linear`` layer stack) so the generic
    consumer binds via ``set_model_ref`` and updates the live weights.
    """
    import torch

    from bioplausible.acceleration import (
        AlgorithmFamily,
        HardwareTarget,
        KernelConfig,
        KernelRegistry,
    )

    config = TrainerConfig(
        model="standard_fa",
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
    assert backend is not None

    before = [p.clone() for p in trainer.model.layers[0].parameters()]
    x = torch.randn(8, 64)
    y = torch.randint(0, 10, (8,))
    metrics = trainer._train_step(x, y)
    assert "loss" in metrics
    assert torch.isfinite(torch.tensor(metrics["loss"]))

    after = list(trainer.model.layers[0].parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after)), (
        "kernel train step did not update the model weights"
    )
    assert trainer._training_path_counts.get("kernel", 0) >= 1


def test_kernel_backend_matches_reference_learning():
    """The kernel-driven FA path learns to match the PyTorch reference.

    ``standard_fa`` trained through ``_train_step`` with ``use_kernel=True``
    must reach the same accuracy as the same model trained without the kernel
    (the backend's fixed feedback weights are shared with the model). This
    guards the consumption seam against silent parity drift.
    """
    import torch

    from bioplausible.acceleration import get_algorithm_kernels

    get_algorithm_kernels()

    torch.manual_seed(42)
    n, dim, classes = 400, 64, 10
    x = torch.randn(n, dim)
    y = torch.randint(0, classes, (n,))
    for c in range(classes):
        mask = y == c
        if mask.any():
            direction = torch.randn(dim)
            direction = direction / direction.norm() * 2.0
            x[mask] += direction * 0.8

    def _final_acc(use_kernel: bool) -> float:
        cfg = TrainerConfig(
            model="standard_fa",
            task="digits",
            model_kwargs={
                "input_dim": dim,
                "hidden_dim": 64,
                "output_dim": classes,
                "num_layers": 2,
            },
            epochs=1,
            use_kernel=use_kernel,
            kernel_backend="cpu",
            track_energy=False,
        )
        trainer = CoreTrainer(cfg)
        trainer.setup()
        dev = next(trainer.model.parameters()).device
        xd, yd = x.to(dev), y.to(dev)
        for _ in range(8):
            perm = torch.randperm(n)
            for i in range(0, n, 32):
                idx = perm[i : i + 32]
                trainer._train_step(xd[idx], yd[idx])
        trainer.model.eval()
        with torch.no_grad():
            return (trainer.model(xd).argmax(1) == yd).float().mean().item()

    kernel_acc = _final_acc(True)
    ref_acc = _final_acc(False)
    assert abs(kernel_acc - ref_acc) <= 0.01, (
        f"kernel acc={kernel_acc:.3f} vs reference acc={ref_acc:.3f} "
        f"drifted beyond 1% parity"
    )
